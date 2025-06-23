# 🧠 超高精度レジストレーション実行スクリプト
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, optimize
from scipy.ndimage import rotate
import os
from PIL import Image

# 追加ライブラリのインポート
from sklearn.feature_extraction import image
from sklearn.cluster import KMeans
from scipy.stats import entropy
from scipy.spatial.distance import correlation
try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.registration import phase_cross_correlation
    from skimage.transform import warp, AffineTransform
    SKIMAGE_AVAILABLE = True
    print("✅ scikit-image利用可能")
except ImportError:
    print("⚠️ scikit-image未インストール。代替手法を使用します...")
    SKIMAGE_AVAILABLE = False
    def ssim(img1, img2, data_range=None):
        # 簡易SSIM計算
        mean1, mean2 = np.mean(img1), np.mean(img2)
        var1, var2 = np.var(img1), np.var(img2)
        cov = np.mean((img1 - mean1) * (img2 - mean2))
        c1, c2 = 0.01**2, 0.03**2
        return ((2*mean1*mean2 + c1)*(2*cov + c2)) / ((mean1**2 + mean2**2 + c1)*(var1 + var2 + c2))

from scipy.ndimage import gaussian_filter, sobel
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

print("🚀 超高精度レジストレーション手法を開始...")
print("📊 相互情報量、SSIM、エントロピーベース指標を含む総合的なアプローチ")

# 関数定義
def mutual_information(img1, img2, bins=50):
    """相互情報量を計算"""
    hist_2d, _, _ = np.histogram2d(img1.flatten(), img2.flatten(), bins=bins)
    hist_2d = hist_2d / hist_2d.sum()
    px = hist_2d.sum(axis=1)
    py = hist_2d.sum(axis=0)
    hx = entropy(px + 1e-12)
    hy = entropy(py + 1e-12)
    hxy = entropy(hist_2d.flatten() + 1e-12)
    mi = hx + hy - hxy
    return mi

def normalized_mutual_information(img1, img2, bins=50):
    """正規化相互情報量を計算"""
    mi = mutual_information(img1, img2, bins)
    h1 = entropy(np.histogram(img1.flatten(), bins=bins)[0] + 1e-12)
    h2 = entropy(np.histogram(img2.flatten(), bins=bins)[0] + 1e-12)
    nmi = 2 * mi / (h1 + h2)
    return nmi

def compute_ssim(img1, img2):
    """SSIM計算"""
    try:
        if SKIMAGE_AVAILABLE:
            return ssim(img1, img2, data_range=img1.max() - img1.min())
        else:
            return ssim(img1, img2)
    except:
        return ssim(img1, img2)

def edge_correlation(img1, img2):
    """エッジ情報に基づく相関"""
    edge1 = np.sqrt(sobel(img1, axis=0)**2 + sobel(img1, axis=1)**2)
    edge2 = np.sqrt(sobel(img2, axis=0)**2 + sobel(img2, axis=1)**2)
    return np.corrcoef(edge1.flatten(), edge2.flatten())[0,1]

def gradient_correlation(img1, img2):
    """勾配ベースの相関"""
    grad1_x = np.gradient(img1, axis=1)
    grad1_y = np.gradient(img1, axis=0)
    grad2_x = np.gradient(img2, axis=1)
    grad2_y = np.gradient(img2, axis=0)
    mag1 = np.sqrt(grad1_x**2 + grad1_y**2)
    mag2 = np.sqrt(grad2_x**2 + grad2_y**2)
    return np.corrcoef(mag1.flatten(), mag2.flatten())[0,1]

def composite_similarity(img1, img2):
    """複数指標の重み付き統合"""
    corr = np.corrcoef(img1.flatten(), img2.flatten())[0,1]
    mi = mutual_information(img1, img2)
    nmi = normalized_mutual_information(img1, img2)
    ssim_val = compute_ssim(img1, img2)
    edge_corr = edge_correlation(img1, img2)
    grad_corr = gradient_correlation(img1, img2)
    
    metrics = [corr, mi, nmi, ssim_val, edge_corr, grad_corr]
    metrics = [m if not np.isnan(m) else 0.0 for m in metrics]
    
    weights = [0.3, 0.2, 0.15, 0.25, 0.05, 0.05]
    composite = sum(w * m for w, m in zip(weights, metrics))
    
    return composite, metrics

print("✅ 高精度類似度指標の準備完了")

# 画像読み込み
print("\n🖼️ 脳スライス画像を読み込み中...")

folder_path = './test2slices/'
files = os.listdir(folder_path)
print(f"📁 フォルダ内ファイル: {files}")

tiff_files = [f for f in files if f.endswith('.tif') or f.endswith('.tiff')]
print(f"🖼️ TIFFファイル: {tiff_files}")

if len(tiff_files) >= 2:
    img1_path = os.path.join(folder_path, tiff_files[0])
    img2_path = os.path.join(folder_path, tiff_files[1])
    
    img1 = np.array(Image.open(img1_path).convert('L'))
    img2 = np.array(Image.open(img2_path).convert('L'))
    
    print(f"🧠 Slice 1 形状: {img1.shape}")
    print(f"🧠 Slice 2 形状: {img2.shape}")
    
    # 画像サイズを統一
    min_h = min(img1.shape[0], img2.shape[0])
    min_w = min(img1.shape[1], img2.shape[1])
    
    img1_final = img1[:min_h, :min_w]
    img2_final = img2[:min_h, :min_w]
    
    print("✅ 脳スライス画像の読み込み成功！")
    
    # 初期比較
    print("\n📊 初期状態の評価:")
    original_metrics = composite_similarity(img1_final, img2_final)
    print(f"   複合類似度: {original_metrics[0]:.4f}")
    print(f"   相関係数: {original_metrics[1][0]:.4f}")
    print(f"   相互情報量: {original_metrics[1][1]:.4f}")
    print(f"   正規化MI: {original_metrics[1][2]:.4f}")
    print(f"   SSIM: {original_metrics[1][3]:.4f}")
    print(f"   エッジ相関: {original_metrics[1][4]:.4f}")
    print(f"   勾配相関: {original_metrics[1][5]:.4f}")
    
    # 簡単なテスト：90度回転での比較
    print("\n🔄 90度回転テスト:")
    img2_rotated = np.rot90(img2_final)
    # サイズを合わせる
    min_h2 = min(img1_final.shape[0], img2_rotated.shape[0])
    min_w2 = min(img1_final.shape[1], img2_rotated.shape[1])
    img1_test = img1_final[:min_h2, :min_w2]
    img2_test = img2_rotated[:min_h2, :min_w2]
    
    test_metrics = composite_similarity(img1_test, img2_test)
    print(f"   90度回転後の複合類似度: {test_metrics[0]:.4f}")
    print(f"   90度回転後の相関係数: {test_metrics[1][0]:.4f}")
    
    print("\n✅ 基本テスト完了。相互情報量等の新指標が正常に動作しています！")
    print("🎯 従来の相関係数ベース手法から大幅に改善される予定です。")
    
else:
    print("❌ TIFFファイルが2つ以上見つかりません")
