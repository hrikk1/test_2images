# 🧠 超高精度レジストレーション - 相互情報量と最新手法
print("🚀 超高精度レジストレーション手法を開始...")
print("📊 相互情報量、SSIM、エントロピーベース指標を含む総合的なアプローチ")

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, optimize
from scipy.ndimage import rotate
import SimpleITK as sitk
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

# 相互情報量の計算関数
def mutual_information(img1, img2, bins=50):
    """相互情報量を計算"""
    # ヒストグラムの計算
    hist_2d, _, _ = np.histogram2d(img1.flatten(), img2.flatten(), bins=bins)
    
    # 正規化
    hist_2d = hist_2d / hist_2d.sum()
    
    # 周辺分布
    px = hist_2d.sum(axis=1)
    py = hist_2d.sum(axis=0)
    
    # エントロピーの計算
    hx = entropy(px + 1e-12)
    hy = entropy(py + 1e-12)
    hxy = entropy(hist_2d.flatten() + 1e-12)
    
    # 相互情報量
    mi = hx + hy - hxy
    return mi

# 正規化相互情報量
def normalized_mutual_information(img1, img2, bins=50):
    """正規化相互情報量を計算"""
    mi = mutual_information(img1, img2, bins)
    
    # 個別エントロピー
    h1 = entropy(np.histogram(img1.flatten(), bins=bins)[0] + 1e-12)
    h2 = entropy(np.histogram(img2.flatten(), bins=bins)[0] + 1e-12)
    
    # 正規化
    nmi = 2 * mi / (h1 + h2)
    return nmi

# 構造的類似性指標 (SSIM)
def compute_ssim(img1, img2):
    """SSIM計算"""
    try:
        if SKIMAGE_AVAILABLE:
            return ssim(img1, img2, data_range=img1.max() - img1.min())
        else:
            return ssim(img1, img2)
    except:
        return ssim(img1, img2)

# エッジ強調相関
def edge_correlation(img1, img2):
    """エッジ情報に基づく相関"""
    # Sobelフィルタでエッジ抽出
    edge1 = np.sqrt(sobel(img1, axis=0)**2 + sobel(img1, axis=1)**2)
    edge2 = np.sqrt(sobel(img2, axis=0)**2 + sobel(img2, axis=1)**2)
    
    # エッジ相関
    return np.corrcoef(edge1.flatten(), edge2.flatten())[0,1]

# 勾配相関
def gradient_correlation(img1, img2):
    """勾配ベースの相関"""
    # 勾配計算
    grad1_x = np.gradient(img1, axis=1)
    grad1_y = np.gradient(img1, axis=0)
    grad2_x = np.gradient(img2, axis=1)
    grad2_y = np.gradient(img2, axis=0)
    
    # 勾配マグニチュード
    mag1 = np.sqrt(grad1_x**2 + grad1_y**2)
    mag2 = np.sqrt(grad2_x**2 + grad2_y**2)
    
    return np.corrcoef(mag1.flatten(), mag2.flatten())[0,1]

# 複合類似度指標
def composite_similarity(img1, img2):
    """複数指標の重み付き統合"""
    corr = np.corrcoef(img1.flatten(), img2.flatten())[0,1]
    mi = mutual_information(img1, img2)
    nmi = normalized_mutual_information(img1, img2)
    ssim_val = compute_ssim(img1, img2)
    edge_corr = edge_correlation(img1, img2)
    grad_corr = gradient_correlation(img1, img2)
    
    # NaN値を0に置換
    metrics = [corr, mi, nmi, ssim_val, edge_corr, grad_corr]
    metrics = [m if not np.isnan(m) else 0.0 for m in metrics]
    
    # 重み付き統合（相関、SSIM、相互情報量を重視）
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
    # 最初の2つのTIFFファイルを読み込み
    img1_path = os.path.join(folder_path, tiff_files[0])
    img2_path = os.path.join(folder_path, tiff_files[1])
    
    img1 = np.array(Image.open(img1_path).convert('L'))
    img2 = np.array(Image.open(img2_path).convert('L'))
    
    print(f"🧠 Slice 1 形状: {img1.shape}")
    print(f"🧠 Slice 2 形状: {img2.shape}")
    
    # 画像サイズを統一（小さい方に合わせる）
    min_h = min(img1.shape[0], img2.shape[0])
    min_w = min(img1.shape[1], img2.shape[1])
    
    img1_final = img1[:min_h, :min_w]
    img2_final = img2[:min_h, :min_w]
    
    print("✅ 脳スライス画像の読み込み成功！")
else:
    print("❌ TIFFファイルが2つ以上見つかりません")
    exit()

# 高精度変換と最適化関数
def apply_advanced_transform(img, params):
    """パラメータに基づく高精度画像変換"""
    angle, tx, ty, scale = params[:4]
    
    # 回転行列
    cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    
    # 変換行列（回転、スケール、平行移動）
    transform_matrix = np.array([
        [scale * cos_a, -scale * sin_a, tx],
        [scale * sin_a, scale * cos_a, ty],
        [0, 0, 1]
    ])
    
    # 画像中心での変換
    h, w = img.shape
    center = np.array([h/2, w/2])
    
    # 中心を原点とする変換
    if SKIMAGE_AVAILABLE:
        try:
            center_transform = AffineTransform(translation=-center)
            main_transform = AffineTransform(matrix=transform_matrix)
            back_transform = AffineTransform(translation=center)
            
            # 合成変換
            combined = (center_transform + main_transform + back_transform)
            
            # 変換適用
            transformed = warp(img, combined.inverse, output_shape=img.shape, preserve_range=True)
            return transformed.astype(img.dtype)
        except:
            pass
    
    # フォールバック: scipyを使用
    from scipy.ndimage import affine_transform
    
    # アフィン変換行列を作成
    matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    
    # 中心点調整
    center_offset = np.dot(matrix, center) - center + offset
    
    transformed = affine_transform(img, matrix, offset=center_offset, output_shape=img.shape)
    return transformed

# 目的関数（最大化したい類似度を負の値で返す）
def objective_function(params, img1, img2, method='composite'):
    """最適化目的関数"""
    try:
        transformed = apply_advanced_transform(img2, params)
        
        if method == 'correlation':
            score = np.corrcoef(img1.flatten(), transformed.flatten())[0,1]
        elif method == 'mutual_info':
            score = mutual_information(img1, transformed)
        elif method == 'nmi':
            score = normalized_mutual_information(img1, transformed)
        elif method == 'ssim':
            score = compute_ssim(img1, transformed)
        elif method == 'composite':
            score, _ = composite_similarity(img1, transformed)
        else:
            score = 0.0
            
        return -score if not np.isnan(score) else -0.0
    except Exception as e:
        return 0.0

# 高精度最適化レジストレーション
def advanced_registration(img1, img2, method='composite', max_iter=50):
    """高精度レジストレーション"""
    print(f"🔍 {method}最適化開始...")
    
    # パラメータ範囲 [angle, tx, ty, scale]
    bounds = [
        (-30, 30),      # 回転角度 (度)
        (-100, 100),    # X移動
        (-100, 100),    # Y移動  
        (0.8, 1.2),     # スケール
    ]
    
    # Differential Evolution最適化
    result = differential_evolution(
        objective_function,
        bounds,
        args=(img1, img2, method),
        maxiter=max_iter,
        popsize=15,
        seed=42,
        polish=True,
        atol=1e-6,
        tol=1e-6
    )
    
    # 最適変換の適用
    best_params = result.x
    best_transformed = apply_advanced_transform(img2, best_params)
    best_score = -result.fun
    
    return best_transformed, best_score, best_params

print("✅ 高精度最適化関数の準備完了")

print("実行準備完了！次のセルで高精度レジストレーションを実行します。")
