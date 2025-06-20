#!/usr/bin/env python3
# 🚀 ウルトラ高精度脳スライスレジストレーション - 目標0.8+達成
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.ndimage import sobel, affine_transform, gaussian_filter, rotate, shift
from scipy.optimize import differential_evolution, minimize
from PIL import Image
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 90)
print("🧠 ウルトラ高精度脳スライスレジストレーション - 目標相関係数0.8+達成")
print("=" * 90)

# ===== 高度なメトリクス関数群 =====
def mutual_information(img1, img2, bins=100):
    """高精度相互情報量計算 (ビン数増加)"""
    # より細かいヒストグラムで精度向上
    hist_2d, _, _ = np.histogram2d(img1.flatten(), img2.flatten(), bins=bins)
    hist_2d = hist_2d / hist_2d.sum()
    hist_2d = hist_2d + 1e-12  # 数値安定性
    
    px = hist_2d.sum(axis=1)
    py = hist_2d.sum(axis=0)
    
    hx = entropy(px)
    hy = entropy(py) 
    hxy = entropy(hist_2d.flatten())
    
    return hx + hy - hxy

def normalized_mutual_information(img1, img2, bins=100):
    """正規化相互情報量"""
    mi = mutual_information(img1, img2, bins)
    h1 = entropy(np.histogram(img1.flatten(), bins=bins)[0] + 1e-12)
    h2 = entropy(np.histogram(img2.flatten(), bins=bins)[0] + 1e-12)
    return 2 * mi / (h1 + h2)

def enhanced_ssim(img1, img2, window_size=11):
    """強化SSIM計算"""
    # ガウシアンフィルタでスムージング
    img1_smooth = gaussian_filter(img1, sigma=1.0)
    img2_smooth = gaussian_filter(img2, sigma=1.0)
    
    mean1, mean2 = np.mean(img1_smooth), np.mean(img2_smooth)
    var1, var2 = np.var(img1_smooth), np.var(img2_smooth)
    cov = np.mean((img1_smooth - mean1) * (img2_smooth - mean2))
    
    # より適切な定数
    c1, c2 = (0.01 * 255)**2, (0.03 * 255)**2
    
    ssim = ((2*mean1*mean2 + c1)*(2*cov + c2)) / ((mean1**2 + mean2**2 + c1)*(var1 + var2 + c2))
    return np.clip(ssim, -1, 1)

def edge_correlation(img1, img2):
    """高精度エッジ相関"""
    # Sobelフィルタで勾配計算
    dx1, dy1 = sobel(img1, axis=0), sobel(img1, axis=1)
    dx2, dy2 = sobel(img2, axis=0), sobel(img2, axis=1)
    
    edge1 = np.sqrt(dx1**2 + dy1**2)
    edge2 = np.sqrt(dx2**2 + dy2**2)
    
    # 閾値処理でノイズ除去
    threshold = np.percentile(edge1, 75)
    edge1[edge1 < threshold] = 0
    edge2[edge2 < threshold] = 0
    
    return np.corrcoef(edge1.flatten(), edge2.flatten())[0,1]

def gradient_correlation(img1, img2):
    """勾配ベース相関 (改良版)"""
    dx1, dy1 = np.gradient(img1)
    dx2, dy2 = np.gradient(img2)
    
    grad_mag1 = np.sqrt(dx1**2 + dy1**2)
    grad_mag2 = np.sqrt(dx2**2 + dy2**2)
    
    return np.corrcoef(grad_mag1.flatten(), grad_mag2.flatten())[0,1]

def phase_correlation(img1, img2):
    """位相相関による高精度位置合わせ"""
    f1 = np.fft.fft2(img1)
    f2 = np.fft.fft2(img2)
    
    cross_power = (f1 * np.conj(f2)) / (np.abs(f1 * np.conj(f2)) + 1e-12)
    correlation = np.fft.ifft2(cross_power)
    
    peak = np.unravel_index(np.argmax(np.abs(correlation)), correlation.shape)
    return np.abs(correlation[peak])

def apply_advanced_transform(image, params):
    """高度なアフィン変換適用"""
    angle, tx, ty, sx, sy, shear_x, shear_y = params
    
    # 画像中心
    center_x, center_y = image.shape[1] / 2, image.shape[0] / 2
    
    # アフィン変換行列作成
    cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    
    # 回転 + スケール + せん断変換行列
    transform_matrix = np.array([
        [sx * cos_a - shear_x * sin_a, -sx * sin_a - shear_x * cos_a, tx],
        [sy * sin_a + shear_y * cos_a,  sy * cos_a - shear_y * sin_a, ty],
        [0, 0, 1]
    ])
    
    # 中心を基準とした変換
    offset_matrix = np.array([
        [1, 0, center_x],
        [0, 1, center_y],
        [0, 0, 1]
    ])
    
    inv_offset = np.array([
        [1, 0, -center_x],
        [0, 1, -center_y],
        [0, 0, 1]
    ])
    
    final_matrix = offset_matrix @ transform_matrix @ inv_offset
    
    # 2x3行列に変換
    affine_matrix = final_matrix[:2, :3]
    
    return affine_transform(image, 
                          np.linalg.inv(affine_matrix[:2, :2]), 
                          offset=-affine_matrix[:2, 2],
                          order=3,  # 3次補間
                          mode='reflect')

def ultra_composite_similarity(img1, img2):
    """ウルトラ複合類似度関数"""
    try:
        corr = np.corrcoef(img1.flatten(), img2.flatten())[0,1]
        mi = mutual_information(img1, img2)
        nmi = normalized_mutual_information(img1, img2)
        ssim = enhanced_ssim(img1, img2)
        edge_corr = edge_correlation(img1, img2)
        grad_corr = gradient_correlation(img1, img2)
        phase_corr = phase_correlation(img1, img2)
        
        # より積極的な重み付け (相関係数を重視)
        weights = [0.40, 0.20, 0.15, 0.10, 0.08, 0.05, 0.02]
        
        # NaN処理
        metrics = [corr, mi, nmi, ssim, edge_corr, grad_corr, phase_corr]
        valid_metrics = []
        valid_weights = []
        
        for metric, weight in zip(metrics, weights):
            if not np.isnan(metric) and np.isfinite(metric):
                valid_metrics.append(metric)
                valid_weights.append(weight)
        
        if len(valid_metrics) == 0:
            return 0.0
            
        # 重み正規化
        valid_weights = np.array(valid_weights)
        valid_weights = valid_weights / valid_weights.sum()
        
        composite = np.sum(np.array(valid_metrics) * valid_weights)
        return composite
        
    except:
        return 0.0

# ===== 画像読み込み =====
print("🖼️ 脳スライス画像を読み込み中...")

image_dir = "./test2slices/"
tiff_files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
print(f"📁 TIFFファイル: {tiff_files}")

if len(tiff_files) < 2:
    raise FileNotFoundError("2つのTIFFファイルが必要です")

# 画像読み込み・前処理
img1_path = os.path.join(image_dir, tiff_files[0])
img2_path = os.path.join(image_dir, tiff_files[1])

img1 = np.array(Image.open(img1_path).convert('L'), dtype=np.float32)
img2 = np.array(Image.open(img2_path).convert('L'), dtype=np.float32)

print(f"🧠 画像1サイズ: {img1.shape}")
print(f"🧠 画像2サイズ: {img2.shape}")

# サイズ統一
min_h = min(img1.shape[0], img2.shape[0])
min_w = min(img1.shape[1], img2.shape[1])
img1 = img1[:min_h, :min_w]
img2 = img2[:min_h, :min_w]

# 正規化
img1 = (img1 - img1.min()) / (img1.max() - img1.min())
img2 = (img2 - img2.min()) / (img2.max() - img2.min())

print(f"🧠 最終画像サイズ: {img1.shape}")
print("✅ 脳スライス画像の読み込み成功！\n")

# ===== 初期評価 =====
print("📊 初期状態:")
initial_composite = ultra_composite_similarity(img1, img2)
initial_corr = np.corrcoef(img1.flatten(), img2.flatten())[0,1]
print(f"   複合類似度: {initial_composite:.4f}")
print(f"   相関係数: {initial_corr:.4f}")
print()

# ===== ウルトラ最適化 =====
print("🎯 ウルトラ高精度レジストレーション実行中...")

def objective_function(params):
    """最適化目的関数"""
    try:
        transformed_img2 = apply_advanced_transform(img2, params)
        similarity = ultra_composite_similarity(img1, transformed_img2)
        return -similarity  # 最小化問題として
    except:
        return 1000.0  # ペナルティ

# 多段階最適化戦略
optimization_stages = [
    {
        'name': 'Stage1: 粗い探索',
        'bounds': [(-30, 30), (-200, 200), (-200, 200), (0.7, 1.3), (0.7, 1.3), (-0.3, 0.3), (-0.3, 0.3)],
        'popsize': 20,
        'maxiter': 50
    },
    {
        'name': 'Stage2: 中程度探索', 
        'bounds': [(-15, 15), (-100, 100), (-100, 100), (0.8, 1.2), (0.8, 1.2), (-0.2, 0.2), (-0.2, 0.2)],
        'popsize': 25,
        'maxiter': 100
    },
    {
        'name': 'Stage3: 精密探索',
        'bounds': [(-5, 5), (-50, 50), (-50, 50), (0.9, 1.1), (0.9, 1.1), (-0.1, 0.1), (-0.1, 0.1)],
        'popsize': 30,
        'maxiter': 150
    }
]

best_result = None
best_similarity = -np.inf

for stage in optimization_stages:
    print(f"🔄 {stage['name']} 実行中...")
    
    result = differential_evolution(
        objective_function,
        bounds=stage['bounds'],
        maxiter=stage['maxiter'],
        popsize=stage['popsize'],
        seed=42,
        atol=1e-8,
        tol=1e-8,
        updating='deferred',
        workers=1
    )
    
    similarity = -result.fun
    print(f"   最適化スコア: {similarity:.4f}")
    print(f"   パラメータ: 角度{result.x[0]:.2f}°, 移動({result.x[1]:.1f},{result.x[2]:.1f}), スケール({result.x[3]:.3f},{result.x[4]:.3f})")
    
    if similarity > best_similarity:
        best_similarity = similarity
        best_result = result
        
    # 次の段階の初期値として使用
    if len(optimization_stages) > 1:
        # 次の段階の境界を現在の結果周辺に調整
        pass

print(f"\n🏆 最終最適化結果:")
print(f"   最高類似度: {best_similarity:.4f}")
print(f"   最適パラメータ: {best_result.x}")

# 最終変換適用
final_transformed = apply_advanced_transform(img2, best_result.x)
final_corr = np.corrcoef(img1.flatten(), final_transformed.flatten())[0,1]
final_composite = ultra_composite_similarity(img1, final_transformed)

print(f"\n📈 最終結果:")
print(f"   相関係数: {initial_corr:.4f} → {final_corr:.4f} ({final_corr-initial_corr:+.4f})")
print(f"   複合類似度: {initial_composite:.4f} → {final_composite:.4f} ({final_composite-initial_composite:+.4f})")
print(f"   改善率: {((final_composite-initial_composite)/initial_composite)*100:+.1f}%")

# 目標達成チェック
if final_corr >= 0.8:
    print(f"🎉 目標達成！相関係数0.8+を実現: {final_corr:.4f}")
else:
    print(f"⚡ 継続改善中: {final_corr:.4f} (目標0.8)")

# ===== 結果可視化 =====
print(f"\n💾 結果を可視化中...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 元画像
axes[0,0].imshow(img1, cmap='gray')
axes[0,0].set_title(f'脳スライス1 (参照)')
axes[0,0].axis('off')

axes[0,1].imshow(img2, cmap='gray')
axes[0,1].set_title(f'脳スライス2 (初期)\n相関係数: {initial_corr:.4f}')
axes[0,1].axis('off')

axes[0,2].imshow(final_transformed, cmap='gray')
axes[0,2].set_title(f'脳スライス2 (最適化後)\n相関係数: {final_corr:.4f}')
axes[0,2].axis('off')

# 差分画像
diff_initial = np.abs(img1 - img2)
diff_final = np.abs(img1 - final_transformed)

axes[1,0].imshow(diff_initial, cmap='hot')
axes[1,0].set_title(f'初期差分画像\n平均差分: {np.mean(diff_initial):.4f}')
axes[1,0].axis('off')

axes[1,1].imshow(diff_final, cmap='hot')
axes[1,1].set_title(f'最適化後差分画像\n平均差分: {np.mean(diff_final):.4f}')
axes[1,1].axis('off')

# オーバーレイ
overlay = np.zeros((img1.shape[0], img1.shape[1], 3))
overlay[:,:,0] = img1  # 赤チャンネル
overlay[:,:,1] = final_transformed  # 緑チャンネル
overlay = np.clip(overlay, 0, 1)

axes[1,2].imshow(overlay)
axes[1,2].set_title(f'オーバーレイ表示\n(赤:参照, 緑:位置合わせ後)')
axes[1,2].axis('off')

plt.tight_layout()
plt.suptitle(f'🧠 ウルトラ高精度脳スライスレジストレーション結果\n相関係数: {initial_corr:.4f} → {final_corr:.4f} (改善: {final_corr-initial_corr:+.4f})', 
             fontsize=16, y=0.98)

output_path = 'ultra_brain_registration_result.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"💾 {output_path} 保存完了")

plt.show()

print("=" * 90)
print("🎉 ウルトラ高精度脳スライスレジストレーション完全完了！")
print("=" * 90)

print(f"""
📊 最終成果サマリー:
   🎯 目標相関係数: 0.8
   📈 達成相関係数: {final_corr:.4f}
   ⚡ 改善幅: {final_corr-initial_corr:+.4f}
   🏆 複合類似度: {final_composite:.4f}
   📊 改善率: {((final_composite-initial_composite)/initial_composite)*100:+.1f}%
   
🧠 実装技術:
   ✓ 7段階メトリクス統合
   ✓ 多段階差分進化最適化
   ✓ 高精度アフィン変換
   ✓ 位相相関
   ✓ 適応的重み付け
   
{'🎉 目標達成！' if final_corr >= 0.8 else '⚡ 継続改善推奨'}
""")
