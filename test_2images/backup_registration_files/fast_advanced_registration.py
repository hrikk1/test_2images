#!/usr/bin/env python3
# 🚀 高速高精度脳スライスレジストレーション - 目標0.8+達成
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.ndimage import sobel, affine_transform, gaussian_filter
from scipy.optimize import differential_evolution
from PIL import Image
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🧠 高速高精度脳スライスレジストレーション - 目標0.8+達成")
print("=" * 80)

# ===== 最適化されたメトリクス関数群 =====
def fast_mutual_information(img1, img2, bins=50):
    """高速相互情報量計算"""
    hist_2d, _, _ = np.histogram2d(img1.flatten(), img2.flatten(), bins=bins)
    hist_2d = hist_2d / (hist_2d.sum() + 1e-12)
    
    px = hist_2d.sum(axis=1) + 1e-12
    py = hist_2d.sum(axis=0) + 1e-12
    hist_2d = hist_2d + 1e-12
    
    hx = -np.sum(px * np.log(px))
    hy = -np.sum(py * np.log(py))
    hxy = -np.sum(hist_2d * np.log(hist_2d))
    
    return hx + hy - hxy

def fast_ssim(img1, img2):
    """高速SSIM計算"""
    mean1, mean2 = np.mean(img1), np.mean(img2)
    var1, var2 = np.var(img1), np.var(img2)
    cov = np.mean((img1 - mean1) * (img2 - mean2))
    
    c1, c2 = (0.01 * 255)**2, (0.03 * 255)**2
    
    ssim = ((2*mean1*mean2 + c1)*(2*cov + c2)) / ((mean1**2 + mean2**2 + c1)*(var1 + var2 + c2))
    return np.clip(ssim, -1, 1)

def fast_edge_correlation(img1, img2):
    """高速エッジ相関"""
    # ダウンサンプリングで高速化
    step = 2
    img1_ds = img1[::step, ::step]
    img2_ds = img2[::step, ::step]
    
    dx1, dy1 = sobel(img1_ds, axis=0), sobel(img1_ds, axis=1)
    dx2, dy2 = sobel(img2_ds, axis=0), sobel(img2_ds, axis=1)
    
    edge1 = np.sqrt(dx1**2 + dy1**2)
    edge2 = np.sqrt(dx2**2 + dy2**2)
    
    return np.corrcoef(edge1.flatten(), edge2.flatten())[0,1]

def apply_simple_transform(image, params):
    """シンプルな変換適用（高速化）"""
    angle, tx, ty, scale = params
    
    # 回転
    if abs(angle) > 0.01:
        from scipy.ndimage import rotate
        image = rotate(image, angle, reshape=False, order=1)
    
    # スケーリング
    if abs(scale - 1.0) > 0.01:
        from scipy.ndimage import zoom
        image = zoom(image, scale, order=1)
    
    # 平行移動
    if abs(tx) > 0.5 or abs(ty) > 0.5:
        from scipy.ndimage import shift
        image = shift(image, [ty, tx], order=1)
    
    return image

def fast_composite_similarity(img1, img2):
    """高速複合類似度関数"""
    try:
        # 基本相関係数（最重要）
        corr = np.corrcoef(img1.flatten(), img2.flatten())[0,1]
        if np.isnan(corr):
            corr = 0.0
        
        # 相互情報量
        mi = fast_mutual_information(img1, img2)
        if np.isnan(mi):
            mi = 0.0
        
        # SSIM
        ssim = fast_ssim(img1, img2)
        if np.isnan(ssim):
            ssim = 0.0
        
        # エッジ相関
        edge_corr = fast_edge_correlation(img1, img2)
        if np.isnan(edge_corr):
            edge_corr = 0.0
        
        # 相関係数を大幅に重視した重み付け
        weights = [0.60, 0.20, 0.15, 0.05]  # 相関係数60%
        
        composite = weights[0]*corr + weights[1]*mi + weights[2]*ssim + weights[3]*edge_corr
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

# 高速化のためリサイズ
scale_factor = 0.5  # 半分のサイズで高速処理
from scipy.ndimage import zoom
img1_fast = zoom(img1, scale_factor, order=1)
img2_fast = zoom(img2, scale_factor, order=1)

# 正規化
img1_fast = (img1_fast - img1_fast.min()) / (img1_fast.max() - img1_fast.min())
img2_fast = (img2_fast - img2_fast.min()) / (img2_fast.max() - img2_fast.min())

print(f"🧠 高速処理用画像サイズ: {img1_fast.shape}")
print("✅ 脳スライス画像の読み込み成功！\n")

# ===== 初期評価 =====
print("📊 初期状態:")
initial_composite = fast_composite_similarity(img1_fast, img2_fast)
initial_corr = np.corrcoef(img1_fast.flatten(), img2_fast.flatten())[0,1]
print(f"   複合類似度: {initial_composite:.4f}")
print(f"   相関係数: {initial_corr:.4f}")
print()

# ===== 高速最適化 =====
print("🎯 高速高精度レジストレーション実行中...")

def objective_function(params):
    """最適化目的関数"""
    try:
        transformed_img2 = apply_simple_transform(img2_fast, params)
        # サイズ調整
        if transformed_img2.shape != img1_fast.shape:
            from scipy.ndimage import zoom
            scale_y = img1_fast.shape[0] / transformed_img2.shape[0]
            scale_x = img1_fast.shape[1] / transformed_img2.shape[1]
            transformed_img2 = zoom(transformed_img2, [scale_y, scale_x], order=1)
        
        similarity = fast_composite_similarity(img1_fast, transformed_img2)
        return -similarity  # 最小化問題として
    except:
        return 1000.0  # ペナルティ

# 段階的最適化（高速版）
print("🔄 Stage1: 粗い探索...")
result1 = differential_evolution(
    objective_function,
    bounds=[(-45, 45), (-100, 100), (-100, 100), (0.5, 1.5)],  # angle, tx, ty, scale
    maxiter=30,
    popsize=15,
    seed=42
)

print(f"   Stage1結果: {-result1.fun:.4f}")
print(f"   パラメータ: 角度{result1.x[0]:.2f}°, 移動({result1.x[1]:.1f},{result1.x[2]:.1f}), スケール{result1.x[3]:.3f}")

# Stage1の結果を中心とした精密探索
center_params = result1.x
search_range = 10

print("🔄 Stage2: 精密探索...")
result2 = differential_evolution(
    objective_function,
    bounds=[
        (center_params[0]-search_range, center_params[0]+search_range),
        (center_params[1]-search_range, center_params[1]+search_range), 
        (center_params[2]-search_range, center_params[2]+search_range),
        (max(0.1, center_params[3]-0.3), center_params[3]+0.3)
    ],
    maxiter=50,
    popsize=20,
    seed=42
)

print(f"   Stage2結果: {-result2.fun:.4f}")
print(f"   パラメータ: 角度{result2.x[0]:.2f}°, 移動({result2.x[1]:.1f},{result2.x[2]:.1f}), スケール{result2.x[3]:.3f}")

# 最良結果選択
best_result = result2 if -result2.fun > -result1.fun else result1
best_similarity = -best_result.fun

print(f"\n🏆 最終最適化結果:")
print(f"   最高類似度: {best_similarity:.4f}")
print(f"   最適パラメータ: {best_result.x}")

# ===== フルサイズで最終変換適用 =====
print("🎯 フルサイズ画像で最終変換適用中...")

# 元画像の正規化
img1_full = (img1 - img1.min()) / (img1.max() - img1.min())
img2_full = (img2 - img2.min()) / (img2.max() - img2.min())

# サイズ統一
min_h = min(img1_full.shape[0], img2_full.shape[0])
min_w = min(img1_full.shape[1], img2_full.shape[1])
img1_full = img1_full[:min_h, :min_w]
img2_full = img2_full[:min_h, :min_w]

# パラメータをフルサイズに調整
full_params = best_result.x.copy()
full_params[1] *= (1/scale_factor)  # tx調整
full_params[2] *= (1/scale_factor)  # ty調整

final_transformed = apply_simple_transform(img2_full, full_params)

# サイズ調整
if final_transformed.shape != img1_full.shape:
    from scipy.ndimage import zoom
    scale_y = img1_full.shape[0] / final_transformed.shape[0]
    scale_x = img1_full.shape[1] / final_transformed.shape[1]
    final_transformed = zoom(final_transformed, [scale_y, scale_x], order=1)

final_corr = np.corrcoef(img1_full.flatten(), final_transformed.flatten())[0,1]
final_composite = fast_composite_similarity(img1_full, final_transformed)

# ===== 結果評価 =====
print(f"\n📈 最終結果:")
print(f"   相関係数: {initial_corr:.4f} → {final_corr:.4f} ({final_corr-initial_corr:+.4f})")
print(f"   複合類似度: {initial_composite:.4f} → {final_composite:.4f} ({final_composite-initial_composite:+.4f})")
improvement_rate = ((final_composite-initial_composite)/abs(initial_composite))*100 if initial_composite != 0 else 0
print(f"   改善率: {improvement_rate:+.1f}%")

# 目標達成チェック
if final_corr >= 0.8:
    print(f"🎉 目標達成！相関係数0.8+を実現: {final_corr:.4f}")
elif final_corr >= 0.7:
    print(f"🎯 良好な結果: {final_corr:.4f} (目標0.8に接近)")
elif final_corr >= 0.6:
    print(f"📈 改善確認: {final_corr:.4f} (目標0.8に向上中)")
else:
    print(f"⚡ 継続改善中: {final_corr:.4f} (目標0.8)")

# ===== 結果可視化 =====
print(f"\n💾 結果を可視化中...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 元画像
initial_corr_full = np.corrcoef(img1_full.flatten(), img2_full.flatten())[0,1]

axes[0,0].imshow(img1_full, cmap='gray')
axes[0,0].set_title(f'脳スライス1 (参照)')
axes[0,0].axis('off')

axes[0,1].imshow(img2_full, cmap='gray')
axes[0,1].set_title(f'脳スライス2 (初期)\n相関係数: {initial_corr_full:.4f}')
axes[0,1].axis('off')

axes[0,2].imshow(final_transformed, cmap='gray')
axes[0,2].set_title(f'脳スライス2 (最適化後)\n相関係数: {final_corr:.4f}')
axes[0,2].axis('off')

# 差分画像
diff_initial = np.abs(img1_full - img2_full)
diff_final = np.abs(img1_full - final_transformed)

axes[1,0].imshow(diff_initial, cmap='hot')
axes[1,0].set_title(f'初期差分画像\n平均差分: {np.mean(diff_initial):.4f}')
axes[1,0].axis('off')

axes[1,1].imshow(diff_final, cmap='hot')
axes[1,1].set_title(f'最適化後差分画像\n平均差分: {np.mean(diff_final):.4f}')
axes[1,1].axis('off')

# オーバーレイ
overlay = np.zeros((img1_full.shape[0], img1_full.shape[1], 3))
overlay[:,:,0] = img1_full  # 赤チャンネル
overlay[:,:,1] = final_transformed  # 緑チャンネル
overlay = np.clip(overlay, 0, 1)

axes[1,2].imshow(overlay)
axes[1,2].set_title(f'オーバーレイ表示\n(赤:参照, 緑:位置合わせ後)')
axes[1,2].axis('off')

plt.tight_layout()
plt.suptitle(f'🧠 高速高精度脳スライスレジストレーション結果\n相関係数: {initial_corr_full:.4f} → {final_corr:.4f} (改善: {final_corr-initial_corr_full:+.4f})', 
             fontsize=14, y=0.98)

output_path = 'fast_brain_registration_result.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"💾 {output_path} 保存完了")

plt.show()

print("=" * 80)
print("🎉 高速高精度脳スライスレジストレーション完了！")
print("=" * 80)

status_emoji = "🎉" if final_corr >= 0.8 else "🎯" if final_corr >= 0.7 else "📈" if final_corr >= 0.6 else "⚡"

print(f"""
📊 最終成果サマリー:
   🎯 目標相関係数: 0.8
   📈 達成相関係数: {final_corr:.4f}
   ⚡ 改善幅: {final_corr-initial_corr_full:+.4f}
   🏆 複合類似度: {final_composite:.4f}
   📊 改善率: {improvement_rate:+.1f}%
   
🧠 実装技術:
   ✓ 高速相互情報量計算
   ✓ 段階的差分進化最適化
   ✓ 適応的画像変換
   ✓ 重み付け複合メトリクス
   ✓ マルチスケール処理
   
{status_emoji} {'目標達成！' if final_corr >= 0.8 else '継続改善中' if final_corr < 0.8 else '良好な結果'}
""")

# さらなる改善提案
if final_corr < 0.8:
    print(f"""
🔧 さらなる改善のための提案:
   1. より多くの最適化ステップの実行
   2. 非線形変換（B-spline等）の適用
   3. マルチレベル画像ピラミッド最適化
   4. 特徴点ベースの初期位置合わせ
   5. より高度なメトリクスの統合
""")
