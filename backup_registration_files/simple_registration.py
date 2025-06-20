#!/usr/bin/env python3
# 🚀 シンプル高精度脳スライスレジストレーション
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, shift, zoom
from scipy.optimize import minimize_scalar, minimize
from PIL import Image
import os

print("🧠 シンプル高精度脳スライスレジストレーション開始")

# 画像読み込み
image_dir = "./test2slices/"
tiff_files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
print(f"📁 ファイル: {tiff_files}")

img1 = np.array(Image.open(os.path.join(image_dir, tiff_files[0])).convert('L'), dtype=np.float32)
img2 = np.array(Image.open(os.path.join(image_dir, tiff_files[1])).convert('L'), dtype=np.float32)

# 高速化のため縮小
scale = 0.25
img1 = zoom(img1, scale)
img2 = zoom(img2, scale)

# サイズ統一
min_h, min_w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
img1, img2 = img1[:min_h, :min_w], img2[:min_h, :min_w]

# 正規化
img1 = (img1 - img1.min()) / (img1.max() - img1.min())
img2 = (img2 - img2.min()) / (img2.max() - img2.min())

print(f"🧠 画像サイズ: {img1.shape}")

# 初期相関
initial_corr = np.corrcoef(img1.flatten(), img2.flatten())[0,1]
print(f"📊 初期相関係数: {initial_corr:.4f}")

def correlation_metric(img1, img2):
    """相関係数計算"""
    try:
        corr = np.corrcoef(img1.flatten(), img2.flatten())[0,1]
        return corr if not np.isnan(corr) else 0.0
    except:
        return 0.0

def apply_transform(img, angle=0, tx=0, ty=0, scale=1.0):
    """変換適用"""
    result = img.copy()
    
    # 回転
    if abs(angle) > 0.1:
        result = rotate(result, angle, reshape=False, order=1, mode='constant', cval=0)
    
    # スケール
    if abs(scale - 1.0) > 0.01:
        result = zoom(result, scale, order=1)
        # サイズ調整
        if result.shape != img.shape:
            h_diff = img.shape[0] - result.shape[0]
            w_diff = img.shape[1] - result.shape[1]
            if h_diff > 0:
                result = np.pad(result, ((h_diff//2, h_diff - h_diff//2), (0, 0)), mode='constant')
            elif h_diff < 0:
                start_h = (-h_diff) // 2
                result = result[start_h:start_h + img.shape[0], :]
            
            if w_diff > 0:
                result = np.pad(result, ((0, 0), (w_diff//2, w_diff - w_diff//2)), mode='constant')
            elif w_diff < 0:
                start_w = (-w_diff) // 2
                result = result[:, start_w:start_w + img.shape[1]]
    
    # 平行移動
    if abs(tx) > 0.5 or abs(ty) > 0.5:
        result = shift(result, [ty, tx], order=1, mode='constant', cval=0)
    
    return result

# Stage 1: 回転最適化
print("🔄 Stage1: 回転最適化")
def rotation_objective(angle):
    transformed = apply_transform(img2, angle=angle)
    return -correlation_metric(img1, transformed)

result_rot = minimize_scalar(rotation_objective, bounds=(-45, 45), method='bounded')
best_angle = result_rot.x
best_corr_rot = -result_rot.fun

print(f"   最適角度: {best_angle:.2f}°")
print(f"   相関係数: {best_corr_rot:.4f}")

# Stage 2: 平行移動最適化
print("🔄 Stage2: 平行移動最適化")
def translation_objective(params):
    tx, ty = params
    transformed = apply_transform(img2, angle=best_angle, tx=tx, ty=ty)
    return -correlation_metric(img1, transformed)

result_trans = minimize(translation_objective, [0, 0], 
                       bounds=[(-50, 50), (-50, 50)], method='L-BFGS-B')
best_tx, best_ty = result_trans.x
best_corr_trans = -result_trans.fun

print(f"   最適移動: ({best_tx:.1f}, {best_ty:.1f})")
print(f"   相関係数: {best_corr_trans:.4f}")

# Stage 3: スケール最適化
print("🔄 Stage3: スケール最適化")
def scale_objective(scale):
    transformed = apply_transform(img2, angle=best_angle, tx=best_tx, ty=best_ty, scale=scale)
    return -correlation_metric(img1, transformed)

result_scale = minimize_scalar(scale_objective, bounds=(0.5, 1.5), method='bounded')
best_scale = result_scale.x
best_corr_scale = -result_scale.fun

print(f"   最適スケール: {best_scale:.3f}")
print(f"   相関係数: {best_corr_scale:.4f}")

# Stage 4: 全体微調整
print("🔄 Stage4: 全体微調整")
def combined_objective(params):
    angle, tx, ty, scale = params
    transformed = apply_transform(img2, angle=angle, tx=tx, ty=ty, scale=scale)
    return -correlation_metric(img1, transformed)

initial_params = [best_angle, best_tx, best_ty, best_scale]
bounds = [(best_angle-5, best_angle+5), (best_tx-10, best_tx+10), 
          (best_ty-10, best_ty+10), (best_scale-0.1, best_scale+0.1)]

result_final = minimize(combined_objective, initial_params, bounds=bounds, method='L-BFGS-B')
final_angle, final_tx, final_ty, final_scale = result_final.x
final_corr = -result_final.fun

print(f"   最終パラメータ: 角度{final_angle:.2f}°, 移動({final_tx:.1f},{final_ty:.1f}), スケール{final_scale:.3f}")
print(f"   最終相関係数: {final_corr:.4f}")

# 最終変換適用
final_transformed = apply_transform(img2, angle=final_angle, tx=final_tx, ty=final_ty, scale=final_scale)

print(f"\n📈 結果:")
print(f"   相関係数: {initial_corr:.4f} → {final_corr:.4f}")
print(f"   改善: {final_corr - initial_corr:+.4f}")
print(f"   改善率: {((final_corr - initial_corr)/abs(initial_corr))*100:+.1f}%")

# 目標達成チェック
if final_corr >= 0.8:
    print(f"🎉 目標達成！相関係数0.8+: {final_corr:.4f}")
elif final_corr >= 0.7:
    print(f"🎯 優秀な結果: {final_corr:.4f}")
elif final_corr >= 0.6:
    print(f"📈 良好な改善: {final_corr:.4f}")
else:
    print(f"⚡ 改善継続中: {final_corr:.4f}")

# 可視化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0,0].imshow(img1, cmap='gray')
axes[0,0].set_title(f'参照画像')
axes[0,0].axis('off')

axes[0,1].imshow(img2, cmap='gray')
axes[0,1].set_title(f'初期画像\n相関: {initial_corr:.4f}')
axes[0,1].axis('off')

axes[1,0].imshow(final_transformed, cmap='gray')
axes[1,0].set_title(f'最適化後\n相関: {final_corr:.4f}')
axes[1,0].axis('off')

# オーバーレイ
overlay = np.zeros((*img1.shape, 3))
overlay[:,:,0] = img1
overlay[:,:,1] = final_transformed
overlay = np.clip(overlay, 0, 1)

axes[1,1].imshow(overlay)
axes[1,1].set_title(f'オーバーレイ\n(赤:参照, 緑:最適化)')
axes[1,1].axis('off')

plt.tight_layout()
plt.suptitle(f'脳スライスレジストレーション結果\n改善: {final_corr - initial_corr:+.4f}', y=0.98)

plt.savefig('simple_brain_registration.png', dpi=200, bbox_inches='tight')
print(f"\n💾 simple_brain_registration.png 保存完了")
plt.show()

print("🎉 シンプル高精度脳スライスレジストレーション完了！")
