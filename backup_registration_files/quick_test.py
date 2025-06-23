#!/usr/bin/env python3
# 🚀 クイック高精度テスト - 目標0.8+確認
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, shift, zoom
from scipy.optimize import minimize_scalar
from PIL import Image
import os

print("🧠 クイック高精度テスト開始")

# 画像読み込み
image_dir = "./test2slices/"
tiff_files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]

img1 = np.array(Image.open(os.path.join(image_dir, tiff_files[0])).convert('L'), dtype=np.float32)
img2 = np.array(Image.open(os.path.join(image_dir, tiff_files[1])).convert('L'), dtype=np.float32)

# 超高速処理のため大幅縮小
scale = 0.1
img1 = zoom(img1, scale)
img2 = zoom(img2, scale)

# サイズ統一・正規化
min_h, min_w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
img1, img2 = img1[:min_h, :min_w], img2[:min_h, :min_w]
img1 = (img1 - img1.min()) / (img1.max() - img1.min())
img2 = (img2 - img2.min()) / (img2.max() - img2.min())

print(f"🧠 テスト画像サイズ: {img1.shape}")

# 初期相関
initial_corr = np.corrcoef(img1.flatten(), img2.flatten())[0,1]
print(f"📊 初期相関係数: {initial_corr:.4f}")

# 簡単な変換テスト
def test_transform(angle=0, tx=0, ty=0, scale=1.0):
    result = img2.copy()
    if abs(angle) > 0.1:
        result = rotate(result, angle, reshape=False, order=0)
    if abs(scale - 1.0) > 0.01:
        result = zoom(result, scale, order=0)
        if result.shape != img2.shape:
            if result.shape[0] < img2.shape[0]:
                pad_h = img2.shape[0] - result.shape[0]
                result = np.pad(result, ((pad_h//2, pad_h - pad_h//2), (0, 0)))
            else:
                start = (result.shape[0] - img2.shape[0]) // 2
                result = result[start:start + img2.shape[0], :]
            if result.shape[1] < img2.shape[1]:
                pad_w = img2.shape[1] - result.shape[1]
                result = np.pad(result, ((0, 0), (pad_w//2, pad_w - pad_w//2)))
            else:
                start = (result.shape[1] - img2.shape[1]) // 2
                result = result[:, start:start + img2.shape[1]]
    if abs(tx) > 0.5 or abs(ty) > 0.5:
        result = shift(result, [ty, tx], order=0)
    return result

# クイックテスト: 様々なパラメータ
test_params = [
    (0, 0, 0, 1.0),      # 無変換
    (15, 0, 0, 1.0),     # 回転のみ
    (-15, 0, 0, 1.0),    # 逆回転
    (0, 10, 0, 1.0),     # X移動
    (0, 0, 10, 1.0),     # Y移動
    (0, 0, 0, 1.1),      # スケール
    (0, 0, 0, 0.9),      # 縮小
    (10, 5, 5, 1.05),    # 組み合わせ1
    (-10, -5, -5, 0.95), # 組み合わせ2
    (30, 10, -10, 1.2),  # 大きな変更
    (-30, -10, 10, 0.8), # 逆方向
]

print("\n🔄 クイックパラメータテスト:")
best_corr = initial_corr
best_params = (0, 0, 0, 1.0)

for i, params in enumerate(test_params):
    angle, tx, ty, scale = params
    transformed = test_transform(angle, tx, ty, scale)
    corr = np.corrcoef(img1.flatten(), transformed.flatten())[0,1]
    
    if not np.isnan(corr) and corr > best_corr:
        best_corr = corr
        best_params = params
        print(f"   ✅ Test{i+1}: {corr:.4f} (角度{angle}°, 移動({tx},{ty}), スケール{scale}) - NEW BEST!")
    else:
        print(f"   📊 Test{i+1}: {corr:.4f} (角度{angle}°, 移動({tx},{ty}), スケール{scale})")

print(f"\n🏆 クイックテスト最良結果:")
print(f"   相関係数: {initial_corr:.4f} → {best_corr:.4f} ({best_corr-initial_corr:+.4f})")
print(f"   最良パラメータ: 角度{best_params[0]}°, 移動({best_params[1]},{best_params[2]}), スケール{best_params[3]}")

# 最良パラメータ周辺でさらに探索
print(f"\n🎯 最良パラメータ周辺探索:")
base_angle, base_tx, base_ty, base_scale = best_params

def angle_objective(angle):
    transformed = test_transform(angle, base_tx, base_ty, base_scale)
    corr = np.corrcoef(img1.flatten(), transformed.flatten())[0,1]
    return -corr if not np.isnan(corr) else 1.0

# 角度の精密最適化
result_angle = minimize_scalar(angle_objective, bounds=(base_angle-10, base_angle+10), method='bounded')
optimal_angle = result_angle.x
optimal_corr_angle = -result_angle.fun

print(f"   角度最適化: {base_angle}° → {optimal_angle:.2f}° (相関: {optimal_corr_angle:.4f})")

# 最終変換適用
final_transformed = test_transform(optimal_angle, base_tx, base_ty, base_scale)
final_corr = np.corrcoef(img1.flatten(), final_transformed.flatten())[0,1]

print(f"\n📈 最終結果:")
print(f"   相関係数: {initial_corr:.4f} → {final_corr:.4f}")
print(f"   改善: {final_corr - initial_corr:+.4f}")
print(f"   改善率: {((final_corr - initial_corr)/abs(initial_corr))*100:+.1f}%")

# 結果判定
if final_corr >= 0.8:
    print(f"🎉 目標達成！0.8+: {final_corr:.4f}")
elif final_corr >= 0.7:
    print(f"🎯 優秀！0.7+: {final_corr:.4f}")
elif final_corr >= 0.6:
    print(f"📈 良好！0.6+: {final_corr:.4f}")
else:
    print(f"⚡ 改善継続: {final_corr:.4f}")

# 簡単な可視化
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(img1, cmap='gray')
axes[0].set_title('参照')
axes[0].axis('off')

axes[1].imshow(img2, cmap='gray')
axes[1].set_title(f'初期\n{initial_corr:.4f}')
axes[1].axis('off')

axes[2].imshow(final_transformed, cmap='gray')
axes[2].set_title(f'最適化\n{final_corr:.4f}')
axes[2].axis('off')

overlay = np.zeros((*img1.shape, 3))
overlay[:,:,0] = img1
overlay[:,:,1] = final_transformed
axes[3].imshow(overlay)
axes[3].set_title('オーバーレイ')
axes[3].axis('off')

plt.tight_layout()
plt.suptitle(f'クイック高精度テスト結果: {initial_corr:.4f} → {final_corr:.4f}', y=1.02)
plt.savefig('quick_test_result.png', dpi=150, bbox_inches='tight')
print(f"\n💾 quick_test_result.png 保存完了")
plt.show()

print("🎉 クイック高精度テスト完了！")

# スケールアップの予測
print(f"\n🔮 フルサイズでの予測性能:")
predicted_improvement = final_corr - initial_corr
print(f"   予測改善幅: {predicted_improvement:+.4f}")
print(f"   フルサイズ予測相関: {0.46 + predicted_improvement:.4f}")  # 元の初期値0.46から

if 0.46 + predicted_improvement >= 0.8:
    print(f"   🎉 フルサイズで目標達成見込み!")
elif 0.46 + predicted_improvement >= 0.7:
    print(f"   🎯 フルサイズで優秀な結果見込み!")
else:
    print(f"   📈 フルサイズでさらなる改善が必要")
