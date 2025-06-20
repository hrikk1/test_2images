#!/usr/bin/env python3
"""
最終最適化された脳スライス画像レジストレーション
目標：相関係数0.8以上を達成
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.optimize import differential_evolution
from sklearn.metrics import mutual_info_score
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_images():
    """画像を読み込み、グレースケールに変換"""
    try:
        img1 = cv2.imread('./test2slices/cropped_MMP_109_x4_largest copy.tif', cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread('./test2slices/cropped_MMP_110_x4_largest copy.tif', cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            raise FileNotFoundError("画像ファイルが見つかりません")
        
        # 画像を0-1の範囲に正規化
        img1 = img1.astype(np.float64) / 255.0
        img2 = img2.astype(np.float64) / 255.0
        
        print(f"画像1のサイズ: {img1.shape}")
        print(f"画像2のサイズ: {img2.shape}")
        
        return img1, img2
    except Exception as e:
        print(f"画像読み込みエラー: {e}")
        return None, None

def pearson_correlation(img1, img2):
    """ピアソン相関係数を計算"""
    # 有効なピクセルのみを使用
    mask = ~(np.isnan(img1) | np.isnan(img2))
    if np.sum(mask) == 0:
        return 0.0
    
    valid_img1 = img1[mask]
    valid_img2 = img2[mask]
    
    # 標準偏差が0の場合をチェック
    if np.std(valid_img1) == 0 or np.std(valid_img2) == 0:
        return 0.0
    
    correlation = np.corrcoef(valid_img1, valid_img2)[0, 1]
    return correlation if not np.isnan(correlation) else 0.0

def mutual_information(img1, img2, bins=256):
    """相互情報量を計算"""
    # 画像を0-255の整数値に変換
    img1_int = np.clip(img1 * 255, 0, 255).astype(int)
    img2_int = np.clip(img2 * 255, 0, 255).astype(int)
    
    # 有効なピクセルのみを使用
    mask = ~(np.isnan(img1) | np.isnan(img2))
    if np.sum(mask) == 0:
        return 0.0
    
    try:
        mi = mutual_info_score(img1_int[mask], img2_int[mask])
        return mi
    except:
        return 0.0

def enhanced_ssim(img1, img2):
    """強化されたSSIM計算"""
    try:
        # 有効な領域を確保
        min_size = min(img1.shape[0], img1.shape[1], img2.shape[0], img2.shape[1])
        if min_size < 7:  # SSIMの最小要件
            return 0.0
        
        # 画像サイズを調整
        h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
        img1_crop = img1[:h, :w]
        img2_crop = img2[:h, :w]
        
        ssim_val = ssim(img1_crop, img2_crop, data_range=1.0, win_size=min(7, min_size))
        return ssim_val if not np.isnan(ssim_val) else 0.0
    except:
        return 0.0

def edge_correlation(img1, img2):
    """エッジ相関を計算"""
    try:
        # Sobelフィルタでエッジを検出
        sobel_x1 = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y1 = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)
        edge1 = np.sqrt(sobel_x1**2 + sobel_y1**2)
        
        sobel_x2 = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y2 = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=3)
        edge2 = np.sqrt(sobel_x2**2 + sobel_y2**2)
        
        return pearson_correlation(edge1, edge2)
    except:
        return 0.0

def gradient_correlation(img1, img2):
    """グラデーション相関を計算"""
    try:
        grad1_x, grad1_y = np.gradient(img1)
        grad2_x, grad2_y = np.gradient(img2)
        
        grad1_mag = np.sqrt(grad1_x**2 + grad1_y**2)
        grad2_mag = np.sqrt(grad2_x**2 + grad2_y**2)
        
        return pearson_correlation(grad1_mag, grad2_mag)
    except:
        return 0.0

def comprehensive_similarity(img1, img2):
    """包括的類似度指標の計算"""
    # 基本指標
    corr = pearson_correlation(img1, img2)
    mi = mutual_information(img1, img2)
    ssim_val = enhanced_ssim(img1, img2)
    edge_corr = edge_correlation(img1, img2)
    grad_corr = gradient_correlation(img1, img2)
    
    # 重み付き合成（相関係数を最重要視）
    composite = (
        0.5 * corr +           # 相関係数（最重要）
        0.15 * (mi / 5.0) +    # 相互情報量（正規化）
        0.15 * ssim_val +      # SSIM
        0.1 * edge_corr +      # エッジ相関
        0.1 * grad_corr        # グラデーション相関
    )
    
    return composite, {
        'correlation': corr,
        'mutual_info': mi,
        'ssim': ssim_val,
        'edge_corr': edge_corr,
        'grad_corr': grad_corr,
        'composite': composite
    }

def apply_transform(image, params):
    """高度な変換を適用"""
    tx, ty, angle, scale_x, scale_y, shear = params
    
    # 画像の中心点
    center = np.array(image.shape[:2][::-1]) / 2
    
    # 変換マトリックス作成
    M = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
    
    # スケーリングとシアーを追加
    scale_matrix = np.array([[scale_x, shear], [0, scale_y]])
    M[:, :2] = M[:, :2] @ scale_matrix
    
    # 平行移動を追加
    M[0, 2] += tx
    M[1, 2] += ty
    
    # 変換を適用
    transformed = cv2.warpAffine(
        image, M, 
        image.shape[:2][::-1], 
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT
    )
    
    return transformed

def objective_function(params, reference_img, moving_img):
    """最適化目的関数"""
    try:
        transformed = apply_transform(moving_img, params)
        similarity, _ = comprehensive_similarity(reference_img, transformed)
        return -similarity  # 最大化のため負値を返す
    except:
        return 1.0  # エラー時はペナルティ

def run_final_optimization():
    """最終最適化実行"""
    print("=== 最終最適化された脳スライス画像レジストレーション ===")
    
    # 画像読み込み
    img1, img2 = load_images()
    if img1 is None or img2 is None:
        print("画像読み込みに失敗しました")
        return
    
    # 初期相関係数
    initial_corr = pearson_correlation(img1, img2)
    print(f"初期相関係数: {initial_corr:.4f}")
    
    # パラメータ範囲を拡張
    bounds = [
        (-300, 300),    # tx (平行移動X)
        (-300, 300),    # ty (平行移動Y)
        (-90, 90),      # angle (回転角度)
        (0.2, 3.0),     # scale_x (X方向スケール)
        (0.2, 3.0),     # scale_y (Y方向スケール)
        (-0.5, 0.5)     # shear (シアー)
    ]
    
    print("最適化を開始します...")
    
    # 多段階最適化
    best_params = None
    best_score = float('inf')
    
    # Stage 1: 粗い探索
    print("Stage 1: 粗い探索...")
    result1 = differential_evolution(
        objective_function,
        bounds,
        args=(img1, img2),
        seed=42,
        maxiter=200,
        popsize=20,
        atol=1e-6,
        tol=1e-6,
        workers=1
    )
    
    if result1.success and result1.fun < best_score:
        best_params = result1.x
        best_score = result1.fun
        print(f"Stage 1 完了: スコア = {-best_score:.4f}")
    
    # Stage 2: 細かい探索（Stage 1の結果周辺）
    if best_params is not None:
        print("Stage 2: 細かい探索...")
        # パラメータ範囲を狭める
        refined_bounds = []
        for i, param in enumerate(best_params):
            if i < 2:  # 平行移動
                range_val = 50
            elif i == 2:  # 回転
                range_val = 15
            elif i < 5:  # スケール
                range_val = 0.3
            else:  # シアー
                range_val = 0.1
            
            refined_bounds.append((
                max(bounds[i][0], param - range_val),
                min(bounds[i][1], param + range_val)
            ))
        
        result2 = differential_evolution(
            objective_function,
            refined_bounds,
            args=(img1, img2),
            seed=43,
            maxiter=300,
            popsize=25,
            atol=1e-8,
            tol=1e-8,
            workers=1
        )
        
        if result2.success and result2.fun < best_score:
            best_params = result2.x
            best_score = result2.fun
            print(f"Stage 2 完了: スコア = {-best_score:.4f}")
    
    # Stage 3: 超精密探索
    if best_params is not None:
        print("Stage 3: 超精密探索...")
        ultra_refined_bounds = []
        for i, param in enumerate(best_params):
            if i < 2:  # 平行移動
                range_val = 20
            elif i == 2:  # 回転
                range_val = 5
            elif i < 5:  # スケール
                range_val = 0.1
            else:  # シアー
                range_val = 0.05
            
            ultra_refined_bounds.append((
                max(bounds[i][0], param - range_val),
                min(bounds[i][1], param + range_val)
            ))
        
        result3 = differential_evolution(
            objective_function,
            ultra_refined_bounds,
            args=(img1, img2),
            seed=44,
            maxiter=400,
            popsize=30,
            atol=1e-10,
            tol=1e-10,
            workers=1
        )
        
        if result3.success and result3.fun < best_score:
            best_params = result3.x
            best_score = result3.fun
            print(f"Stage 3 完了: スコア = {-best_score:.4f}")
    
    if best_params is None:
        print("最適化に失敗しました")
        return
    
    # 最適変換を適用
    print("最適変換を適用中...")
    transformed_img2 = apply_transform(img2, best_params)
    
    # 最終結果の評価
    final_similarity, metrics = comprehensive_similarity(img1, transformed_img2)
    final_corr = metrics['correlation']
    
    print("\n=== 最終結果 ===")
    print(f"初期相関係数: {initial_corr:.4f}")
    print(f"最終相関係数: {final_corr:.4f}")
    print(f"改善: {final_corr - initial_corr:.4f}")
    print(f"目標達成: {'✓' if final_corr >= 0.8 else '✗'} (目標: 0.8以上)")
    
    print(f"\n詳細メトリクス:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\n最適パラメータ:")
    param_names = ['tx', 'ty', 'angle', 'scale_x', 'scale_y', 'shear']
    for name, value in zip(param_names, best_params):
        print(f"  {name}: {value:.4f}")
    
    # 結果を可視化
    visualize_results(img1, img2, transformed_img2, metrics, best_params)
    
    return final_corr >= 0.8

def visualize_results(img1, img2, transformed_img2, metrics, params):
    """結果を可視化"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 元画像
    axes[0, 0].imshow(img1, cmap='gray')
    axes[0, 0].set_title('参照画像 (Image 1)')
    axes[0, 0].axis('off')
    
    # 移動画像（変換前）
    axes[0, 1].imshow(img2, cmap='gray')
    axes[0, 1].set_title('移動画像 (Image 2 - 変換前)')
    axes[0, 1].axis('off')
    
    # 変換後画像
    axes[0, 2].imshow(transformed_img2, cmap='gray')
    axes[0, 2].set_title('変換後画像 (Image 2 - 変換後)')
    axes[0, 2].axis('off')
    
    # オーバーレイ（変換前）
    overlay_before = np.zeros((*img1.shape, 3))
    overlay_before[:, :, 0] = img1
    overlay_before[:, :, 1] = img2[:img1.shape[0], :img1.shape[1]] if img2.shape[0] >= img1.shape[0] and img2.shape[1] >= img1.shape[1] else np.zeros_like(img1)
    axes[1, 0].imshow(overlay_before)
    axes[1, 0].set_title(f'オーバーレイ（変換前）\n相関: {pearson_correlation(img1, img2):.4f}')
    axes[1, 0].axis('off')
    
    # オーバーレイ（変換後）
    overlay_after = np.zeros((*img1.shape, 3))
    overlay_after[:, :, 0] = img1
    overlay_after[:, :, 1] = transformed_img2
    axes[1, 1].imshow(overlay_after)
    axes[1, 1].set_title(f'オーバーレイ（変換後）\n相関: {metrics["correlation"]:.4f}')
    axes[1, 1].axis('off')
    
    # 差分画像
    diff = np.abs(img1 - transformed_img2)
    axes[1, 2].imshow(diff, cmap='hot')
    axes[1, 2].set_title('差分画像')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f'脳スライス画像レジストレーション結果\n最終相関係数: {metrics["correlation"]:.4f}', 
                 fontsize=16, y=0.98)
    
    # 結果を保存
    plt.savefig('final_registration_result.png', dpi=300, bbox_inches='tight')
    print(f"結果を 'final_registration_result.png' に保存しました")
    
    plt.show()

if __name__ == "__main__":
    success = run_final_optimization()
    if success:
        print("\n🎉 目標達成！相関係数0.8以上を達成しました！")
    else:
        print("\n⚠️ 目標未達成。さらなる最適化が必要です。")
