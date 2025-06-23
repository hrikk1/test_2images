#!/usr/bin/env python3
"""
統合された高精度脳スライス画像レジストレーション
目標: 相関係数0.8+を達成する
"""

import numpy as np
import cv2
from scipy.optimize import differential_evolution, minimize
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class AdvancedImageRegistration:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.best_score = -np.inf
        self.iteration = 0
        
    def load_and_preprocess(self, img1_path, img2_path):
        """画像の読み込みと前処理"""
        if self.verbose:
            print("画像を読み込み中...")
        
        # 画像読み込み
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            raise ValueError("画像の読み込みに失敗しました")
        
        # ノイズ除去とコントラスト強化
        img1 = cv2.bilateralFilter(img1, 9, 75, 75)
        img2 = cv2.bilateralFilter(img2, 9, 75, 75)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img1 = clahe.apply(img1)
        img2 = clahe.apply(img2)
        
        # 正規化
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0
        
        # ガウシアンスムージング
        img1 = gaussian_filter(img1, sigma=0.5)
        img2 = gaussian_filter(img2, sigma=0.5)
        
        if self.verbose:
            print(f"画像1形状: {img1.shape}, 画像2形状: {img2.shape}")
        
        return img1, img2
    
    def create_transformation_matrix(self, params):
        """変換行列の作成"""
        tx, ty, rotation, scale_x, scale_y, shear_x, shear_y = params
        
        # 中心点
        cx, cy = 256, 256  # 画像中心と仮定
        
        # 変換行列の構築
        # 1. 中心への移動
        T1 = np.array([[1, 0, -cx],
                       [0, 1, -cy],
                       [0, 0, 1]], dtype=np.float32)
        
        # 2. 回転
        cos_r, sin_r = np.cos(rotation), np.sin(rotation)
        R = np.array([[cos_r, -sin_r, 0],
                      [sin_r, cos_r, 0],
                      [0, 0, 1]], dtype=np.float32)
        
        # 3. スケーリング
        S = np.array([[scale_x, 0, 0],
                      [0, scale_y, 0],
                      [0, 0, 1]], dtype=np.float32)
        
        # 4. せん断
        Sh = np.array([[1, shear_x, 0],
                       [shear_y, 1, 0],
                       [0, 0, 1]], dtype=np.float32)
        
        # 5. 平行移動
        T2 = np.array([[1, 0, tx],
                       [0, 1, ty],
                       [0, 0, 1]], dtype=np.float32)
        
        # 6. 中心からの移動
        T3 = np.array([[1, 0, cx],
                       [0, 1, cy],
                       [0, 0, 1]], dtype=np.float32)
        
        # 全変換の合成
        M = T3 @ T2 @ Sh @ S @ R @ T1
        return M[:2, :]
    
    def apply_transformation(self, img, params):
        """画像に変換を適用"""
        M = self.create_transformation_matrix(params)
        
        # アフィン変換の適用
        transformed = cv2.warpAffine(
            img, M, (img.shape[1], img.shape[0]),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT
        )
        
        return transformed
    
    def calculate_mutual_information(self, img1, img2, bins=64):
        """相互情報量の計算"""
        # ヒストグラムビンの準備
        hist_2d, _, _ = np.histogram2d(
            img1.ravel(), img2.ravel(), bins=bins,
            range=[[0, 1], [0, 1]]
        )
        
        # 正規化
        hist_2d = hist_2d + 1e-10  # ゼロ除算回避
        hist_2d = hist_2d / np.sum(hist_2d)
        
        # 周辺分布
        px = np.sum(hist_2d, axis=1)
        py = np.sum(hist_2d, axis=0)
        
        # 相互情報量の計算
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if hist_2d[i, j] > 0:
                    mi += hist_2d[i, j] * np.log(hist_2d[i, j] / (px[i] * py[j]))
        
        return mi
    
    def calculate_normalized_mi(self, img1, img2, bins=64):
        """正規化相互情報量の計算"""
        mi = self.calculate_mutual_information(img1, img2, bins)
        
        # エントロピーの計算
        h1 = -np.sum(np.histogram(img1, bins=bins, range=(0, 1), density=True)[0] * 
                     np.log(np.histogram(img1, bins=bins, range=(0, 1), density=True)[0] + 1e-10)) / bins
        h2 = -np.sum(np.histogram(img2, bins=bins, range=(0, 1), density=True)[0] * 
                     np.log(np.histogram(img2, bins=bins, range=(0, 1), density=True)[0] + 1e-10)) / bins
        
        nmi = 2 * mi / (h1 + h2)
        return nmi
    
    def calculate_edge_correlation(self, img1, img2):
        """エッジベース相関の計算"""
        # Sobelエッジ検出
        edge1_x = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
        edge1_y = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)
        edge1 = np.sqrt(edge1_x**2 + edge1_y**2)
        
        edge2_x = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=3)
        edge2_y = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=3)
        edge2 = np.sqrt(edge2_x**2 + edge2_y**2)
        
        # 相関計算
        correlation = np.corrcoef(edge1.ravel(), edge2.ravel())[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def calculate_gradient_correlation(self, img1, img2):
        """勾配ベース相関の計算"""
        grad1_x, grad1_y = np.gradient(img1)
        grad2_x, grad2_y = np.gradient(img2)
        
        grad1_mag = np.sqrt(grad1_x**2 + grad1_y**2)
        grad2_mag = np.sqrt(grad2_x**2 + grad2_y**2)
        
        correlation = np.corrcoef(grad1_mag.ravel(), grad2_mag.ravel())[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def calculate_ssim_score(self, img1, img2):
        """SSIM スコアの計算"""
        return ssim(img1, img2, data_range=1.0, gaussian_weights=True)
    
    def composite_similarity(self, img1, img2):
        """複合類似度の計算"""
        # 基本相関
        corr = np.corrcoef(img1.ravel(), img2.ravel())[0, 1]
        if np.isnan(corr):
            corr = 0.0
        
        # 正規化相互情報量
        nmi = self.calculate_normalized_mi(img1, img2)
        
        # SSIM
        ssim_score = self.calculate_ssim_score(img1, img2)
        
        # エッジ相関
        edge_corr = self.calculate_edge_correlation(img1, img2)
        
        # 勾配相関
        grad_corr = self.calculate_gradient_correlation(img1, img2)
        
        # 重み付き複合スコア（最適化済み重み）
        composite_score = (
            0.40 * corr +          # 基本相関
            0.25 * nmi +           # 正規化相互情報量
            0.20 * ssim_score +    # SSIM
            0.10 * edge_corr +     # エッジ相関
            0.05 * grad_corr       # 勾配相関
        )
        
        return composite_score, {
            'correlation': corr,
            'nmi': nmi,
            'ssim': ssim_score,
            'edge_correlation': edge_corr,
            'gradient_correlation': grad_corr
        }
    
    def objective_function(self, params, fixed_img, moving_img):
        """最適化目的関数"""
        try:
            # 変換適用
            transformed = self.apply_transformation(moving_img, params)
            
            # 複合類似度計算
            score, metrics = self.composite_similarity(fixed_img, transformed)
            
            # 最適化のため負の値を返す（最小化）
            negative_score = -score
            
            # 進捗表示
            self.iteration += 1
            if score > self.best_score:
                self.best_score = score
                if self.verbose and self.iteration % 50 == 0:
                    print(f"反復 {self.iteration}: 最良スコア = {score:.4f}, 相関 = {metrics['correlation']:.4f}")
            
            return negative_score
            
        except Exception as e:
            if self.verbose:
                print(f"エラー: {e}")
            return 1000.0  # 大きなペナルティ
    
    def multi_stage_optimization(self, fixed_img, moving_img):
        """多段階最適化"""
        if self.verbose:
            print("多段階最適化を開始...")
        
        # Stage 1: 粗い探索
        if self.verbose:
            print("Stage 1: 粗い探索")
        bounds = [
            (-50, 50),    # tx
            (-50, 50),    # ty
            (-0.5, 0.5),  # rotation
            (0.7, 1.3),   # scale_x
            (0.7, 1.3),   # scale_y
            (-0.3, 0.3),  # shear_x
            (-0.3, 0.3)   # shear_y
        ]
        
        result1 = differential_evolution(
            self.objective_function,
            bounds,
            args=(fixed_img, moving_img),
            maxiter=200,
            popsize=20,
            seed=42,
            atol=1e-6,
            tol=1e-6
        )
        
        if self.verbose:
            print(f"Stage 1 完了: スコア = {-result1.fun:.4f}")
        
        # Stage 2: 細かい探索
        if self.verbose:
            print("Stage 2: 細かい探索")
        
        # Stage 1の結果を中心とした狭い範囲で探索
        center = result1.x
        bounds2 = [
            (center[0] - 10, center[0] + 10),
            (center[1] - 10, center[1] + 10),
            (center[2] - 0.1, center[2] + 0.1),
            (center[3] - 0.1, center[3] + 0.1),
            (center[4] - 0.1, center[4] + 0.1),
            (center[5] - 0.1, center[5] + 0.1),
            (center[6] - 0.1, center[6] + 0.1)
        ]
        
        result2 = differential_evolution(
            self.objective_function,
            bounds2,
            args=(fixed_img, moving_img),
            maxiter=150,
            popsize=15,
            seed=123,
            atol=1e-8,
            tol=1e-8
        )
        
        if self.verbose:
            print(f"Stage 2 完了: スコア = {-result2.fun:.4f}")
        
        # Stage 3: 局所最適化
        if self.verbose:
            print("Stage 3: 局所最適化")
        
        result3 = minimize(
            self.objective_function,
            result2.x,
            args=(fixed_img, moving_img),
            method='L-BFGS-B',
            bounds=bounds2,
            options={'maxiter': 100, 'ftol': 1e-10}
        )
        
        if self.verbose:
            print(f"Stage 3 完了: 最終スコア = {-result3.fun:.4f}")
        
        return result3.x
    
    def visualize_results(self, fixed_img, moving_img, optimal_params):
        """結果の可視化"""
        transformed_img = self.apply_transformation(moving_img, optimal_params)
        
        # 複合類似度とメトリクス計算
        final_score, metrics = self.composite_similarity(fixed_img, transformed_img)
        
        # 差分画像
        diff_before = np.abs(fixed_img - moving_img)
        diff_after = np.abs(fixed_img - transformed_img)
        
        # プロット
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        # 元画像
        axes[0, 0].imshow(fixed_img, cmap='gray')
        axes[0, 0].set_title('固定画像')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(moving_img, cmap='gray')
        axes[0, 1].set_title('移動画像（変換前）')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(transformed_img, cmap='gray')
        axes[0, 2].set_title('移動画像（変換後）')
        axes[0, 2].axis('off')
        
        # 差分画像
        axes[1, 0].imshow(diff_before, cmap='hot')
        axes[1, 0].set_title('差分（変換前）')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(diff_after, cmap='hot')
        axes[1, 1].set_title('差分（変換後）')
        axes[1, 1].axis('off')
        
        # オーバーレイ
        overlay = np.zeros((fixed_img.shape[0], fixed_img.shape[1], 3))
        overlay[:, :, 0] = fixed_img
        overlay[:, :, 1] = transformed_img
        axes[1, 2].imshow(overlay)
        axes[1, 2].set_title('オーバーレイ（赤：固定、緑：変換後）')
        axes[1, 2].axis('off')
        
        # 結果表示
        result_text = f"""
        最終結果:
        複合スコア: {final_score:.4f}
        相関係数: {metrics['correlation']:.4f}
        正規化MI: {metrics['nmi']:.4f}
        SSIM: {metrics['ssim']:.4f}
        エッジ相関: {metrics['edge_correlation']:.4f}
        勾配相関: {metrics['gradient_correlation']:.4f}
        
        変換パラメータ:
        平行移動: ({optimal_params[0]:.2f}, {optimal_params[1]:.2f})
        回転: {np.degrees(optimal_params[2]):.2f}°
        スケール: ({optimal_params[3]:.3f}, {optimal_params[4]:.3f})
        せん断: ({optimal_params[5]:.3f}, {optimal_params[6]:.3f})
        """
        
        axes[2, 0].text(0.1, 0.5, result_text, fontsize=10, verticalalignment='center')
        axes[2, 0].axis('off')
        
        # ヒストグラム比較
        axes[2, 1].hist(fixed_img.ravel(), bins=50, alpha=0.5, label='固定画像', color='blue')
        axes[2, 1].hist(transformed_img.ravel(), bins=50, alpha=0.5, label='変換後画像', color='red')
        axes[2, 1].set_title('輝度ヒストグラム')
        axes[2, 1].legend()
        
        # 散布図
        sample_indices = np.random.choice(fixed_img.size, 5000, replace=False)
        axes[2, 2].scatter(
            fixed_img.ravel()[sample_indices], 
            transformed_img.ravel()[sample_indices], 
            alpha=0.1
        )
        axes[2, 2].plot([0, 1], [0, 1], 'r--')
        axes[2, 2].set_xlabel('固定画像の輝度')
        axes[2, 2].set_ylabel('変換後画像の輝度')
        axes[2, 2].set_title(f'輝度散布図 (相関: {metrics["correlation"]:.4f})')
        
        plt.tight_layout()
        plt.savefig('/Users/horiieikkei/Desktop/VS code/final_registration_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return final_score, metrics

def main():
    """メイン実行関数"""
    print("=== 高精度脳スライス画像レジストレーション ===")
    print("目標: 相関係数 0.8+ を達成")
    print()
    
    # 画像パス
    img1_path = '/Users/horiieikkei/Desktop/VS code/test2slices/cropped_MMP_109_x4_largest copy.tif'
    img2_path = '/Users/horiieikkei/Desktop/VS code/test2slices/cropped_MMP_110_x4_largest copy.tif'
    
    try:
        # レジストレーションインスタンス作成
        registration = AdvancedImageRegistration(verbose=True)
        
        # 画像読み込みと前処理
        fixed_img, moving_img = registration.load_and_preprocess(img1_path, img2_path)
        
        # 初期状態の評価
        print("初期状態の評価中...")
        initial_score, initial_metrics = registration.composite_similarity(fixed_img, moving_img)
        print(f"初期複合スコア: {initial_score:.4f}")
        print(f"初期相関係数: {initial_metrics['correlation']:.4f}")
        print()
        
        # 多段階最適化実行
        optimal_params = registration.multi_stage_optimization(fixed_img, moving_img)
        
        print("\n=== 最適化完了 ===")
        print("結果を可視化中...")
        
        # 結果可視化
        final_score, final_metrics = registration.visualize_results(
            fixed_img, moving_img, optimal_params
        )
        
        print(f"\n=== 最終結果 ===")
        print(f"複合スコア: {initial_score:.4f} → {final_score:.4f}")
        print(f"相関係数: {initial_metrics['correlation']:.4f} → {final_metrics['correlation']:.4f}")
        print(f"改善: {final_score - initial_score:.4f}")
        
        # 目標達成判定
        if final_metrics['correlation'] >= 0.8:
            print("\n🎉 目標達成！相関係数 0.8+ を達成しました！")
        else:
            print(f"\n目標まであと {0.8 - final_metrics['correlation']:.4f} です。")
        
        print(f"\n結果画像が保存されました: /Users/horiieikkei/Desktop/VS code/final_registration_results.png")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
