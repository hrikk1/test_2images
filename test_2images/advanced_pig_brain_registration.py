#!/usr/bin/env python3
"""
🐷 豚の脳スライス高精度位置合わせシステム
目標: 相関係数0.8+を達成し、変形を最小限に抑制

複数のレジストレーション手法を比較評価:
1. Rigid変換（回転+平行移動のみ）- 変形なし
2. Similarity変換（回転+平行移動+スケール）- 最小変形
3. Affine変換（より柔軟、ただし変形制御）
4. B-spline非線形変換（局所的変形制御）

Created: 2025-06-23
Author: 3D Brain Mapping System
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import gaussian_filter, rotate, shift
from skimage import exposure, filters, measure, feature
from skimage.registration import phase_cross_correlation
from skimage.feature import match_descriptors, ORB, SIFT
from skimage.transform import AffineTransform, SimilarityTransform, estimate_transform
from skimage.measure import ransac
import SimpleITK as sitk
import cv2
import os
import warnings
import time
from datetime import datetime
from PIL import Image
import json

warnings.filterwarnings('ignore')

class PigBrainRegistration:
    """豚の脳スライス高精度位置合わせクラス"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.results = {}
        self.start_time = None
        self.evaluation_metrics = {}
        
    def log(self, message):
        """ログ出力"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")
    
    def load_images(self, image_dir="./test2slices"):
        """豚の脳スライス画像の読み込み"""
        self.log("🐷 豚の脳スライス画像を読み込み中...")
        
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"画像ディレクトリが見つかりません: {image_dir}")
        
        # TIFFファイルを検索
        tiff_files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
        tiff_files.sort()
        
        if len(tiff_files) < 2:
            raise ValueError(f"最低2つのTIFFファイルが必要です。見つかったファイル: {len(tiff_files)}")
        
        self.log(f"📁 連続スライス: {tiff_files[:2]}")
        
        # 最初の2つの画像を読み込み
        img1_path = os.path.join(image_dir, tiff_files[0])
        img2_path = os.path.join(image_dir, tiff_files[1])
        
        img1 = np.array(Image.open(img1_path).convert('L'))
        img2 = np.array(Image.open(img2_path).convert('L'))
        
        self.log(f"🧠 読み込み完了 - Slice1: {img1.shape}, Slice2: {img2.shape}")
        
        # サイズ統一（小さい方に合わせる）
        min_h = min(img1.shape[0], img2.shape[0])
        min_w = min(img1.shape[1], img2.shape[1])
        
        if img1.shape != (min_h, min_w) or img2.shape != (min_h, min_w):
            self.log(f"📏 画像サイズを統一: ({min_h}, {min_w})")
            img1 = np.array(Image.fromarray(img1).resize((min_w, min_h), Image.Resampling.LANCZOS))
            img2 = np.array(Image.fromarray(img2).resize((min_w, min_h), Image.Resampling.LANCZOS))
        
        # 正規化
        img1 = img1.astype(np.float64) / 255.0
        img2 = img2.astype(np.float64) / 255.0
        
        self.img1_original = img1.copy()
        self.img2_original = img2.copy()
        
        return img1, img2
    
    def preprocess_images(self, img1, img2, method="adaptive"):
        """適応的画像前処理"""
        self.log(f"🔧 画像前処理実行中... (方法: {method})")
        
        if method == "adaptive":
            # 適応的ヒストグラム均等化
            img1_proc = exposure.equalize_adapthist(img1, clip_limit=0.03)
            img2_proc = exposure.equalize_adapthist(img2, clip_limit=0.03)
            
            # 軽微なガウシアンフィルタ（ノイズ除去）
            img1_proc = gaussian_filter(img1_proc, sigma=0.5)
            img2_proc = gaussian_filter(img2_proc, sigma=0.5)
            
        elif method == "conservative":
            # 保守的な前処理（最小限の変更）
            img1_proc = gaussian_filter(img1, sigma=0.3)
            img2_proc = gaussian_filter(img2, sigma=0.3)
            
        elif method == "aggressive":
            # 積極的な前処理
            img1_proc = exposure.equalize_hist(img1)
            img2_proc = exposure.equalize_hist(img2)
            img1_proc = gaussian_filter(img1_proc, sigma=1.0)
            img2_proc = gaussian_filter(img2_proc, sigma=1.0)
        
        self.log("✅ 前処理完了")
        return img1_proc, img2_proc
    
    def calculate_comprehensive_metrics(self, fixed, moving, aligned, transform_params=None):
        """包括的評価指標の計算"""
        metrics = {}
        
        # 1. 相関係数（メイン指標）
        mask = ~(np.isnan(fixed) | np.isnan(aligned))
        if mask.sum() > 0:
            correlation = np.corrcoef(fixed[mask].flatten(), aligned[mask].flatten())[0, 1]
            metrics['correlation'] = correlation if not np.isnan(correlation) else 0.0
        else:
            metrics['correlation'] = 0.0
        
        # 2. 正規化相互情報量（NMI）
        try:
            fixed_int = (fixed * 255).astype(np.uint8)
            aligned_int = (aligned * 255).astype(np.uint8)
            
            hist_2d, _, _ = np.histogram2d(fixed_int.ravel(), aligned_int.ravel(), bins=256)
            
            # 相互情報量計算
            pxy = hist_2d / float(np.sum(hist_2d))
            px = np.sum(pxy, axis=1)
            py = np.sum(pxy, axis=0)
            px_py = px[:, None] * py[None, :]
            
            nzs = pxy > 0
            mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
            
            hx = -np.sum(px * np.log(px + 1e-10))
            hy = -np.sum(py * np.log(py + 1e-10))
            
            nmi = 2 * mi / (hx + hy)
            metrics['nmi'] = nmi
        except:
            metrics['nmi'] = 0.0
        
        # 3. 構造類似性指数（SSIM）
        try:
            from skimage.metrics import structural_similarity
            ssim_val = structural_similarity(fixed, aligned, data_range=1.0)
            metrics['ssim'] = ssim_val
        except:
            metrics['ssim'] = 0.0
        
        # 4. 平均二乗誤差（MSE）
        mse = np.mean((fixed - aligned) ** 2)
        metrics['mse'] = mse
        
        # 5. ピーク信号対雑音比（PSNR）
        if mse > 0:
            psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        else:
            psnr = float('inf')
        metrics['psnr'] = psnr
        
        # 6. 変形度評価
        if transform_params:
            metrics['deformation_score'] = self.calculate_deformation_score(transform_params)
        
        # 7. エッジ保存度
        try:
            fixed_edges = feature.canny(fixed)
            aligned_edges = feature.canny(aligned)
            edge_preservation = np.sum(fixed_edges & aligned_edges) / np.sum(fixed_edges | aligned_edges)
            metrics['edge_preservation'] = edge_preservation
        except:
            metrics['edge_preservation'] = 0.0
        
        return metrics
    
    def calculate_deformation_score(self, transform_params):
        """変形度の評価（低いほど良い）"""
        if 'rotation' in transform_params:
            rotation_penalty = abs(transform_params['rotation']) / 180.0  # 0-1スケール
        else:
            rotation_penalty = 0.0
        
        if 'translation' in transform_params:
            translation_penalty = np.linalg.norm(transform_params['translation']) / 100.0  # 正規化
        else:
            translation_penalty = 0.0
        
        if 'scale' in transform_params:
            scale_penalty = abs(1.0 - transform_params['scale'])  # 1からの乖離
        else:
            scale_penalty = 0.0
        
        return rotation_penalty + translation_penalty + scale_penalty
    
    def rigid_registration_sitk(self, fixed, moving):
        """剛体変換（Rigid）- 変形なし"""
        self.log("🔄 剛体変換レジストレーション実行中...")
        
        fixed_sitk = sitk.GetImageFromArray(fixed)
        moving_sitk = sitk.GetImageFromArray(moving)
        
        # レジストレーション設定
        registration_method = sitk.ImageRegistrationMethod()
        
        # メトリクス: 相互情報量
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.01)
        
        # 剛体変換（Euler2D）
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_sitk, moving_sitk, 
            sitk.Euler2DTransform(), 
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        registration_method.SetInitialTransform(initial_transform, inPlace=False)
        
        # オプティマイザー
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=1.0, 
            numberOfIterations=200,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10
        )
        
        # マルチスケール
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        
        # 実行
        final_transform = registration_method.Execute(fixed_sitk, moving_sitk)
        
        # 結果適用
        moving_resampled = sitk.Resample(
            moving_sitk, fixed_sitk, final_transform, 
            sitk.sitkLinear, 0.0, moving_sitk.GetPixelID()
        )
        
        aligned = sitk.GetArrayFromImage(moving_resampled)
        
        # 変換パラメータ取得
        transform_params = {
            'rotation': np.degrees(final_transform.GetAngle()),
            'translation': final_transform.GetTranslation()
        }
        
        return aligned, transform_params
    
    def similarity_registration_sitk(self, fixed, moving):
        """類似変換（Similarity）- スケール+回転+平行移動"""
        self.log("📏 類似変換レジストレーション実行中...")
        
        fixed_sitk = sitk.GetImageFromArray(fixed)
        moving_sitk = sitk.GetImageFromArray(moving)
        
        registration_method = sitk.ImageRegistrationMethod()
        
        # メトリクス
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.01)
        
        # 類似変換
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_sitk, moving_sitk, 
            sitk.Similarity2DTransform(), 
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        registration_method.SetInitialTransform(initial_transform, inPlace=False)
        
        # オプティマイザー
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=1.0, 
            numberOfIterations=200,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10
        )
        
        # マルチスケール
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        
        final_transform = registration_method.Execute(fixed_sitk, moving_sitk)
        
        moving_resampled = sitk.Resample(
            moving_sitk, fixed_sitk, final_transform, 
            sitk.sitkLinear, 0.0, moving_sitk.GetPixelID()
        )
        
        aligned = sitk.GetArrayFromImage(moving_resampled)
        
        transform_params = {
            'rotation': np.degrees(final_transform.GetAngle()),
            'translation': final_transform.GetTranslation(),
            'scale': final_transform.GetScale()
        }
        
        return aligned, transform_params
    
    def affine_registration_sitk(self, fixed, moving):
        """アフィン変換 - 柔軟性と変形のバランス"""
        self.log("🔧 アフィン変換レジストレーション実行中...")
        
        fixed_sitk = sitk.GetImageFromArray(fixed)
        moving_sitk = sitk.GetImageFromArray(moving)
        
        registration_method = sitk.ImageRegistrationMethod()
        
        # より保守的な設定
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.005)  # サンプリング率を下げる
        
        # アフィン変換
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_sitk, moving_sitk, 
            sitk.AffineTransform(fixed_sitk.GetDimension()), 
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        registration_method.SetInitialTransform(initial_transform, inPlace=False)
        
        # より保守的なオプティマイザー
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=0.5,  # 学習率を下げる
            numberOfIterations=150,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10
        )
        
        # マルチスケール
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        
        final_transform = registration_method.Execute(fixed_sitk, moving_sitk)
        
        moving_resampled = sitk.Resample(
            moving_sitk, fixed_sitk, final_transform, 
            sitk.sitkLinear, 0.0, moving_sitk.GetPixelID()
        )
        
        aligned = sitk.GetArrayFromImage(moving_resampled)
        
        # アフィンパラメータ取得
        matrix = np.array(final_transform.GetMatrix()).reshape(2, 2)
        translation = final_transform.GetTranslation()
        
        transform_params = {
            'matrix': matrix,
            'translation': translation,
            'determinant': np.linalg.det(matrix)  # 変形度の指標
        }
        
        return aligned, transform_params
    
    def bspline_registration_sitk(self, fixed, moving):
        """B-spline非線形変換 - 局所変形制御"""
        self.log("🌊 B-spline非線形レジストレーション実行中...")
        
        fixed_sitk = sitk.GetImageFromArray(fixed)
        moving_sitk = sitk.GetImageFromArray(moving)
        
        registration_method = sitk.ImageRegistrationMethod()
        
        # メトリクス
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.001)  # 非線形なのでさらに少なく
        
        # B-spline変換（グリッド間隔を大きくして変形を制限）
        transformDomainMeshSize = [8, 8]  # 粗いグリッド
        initial_transform = sitk.BSplineTransformInitializer(
            fixed_sitk, 
            transformDomainMeshSize
        )
        registration_method.SetInitialTransform(initial_transform, inPlace=True)
        
        # 保守的なオプティマイザー
        registration_method.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-5,
            numberOfIterations=100
        )
        
        # マルチスケール
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1])
        
        final_transform = registration_method.Execute(fixed_sitk, moving_sitk)
        
        moving_resampled = sitk.Resample(
            moving_sitk, fixed_sitk, final_transform, 
            sitk.sitkLinear, 0.0, moving_sitk.GetPixelID()
        )
        
        aligned = sitk.GetArrayFromImage(moving_resampled)
        
        # 変形フィールドの統計
        displacement_field = sitk.TransformToDisplacementField(
            final_transform, sitk.sitkVectorFloat64, 
            fixed_sitk.GetSize(), fixed_sitk.GetOrigin(), 
            fixed_sitk.GetSpacing(), fixed_sitk.GetDirection()
        )
        displacement_array = sitk.GetArrayFromImage(displacement_field)
        
        transform_params = {
            'max_displacement': np.max(np.linalg.norm(displacement_array, axis=-1)),
            'mean_displacement': np.mean(np.linalg.norm(displacement_array, axis=-1)),
            'displacement_field_shape': displacement_array.shape
        }
        
        return aligned, transform_params
    
    def phase_correlation_registration(self, fixed, moving):
        """位相相関レジストレーション - 高速・ロバスト"""
        self.log("⚡ 位相相関レジストレーション実行中...")
        
        # 位相相関による位置合わせ
        shift, error, diffphase = phase_cross_correlation(fixed, moving, upsample_factor=100)
        
        # 結果適用
        aligned = ndimage.shift(moving, shift, order=3)
        
        transform_params = {
            'translation': shift,
            'error': error
        }
        
        return aligned, transform_params
    
    def run_comprehensive_registration(self, image_dir="./test2slices"):
        """包括的レジストレーション評価実行"""
        self.start_time = time.time()
        
        print("=" * 80)
        print("🐷 豚の脳スライス高精度位置合わせシステム")
        print("=" * 80)
        print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"目標: 相関係数 0.8+ & 最小変形")
        print("=" * 80)
        print()
        
        try:
            # 1. 画像読み込み
            img1, img2 = self.load_images(image_dir)
            
            # 2. 初期評価
            initial_metrics = self.calculate_comprehensive_metrics(img1, img2, img2)
            self.log(f"📊 初期相関係数: {initial_metrics['correlation']:.4f}")
            
            # 3. 前処理オプション
            preprocessing_methods = ["conservative", "adaptive", "aggressive"]
            methods_to_test = [
                ("位相相関", self.phase_correlation_registration),
                ("剛体変換", self.rigid_registration_sitk),
                ("類似変換", self.similarity_registration_sitk),
                ("アフィン変換", self.affine_registration_sitk),
                ("B-spline変換", self.bspline_registration_sitk)
            ]
            
            all_results = []
            
            for prep_method in preprocessing_methods:
                self.log(f"🔄 前処理方法: {prep_method}")
                img1_proc, img2_proc = self.preprocess_images(img1, img2, prep_method)
                
                for method_name, method_func in methods_to_test:
                    try:
                        self.log(f"  📌 {method_name} 実行中...")
                        
                        if method_name == "位相相関":
                            aligned, transform_params = method_func(img1_proc, img2_proc)
                        else:
                            aligned, transform_params = method_func(img1_proc, img2_proc)
                        
                        # 評価
                        metrics = self.calculate_comprehensive_metrics(
                            img1_proc, img2_proc, aligned, transform_params
                        )
                        
                        # 結果保存
                        result = {
                            'preprocessing': prep_method,
                            'method': method_name,
                            'metrics': metrics,
                            'transform_params': transform_params,
                            'aligned_image': aligned,
                            'processed_fixed': img1_proc,
                            'processed_moving': img2_proc
                        }
                        
                        all_results.append(result)
                        
                        self.log(f"    ✅ 相関: {metrics['correlation']:.4f}, SSIM: {metrics['ssim']:.4f}")
                        
                    except Exception as e:
                        self.log(f"    ❌ {method_name} エラー: {e}")
            
            # 4. 結果分析と最適手法選択
            self.results = all_results
            self.analyze_and_display_results(initial_metrics)
            
            return all_results
            
        except Exception as e:
            self.log(f"❌ エラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def analyze_and_display_results(self, initial_metrics):
        """結果分析と表示"""
        if not self.results:
            print("分析する結果がありません")
            return
        
        print("\n" + "=" * 80)
        print("📊 包括的レジストレーション結果分析")
        print("=" * 80)
        
        # 結果をソート（相関係数でソート）
        sorted_results = sorted(self.results, key=lambda x: x['metrics']['correlation'], reverse=True)
        
        print(f"📈 初期相関係数: {initial_metrics['correlation']:.4f}")
        print()
        print("🏆 トップ5結果:")
        print("-" * 80)
        print(f"{'順位':<4} {'前処理':<12} {'手法':<12} {'相関':<8} {'SSIM':<8} {'変形度':<8} {'エッジ保存':<10}")
        print("-" * 80)
        
        for i, result in enumerate(sorted_results[:5]):
            metrics = result['metrics']
            deformation = metrics.get('deformation_score', 0.0)
            
            print(f"{i+1:<4} {result['preprocessing']:<12} {result['method']:<12} "
                  f"{metrics['correlation']:.4f}   {metrics['ssim']:.4f}   "
                  f"{deformation:.4f}   {metrics['edge_preservation']:.4f}")
        
        # 目標達成分析
        print("\n🎯 目標達成分析:")
        target_results = [r for r in sorted_results if r['metrics']['correlation'] >= 0.8]
        
        if target_results:
            print(f"✅ 相関係数0.8+を達成した手法: {len(target_results)}個")
            best_result = target_results[0]
            print(f"🏆 最高性能: {best_result['preprocessing']} + {best_result['method']}")
            print(f"   相関係数: {best_result['metrics']['correlation']:.4f}")
            print(f"   変形度: {best_result['metrics'].get('deformation_score', 0.0):.4f}")
        else:
            print("❌ 相関係数0.8+を達成した手法はありませんでした")
            print(f"   最高値: {sorted_results[0]['metrics']['correlation']:.4f}")
        
        # 低変形度分析
        print("\n🔄 低変形度分析:")
        low_deformation = [r for r in sorted_results 
                          if r['metrics'].get('deformation_score', 1.0) < 0.3 
                          and r['metrics']['correlation'] > 0.6]
        
        if low_deformation:
            print(f"✅ 低変形度(0.3未満)かつ高相関(0.6+): {len(low_deformation)}個")
            for result in low_deformation[:3]:
                print(f"   {result['preprocessing']} + {result['method']}: "
                      f"相関{result['metrics']['correlation']:.4f}, "
                      f"変形{result['metrics'].get('deformation_score', 0.0):.4f}")
        
        # 可視化
        self.visualize_comprehensive_results(sorted_results[:3])
        
        # 結果保存
        self.save_results_to_file(sorted_results)
    
    def visualize_comprehensive_results(self, top_results):
        """上位結果の可視化"""
        if not top_results:
            return
        
        n_results = len(top_results)
        fig, axes = plt.subplots(n_results + 1, 4, figsize=(16, 4 * (n_results + 1)))
        
        if n_results == 1:
            axes = axes.reshape(1, -1)
        
        # 元画像表示
        axes[0, 0].imshow(self.img1_original, cmap='gray')
        axes[0, 0].set_title('固定画像 (Slice 1)')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(self.img2_original, cmap='gray')
        axes[0, 1].set_title('移動画像 (Slice 2)')
        axes[0, 1].axis('off')
        
        # 重ね合わせ（位置合わせ前）
        axes[0, 2].imshow(self.img1_original, cmap='Reds', alpha=0.7)
        axes[0, 2].imshow(self.img2_original, cmap='Blues', alpha=0.7)
        axes[0, 2].set_title('位置合わせ前')
        axes[0, 2].axis('off')
        
        # 空白
        axes[0, 3].axis('off')
        
        # 各結果の表示
        for i, result in enumerate(top_results):
            row = i + 1
            metrics = result['metrics']
            
            # 位置合わせ結果
            axes[row, 0].imshow(result['aligned_image'], cmap='gray')
            axes[row, 0].set_title(f"{result['method']}\n相関: {metrics['correlation']:.4f}")
            axes[row, 0].axis('off')
            
            # 重ね合わせ（位置合わせ後）
            axes[row, 1].imshow(result['processed_fixed'], cmap='Reds', alpha=0.7)
            axes[row, 1].imshow(result['aligned_image'], cmap='Blues', alpha=0.7)
            axes[row, 1].set_title(f"重ね合わせ結果\nSSIM: {metrics['ssim']:.4f}")
            axes[row, 1].axis('off')
            
            # 差分画像
            diff = np.abs(result['processed_fixed'] - result['aligned_image'])
            axes[row, 2].imshow(diff, cmap='hot')
            axes[row, 2].set_title(f"差分画像\nMSE: {metrics['mse']:.6f}")
            axes[row, 2].axis('off')
            
            # メトリクス表示
            metric_text = f"前処理: {result['preprocessing']}\n"
            metric_text += f"相関: {metrics['correlation']:.4f}\n"
            metric_text += f"SSIM: {metrics['ssim']:.4f}\n"
            metric_text += f"NMI: {metrics['nmi']:.4f}\n"
            metric_text += f"PSNR: {metrics['psnr']:.2f}\n"
            if 'deformation_score' in metrics:
                metric_text += f"変形度: {metrics['deformation_score']:.4f}"
            
            axes[row, 3].text(0.1, 0.5, metric_text, fontsize=9, 
                             verticalalignment='center',
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            axes[row, 3].set_xlim(0, 1)
            axes[row, 3].set_ylim(0, 1)
            axes[row, 3].axis('off')
        
        plt.tight_layout()
        
        # 保存
        output_path = f"/Users/horiieikkei/Desktop/VS code/pig_brain_registration_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        self.log(f"💾 結果画像を保存: {output_path}")
        
        plt.show()
    
    def save_results_to_file(self, results):
        """結果をJSONファイルに保存"""
        output_data = []
        
        for result in results:
            # 画像データを除いてメタデータのみ保存
            save_data = {
                'preprocessing': result['preprocessing'],
                'method': result['method'],
                'metrics': result['metrics'],
                'transform_params': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                   for k, v in result['transform_params'].items()}
            }
            output_data.append(save_data)
        
        output_file = f"/Users/horiieikkei/Desktop/VS code/pig_brain_registration_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        self.log(f"💾 結果データを保存: {output_file}")

def main():
    """メイン実行関数"""
    try:
        # システム初期化
        registrator = PigBrainRegistration(verbose=True)
        
        # 包括的レジストレーション実行
        results = registrator.run_comprehensive_registration()
        
        print("\n🎉 豚の脳スライス高精度位置合わせ評価が完了しました！")
        
        return registrator, results
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    registrator, results = main()
