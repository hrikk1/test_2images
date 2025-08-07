#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, optimize
from scipy.ndimage import gaussian_filter, rotate, shift, affine_transform
from skimage import exposure, filters, feature, transform, measure
from skimage.registration import phase_cross_correlation
from skimage.feature import match_descriptors, ORB
from skimage.transform import ProjectiveTransform, warp
from skimage.measure import ransac
from PIL import Image, ImageEnhance
import cv2
import os
import warnings
import time
from datetime import datetime

warnings.filterwarnings('ignore')

class BrainSliceRegistration:
    """脳スライス画像位置合わせのメインクラス"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.results_history = []
        self.start_time = None
        
    def log(self, message):
        """ログ出力"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")
    
    def load_images(self, image_dir="./test2slices"):
        """脳スライス画像の読み込み"""
        self.log("🖼️ 脳スライス画像を読み込み中...")
        
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"画像ディレクトリが見つかりません: {image_dir}")
        
        # TIFFファイルを検索
        tiff_files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
        tiff_files.sort()
        
        if len(tiff_files) < 2:
            raise ValueError(f"最低2つのTIFFファイルが必要です。見つかったファイル: {len(tiff_files)}")
        
        self.log(f"📁 発見されたTIFFファイル: {tiff_files[:2]}")
        
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
    
    def preprocess_images(self, img1, img2):
        """画像前処理：ノイズ除去とコントラスト強化"""
        self.log("🔧 画像前処理実行中...")
        
        # ガウシアンフィルタでノイズ除去
        img1_proc = gaussian_filter(img1, sigma=1.0)
        img2_proc = gaussian_filter(img2, sigma=1.0)
        
        # ヒストグラム均等化でコントラスト向上
        img1_proc = exposure.equalize_hist(img1_proc)
        img2_proc = exposure.equalize_hist(img2_proc)
        
        self.log("✅ 前処理完了（ノイズ除去・コントラスト強化）")
        
        return img1_proc, img2_proc
    
    def calculate_correlation(self, img1, img2):
        """相関係数の計算（NaN対応）"""
        mask = ~(np.isnan(img1) | np.isnan(img2))
        if mask.sum() == 0:
            return 0.0
        
        correlation = np.corrcoef(img1[mask].flatten(), img2[mask].flatten())[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def coarse_rotation_search(self, img1, img2, angle_range=(-30, 30), step=5):
        """粗い回転角度探索"""
        self.log(f"🔄 粗い回転探索: {angle_range[0]}°〜{angle_range[1]}° ({step}°刻み)")
        
        angles = np.arange(angle_range[0], angle_range[1] + step, step)
        best_angle = 0
        best_score = -1
        best_rotated = img2.copy()
        
        for angle in angles:
            rotated = rotate(img2, angle, reshape=False, order=1)
            score = self.calculate_correlation(img1, rotated)
            
            if score > best_score:
                best_score = score
                best_angle = angle
                best_rotated = rotated
            
            if self.verbose and angle % 10 == 0:
                self.log(f"   角度 {angle:+3d}°: 相関 = {score:.4f}")
        
        self.log(f"✅ 粗い回転結果: {best_angle:+3d}° (相関: {best_score:.4f})")
        return best_rotated, best_angle, best_score
    
    def fine_rotation_search(self, img1, img2, coarse_angle, search_range=3, step=0.1):
        """細かい回転角度探索"""
        self.log(f"🎯 細密回転探索: {coarse_angle-search_range:.1f}°〜{coarse_angle+search_range:.1f}° ({step}°刻み)")
        
        angles = np.arange(coarse_angle - search_range, coarse_angle + search_range + step, step)
        best_angle = coarse_angle
        best_score = -1
        best_rotated = img2.copy()
        
        for angle in angles:
            rotated = rotate(img2, angle, reshape=False, order=3)
            score = self.calculate_correlation(img1, rotated)
            
            if score > best_score:
                best_score = score
                best_angle = angle
                best_rotated = rotated
        
        self.log(f"✅ 細密回転結果: {best_angle:.2f}° (相関: {best_score:.4f})")
        return best_rotated, best_angle, best_score
    
    def translation_optimization(self, img1, img2, max_shift=30, step=2):
        """平行移動最適化（軽量版）"""
        self.log(f"📍 平行移動最適化: ±{max_shift}ピクセル ({step}ピクセル刻み)")
        
        best_score = -1
        best_shift = (0, 0)
        best_shifted = img2.copy()
        
        shifts = np.arange(-max_shift, max_shift + step, step)
        
        for dy in shifts:
            for dx in shifts:
                shifted = shift(img2, [dy, dx], order=1)
                score = self.calculate_correlation(img1, shifted)
                
                if score > best_score:
                    best_score = score
                    best_shift = (dy, dx)
                    best_shifted = shifted
        
        self.log(f"✅ 平行移動結果: shift=({best_shift[0]:+.0f}, {best_shift[1]:+.0f}) (相関: {best_score:.4f})")
        return best_shifted, best_shift, best_score
    
    def run_registration(self, image_dir="./test2slices"):
        """完全な位置合わせパイプライン実行"""
        self.start_time = time.time()
        
        print("=" * 80)
        print("🧠 高精度脳スライス画像位置合わせシステム v2.0")
        print("=" * 80)
        print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"目標: 相関係数 0.8+")
        print()
        
        try:
            # 1. 画像読み込み
            img1, img2 = self.load_images(image_dir)
            initial_correlation = self.calculate_correlation(img1, img2)
            
            self.log(f"📊 初期相関係数: {initial_correlation:.4f}")
            
            # 2. 画像前処理
            img1_proc, img2_proc = self.preprocess_images(img1, img2)
            
            # 3. 段階的最適化
            results = []
            
            # 粗い回転探索
            rotated_img, coarse_angle, coarse_score = self.coarse_rotation_search(img1_proc, img2_proc)
            results.append(("粗い回転", coarse_angle, coarse_score, rotated_img))
            
            # 細かい回転探索
            fine_rotated, fine_angle, fine_score = self.fine_rotation_search(img1_proc, rotated_img, coarse_angle)
            results.append(("細密回転", fine_angle, fine_score, fine_rotated))
            
            # 平行移動最適化
            translated_img, shift, trans_score = self.translation_optimization(img1_proc, fine_rotated)
            results.append(("平行移動", fine_angle, trans_score, translated_img))
            
            # 最良結果の選択
            best_result = max(results, key=lambda x: x[2])
            method, angle, score, final_img = best_result
            
            # 結果保存
            self.results_history = results
            self.final_result = {
                'method': method,
                'angle': angle,
                'score': score,
                'image': final_img,
                'initial_score': initial_correlation,
                'improvement': score - initial_correlation
            }
            
            # 結果表示
            self.display_results()
            
            return final_img, score
            
        except Exception as e:
            self.log(f"❌ エラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def display_results(self):
        """結果の表示と可視化"""
        elapsed_time = time.time() - self.start_time
        
        print("\n" + "=" * 60)
        print("📊 位置合わせ結果サマリー")
        print("=" * 60)
        
        print(f"🏆 最良手法: {self.final_result['method']}")
        print(f"📈 最終相関係数: {self.final_result['score']:.4f}")
        print(f"📊 初期相関係数: {self.final_result['initial_score']:.4f}")
        print(f"⬆️ 改善幅: {self.final_result['improvement']:+.4f}")
        print(f"⏱️ 処理時間: {elapsed_time:.1f}秒")
        
        # 目標達成判定
        if self.final_result['score'] >= 0.8:
            print("🎉 目標達成！相関係数0.8+を実現しました！")
        elif self.final_result['score'] >= 0.7:
            print("🚀 優秀な結果！相関係数0.7+を達成しました！")
        elif self.final_result['score'] >= 0.6:
            print("📈 良好な改善！相関係数0.6+を達成しました！")
        else:
            print("💪 継続改善中...さらなる最適化が必要です")
        
        print("\n📋 各手法の結果:")
        for method, angle, score, _ in self.results_history:
            print(f"   {method:12}: 相関 = {score:.4f}")
        
        print("=" * 60)
        
        # 可視化
        self.visualize_results()
    
    def visualize_results(self):
        """結果の可視化"""
        if not hasattr(self, 'final_result'):
            print("表示する結果がありません")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 元画像
        axes[0,0].imshow(self.img1_original, cmap='gray')
        axes[0,0].set_title('固定画像 (Slice 1)')
        axes[0,0].axis('off')
        
        axes[0,1].imshow(self.img2_original, cmap='gray')
        axes[0,1].set_title('移動画像 (Slice 2)')
        axes[0,1].axis('off')
        
        # 位置合わせ結果
        axes[0,2].imshow(self.final_result['image'], cmap='gray')
        axes[0,2].set_title(f'位置合わせ結果\n{self.final_result["method"]}')
        axes[0,2].axis('off')
        
        # 重ね合わせ比較
        axes[1,0].imshow(self.img1_original, cmap='Reds', alpha=0.7)
        axes[1,0].imshow(self.img2_original, cmap='Blues', alpha=0.7)
        axes[1,0].set_title(f'位置合わせ前\n相関: {self.final_result["initial_score"]:.3f}')
        axes[1,0].axis('off')
        
        axes[1,1].imshow(self.img1_original, cmap='Reds', alpha=0.7)
        axes[1,1].imshow(self.final_result['image'], cmap='Blues', alpha=0.7)
        axes[1,1].set_title(f'位置合わせ後\n相関: {self.final_result["score"]:.3f}')
        axes[1,1].axis('off')
        
        # 手法比較グラフ
        methods = [r[0] for r in self.results_history]
        scores = [r[2] for r in self.results_history]
        
        axes[1,2].bar(range(len(methods)), scores, 
                     color=['green' if s >= 0.8 else 'orange' if s >= 0.6 else 'red' for s in scores])
        axes[1,2].axhline(y=0.8, color='blue', linestyle='--', alpha=0.7, label='目標 (0.8)')
        axes[1,2].set_ylabel('相関係数')
        axes[1,2].set_title('手法別性能比較')
        axes[1,2].set_xticks(range(len(methods)))
        axes[1,2].set_xticklabels(methods, rotation=45, ha='right')
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # ファイル保存
        output_path = '/Users/horiieikkei/Desktop/VS code/brain_registration_results.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 結果画像を保存しました: {output_path}")
        
        plt.show()

def main():
    """メイン実行関数"""
    try:
        # 位置合わせシステム初期化
        registrator = BrainSliceRegistration(verbose=True)
        
        # 位置合わせ実行
        final_image, final_score = registrator.run_registration()
        
        print("\n🎉 脳スライス画像位置合わせが完了しました！")
        
        return registrator
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()
