#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脳スライス画像の位置合わせと3Dモデル化
Image Registration and 3D Modeling for Brain Slices
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
import os

def main():
    print("🧠 脳スライス画像の位置合わせと3Dモデル化")
    print("=" * 50)
    
    # 画像ファイルの読み込み
    img_dir = './test2slices'
    if not os.path.exists(img_dir):
        print(f"❌ エラー: フォルダ {img_dir} が見つかりません")
        return
    
    img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.tif')]
    img_files.sort()
    
    if len(img_files) < 2:
        print("❌ エラー: TIFFファイルが2つ未満です")
        return
    
    print(f"📁 発見したファイル: {[os.path.basename(f) for f in img_files[:2]]}")
    
    # 画像読み込み
    img1 = Image.open(img_files[0]).convert('L')
    img2 = Image.open(img_files[1]).convert('L')
    
    img1_array = np.array(img1)
    img2_array = np.array(img2)
    
    print(f"🧠 Slice 1 形状: {img1_array.shape}")
    print(f"🧠 Slice 2 形状: {img2_array.shape}")
    
    # 画像サイズを合わせる
    h1, w1 = img1_array.shape
    h2, w2 = img2_array.shape
    target_h = min(h1, h2)
    target_w = min(w1, w2)
    
    print(f"🎯 目標サイズ: ({target_h}, {target_w})")
    
    # リサイズ
    img1_resized = Image.fromarray(img1_array).resize((target_w, target_h), Image.LANCZOS)
    img2_resized = Image.fromarray(img2_array).resize((target_w, target_h), Image.LANCZOS)
    
    img1_final = np.array(img1_resized)
    img2_final = np.array(img2_resized)
    
    print("✅ リサイズ完了")
    
    # 回転角度テスト
    test_angles = np.arange(-20, 25, 5)
    print(f"🔄 テスト角度: {test_angles.tolist()}")
    
    best_angle = 0
    best_score = -1
    best_rotated = img2_final.copy()
    results = []
    
    print("\n回転テスト実行中...")
    for angle in test_angles:
        try:
            # 画像2を回転
            rotated = ndimage.rotate(img2_final, angle, reshape=False, order=1)
            
            # 相関係数計算
            correlation = np.corrcoef(img1_final.flatten(), rotated.flatten())[0, 1]
            
            if np.isnan(correlation):
                correlation = 0.0
            
            results.append((angle, correlation))
            print(f"角度 {angle:+3d}°: 相関係数 = {correlation:+.4f}")
            
            if correlation > best_score:
                best_score = correlation
                best_angle = angle
                best_rotated = rotated
                
        except Exception as e:
            print(f"角度 {angle:+3d}°: エラー - {e}")
            results.append((angle, 0.0))
            continue
    
    print(f"\n🏆 最適結果: {best_angle:+.1f}° (相関係数: {best_score:+.4f})")
    
    # 結果の可視化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 元画像
    axes[0,0].imshow(img1_final, cmap='gray')
    axes[0,0].set_title('Brain Slice 1 (Fixed)')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(img2_final, cmap='gray')
    axes[0,1].set_title('Brain Slice 2 (Original)')
    axes[0,1].axis('off')
    
    # 位置合わせ後
    axes[0,2].imshow(best_rotated, cmap='gray')
    axes[0,2].set_title(f'Slice 2 Aligned ({best_angle:+.1f}°)')
    axes[0,2].axis('off')
    
    # 重ね合わせ
    axes[1,0].imshow(img1_final, cmap='Reds', alpha=0.7)
    axes[1,0].imshow(img2_final, cmap='Blues', alpha=0.7)
    axes[1,0].set_title('Before Alignment')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(img1_final, cmap='Reds', alpha=0.7)
    axes[1,1].imshow(best_rotated, cmap='Blues', alpha=0.7)
    axes[1,1].set_title(f'After Alignment\nCorrelation: {best_score:.3f}')
    axes[1,1].axis('off')
    
    # 差分画像
    diff_img = np.abs(img1_final.astype(float) - best_rotated.astype(float))
    axes[1,2].imshow(diff_img, cmap='hot')
    axes[1,2].set_title('Difference Map')
    axes[1,2].axis('off')
    
    plt.tight_layout()
    plt.savefig('brain_slice_alignment_result.png', dpi=300, bbox_inches='tight')
    print("✅ 結果画像を保存しました: brain_slice_alignment_result.png")
    plt.show()
    
    # 角度と相関係数のグラフ
    if results:
        angles, correlations = zip(*results)
        plt.figure(figsize=(10, 6))
        plt.plot(angles, correlations, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('回転角度 (°)')
        plt.ylabel('相関係数')
        plt.title('回転角度 vs 類似度')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=best_score, color='r', linestyle='--', alpha=0.7, label=f'Max: {best_score:.3f}')
        plt.axvline(x=best_angle, color='r', linestyle='--', alpha=0.7, label=f'Best: {best_angle}°')
        plt.legend()
        plt.savefig('rotation_optimization_curve.png', dpi=300, bbox_inches='tight')
        print("✅ 最適化グラフを保存しました: rotation_optimization_curve.png")
        plt.show()
    
    # 3D可視化
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        stack = np.stack([img1_final, best_rotated], axis=0)
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for i in range(stack.shape[0]):
            slice_img = stack[i]
            h, w = slice_img.shape
            
            step = max(15, min(h, w) // 80)
            x = np.arange(0, w, step)
            y = np.arange(0, h, step)
            X, Y = np.meshgrid(x, y)
            Z = np.full_like(X, i * 50)
            
            colors = slice_img[::step, ::step]
            if colors.max() > 0:
                colors = colors / colors.max()
            
            ax.plot_surface(X, Y, Z, facecolors=plt.cm.gray(colors), 
                           linewidth=0, antialiased=True, alpha=0.8)
        
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_zlabel('Z (depth)')
        ax.set_title(f'3D Brain Stack Model\n(Rotation: {best_angle:+.1f}°)')
        ax.view_init(elev=25, azim=45)
        
        plt.savefig('brain_3d_model.png', dpi=300, bbox_inches='tight')
        print("✅ 3Dモデル画像を保存しました: brain_3d_model.png")
        plt.show()
        
    except Exception as e:
        print(f"⚠️ 3D表示エラー: {e}")
    
    # 最終結果レポート
    print("\n" + "=" * 60)
    print("🧠 脳スライス IMAGE REGISTRATION 結果レポート")
    print("=" * 60)
    
    print(f"🖼️ 元画像情報:")
    print(f"  • Slice 1: {img1_array.shape} pixels")
    print(f"  • Slice 2: {img2_array.shape} pixels")
    print(f"  • 処理後: {img1_final.shape} pixels")
    
    print(f"\n🔄 回転角最適化:")
    print(f"  • 最適角度: {best_angle:+.1f}°")
    print(f"  • 相関係数: {best_score:.4f}")
    
    # 品質評価
    if best_score > 0.8:
        quality = "🏆 優秀"
        comment = "非常に高精度な位置合わせが達成されました！"
    elif best_score > 0.6:
        quality = "🚀 良好"
        comment = "適度な精度の位置合わせです。実用的なレベルです。"
    elif best_score > 0.3:
        quality = "🔶 改善需"
        comment = "位置合わせは達成されましたが、さらなる精度向上が可能です。"
    else:
        quality = "⚠️ 要改善"
        comment = "位置合わせが不十分です。パラメータの調整が必要です。"
    
    print(f"\n{quality}")
    print(f"💬 {comment}")
    
    print(f"\n🤖 使用アルゴリズム:")
    print(f"  • 画像サイズ正規化 (PIL LANCZOS)")
    print(f"  • 回転角度スキャン ({test_angles.min()}° to {test_angles.max()}°)")
    print(f"  • SciPy ndimage.rotate() 回転")
    print(f"  • 正規化相互相関で類似度評価")
    
    print(f"\n🎉 3D脳スライスモデルの作成完了！")
    print("=" * 60)

if __name__ == "__main__":
    main()
