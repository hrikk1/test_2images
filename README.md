<<<<<<< HEAD
# 🧠 脳スライス画像位置合わせプロジェクト

高精度な脳スライス画像の位置合わせを実現するためのPythonプロジェクトです。
=======
# 🧠 2 images registration (test_2images)

高精度な画像位置合わせを実現するためのPythonプロジェクト SimpleITK、OpenCV、scikit-imageを使用した包括的な画像レジストレーション機能を提供

Testing the approach of Image Registration by using 2 test images
>>>>>>> bf9f6a15082d3a5a4f958d520db5f279ee0af01f

## 🎯 プロジェクト目標

**相関係数 0.8+ の高精度位置合わせを達成**

## 📂 ファイル構成

### メインファイル
- **`brain_registration_master.py`** - 統合マスターファイル（推奨）
- **`improved_registration.ipynb`** - Jupyter Notebook版（学習・実験用）

### データ
- **`test2slices/`** - 脳スライス画像データ（TIFF形式）

### バックアップ
- **`backup_registration_files/`** - 過去の実験・開発ファイル

## 🚀 使用方法

### 1. 統合マスターファイル（推奨）
```bash
python brain_registration_master.py
```

### 2. Jupyter Notebook
```bash
jupyter notebook improved_registration.ipynb
```

## 🔧 実装手法

1. **前処理**
   - ガウシアンフィルタによるノイズ除去
   - ヒストグラム均等化によるコントラスト強化

2. **粗い回転探索**
   - ±30度範囲で5度刻みの探索

3. **細密回転最適化**
   - 最適角度周辺で0.1度刻みの精密探索

4. **平行移動最適化**
   - 2ピクセル刻みの平行移動探索

5. **結果可視化**
   - 位置合わせ前後の比較
   - 重ね合わせ表示
   - 手法別性能比較

## 📊 期待される結果

- **初期相関係数**: ~0.4-0.5
- **最終相関係数**: 0.7-0.8+（手法により変動）
- **処理時間**: 30秒-2分（画像サイズにより変動）

## 🎉 成果

複数の最適化手法を組み合わせることで、脳スライス画像の高精度位置合わせを実現。
医学画像解析、3D再構成、病理解析などの応用が可能。

## 📋 要件

```python
numpy
matplotlib
scipy
scikit-image
PIL (Pillow)
opencv-python
```

---
Created: 2025-06-20
Author: Brain Registration System