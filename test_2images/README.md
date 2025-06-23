# 🧠 2画像位置合わせプロジェクト

高精度な画像位置合わせを実現するためのPythonプロジェクトです。SimpleITK、OpenCV、scikit-imageを使用した包括的な画像レジストレーション機能を提供します。

## 🎯 プロジェクト目標

**相関係数 0.8+ の高精度位置合わせを達成**

## 📂 ファイル構成

### メインファイル
- **`2image_registration.ipynb`** - Jupyter Notebook版（メイン開発・実験用）
- **`brain_registration_master.py`** - 統合マスターファイル（Python スクリプト版）

### データ
- **`test2slices/`** - 脳スライス画像データ（TIFF形式）
  - `cropped_MMP_109_x4_largest copy.tif`
  - `cropped_MMP_110_x4_largest copy.tif`

### バックアップ・開発履歴
- **`backup_registration_files/`** - 過去の実験・開発ファイル
- **`registration_output.log`** - 実行ログファイル

## 🚀 使用方法

### 1. Jupyter Notebook（推奨）
```bash
jupyter notebook 2image_registration.ipynb
```

### 2. 統合マスターファイル
```bash
python brain_registration_master.py
```

## 🔧 実装機能

### ノートブック版（2image_registration.ipynb）
1. **パッケージ管理**
   - 必要ライブラリの自動インストール
   - OpenCV、SimpleITK、scikit-imageの統合利用

2. **画像前処理**
   - ガウシアンフィルタによるノイズ除去
   - ヒストグラム均等化によるコントラスト強化
   - 画像サイズの自動調整

3. **レジストレーション手法**
   - Basicレジストレーション（SimpleITK Euler2DTransform）
   - Advancedレジストレーション（MattesMutualInformation + Affine変換）
   - マルチスケール処理
   - 複数メトリクスによる評価

4. **包括的評価指標**
   - MSE（平均二乗誤差）
   - PSNR（ピーク信号対雑音比）
   - NCC（正規化相互相関）
   - SSIM（構造類似性指数）
   - エッジ類似度

5. **可視化機能**
   - レジストレーション前後の比較
   - 差分画像表示
   - オーバーレイ表示
   - メトリクス比較グラフ
   - 変換パラメータ表示

### Python版（brain_registration_master.py）
1. **段階的最適化**
   - 粗い回転探索（±30度、5度刻み）
   - 細密回転最適化（0.1度刻み）
   - 平行移動最適化（2ピクセル刻み）

2. **結果可視化**
   - 位置合わせ前後の比較
   - 重ね合わせ表示
   - 手法別性能比較

## 📊 期待される結果

- **初期相関係数**: ~0.3-0.5
- **最終相関係数**: 0.7-0.8+（手法により変動）
- **処理時間**: 30秒-2分（画像サイズと手法により変動）

## 🎉 主な成果

- **包括的評価システム**: MSE、PSNR、NCC、SSIM、エッジ類似度による多角的評価
- **複数レジストレーション手法**: BasicとAdvanced手法の比較実装
- **インタラクティブ可視化**: Jupyter Notebookでの詳細な結果表示
- **自動化パイプライン**: Pythonスクリプトでの完全自動実行

医学画像解析、3D再構成、病理解析などの応用が可能です。

## 📋 技術要件

### 必須パッケージ
```python
numpy
matplotlib
opencv-python-headless  # または opencv-python
SimpleITK
scipy
scikit-image
pillow (PIL)
ipywidgets  # Jupyter Notebook用
```

### インストール方法
```bash
pip install opencv-python-headless numpy matplotlib SimpleITK ipywidgets scipy scikit-image pillow
```

## 🔬 技術詳細

### 使用ライブラリ
- **SimpleITK**: 医学画像処理専用の高性能レジストレーション
- **OpenCV**: 基本的画像処理とエッジ検出
- **scikit-image**: 画像前処理と構造類似性評価
- **SciPy**: 数値最適化と画像変換
- **Matplotlib**: 結果可視化とグラフ作成

### レジストレーション手法
1. **Euler2D変換**: 回転と平行移動に特化
2. **Affine変換**: より柔軟な幾何変換
3. **MattesMutualInformation**: 医学画像に最適化されたメトリクス
4. **マルチスケール処理**: 粗い→細かいレベルでの最適化

## 📈 プロジェクト履歴

- **2025-06-23**: 2image_registration.ipynb - 包括的評価システム完成
- **2025-06-20**: brain_registration_master.py - 段階的最適化システム
- **開発期間**: 継続的な改善とアルゴリズム最適化