# 🧠 Feature-Based TPS Brain Image Registration

豚脳スライス画像の特徴ベース非剛体レジストレーション（位置合わせ）を行うPythonプロジェクトです。OpenCVのAKAZE特徴検出器とThin Plate Spline（TPS）変換を使用した高精度な画像レジストレーション機能を提供します。

## 🎯 プロジェクト目標

**AKAZE特徴点を使った堅牢なTPS非剛体レジストレーションの実現**

## 📂 ファイル構成

### メインファイル
- **`feature_based_tps_registration_notebook.ipynb`** - メインのTPS特徴ベースレジストレーション（Jupyter Notebook）
- **`brain_registration_master.py`** - レガシー統合マスターファイル
- **`feature_based_tps.py`** - TPS機能のモジュール化されたバージョン

### その他のノートブック
- **`pig_brain_3step_registration.ipynb`** - 3段階レジストレーションアプローチ
- **`pig_brain_feature_based_registration.ipynb`** - 特徴ベースレジストレーション実験
- **`pig_brain_feature_registration_complete.ipynb`** - 完全版特徴レジストレーション

### データ
- **`test2slices/`** - 豚脳スライス画像データ（TIFF形式）
  - `cropped_MMP_109_x4_largest copy.tif`
  - `cropped_MMP_110_x4_largest copy.tif`

### バックアップ・開発履歴
- **`backup_registration_files/`** - 過去の実験・開発ファイル

## 🚀 使用方法

### メインノートブック（推奨）
```bash
jupyter notebook feature_based_tps_registration_notebook.ipynb
```

### Python環境要件
- Python 3.13.2 (Conda)
- 必要パッケージ（下記の詳細バージョン参照）

## 🔧 実装機能

### メインノートブック（feature_based_tps_registration_notebook.ipynb）

1. **画像読み込みと前処理**
   - TIFF画像の読み込み
   - グレースケール変換
   - サイズ正規化

2. **AKAZE特徴検出**
   - OpenCVのAKAZE検出器による特徴点抽出
   - 特徴点の品質診断と可視化
   - 特徴量の分布分析

3. **特徴マッチング**
   - Brute Force Matcherによる特徴点マッチング
   - Lowe's ratioテストによるフィルタリング
   - マッチング品質の評価と診断

4. **TPS（Thin Plate Spline）変換**
   - 特徴点ペアからのTPS係数推定
   - 非剛体変換による画像ワーピング
   - 複数のフォールバック戦略：
     - RANSAC外れ値除去
     - 制御点数削減
     - アフィン/透視変換フォールバック

5. **堅牢性向上機能**
   - TPS出力の統計診断（範囲、標準偏差、コントラスト）
   - 過度の平滑化検出とコントラスト強化
   - 段階的フォールバック処理

6. **評価と可視化**
   - レジストレーション前後の比較
   - オーバーレイ表示（サイズ正規化対応）
   - 特徴点マッチング可視化
   - 定量的評価指標
   - 詳細なチェックポイント出力

### レガシーファイル（brain_registration_master.py）
1. **段階的最適化**
   - 粗い回転探索（±30度、5度刻み）
   - 細密回転最適化（0.1度刻み）
   - 平行移動最適化（2ピクセル刻み）

2. **結果可視化**
   - 位置合わせ前後の比較
   - 重ね合わせ表示
   - 手法別性能比較

## 📊 期待される結果

- **特徴検出**: AKAZE特徴点の安定した検出
- **マッチング**: 高品質な特徴点マッチング（Lowe's ratio < 0.75）
- **TPS変換**: 非剛体変形による柔軟な位置合わせ
- **フォールバック**: 堅牢な処理継続（マッチング失敗時のアフィン変換等）
- **処理時間**: 1-3分（画像サイズと特徴点数により変動）

## 🎉 主な成果

- **特徴ベースTPS実装**: AKAZE + TPS変換による高精度非剛体レジストレーション
- **堅牢なフォールバック**: RANSAC、制御点削減、変換手法降格による安定処理
- **包括的診断**: 特徴検出、マッチング、変換の各段階での詳細診断
- **インタラクティブ可視化**: Jupyter Notebookでの段階的結果表示
- **エラーハンドリング**: 黒画像問題、サイズ不一致等の解決

豚脳病理解析、医学画像レジストレーション、組織学的解析などの応用が可能です。

## 📋 技術要件

### Python環境
- **Python**: 3.13.2 (Conda)
- **OS**: macOS対応（Linuxでも動作確認済み）

### 必須パッケージ（確認済みバージョン）
```python
# 画像処理・コンピュータビジョン
opencv-contrib-python==4.11.0.86
scikit-image==0.25.2

# 数値計算・科学計算
numpy==2.2.6
scipy==1.15.3

# 可視化
matplotlib==3.10.3

# Jupyter Notebook
jupyter==1.1.1
ipywidgets==8.1.5

# 画像IO
pillow==11.1.0
tifffile==2025.2.13
```

### インストール方法

#### オプション1: requirements.txtを使用（推奨）
```bash
# リポジトリをクローンした後
pip install -r requirements.txt
```

#### オプション2: Condaを使用
```bash
conda install opencv numpy scipy matplotlib scikit-image jupyter ipywidgets pillow tifffile
```

#### オプション3: 個別インストール
```bash
pip install opencv-contrib-python==4.11.0.86 numpy==2.2.6 scipy==1.15.3 matplotlib==3.10.3 scikit-image==0.25.2 jupyter ipywidgets pillow tifffile
```

## 🔬 技術詳細

### 使用ライブラリ
- **OpenCV**: AKAZE特徴検出、特徴マッチング、TPS変換実装
- **scikit-image**: 画像I/O、前処理、画像品質評価
- **NumPy**: 数値計算、行列演算、配列操作
- **SciPy**: 科学計算、最適化、統計処理
- **Matplotlib: 結果可視化、グラフ作成、画像表示

### アルゴリズム詳細
1. **AKAZE特徴検出**: 非線形スケール空間による堅牢な特徴抽出
2. **Brute Force Matching**: 全探索による正確な特徴マッチング
3. **Lowe's Ratio Test**: 曖昧なマッチングの除去（閾値: 0.75）
4. **TPS（Thin Plate Spline）**: 放射基底関数による非剛体変換
5. **RANSAC**: 外れ値に堅牢な変換パラメータ推定
6. **多段階フォールバック**: TPS → アフィン → 透視 → 並進変換

### 診断・デバッグ機能
- 特徴点数と品質の評価
- マッチング比率と幾何学的分布
- TPS出力の統計解析（平均、標準偏差、範囲）
- コントラスト不足の自動検出と補正
- 段階的チェックポイント出力

## 📈 プロジェクト履歴

- **2025-01**: `feature_based_tps_registration_notebook.ipynb` - AKAZE+TPS特徴ベースレジストレーション完成
- **開発内容**: 
  - 豚脳スライス画像の非剛体レジストレーション実装
  - 黒画像問題の診断・解決
  - フォールバック機能による堅牢性向上
  - 包括的診断システムの構築
- **技術改良**: 継続的なアルゴリズム最適化とエラーハンドリング強化

## 🐛 既知の問題と解決策

### 問題1: TPS変換後の黒画像
**原因**: 特徴点不足、不適切な制御点配置、変換係数の異常値
**解決策**: 
- RANSAC外れ値除去
- 制御点数の削減
- アフィン変換へのフォールバック
- コントラスト強化

### 問題2: 画像サイズ不一致によるオーバーレイエラー
**原因**: 異なるサイズの画像の重ね合わせ
**解決策**: 表示前の画像サイズ正規化

### 問題3: 特徴マッチング失敗
**原因**: 画像間の類似性不足、特徴点の品質低下
**解決策**: AKAZE閾値調整、代替変換手法の利用

## 🚀 今後の改善予定

- [ ] GPU加速による処理高速化
- [ ] 他の特徴検出器（SIFT、ORB）との比較
- [ ] バッチ処理機能の追加
- [ ] 3D画像レジストレーションへの拡張
- [ ] 深層学習ベースの手法との比較