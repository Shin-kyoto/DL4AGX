# VAD Mock

Vectorized Autonomous Driving (VAD) モデルのテスト用モックシステムです。このシステムは実際の推論を行う代わりに、受信したセンサーデータをログファイルに記録し、推論を模擬します。

## 機能

- 6つのカメラ画像、IMU、Odometryデータを時刻同期
- 設定ファイルで指定された時刻オフセットに基づいたデータ処理
- 画像欠損時の代替戦略（黒塗り画像または過去画像の使用）
- 推論ログの詳細な記録
- ステータス監視と診断情報

## 前提条件

- ROS2環境（Foxy以上）
- Python 3.6+
- OpenCV、NumPy、cv_bridge

## パッケージ構成

```
vad_mock/
├── CMakeLists.txt           # CMakeビルドスクリプト
├── package.xml              # パッケージマニフェスト
├── config/
│   └── vad_mock_params.yaml # 設定ファイル
├── launch/
│   └── vad_mock.launch.xml  # 起動ファイル
├── scripts/
│   └── vad_mock_node        # 実行スクリプト
├── vad_mock/
│   ├── __init__.py          # パッケージ初期化
│   └── vad_mock.py          # メインクラス実装
└── README.md                # このファイル
```

## インストール方法

1. ROS2ワークスペースにクローン：

```bash
cd ~/ros2_ws/src
git clone <repository_url> vad_mock
```

2. パッケージをビルド：

```bash
cd ~/ros2_ws
colcon build --packages-select vad_mock
source ~/ros2_ws/install/setup.bash
```

## 使用方法

### 基本的な起動

```bash
ros2 launch vad_mock vad_mock.launch.xml
```

### パラメータを指定して起動

```bash
ros2 launch vad_mock vad_mock.launch.xml drop_strategy:=black
```

### ROSバッグの再生と組み合わせた実行

```bash
# ターミナル1でVADモックを起動
ros2 launch vad_mock vad_mock.launch.xml

# ターミナル2でROSバッグを再生
ros2 bag play /path/to/your/rosbag.bag
```

## 設定パラメータ

`config/vad_mock_params.yaml`で以下のパラメータを設定できます：

| パラメータ      | 説明                                        | デフォルト値       |
|----------------|---------------------------------------------|-------------------|
| drop_strategy  | 画像ドロップ時の戦略（'black'/'past'）        | 'past'            |
| max_slop       | 最大許容時間差（秒）                          | 0.05              |
| queue_size     | メッセージキューサイズ                        | 10                |
| buffer_size    | バッファサイズ                               | 5                 |
| timeout_sec    | タイムアウト時間（秒）                        | 0.1               |
| log_dir        | ログの出力ディレクトリ                        | '/tmp/vad_mock_logs' |
| time_offsets   | カメラのオフセット設定                        | [表示済みの設定]   |

## ログファイル

推論のログは`log_dir`で指定されたディレクトリに保存されます。各ログファイルには以下の情報が含まれます：

- 推論実行時の時刻（ROSの時間）
- 現在のROS wall time
- 各カメラ画像のトピック名とタイムスタンプ
- 画像ドロップが発生した場合の対処方法

## 診断情報

5秒ごとに以下の診断情報が出力されます：

- 処理されたフレーム数
- ドロップしたフレーム数
- 前回の処理からの経過時間
- 各カメラバッファのサイズ

## カスタマイズ

VADモックの動作をカスタマイズするには、以下のファイルを編集します：

- `vad_mock/vad_mock.py`: メインクラスの実装
- `config/vad_mock_params.yaml`: 設定パラメータ

## よくある問題と解決策

### データ同期の問題

- **症状**: カメラ画像が同期されない
- **解決策**: `max_slop`パラメータを大きくして許容時間差を広げる

### 欠損データの処理

- **症状**: 多くの画像ドロップが検出される
- **解決策**: `buffer_size`を増やして過去画像の保持数を増やす、または`drop_strategy`を変更する

### パフォーマンスの問題

- **症状**: 処理が遅い、CPUの使用率が高い
- **解決策**: 不要な画像処理を減らす、バッファサイズを最適化する

## ライセンス

Apache License 2.0
