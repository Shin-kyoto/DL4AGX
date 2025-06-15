// Copyright 2025 Shin-kyoto.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef AUTOWARE_TENSORRT_VAD_VAD_TRT_HPP_
#define AUTOWARE_TENSORRT_VAD_VAD_TRT_HPP_

#include <optional>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <nlohmann/json.hpp>
#include "net.h"

using json = nlohmann::json;

namespace autoware::tensorrt_vad
{

// ログレベル定義
enum class LogLevel {
  DEBUG,
  INFO,
  WARN,
  ERROR
};

// ロガーインターフェース
class VadLogger {
public:
  virtual ~VadLogger() = default;
  virtual void log(LogLevel level, const std::string& message) = 0;
  
  // 便利なメソッド
  void debug(const std::string& message) { log(LogLevel::DEBUG, message); }
  void info(const std::string& message) { log(LogLevel::INFO, message); }
  void warn(const std::string& message) { log(LogLevel::WARN, message); }
  void error(const std::string& message) { log(LogLevel::ERROR, message); }
};

// VAD推論の入力データ構造
struct VadInputData
{
  // カメラ画像データ（複数カメラ対応）
  std::vector<float> camera_images_;

  // 前回のBEV特徴量（時系列処理用）
  // nets["head_no_prev"]->bindings["out.bev_embed"]
  // std::vector<float> prev_bev_{};

  // シフト情報（img_metas.0[shift]）
  std::vector<float> shift_;

  // LiDAR座標系からカメラ画像座標系への変換行列（img_metas.0[lidar2img]）
  std::vector<float> lidar2img_;

  // CAN-BUSデータ（車両状態情報：速度、角速度など）(img_metas.0[can_bus])
  std::vector<float> can_bus_;

  // コマンドインデックス（軌道選択用）
  int32_t command_{2};
};

// VAD推論の出力データ構造
struct VadOutputData
{
  // 予測された軌道（6つの2D座標点、累積座標として表現）
  // planning[0,1] = 1st point (x,y), planning[2,3] = 2nd point (x,y), ...
  std::vector<float> predicted_trajectory_{};  // size: 12 (6 points * 2 coordinates)

  // // 検出されたオブジェクト
  // std::vector<std::vector<float>> detected_objects_{};

  // // コマンドインデックス（選択された軌道のインデックス）
  // int32_t selected_command_index_{2};
};

// config for Net class
struct NetConfig
{
  std::string name;
  std::string engine_file;
  bool use_graph;
  std::map<std::string, std::map<std::string, std::string>> inputs;
};

// config for VadModel class
struct VadConfig
{
  std::string plugins_path;
  int32_t warm_up_num;
  std::vector<NetConfig> nets_config;
};

class NetworkParam
{
public:
  NetworkParam(std::string onnx_path, std::string engine_path, std::string trt_precision)
  : onnx_path_(std::move(onnx_path)), engine_path_(std::move(engine_path)), trt_precision_(std::move(trt_precision))
  {
  }

  std::string onnx_path() const { return onnx_path_; }
  std::string engine_path() const { return engine_path_; }
  std::string trt_precision() const { return trt_precision_; }

private:
  std::string onnx_path_;
  std::string engine_path_;
  std::string trt_precision_;
};

// VADモデルクラス - CUDA/TensorRTを用いた推論を担当
// 注意: デフォルトはサイレントロガーです。実際の使用ではRosVadLoggerの使用を推奨します。
// 例: VadModel model(config, std::make_shared<RosVadLogger>(node));
class VadModel
{
public:
  // コンストラクタ（サイレントロガーを使用）
  VadModel(const VadConfig& config);

  // ロガー付きコンストラクタ（推奨）
  VadModel(const VadConfig& config, std::shared_ptr<VadLogger> logger);

  // デストラクタ
  ~VadModel();

  // モデルの初期化
  [[nodiscard]] bool initialize(const std::string & model_path);

  // メイン推論API
  [[nodiscard]] std::optional<VadOutputData> infer(const VadInputData & input);

  // メンバ変数
  std::unique_ptr<nvinfer1::IRuntime, std::function<void(nvinfer1::IRuntime*)>> runtime_;
  cudaStream_t stream_;
  std::unordered_map<std::string, std::shared_ptr<nv::Net>> nets_;
  bool initialized_;

  // 前回のBEV特徴量保存用
  std::shared_ptr<nv::Tensor> saved_prev_bev_;
  bool is_first_frame_;

  // 設定情報の保存
  VadConfig config_;

  // ロガーインスタンス
  std::shared_ptr<VadLogger> logger_;

private:
  // メンバ関数
  std::unique_ptr<nvinfer1::IRuntime, std::function<void(nvinfer1::IRuntime*)>> create_runtime();
  bool load_plugin(const std::string& plugin_dir);
  std::unordered_map<std::string, std::shared_ptr<nv::Net>> init_engines(
    const std::vector<NetConfig>& nets_config);
  void warm_up(int32_t warm_up_num);
  
  // infer関数で使用するヘルパー関数
  void load_inputs(const VadInputData& vad_input, const std::string& head_name);
  void enqueue(const std::string& head_name);
  std::shared_ptr<nv::Tensor> save_prev_bev(const std::string& head_name);
  void release_network(const std::string& network_name);
  void load_head();
  VadOutputData postprocess(const std::string& head_name, int32_t cmd);
};

}  // namespace autoware::tensorrt_vad

#endif  // AUTOWARE_TENSORRT_VAD_VAD_TRT_HPP_
