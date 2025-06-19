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

#ifndef AUTOWARE_TENSORRT_VAD_VAD_INTERFACE_HPP_
#define AUTOWARE_TENSORRT_VAD_VAD_INTERFACE_HPP_

#include <vector>
#include <map>

#include "rclcpp/time.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "tf2_msgs/msg/tf_message.hpp"
#include "autoware_planning_msgs/msg/trajectory.hpp"
#include "autoware_perception_msgs/msg/detected_objects.hpp"

namespace autoware::tensorrt_vad
{

struct CameraMapping {
    std::int64_t autoware_camera_id;
    int vad_camera_id;
};

struct ImageProcessingParams
{
  int target_width;
  int target_height;
  std::vector<double> mean;
  std::vector<double> std;
};

struct VadTopicData
{
  // このデータセットの基準となるタイムスタンプ
  rclcpp::Time stamp;

  // 複数のカメラからの画像データ。
  // launchファイルでリマップされた ~/input/image0, ~/input/image1, ... に対応します。
  // vectorのインデックスが autoware_camera_id となります。
  std::vector<sensor_msgs::msg::Image::ConstSharedPtr> images;

  // 上記の各画像に対応するカメラのキャリブレーション情報
  std::vector<sensor_msgs::msg::CameraInfo::ConstSharedPtr> camera_infos;

  // 静的な座標変換情報（例: lidar_to_camera）
  tf2_msgs::msg::TFMessage::ConstSharedPtr tf_static;

  // 車両の運動状態データ（/localization/kinematic_state などから）
  nav_msgs::msg::Odometry::ConstSharedPtr kinematic_state;

  // IMUデータ（/sensing/imu/tamagawa/imu_raw などから）
  sensor_msgs::msg::Imu::ConstSharedPtr imu_raw;

  // フレームが完成しているかをチェック
  bool is_complete() const {
    if (images.size() != 6 || camera_infos.size() != 6) return false;
    
    for (int i = 0; i < 6; ++i) {
      if (!images[i] || !camera_infos[i]) return false;
    }
    
    return tf_static != nullptr && 
           kinematic_state != nullptr && 
           imu_raw != nullptr;
  }
};

class VadInterface {
  /**
   * @brief コンストラクタ。各種設定情報を受け取り、内部に保持する。
   * @param mappings カメラのマッピング情報
   * @param image_params 画像処理のパラメータ
   */
  explicit VadInterface(
    const std::vector<CameraMapping> & mappings, const ImageProcessingParams & image_params);

    VadInputData convert_ros_to_vad_input(const VadTopicData & topic_data) const;

    std::tuple<
        std::optional<autoware_planning_msgs::msg::Trajectory>,
        std::optional<autoware_perception_msgs::msg::DetectedObjects>>
    convert_vad_output_to_ros(const VadOutputData & vad_output) const;

private:
    // --- Private Helper Methods ---

    [[nodiscard]] std::vector<float> convert_to_camera_images(
        const std::vector<sensor_msgs::msg::Image::ConstSharedPtr> & images) const;

    [[nodiscard]] std::vector<float> convert_to_lidar2img(
        const std::vector<sensor_msgs::msg::CameraInfo::ConstSharedPtr> & camera_infos,
        const tf2_msgs::msg::TFMessage::ConstSharedPtr & tf_static) const;

    [[nodiscard]] std::tuple<std::vector<float>, std::vector<float>> convert_to_can_bus_and_shift(
        const nav_msgs::msg::Odometry::ConstSharedPtr & kinematic_state,
        const sensor_msgs::msg::Imu::ConstSharedPtr & imu_raw) const;

    // --- Member Variables ---

    // メンバとしてはmapで保持する
    std::map<std::int64_t, int> camera_mapping_;
    const ImageProcessingParams image_params_;
};

}  // namespace autoware::tensorrt_vad

#endif  // AUTOWARE_TENSORRT_VAD_VAD_INTERFACE_HPP_
