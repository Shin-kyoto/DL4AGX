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
#include <unordered_map>
#include <optional>
#include <tuple>

#include "rclcpp/time.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "tf2_msgs/msg/tf_message.hpp"
#include "tf2_ros/buffer.h"
#include <tf2_eigen/tf2_eigen.h>

#include "vad_model.hpp" 

namespace autoware::tensorrt_vad
{

struct VadInputTopicData
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
    
    for (int32_t i = 0; i < 6; ++i) {
      if (!images[i] || !camera_infos[i]) return false;
    }
    
    return tf_static != nullptr && 
           kinematic_state != nullptr && 
           imu_raw != nullptr;
  }
};

// 各process_*メソッドの戻り値となるデータ構造
using CameraImagesData = std::vector<float>;
using ShiftData = std::vector<float>;
using Lidar2ImgData = std::vector<float>;
using CanBusData = std::vector<float>;

/**
 * @class VadInterface
 * @brief ROS topicデータをVADの入力形式に変換するインターフェース
 */
class VadInterface
{
public:
  explicit VadInterface(
    int32_t input_image_width, int32_t input_image_height,
    int32_t target_image_width, int32_t target_image_height,
    const std::vector<double>& point_cloud_range,
    int32_t bev_h, int32_t bev_w,
    double default_patch_angle,
    int32_t default_command,
    const std::vector<double>& default_shift,
    const std::vector<double>& image_normalization_param_mean,
    const std::vector<double>& image_normalization_param_std);

  VadInputData convert(const VadInputTopicData & vad_input_topic_data, const std::vector<float> & prev_can_bus = {});

private:
  // --- クラスの不変条件 (メンバ変数) ---
  std::unordered_map<int32_t, int32_t> autoware_to_vad_;
  int32_t target_image_width_, target_image_height_;
  int32_t input_image_width_, input_image_height_;
  float point_cloud_range_[6];
  int32_t bev_h_, bev_w_;
  float default_patch_angle_;
  int32_t default_command_;
  std::vector<float> default_shift_;
  // 正規化のパラメータ
  float image_normalization_param_mean_[3];
  float image_normalization_param_std_[3];

  // --- 内部処理関数 ---
  void add_vad_to_base_link_transform(tf2_ros::Buffer & buffer, const rclcpp::Time & stamp) const;
  
  std::optional<Eigen::Matrix4f> lookup_vad_to_base_rt(tf2_ros::Buffer & buffer) const;
  std::optional<Eigen::Matrix4f> lookup_base_to_camera_rt(tf2_ros::Buffer & buffer, int32_t autoware_camera_id) const;
  Eigen::Matrix4f create_viewpad(const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info) const;
  Eigen::Matrix4f apply_scaling(const Eigen::Matrix4f & lidar2img, float scale_width, float scale_height) const;
  std::vector<float> matrix_to_flat(const Eigen::Matrix4f & matrix) const;
  
  std::optional<std::tuple<unsigned char*, int32_t, int32_t>> resize_image(
    unsigned char *image_data, int32_t width, int32_t height, int32_t channels, 
    int32_t target_image_width, int32_t target_image_height) const;
  std::vector<float> normalize_image(unsigned char *image_data, int32_t width, int32_t height) const;
  
  std::vector<float> calculate_can_bus(
    const nav_msgs::msg::Odometry::ConstSharedPtr & kinematic_state,
    const sensor_msgs::msg::Imu::ConstSharedPtr & imu_raw,
    const std::vector<float> & prev_can_bus) const;
  
  std::vector<float> calculate_shift(float delta_x, float delta_y, float patch_angle_rad) const;
  
  Lidar2ImgData process_lidar2img(
    const tf2_msgs::msg::TFMessage::ConstSharedPtr & tf_static,
    const std::vector<sensor_msgs::msg::CameraInfo::ConstSharedPtr> & camera_infos,
    float scale_width, float scale_height) const;

  std::tuple<CanBusData, ShiftData> process_can_bus_shift(
    const nav_msgs::msg::Odometry::ConstSharedPtr & kinematic_state,
    const sensor_msgs::msg::Imu::ConstSharedPtr & imu_raw,
    const std::vector<float> & prev_can_bus) const;

  CameraImagesData process_image(
    const std::vector<sensor_msgs::msg::Image::ConstSharedPtr> & images) const;

  // --- 静的ヘルパーメソッド ---
  static std::pair<float, float> aw2ns_xy(float aw_x, float aw_y);
  static Eigen::Quaternionf aw2ns_quaternion(const Eigen::Quaternionf & q_aw);
};

}  // namespace autoware::tensorrt_vad

#endif  // AUTOWARE_TENSORRT_VAD_VAD_INTERFACE_HPP_
