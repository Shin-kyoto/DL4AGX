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
#include <functional>
#include <memory>

#include "rclcpp/time.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "tf2_msgs/msg/tf_message.hpp"
#include "geometry_msgs/msg/transform.hpp"
#include "Eigen/Dense"
#include "vad_model.hpp"

namespace autoware::tensorrt_vad
{

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

// Lidar2Cam変換クラス
class Lidar2Cam {
private:
    Eigen::Matrix4f lidar2cam_;
    bool is_valid_ = false;
    
    // 座標変換関数へのポインタ
    std::function<std::pair<float, float>(float, float)> transform_autoware2vad_xy_;
    std::function<Eigen::Quaternionf(const Eigen::Quaternionf&)> transform_autoware2vad_q_;
    
    // デバッグ用メンバ変数
    geometry_msgs::msg::Transform transform_;
    Eigen::Vector3f translation_autoware_;
    Eigen::Quaternionf q_autoware_;
    Eigen::Vector3f translation_vad_;
    Eigen::Quaternionf q_vad_;
    
public:
    /**
     * @brief Lidar2Cam constructor
     * @param transform Transform message
     * @param transform_autoware2vad_xy Function to transform x,y coordinates from Autoware to VAD
     * @param transform_autoware2vad_q Function to transform quaternion from Autoware to VAD
     */
    Lidar2Cam(const geometry_msgs::msg::Transform& transform,
              std::function<std::pair<float, float>(float, float)> transform_autoware2vad_xy,
              std::function<Eigen::Quaternionf(const Eigen::Quaternionf&)> transform_autoware2vad_q);
    
    void load_from_tf(const geometry_msgs::msg::Transform& transform);
    
    Eigen::Matrix4f get_matrix() const { return lidar2cam_; }
    Eigen::Matrix4f get_transpose() const { return lidar2cam_.transpose(); }
    bool is_valid() const { return is_valid_; }
    
    void reset();
    
    /**
     * @brief デバッグ情報をVadLoggerに出力
     * @param logger VadLogger
     * @param camera_id カメラID（デバッグ情報に含める）
     */
    void log_debug_info(const VadLogger& logger, int camera_id = -1) const;
    
protected:
    /**
     * @brief Transform translation from Autoware to VAD coordinates
     * @param translation_autoware Translation in Autoware coordinates
     * @return Translation in VAD coordinates
     */
    Eigen::Vector3f transform_translation_to_vad(const Eigen::Vector3f& translation_autoware) const;
    
    /**
     * @brief Transform quaternion from Autoware to VAD coordinates
     * @param q_autoware Quaternion in Autoware coordinates
     * @return Quaternion in VAD coordinates
     */
    Eigen::Quaternionf transform_quaternion_to_vad(const Eigen::Quaternionf& q_autoware) const;
};

// CameraProjectionMatrix変換クラス
class CameraProjectionMatrix {
private:
    Eigen::Matrix4f projection_matrix_;
    bool is_valid_ = false;
    
    // デバッグ用メンバ変数
    sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info_;
    Eigen::Matrix3f k_matrix_;
    
public:
    /**
     * @brief CameraProjectionMatrix constructor
     * @param camera_info Camera information containing intrinsic parameters
     * 
     * Creates a 4x4 projection matrix from camera intrinsic parameters.
     * a.k.a. viewpad (nuScenes terminology)
     */
    explicit CameraProjectionMatrix(const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info);
    
    Eigen::Matrix4f get_matrix() const { return projection_matrix_; }
    bool is_valid() const { return is_valid_; }
    
    void reset();
    
    /**
     * @brief デバッグ情報をVadLoggerに出力
     * @param logger VadLogger
     * @param camera_id カメラID（デバッグ情報に含める）
     */
    void log_debug_info(const VadLogger& logger, int camera_id = -1) const;
    
protected:
    /**
     * @brief Create 4x4 projection matrix from camera info
     * @param camera_info Camera information
     * @return 4x4 projection matrix (a.k.a. viewpad)
     */
    Eigen::Matrix4f create_projection_matrix(const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info) const;
};

// Lidar2Img変換クラス
class Lidar2Img {
private:
    Eigen::Matrix4f lidar2img_;
    bool is_valid_ = false;
    
    // デバッグ用メンバ変数
    Eigen::Matrix4f projection_matrix_;
    Eigen::Matrix4f lidar2cam_matrix_;
    Eigen::Matrix4f lidar2cam_transpose_;
    Eigen::Matrix4f lidar2img_raw_;
    Eigen::Matrix4f lidar2img_scaled_;
    std::vector<float> lidar2img_scaled_flat_;
    
public:
    /**
     * @brief Lidar2Img constructor
     * @param projection_matrix Camera projection matrix (a.k.a. viewpad)
     * @param lidar2cam Lidar to camera transformation matrix
     */
    Lidar2Img(const CameraProjectionMatrix& projection_matrix, const Lidar2Cam& lidar2cam);
    
    void apply_scaling(float scale_width, float scale_height);
    
    std::vector<float> to_flat_vector() const;
    
    Eigen::Matrix4f get_matrix() const { return lidar2img_; }
    bool is_valid() const { return is_valid_; }
    
    void reset();
    
    /**
     * @brief デバッグ情報をVadLoggerに出力
     * @param logger VadLogger
     * @param camera_id カメラID（デバッグ情報に含める）
     */
    void log_debug_info(const VadLogger& logger, int camera_id = -1) const;
    
protected:
    /**
     * @brief Calculate lidar2img matrix from components
     * @param projection_matrix Camera projection matrix
     * @param lidar2cam_transpose Transpose of lidar2cam matrix
     * @return lidar2img matrix
     */
    Eigen::Matrix4f calculate_lidar2img(const Eigen::Matrix4f& projection_matrix, 
                                       const Eigen::Matrix4f& lidar2cam_transpose) const;
    
    /**
     * @brief Apply scaling to lidar2img matrix
     * @param lidar2img Original lidar2img matrix
     * @param scale_width Width scaling factor
     * @param scale_height Height scaling factor
     * @return Scaled lidar2img matrix
     */
    Eigen::Matrix4f apply_scaling_to_matrix(const Eigen::Matrix4f& lidar2img, 
                                           float scale_width, float scale_height) const;
    
    /**
     * @brief Convert 4x4 matrix to flat vector
     * @param matrix 4x4 matrix
     * @return Flat vector of 16 elements
     */
    std::vector<float> matrix_to_flat_vector(const Eigen::Matrix4f& matrix) const;
};

}  // namespace autoware::tensorrt_vad

#endif  // AUTOWARE_TENSORRT_VAD_VAD_INTERFACE_HPP_
