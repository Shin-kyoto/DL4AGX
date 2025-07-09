#include "vad_interface.hpp"
#include <stb/stb_image.h>
#include <stb/stb_image_resize.h>
#include <rclcpp/rclcpp.hpp>
#include <tf2/exceptions.h>
#include <cmath>

namespace autoware::tensorrt_vad
{

VadInterface::VadInterface(
    int32_t input_image_width, int32_t input_image_height,
    int32_t target_image_width, int32_t target_image_height,
    const std::vector<double>& point_cloud_range,
    int32_t bev_h, int32_t bev_w,
    double default_patch_angle,
    int32_t default_command,
    const std::vector<double>& default_shift,
    const std::vector<double>& image_normalization_param_mean,
    const std::vector<double>& image_normalization_param_std) 
  : target_image_width_(target_image_width),
    target_image_height_(target_image_height),
    input_image_width_(input_image_width),
    input_image_height_(input_image_height),
    bev_h_(bev_h),
    bev_w_(bev_w),
    default_patch_angle_(static_cast<float>(default_patch_angle)),
    default_command_(default_command)
{
  // default_shiftをdoubleからfloatに変換してコピー
  default_shift_.resize(default_shift.size());
  for (size_t i = 0; i < default_shift.size(); ++i) {
    default_shift_[i] = static_cast<float>(default_shift[i]);
  }
  
  // point_cloud_rangeの配列への代入（doubleからfloatに変換）
  for (size_t i = 0; i < 6 && i < point_cloud_range.size(); ++i) {
    point_cloud_range_[i] = static_cast<float>(point_cloud_range[i]);
  }
  
  // image_normalization_param_meanの配列への代入（doubleからfloatに変換）
  for (size_t i = 0; i < 3 && i < image_normalization_param_mean.size(); ++i) {
    image_normalization_param_mean_[i] = static_cast<float>(image_normalization_param_mean[i]);
  }
  
  // image_normalization_param_stdの配列への代入（doubleからfloatに変換）
  for (size_t i = 0; i < 3 && i < image_normalization_param_std.size(); ++i) {
    image_normalization_param_std_[i] = static_cast<float>(image_normalization_param_std[i]);
  }
  
  // AutowareカメラインデックスからVADカメラインデックスへのマッピング
  autoware_to_vad_ = {
    {0, 0}, // FRONT
    {4, 1}, // FRONT_RIGHT
    {2, 2}, // FRONT_LEFT
    {1, 3}, // BACK
    {3, 4}, // BACK_LEFT
    {5, 5}  // BACK_RIGHT
  };
}

VadInputData VadInterface::convert(const VadInputTopicData & vad_input_topic_data, const std::vector<float> & prev_can_bus)
{
  VadInputData vad_input_data;

  float scale_width = target_image_width_ / static_cast<float>(input_image_width_);
  float scale_height = target_image_height_ / static_cast<float>(input_image_height_);

  // Process lidar2img transformation
  vad_input_data.lidar2img_ = process_lidar2img(
    vad_input_topic_data.tf_static,
    vad_input_topic_data.camera_infos,
    scale_width, scale_height
  );
  
  // Process can_bus and shift data
  auto [can_bus, shift] = process_can_bus_shift(
    vad_input_topic_data.kinematic_state,
    vad_input_topic_data.imu_raw,
    prev_can_bus
  );
  vad_input_data.can_bus_ = can_bus;
  vad_input_data.shift_ = shift;
  
  // Process image data
  vad_input_data.camera_images_ = process_image(vad_input_topic_data.images);
  
  // Set default command
  vad_input_data.command_ = default_command_;
  
  return vad_input_data;
}

void VadInterface::add_vad_to_base_link_transform(tf2_ros::Buffer & buffer, const rclcpp::Time & stamp) const
{
  geometry_msgs::msg::TransformStamped transform;
  transform.header.stamp = stamp;
  transform.header.frame_id = "vad_base_link";
  transform.child_frame_id = "base_link";
  transform.transform.translation.x = 0.0;
  transform.transform.translation.y = 0.0;
  transform.transform.translation.z = 0.0;
  // nuScenes(vad) -> Autoware(base) は +90度 Z回転
  Eigen::Quaternionf q_rot(Eigen::AngleAxisf(static_cast<float>(M_PI / 2.0), Eigen::Vector3f::UnitZ()));
  transform.transform.rotation = tf2::toMsg(q_rot.cast<double>());
  buffer.setTransform(transform, "default_authority", true); // 静的変換として登録
}

std::optional<Eigen::Matrix4f> VadInterface::lookup_vad_to_base_rt(tf2_ros::Buffer & buffer) const
{
  std::string target_frame = "base_link";
  std::string source_frame = "vad_base_link";

  try {
    geometry_msgs::msg::TransformStamped lookup_result =
        buffer.lookupTransform(target_frame, source_frame, tf2::TimePointZero);
    
    // geometry_msgs::msg::TransformからEigen::Matrix4fへの手動変換
    Eigen::Matrix4f transform_matrix = Eigen::Matrix4f::Identity();
    
    // 並進部分
    transform_matrix(0, 3) = lookup_result.transform.translation.x;
    transform_matrix(1, 3) = lookup_result.transform.translation.y;
    transform_matrix(2, 3) = lookup_result.transform.translation.z;
    
    // 回転部分（クォータニオンから回転行列への変換）
    Eigen::Quaternionf q(
        lookup_result.transform.rotation.w,
        lookup_result.transform.rotation.x,
        lookup_result.transform.rotation.y,
        lookup_result.transform.rotation.z);
    transform_matrix.block<3, 3>(0, 0) = q.toRotationMatrix();
    
    return transform_matrix;

  } catch (const tf2::TransformException &ex) {
    RCLCPP_ERROR(rclcpp::get_logger("VadInterface"), "TF変換の取得に失敗: %s -> %s. Reason: %s",
                 source_frame.c_str(), target_frame.c_str(), ex.what());
    return std::nullopt;
  }
}

std::optional<Eigen::Matrix4f> VadInterface::lookup_base_to_camera_rt(tf2_ros::Buffer & buffer, int32_t autoware_camera_id) const
{
  std::string target_frame = "camera" + std::to_string(autoware_camera_id) + "/camera_optical_link";
  std::string source_frame = "base_link";

  try {
    geometry_msgs::msg::TransformStamped lookup_result =
        buffer.lookupTransform(target_frame, source_frame, tf2::TimePointZero);
    
    // geometry_msgs::msg::TransformからEigen::Matrix4fへの手動変換
    Eigen::Matrix4f transform_matrix = Eigen::Matrix4f::Identity();
    
    // 並進部分
    transform_matrix(0, 3) = lookup_result.transform.translation.x;
    transform_matrix(1, 3) = lookup_result.transform.translation.y;
    transform_matrix(2, 3) = lookup_result.transform.translation.z;
    
    // 回転部分（クォータニオンから回転行列への変換）
    Eigen::Quaternionf q(
        lookup_result.transform.rotation.w,
        lookup_result.transform.rotation.x,
        lookup_result.transform.rotation.y,
        lookup_result.transform.rotation.z);
    transform_matrix.block<3, 3>(0, 0) = q.toRotationMatrix();
    
    return transform_matrix;

  } catch (const tf2::TransformException &ex) {
    RCLCPP_ERROR(rclcpp::get_logger("VadInterface"), "TF変換の取得に失敗: %s -> %s. Reason: %s",
                 source_frame.c_str(), target_frame.c_str(), ex.what());
    return std::nullopt;
  }
}

Eigen::Matrix4f VadInterface::create_viewpad(const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info) const
{
  Eigen::Matrix3f k_matrix;
  for (int32_t i = 0; i < 3; ++i) {
    for (int32_t j = 0; j < 3; ++j) {
      k_matrix(i, j) = camera_info->k[i * 3 + j];
    }
  }
  
  // viewpadを作成
  Eigen::Matrix4f viewpad = Eigen::Matrix4f::Zero();
  viewpad.block<3, 3>(0, 0) = k_matrix;
  viewpad(3, 3) = 1.0f;
  
  return viewpad;
}

Eigen::Matrix4f VadInterface::apply_scaling(const Eigen::Matrix4f & lidar2img, float scale_width, float scale_height) const
{
  Eigen::Matrix4f scale_matrix = Eigen::Matrix4f::Identity();
  scale_matrix(0, 0) = scale_width;
  scale_matrix(1, 1) = scale_height;
  return scale_matrix * lidar2img;
}

std::vector<float> VadInterface::matrix_to_flat(const Eigen::Matrix4f & matrix) const
{
  std::vector<float> flat(16);
  int32_t k = 0;
  for (int32_t i = 0; i < 4; ++i) {
    for (int32_t j = 0; j < 4; ++j) {
      flat[k++] = matrix(i, j);
    }
  }
  return flat;
}

Lidar2ImgData VadInterface::process_lidar2img(
  const tf2_msgs::msg::TFMessage::ConstSharedPtr & tf_static,
  const std::vector<sensor_msgs::msg::CameraInfo::ConstSharedPtr> & camera_infos,
  float scale_width, float scale_height) const
{
  std::vector<float> frame_lidar2img(16 * 6, 0.0f); // 6カメラ分のスペースを確保

  // TFバッファの初期化
  auto clock = std::make_shared<rclcpp::Clock>(RCL_ROS_TIME);
  tf2_ros::Buffer tf_buffer(clock);

  // rosbagから読み込んだ変換 (base_link -> camera) をバッファに登録
  for (const auto &transform : tf_static->transforms) {
    tf_buffer.setTransform(transform, "default_authority", true);
  }

  // vad_base_link -> base_link の変換をバッファに登録
  rclcpp::Time stamp(0, 0, RCL_ROS_TIME);
  if (!tf_static->transforms.empty()) {
    stamp = tf_static->transforms[0].header.stamp;
  }
  add_vad_to_base_link_transform(tf_buffer, stamp);

  // 各カメラの処理
  for (int32_t autoware_camera_id = 0; autoware_camera_id < 6; ++autoware_camera_id) {
    if (!camera_infos[autoware_camera_id]) {
      continue;
    }

    auto base_to_camera_rt_opt = lookup_base_to_camera_rt(tf_buffer, autoware_camera_id);
    if (!base_to_camera_rt_opt) continue;
    Eigen::Matrix4f base_to_camera_rt = *base_to_camera_rt_opt;

    auto vad_to_base_rt_opt = lookup_vad_to_base_rt(tf_buffer);
    if (!vad_to_base_rt_opt) continue;
    Eigen::Matrix4f vad_to_base_rt = *vad_to_base_rt_opt;

    Eigen::Matrix4f viewpad = create_viewpad(camera_infos[autoware_camera_id]);
    Eigen::Matrix4f lidar2cam_rt = base_to_camera_rt * vad_to_base_rt;
    Eigen::Matrix4f lidar2img = viewpad * lidar2cam_rt;

    // スケーリングを適用
    lidar2img = apply_scaling(lidar2img, scale_width, scale_height);

    // 結果を格納
    std::vector<float> lidar2img_flat = matrix_to_flat(lidar2img);

    // lidar2imgの計算後、VADカメラIDの位置に格納
    int32_t vad_camera_id = autoware_to_vad_.at(autoware_camera_id);
    if (vad_camera_id >= 0 && vad_camera_id < 6) {
      std::copy(lidar2img_flat.begin(), lidar2img_flat.end(),
                frame_lidar2img.begin() + vad_camera_id * 16);
    }
  }

  return frame_lidar2img;
}

std::optional<std::tuple<unsigned char*, int32_t, int32_t>> VadInterface::resize_image(
  unsigned char *image_data, int32_t width, int32_t height, int32_t channels, 
  int32_t target_image_width, int32_t target_image_height) const
{
  unsigned char *resized_data = (unsigned char *)malloc(target_image_width * target_image_height * channels);
  
  int32_t resize_result = stbir_resize_uint8(image_data, width, height, 0, resized_data,
                                        target_image_width, target_image_height, 0, channels);
  
  if (!resize_result) {
    free(resized_data);
    return std::nullopt;
  }
  
  // 元のデータを解放
  stbi_image_free(image_data);
  
  return std::make_tuple(resized_data, target_image_width, target_image_height);
}

std::vector<float> VadInterface::normalize_image(unsigned char *image_data, int32_t width, int32_t height) const
{
  std::vector<float> normalized_image_data(width * height * 3);
  
  // BGRの順で処理
  for (int32_t c = 0; c < 3; ++c) {
    for (int32_t h = 0; h < height; ++h) {
      for (int32_t w = 0; w < width; ++w) {
        int32_t src_idx = (h * width + w) * 3 + (2 - c); // BGR -> RGB
        int32_t dst_idx = c * height * width + h * width + w; // CHW形式
        float pixel_value = static_cast<float>(image_data[src_idx]);
        normalized_image_data[dst_idx] = (pixel_value - image_normalization_param_mean_[c]) / image_normalization_param_std_[c];
      }
    }
  }
  
  return normalized_image_data;
}

CameraImagesData VadInterface::process_image(
  const std::vector<sensor_msgs::msg::Image::ConstSharedPtr> & images) const
{
  std::vector<std::vector<float>> frame_images;
  frame_images.resize(6); // VADカメラ順序で初期化

  // 各カメラの画像を処理
  for (int32_t autoware_idx = 0; autoware_idx < 6; ++autoware_idx) {
    const auto &image_msg = images[autoware_idx];

    // 画像データの処理
    int32_t width, height, channels;
    unsigned char *image_data = stbi_load_from_memory(
        image_msg->data.data(), static_cast<int>(image_msg->data.size()),
        &width, &height, &channels, STBI_rgb); // RGBとして読み込む

    // サイズが目標と違う場合はリサイズする
    if (width != target_image_width_ || height != target_image_height_) {
      auto resize_result = resize_image(image_data, width, height, channels, target_image_width_, target_image_height_);
      
      if (resize_result.has_value()) {
        // リサイズされたデータとサイズを取得
        auto [new_image_data, new_width, new_height] = resize_result.value();
        image_data = new_image_data;
        width = new_width;
        height = new_height;
      }
    }

    // 画像を正規化
    std::vector<float> normalized_image_data = normalize_image(image_data, width, height);

    // VADカメラ順序で格納
    int32_t vad_idx = autoware_to_vad_.at(autoware_idx);
    frame_images[vad_idx] = normalized_image_data;
  }

  // 画像データを連結
  std::vector<float> concatenated_data;
  size_t single_camera_size = 3 * target_image_height_ * target_image_width_;
  concatenated_data.reserve(single_camera_size * 6);

  // カメラの順序: {0, 1, 2, 3, 4, 5}
  for (int32_t camera_idx = 0; camera_idx < 6; ++camera_idx) {
    const auto &img_data = frame_images[camera_idx];
    if (img_data.size() != single_camera_size) {
      throw std::runtime_error("画像サイズが不正です: " +
                               std::to_string(camera_idx));
    }
    concatenated_data.insert(concatenated_data.end(), img_data.begin(),
                             img_data.end());
  }

  return concatenated_data;
}

std::vector<float> VadInterface::calculate_can_bus(
  const nav_msgs::msg::Odometry::ConstSharedPtr & kinematic_state,
  const sensor_msgs::msg::Imu::ConstSharedPtr & imu_raw,
  const std::vector<float> & prev_can_bus) const
{
  std::vector<float> can_bus(18, 0.0f);

  // Apply Autoware to nuScenes coordinate transformation to position
  auto [ns_x, ns_y] =
      aw2ns_xy(kinematic_state->pose.pose.position.x,
                kinematic_state->pose.pose.position.y);

  // translation (0:3)
  can_bus[0] = ns_x;
  can_bus[1] = ns_y;
  can_bus[2] = static_cast<float>(kinematic_state->pose.pose.position.z);

  // Apply Autoware to nuScenes coordinate transformation to orientation
  Eigen::Quaternionf q_aw(
      kinematic_state->pose.pose.orientation.w,
      kinematic_state->pose.pose.orientation.x,
      kinematic_state->pose.pose.orientation.y,
      kinematic_state->pose.pose.orientation.z);

  Eigen::Quaternionf q_ns = aw2ns_quaternion(q_aw);

  // rotation (3:7)
  can_bus[3] = q_ns.x();
  can_bus[4] = q_ns.y();
  can_bus[5] = q_ns.z();
  can_bus[6] = q_ns.w();

  // Apply Autoware to nuScenes coordinate transformation to acceleration
  auto [ns_ax, ns_ay] =
      aw2ns_xy(imu_raw->linear_acceleration.x,
                imu_raw->linear_acceleration.y);

  // acceleration (7:10)
  can_bus[7] = ns_ax;
  can_bus[8] = ns_ay;
  can_bus[9] = static_cast<float>(imu_raw->linear_acceleration.z);

  // Apply Autoware to nuScenes coordinate transformation to angular velocity
  auto [ns_wx, ns_wy] =
      aw2ns_xy(kinematic_state->twist.twist.angular.x,
                kinematic_state->twist.twist.angular.y);

  // angular velocity (10:13)
  can_bus[10] = ns_wx;
  can_bus[11] = ns_wy;
  can_bus[12] = static_cast<float>(kinematic_state->twist.twist.angular.z);

  // Apply Autoware to nuScenes coordinate transformation to velocity
  auto [ns_vx, ns_vy] =
      aw2ns_xy(kinematic_state->twist.twist.linear.x,
                kinematic_state->twist.twist.linear.y);

  // velocity (13:16)
  can_bus[13] = ns_vx;
  can_bus[14] = ns_vy;
  can_bus[15] = 0.0f; // z方向の速度は0とする

  // patch_angle[rad]の計算 (16)
  double yaw = std::atan2(
      2.0 * (can_bus[6] * can_bus[5] + can_bus[3] * can_bus[4]),
      1.0 - 2.0 * (can_bus[4] * can_bus[4] + can_bus[5] * can_bus[5]));
  if (yaw < 0)
    yaw += 2 * M_PI;
  can_bus[16] = static_cast<float>(yaw);

  // patch_angle[deg]の計算 (17)
  if (!prev_can_bus.empty()) {
    float prev_angle = prev_can_bus[16];
    can_bus[17] = (yaw - prev_angle) * 180.0f / M_PI;
  } else {
    can_bus[17] = default_patch_angle_; // 最初のフレームのデフォルト値
  }

  return can_bus;
}

std::vector<float> VadInterface::calculate_shift(float delta_x, float delta_y, float patch_angle_rad) const
{
  float real_w = point_cloud_range_[3] - point_cloud_range_[0];
  float real_h = point_cloud_range_[4] - point_cloud_range_[1];
  float grid_length[] = {real_h / bev_h_, real_w / bev_w_};

  float ego_angle = patch_angle_rad / M_PI * 180.0;
  float grid_length_y = grid_length[0];
  float grid_length_x = grid_length[1];

  float translation_length = std::sqrt(delta_x * delta_x + delta_y * delta_y);
  float translation_angle = std::atan2(delta_y, delta_x) / M_PI * 180.0;
  float bev_angle = ego_angle - translation_angle;

  float shift_y = translation_length * std::cos(bev_angle / 180.0 * M_PI) /
                  grid_length_y / bev_h_;
  float shift_x = translation_length * std::sin(bev_angle / 180.0 * M_PI) /
                  grid_length_x / bev_w_;

  return {shift_x, shift_y};
}

std::tuple<CanBusData, ShiftData> VadInterface::process_can_bus_shift(
  const nav_msgs::msg::Odometry::ConstSharedPtr & kinematic_state,
  const sensor_msgs::msg::Imu::ConstSharedPtr & imu_raw,
  const std::vector<float> & prev_can_bus) const
{
  std::vector<float> shift;

  // can_busデータの計算
  std::vector<float> can_bus = calculate_can_bus(kinematic_state, imu_raw, prev_can_bus);

  // シフトデータの計算
  if (!prev_can_bus.empty()) {
    float delta_x = can_bus[0] - prev_can_bus[0];  // translation[0]
    float delta_y = can_bus[1] - prev_can_bus[1];  // translation[1]

    double yaw = std::atan2(
        2.0 * (can_bus[6] * can_bus[5] + can_bus[3] * can_bus[4]),
        1.0 - 2.0 * (can_bus[4] * can_bus[4] + can_bus[5] * can_bus[5]));
    if (yaw < 0)
      yaw += 2 * M_PI;

    shift = calculate_shift(delta_x, delta_y, yaw);
  } else {
    shift = default_shift_;
  }

  return std::make_tuple(can_bus, shift);
}

std::pair<float, float> VadInterface::aw2ns_xy(float aw_x, float aw_y)
{
  float ns_x = -aw_y;
  float ns_y = aw_x;
  return {ns_x, ns_y};
}

Eigen::Quaternionf VadInterface::aw2ns_quaternion(const Eigen::Quaternionf & q_aw)
{
  // Create a -90-degree rotation around Z-axis (Autoware -> nuScenes)
  Eigen::Quaternionf q_rotation(
      Eigen::AngleAxisf(-M_PI / 2, Eigen::Vector3f::UnitZ()));

  // Apply the rotation
  return q_rotation * q_aw;
}

}  // namespace autoware::tensorrt_vad
