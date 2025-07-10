#ifndef AUTOWARE_TENSORRT_VAD_VAD_INTERFACE_CONFIG_HPP_
#define AUTOWARE_TENSORRT_VAD_VAD_INTERFACE_CONFIG_HPP_

#include <vector>
#include <memory>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <tf2_ros/buffer.h>

namespace autoware::tensorrt_vad {

class VadInterfaceConfig {
public:
  int32_t input_image_width_;
  int32_t input_image_height_;
  int32_t target_image_width_;
  int32_t target_image_height_;
  std::array<float, 6> point_cloud_range_;
  int32_t bev_h_;
  int32_t bev_w_;
  float default_patch_angle_;
  int32_t default_command_;
  std::vector<float> default_shift_;
  std::array<float, 3> image_normalization_param_mean_;
  std::array<float, 3> image_normalization_param_std_;
  Eigen::Matrix4f vad2base_;
  Eigen::Matrix4f base2vad_;
  std::unordered_map<int32_t, int32_t> autoware_to_vad_camera_mapping_;

  // ROS 2のdeclare_parameterで，std::vector<float>やstd::vector<int32_t>を受け取ることができないため，doubleやint64_tを使用
  VadInterfaceConfig(
    int32_t input_image_width, int32_t input_image_height,
    int32_t target_image_width, int32_t target_image_height,
    const std::vector<double>& point_cloud_range,
    int32_t bev_h, int32_t bev_w,
    double default_patch_angle,
    int32_t default_command,
    const std::vector<double>& default_shift,
    const std::vector<double>& image_normalization_param_mean,
    const std::vector<double>& image_normalization_param_std,
    const std::vector<double>& vad2base,
    const std::vector<int64_t>& autoware_to_vad_camera_mapping)
    : input_image_width_(input_image_width),
      input_image_height_(input_image_height),
      target_image_width_(target_image_width),
      target_image_height_(target_image_height),
      bev_h_(bev_h),
      bev_w_(bev_w),
      default_patch_angle_(static_cast<float>(default_patch_angle)),
      default_command_(default_command)
  {
    // point_cloud_range: 6 elements
    for (int i = 0; i < 6; ++i) {
      point_cloud_range_[i] = static_cast<float>(point_cloud_range[i]);
    }
    // default_shift: copy and convert
    default_shift_.clear();
    for (auto v : default_shift) default_shift_.push_back(static_cast<float>(v));
    // normalization mean/std: 3 elements
    for (int i = 0; i < 3; ++i) {
      image_normalization_param_mean_[i] = static_cast<float>(image_normalization_param_mean[i]);
      image_normalization_param_std_[i] = static_cast<float>(image_normalization_param_std[i]);
    }
    // vad2base: 16 elements, row-major
    vad2base_ = Eigen::Matrix4f::Identity();
    for (int i = 0; i < 16; ++i) {
      vad2base_(i/4, i%4) = static_cast<float>(vad2base[i]);
    }
    // base2vad: inverse
    base2vad_ = vad2base_.inverse();
    // camera mapping: convert from vector to map
    autoware_to_vad_camera_mapping_.clear();
    for (size_t i = 0; i + 1 < autoware_to_vad_camera_mapping.size(); i += 2) {
        int32_t key = static_cast<int32_t>(autoware_to_vad_camera_mapping[i]);
        int32_t value = static_cast<int32_t>(autoware_to_vad_camera_mapping[i + 1]);
        autoware_to_vad_camera_mapping_[key] = value;
    }
  }
};

} // namespace autoware::tensorrt_vad

#endif // AUTOWARE_TENSORRT_VAD_VAD_INTERFACE_CONFIG_HPP_
