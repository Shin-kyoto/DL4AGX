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
  int32_t input_image_width;
  int32_t input_image_height;
  int32_t target_image_width;
  int32_t target_image_height;
  std::array<float, 6> point_cloud_range;
  int32_t bev_h;
  int32_t bev_w;
  float default_patch_angle;
  int32_t default_command;
  std::vector<float> default_shift;
  std::array<float, 3> image_normalization_param_mean;
  std::array<float, 3> image_normalization_param_std;
  Eigen::Matrix4f vad2base;
  Eigen::Matrix4f base2vad;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer;

  // Constructor: accepts all params as double/int, does conversion inside
  VadInterfaceConfig(
    int32_t input_image_width_, int32_t input_image_height_,
    int32_t target_image_width_, int32_t target_image_height_,
    const std::array<double, 6>& point_cloud_range_,
    int32_t bev_h_, int32_t bev_w_,
    double default_patch_angle_,
    int32_t default_command_,
    const std::vector<double>& default_shift_,
    const std::array<double, 3>& image_normalization_param_mean_,
    const std::array<double, 3>& image_normalization_param_std_,
    const std::vector<double>& vad2base_,
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_)
    : input_image_width(input_image_width_),
      input_image_height(input_image_height_),
      target_image_width(target_image_width_),
      target_image_height(target_image_height_),
      bev_h(bev_h_),
      bev_w(bev_w_),
      default_patch_angle(static_cast<float>(default_patch_angle_)),
      default_command(default_command_),
      tf_buffer(tf_buffer_)
  {
    // point_cloud_range: 6 elements
    for (int i = 0; i < 6; ++i) {
      point_cloud_range[i] = static_cast<float>(point_cloud_range_[i]);
    }
    // default_shift: copy and convert
    default_shift.clear();
    for (auto v : default_shift_) default_shift.push_back(static_cast<float>(v));
    // normalization mean/std: 3 elements
    for (int i = 0; i < 3; ++i) {
      image_normalization_param_mean[i] = static_cast<float>(image_normalization_param_mean_[i]);
      image_normalization_param_std[i] = static_cast<float>(image_normalization_param_std_[i]);
    }
    // vad2base: 16 elements, row-major
    vad2base = Eigen::Matrix4f::Identity();
    for (int i = 0; i < 16; ++i) {
      vad2base(i/4, i%4) = static_cast<float>(vad2base_[i]);
    }
    // base2vad: inverse
    base2vad = vad2base.inverse();
  }
};

} // namespace autoware::tensorrt_vad

#endif // AUTOWARE_TENSORRT_VAD_VAD_INTERFACE_CONFIG_HPP_
