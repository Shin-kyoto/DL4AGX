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

#ifndef AUTOWARE_TENSORRT_VAD_VAD_NODE_HPP_
#define AUTOWARE_TENSORRT_VAD_VAD_NODE_HPP_

#include "autoware/tensorrt_vad/vad_model.hpp"
#include "autoware/tensorrt_vad/vad_interface.hpp"
#include "autoware/tensorrt_vad/vad_interface_config.hpp"
#include "ros_vad_logger.hpp"

#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include <autoware_perception_msgs/msg/detected_objects.hpp>
#include <autoware_planning_msgs/msg/trajectory.hpp>
#include <autoware_internal_planning_msgs/msg/candidate_trajectories.hpp>
#include <autoware_internal_planning_msgs/msg/candidate_trajectory.hpp>
#include <autoware_internal_planning_msgs/msg/generator_info.hpp>
#include <autoware_planning_msgs/msg/trajectory_point.hpp>
#include <autoware_utils_uuid/uuid_helper.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/quaternion.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_msgs/msg/tf_message.hpp>
#include <cv_bridge/cv_bridge.h>

#include <cmath>
#include <memory>
#include <vector>
#include <map>
#include <mutex>

namespace autoware::tensorrt_vad
{

class VadNode : public rclcpp::Node
{
public:
  explicit VadNode(const rclcpp::NodeOptions & options);

private:
  // Config loading function
  void load_vad_config();
  void load_net_configs();
  void initialize_vad_model();

  // Publisher methods
  void publish_trajectory(const autoware_planning_msgs::msg::Trajectory & trajectory);
  void publish_trajectories(const autoware_internal_planning_msgs::msg::CandidateTrajectories & candidate_trajectories);

  // Conversion methods

  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg, std::size_t camera_id);
  void camera_info_callback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg, std::size_t camera_id);
  void odometry_callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg);
  void imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr msg);
  void tf_static_callback(const tf2_msgs::msg::TFMessage::ConstSharedPtr msg);

  // Helper functions for data synchronization
  void check_and_process_frame();
  void reset_current_frame();
  void frame_timeout_callback();

  // Member variables in declaration order to match constructor initialization
  VadInterfaceConfig vad_interface_config_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_{tf_buffer_};

  // VAD interface config and previous can bus data
  std::vector<float> prev_can_bus_;

  // Subscribers for images (using image_transport)
  std::vector<image_transport::Subscriber> camera_image_subs_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr> camera_info_subs_;

  // Subscribers for odometry and IMU data
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_static_sub_;

  // VAD model - 具体的なロガー型を指定
  std::unique_ptr<VadModel<RosVadLogger>> vad_model_ptr_{};

  // VAD interface
  std::unique_ptr<VadInterface> vad_interface_ptr_{};

  // VAD config
  VadConfig vad_config_;

  // trajectory_timestep parameter
  double trajectory_timestep_;

  // Publishers
  rclcpp::Publisher<autoware_planning_msgs::msg::Trajectory>::SharedPtr trajectory_publisher_;
  rclcpp::Publisher<autoware_internal_planning_msgs::msg::CandidateTrajectories>::SharedPtr candidate_trajectories_publisher_;
  rclcpp::Publisher<autoware_perception_msgs::msg::DetectedObjects>::SharedPtr objects_publisher_;

  // Current frame data accumulation
  VadInputTopicData current_frame_;
  bool frame_started_;
  std::mutex frame_mutex_;
  
  // Timeout timer for frame completion
  rclcpp::TimerBase::SharedPtr frame_timeout_timer_;
  static constexpr std::chrono::milliseconds FRAME_TIMEOUT{180}; // 180ms timeout

  // 推論を実行するメソッド
  std::optional<VadOutputTopicData> execute_inference(const VadInputTopicData & vad_topic_data);
  void publish(const VadInputTopicData & vad_topic_data);
};
}  // namespace autoware::tensorrt_vad

#endif  // AUTOWARE_TENSORRT_VAD_VAD_NODE_HPP_
