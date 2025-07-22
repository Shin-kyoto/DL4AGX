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

#include "autoware/tensorrt_vad/vad_node.hpp"

#include "autoware/tensorrt_vad/utils.hpp"

#include <rclcpp_components/register_node_macro.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.hpp>
#include <geometry_msgs/msg/accel_with_covariance_stamped.hpp>

namespace autoware::tensorrt_vad
{
std::pair<Eigen::Matrix4f, Eigen::Matrix4f> get_transform_matrix(
  const nav_msgs::msg::Odometry & msg)
{
  // Extract position
  double x = msg.pose.pose.position.x;
  double y = msg.pose.pose.position.y;
  double z = msg.pose.pose.position.z;

  // Create Eigen quaternion and normalize it just in case
  Eigen::Quaternionf q = std::invoke([&msg]() -> Eigen::Quaternionf {
    double qx = msg.pose.pose.orientation.x;
    double qy = msg.pose.pose.orientation.y;
    double qz = msg.pose.pose.orientation.z;
    double qw = msg.pose.pose.orientation.w;

    // Create Eigen quaternion and normalize it just in case
    Eigen::Quaternionf q(qw, qx, qy, qz);
    return (q.norm() < std::numeric_limits<float>::epsilon()) ? Eigen::Quaternionf::Identity()
                                                              : q.normalized();
  });

  // Rotation matrix (3x3)
  Eigen::Matrix3f R = q.toRotationMatrix();

  // Translation vector
  Eigen::Vector3f t(x, y, z);

  // Base_link → Map (forward)
  Eigen::Matrix4f bl2map = Eigen::Matrix4f::Identity();
  bl2map.block<3, 3>(0, 0) = R;
  bl2map.block<3, 1>(0, 3) = t;

  // Map → Base_link (inverse)
  Eigen::Matrix4f map2bl = Eigen::Matrix4f::Identity();
  map2bl.block<3, 3>(0, 0) = R.transpose();
  map2bl.block<3, 1>(0, 3) = -R.transpose() * t;

  return {bl2map, map2bl};
}

VadNode::VadNode(const rclcpp::NodeOptions & options) 
  : Node("vad_node", options), 
    tf_buffer_(this->get_clock()),
    num_cameras_(declare_parameter<int32_t>("node_params.num_cameras")),
    vad_interface_config_(
      declare_parameter<int32_t>("interface_params.input_image_width"),
      declare_parameter<int32_t>("interface_params.input_image_height"),
      declare_parameter<int32_t>("interface_params.target_image_width"),
      declare_parameter<int32_t>("interface_params.target_image_height"),
      declare_parameter<std::vector<double>>("interface_params.point_cloud_range"),
      declare_parameter<int32_t>("interface_params.bev_h"),
      declare_parameter<int32_t>("interface_params.bev_w"),
      declare_parameter<double>("interface_params.default_patch_angle"),
      declare_parameter<int32_t>("model_params.default_command"),
      declare_parameter<std::vector<double>>("interface_params.default_shift"),
      declare_parameter<std::vector<double>>("interface_params.default_can_bus"),
      declare_parameter<std::vector<double>>("interface_params.image_normalization_param_mean"),
      declare_parameter<std::vector<double>>("interface_params.image_normalization_param_std"),
      declare_parameter<std::vector<double>>("interface_params.vad2base"),
      declare_parameter<std::vector<int64_t>>("interface_params.autoware_to_vad_camera_mapping")
    ),
    trajectory_timestep_(declare_parameter<double>("interface_params.trajectory_timestep", 1.0)),
    current_frame_(num_cameras_),
    frame_started_(false)
{
  // Publishers
  trajectory_publisher_ =
      this->create_publisher<autoware_planning_msgs::msg::Trajectory>(
          "~/output/trajectory", rclcpp::QoS(1));

  candidate_trajectories_publisher_ =
      this->create_publisher<autoware_internal_planning_msgs::msg::CandidateTrajectories>(
          "~/output/trajectories", rclcpp::QoS(1));

  objects_publisher_ =
      this->create_publisher<autoware_perception_msgs::msg::DetectedObjects>(
          "~/output/objects",
          rclcpp::QoS(1));

  // Create QoS profiles for sensor data (best effort for compatibility with typical sensor topics)
  auto sensor_qos = rclcpp::QoS(1).reliability(rclcpp::ReliabilityPolicy::BestEffort);
  auto camera_info_qos = rclcpp::QoS(10).reliability(rclcpp::ReliabilityPolicy::BestEffort);
  auto reliable_qos = rclcpp::QoS(1).reliability(rclcpp::ReliabilityPolicy::Reliable);

  // Subscribers for each camera
  create_camera_image_subscribers(sensor_qos);

  // Subscribers for camera info
  create_camera_info_subscribers(camera_info_qos);

  // Odometry subscriber (kinematic state is usually reliable)
  odometry_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "~/input/kinematic_state", reliable_qos,
      std::bind(&VadNode::odometry_callback, this, std::placeholders::_1));

  // Acceleration subscriber (sensor data typically uses best effort)
  acceleration_sub_ = this->create_subscription<geometry_msgs::msg::AccelWithCovarianceStamped>(
      "~/input/acceleration", sensor_qos,
      std::bind(&VadNode::acceleration_callback, this, std::placeholders::_1));

  // TF static subscriber (transient local for persistence)
  auto tf_static_qos = rclcpp::QoS(1).reliability(rclcpp::ReliabilityPolicy::Reliable)
                                     .durability(rclcpp::DurabilityPolicy::TransientLocal);
  tf_static_sub_ = this->create_subscription<tf2_msgs::msg::TFMessage>(
      "/tf_static", tf_static_qos,
      std::bind(&VadNode::tf_static_callback, this, std::placeholders::_1));

  // Load VadConfig
  load_vad_config();

  // Initialize VAD model on first complete frame
  if (!vad_model_ptr_) {
    RCLCPP_INFO(this->get_logger(), "Initializing VAD model with first complete frame");
    initialize_vad_model();
  }

  // Initialize timeout timer
  frame_timeout_timer_ = this->create_wall_timer(
    FRAME_TIMEOUT,
    std::bind(&VadNode::frame_timeout_callback, this));
  
  // Start timer as stopped initially
  frame_timeout_timer_->cancel();

  RCLCPP_INFO(this->get_logger(), "VAD Node has been initialized - VAD model will be initialized after first callback");
}

void VadNode::image_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg, std::size_t camera_id)
{
  // Validate camera_id
  if (static_cast<int32_t>(camera_id) >= num_cameras_) {
    RCLCPP_ERROR(this->get_logger(), "Invalid camera_id: %zu. Expected 0-%d", camera_id, num_cameras_ - 1);
    return;
  }

  std::lock_guard<std::mutex> lock(frame_mutex_);

  // Initialize frame if not started
  if (!frame_started_) {
    current_frame_.stamp = msg->header.stamp;
    frame_started_ = true;
    // Start timeout timer for this frame
    frame_timeout_timer_->reset();
  }

  // Store as ConstSharedPtr directly
  current_frame_.images[camera_id] = msg;
    
  auto vad_output_topic_data = trigger_inference();
  if (vad_output_topic_data.has_value()) {
    publish(vad_output_topic_data.value());
  }

  RCLCPP_DEBUG(this->get_logger(), "Received image from camera %zu", camera_id);
}

void VadNode::camera_info_callback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg, std::size_t camera_id)
{
  // Validate camera_id
  if (static_cast<int32_t>(camera_id) >= num_cameras_) {
    RCLCPP_ERROR(this->get_logger(), "Invalid camera_id: %zu. Expected 0-%d", camera_id, num_cameras_ - 1);
    return;
  }

  std::lock_guard<std::mutex> lock(frame_mutex_);

  // Initialize frame if not started
  if (!frame_started_) {
    current_frame_.stamp = msg->header.stamp;
    frame_started_ = true;
    // Start timeout timer for this frame
    frame_timeout_timer_->reset();
  }

  current_frame_.camera_infos[camera_id] = msg;
  auto vad_output_topic_data = trigger_inference();
  if (vad_output_topic_data.has_value()) {
    publish(vad_output_topic_data.value());
  }

  RCLCPP_DEBUG(this->get_logger(), "Received camera info from camera %zu", camera_id);
}

void VadNode::odometry_callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(frame_mutex_);

  // Initialize frame if not started
  if (!frame_started_) {
    current_frame_.stamp = msg->header.stamp;
    frame_started_ = true;
    // Start timeout timer for this frame
    frame_timeout_timer_->reset();
  }

  current_frame_.kinematic_state = msg;
  auto vad_output_topic_data = trigger_inference();
  if (vad_output_topic_data.has_value()) {
    publish(vad_output_topic_data.value());
  }

  RCLCPP_DEBUG(this->get_logger(), "Received odometry data");
}

void VadNode::acceleration_callback(const geometry_msgs::msg::AccelWithCovarianceStamped::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(frame_mutex_);

  // Initialize frame if not started
  if (!frame_started_) {
    current_frame_.stamp = msg->header.stamp;
    frame_started_ = true;
    // Start timeout timer for this frame
    frame_timeout_timer_->reset();
  }

  current_frame_.acceleration = msg;
  auto vad_output_topic_data = trigger_inference();
  if (vad_output_topic_data.has_value()) {
    publish(vad_output_topic_data.value());
  }

  RCLCPP_DEBUG(this->get_logger(), "Received acceleration data");
}

void VadNode::tf_static_callback(const tf2_msgs::msg::TFMessage::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(frame_mutex_);

  // Initialize frame if not started (though tf_static typically comes first)
  if (!frame_started_) {
    current_frame_.stamp = rclcpp::Time(0);  // tf_static doesn't have timestamp
    frame_started_ = true;
    // Start timeout timer for this frame
    frame_timeout_timer_->reset();
  }

  current_frame_.tf_static = msg;

  // Register transforms in tf_buffer
  for (const auto &transform : msg->transforms) {
    tf_buffer_.setTransform(transform, "default_authority", true);
  }

  auto vad_output_topic_data = trigger_inference();
  if (vad_output_topic_data.has_value()) {
    publish(vad_output_topic_data.value());
  }

  RCLCPP_DEBUG(this->get_logger(), "Received TF static data");
}

std::optional<VadOutputTopicData> VadNode::trigger_inference()
{
  // Use the built-in is_complete() method to check if we have all required data
  if (current_frame_.is_complete()) {
    
    // Cancel timeout timer since frame is complete
    frame_timeout_timer_->cancel();

    // Execute inference
    auto vad_output_topic_data = execute_inference(current_frame_);

    // Reset frame for next data collection
    reset_current_frame();

    return vad_output_topic_data;
  }
  
  return std::nullopt;
}

void VadNode::reset_current_frame()
{
  current_frame_.reset();
  frame_started_ = false;
}

void VadNode::initialize_vad_model()
{
  // Initialize VAD interface and model
  auto tf_buffer_shared = std::shared_ptr<tf2_ros::Buffer>(&tf_buffer_, [](tf2_ros::Buffer*){});
  vad_interface_ptr_ = std::make_unique<VadInterface>(vad_interface_config_, tf_buffer_shared);

  // Create RosVadLogger using the logger
  auto ros_logger = std::make_shared<RosVadLogger>(this->get_logger());
  vad_model_ptr_ = std::make_unique<VadModel<RosVadLogger>>(vad_config_, ros_logger);

  RCLCPP_INFO(this->get_logger(), "VAD model and interface initialized successfully");
}

void VadNode::load_vad_config()
{
  // Load VAD config parameters
  vad_config_.plugins_path =
      this->declare_parameter<std::string>("model_params.plugins_path", "");
  vad_config_.warm_up_num =
      this->declare_parameter<int>("model_params.warm_up_num", 20);

  // Load network configurations
  load_net_configs();
}

void VadNode::load_net_configs()
{
  // Load network configurations
  
  // backbone configuration
  NetConfig backbone_config;
  backbone_config.name = this->declare_parameter<std::string>(
      "model_params.nets.backbone.name", "backbone");
  backbone_config.engine_file = this->declare_parameter<std::string>(
      "model_params.nets.backbone.engine_file", "");
  backbone_config.use_graph = this->declare_parameter<bool>(
      "model_params.nets.backbone.use_graph", true);

  // head configuration
  NetConfig head_config;
  head_config.name = this->declare_parameter<std::string>(
      "model_params.nets.head.name", "head");
  head_config.engine_file = this->declare_parameter<std::string>(
      "model_params.nets.head.engine_file", "");
  head_config.use_graph = this->declare_parameter<bool>(
      "model_params.nets.head.use_graph", true);

  // head inputs
  std::string input_feature = this->declare_parameter<std::string>(
      "model_params.nets.head.inputs.input_feature", "mlvl_feats.0");
  std::string net_param = this->declare_parameter<std::string>(
      "model_params.nets.head.inputs.net", "backbone");
  std::string name_param = this->declare_parameter<std::string>(
      "model_params.nets.head.inputs.name", "out.0");
  head_config.inputs[input_feature]["net"] = net_param;
  head_config.inputs[input_feature]["name"] = name_param;

  // head_no_prev configuration
  NetConfig head_no_prev_config;
  head_no_prev_config.name = this->declare_parameter<std::string>(
      "model_params.nets.head_no_prev.name", "head_no_prev");
  head_no_prev_config.engine_file = this->declare_parameter<std::string>(
      "model_params.nets.head_no_prev.engine_file", "");
  head_no_prev_config.use_graph = this->declare_parameter<bool>(
      "model_params.nets.head_no_prev.use_graph", true);

  // head_no_prev inputs
  std::string input_feature_no_prev = this->declare_parameter<std::string>(
      "model_params.nets.head_no_prev.inputs.input_feature", "mlvl_feats.0");
  std::string net_param_no_prev = this->declare_parameter<std::string>(
      "model_params.nets.head_no_prev.inputs.net", "backbone");
  std::string name_param_no_prev = this->declare_parameter<std::string>(
      "model_params.nets.head_no_prev.inputs.name", "out.0");
  head_no_prev_config.inputs[input_feature_no_prev]["net"] = net_param_no_prev;
  head_no_prev_config.inputs[input_feature_no_prev]["name"] = name_param_no_prev;

  vad_config_.nets_config.push_back(backbone_config);
  vad_config_.nets_config.push_back(head_config);
  vad_config_.nets_config.push_back(head_no_prev_config);
}

void VadNode::publish_trajectories(const autoware_internal_planning_msgs::msg::CandidateTrajectories & candidate_trajectories)
{
  auto candidate_trajectories_msg = std::make_unique<autoware_internal_planning_msgs::msg::CandidateTrajectories>(candidate_trajectories);
  candidate_trajectories_publisher_->publish(std::move(candidate_trajectories_msg));
}

void VadNode::publish_trajectory(const autoware_planning_msgs::msg::Trajectory & trajectory)
{
  auto trajectory_msg = std::make_unique<autoware_planning_msgs::msg::Trajectory>(trajectory);
  trajectory_publisher_->publish(std::move(trajectory_msg));
}

std::optional<VadOutputTopicData> VadNode::execute_inference(const VadInputTopicData & vad_topic_data)
{
  if (!vad_interface_ptr_ || !vad_model_ptr_) {
    RCLCPP_ERROR(this->get_logger(), "VAD interface or model not initialized");
    return std::nullopt;
  }

  // VadInterfaceを通じてVadInputDataに変換
  // scalingされた状態の画像を含む
  const auto vad_input = vad_interface_ptr_->convert_input(vad_topic_data);

  // VadModelで推論実行
  const auto vad_output = vad_model_ptr_->infer(vad_input);

  const auto [base2map_transform, map2base_transform] = get_transform_matrix(*current_frame_.kinematic_state);
  // VadInterfaceを通じてROS型に変換
  if (vad_output.has_value()) {
    const auto vad_output_topic_data = vad_interface_ptr_->convert_output(
      *vad_output, this->now(), trajectory_timestep_, base2map_transform);
    // Return VadOutputTopicData
    return vad_output_topic_data;
  }

  return std::nullopt;
}

void VadNode::publish(const VadOutputTopicData & vad_output_topic_data)
{
  // Publish individual trajectory using the dedicated method
  publish_trajectory(vad_output_topic_data.trajectory);

  // Publish candidate trajectories
  publish_trajectories(vad_output_topic_data.candidate_trajectories);

  RCLCPP_DEBUG(this->get_logger(), "Published trajectories and candidate trajectories");
}

void VadNode::frame_timeout_callback()
{
  std::lock_guard<std::mutex> lock(frame_mutex_);
  
  if (frame_started_) {
    // Check how much data we have
    int valid_images = 0;
    int valid_camera_infos = 0;
    
    for (size_t i = 0; i < current_frame_.images.size(); ++i) {
      if (current_frame_.images[i]) valid_images++;
      if (current_frame_.camera_infos[i]) valid_camera_infos++;
    }
    
    RCLCPP_WARN(this->get_logger(), 
                "Frame timeout reached. Have %d/%d images, %d/%d camera_infos, kinematic_state: %s, acceleration: %s, tf_static: %s",
                valid_images, num_cameras_, valid_camera_infos, num_cameras_,
                current_frame_.kinematic_state ? "yes" : "no",
                current_frame_.acceleration ? "yes" : "no",
                current_frame_.tf_static ? "yes" : "no");
    
    // Reset frame if incomplete
    reset_current_frame();
  }
  
  // Stop timer until next frame starts
  frame_timeout_timer_->cancel();
}

void VadNode::create_camera_image_subscribers(const rclcpp::QoS& sensor_qos)
{
  camera_image_subs_.resize(num_cameras_);
  std::vector<bool> use_raw_cameras = this->declare_parameter<std::vector<bool>>("node_params.use_raw", std::vector<bool>(num_cameras_, false));
  auto resolve_topic_name = [this](const std::string & query) {
    return this->get_node_topics_interface()->resolve_topic_name(query);
  };
  for (int32_t i = 0; i < num_cameras_; ++i) {
    const auto transport = use_raw_cameras[i] ? "raw" : "compressed";
    auto callback =
        [this, i](const sensor_msgs::msg::Image::ConstSharedPtr msg) {
          this->image_callback(msg, i);
        };
    // image_transport::create_subscription を使用
    const auto image_topic = resolve_topic_name("~/input/image" + std::to_string(i));
    camera_image_subs_[i] = image_transport::create_subscription(
        this,
        image_topic,
        callback,
        transport,
        sensor_qos.get_rmw_qos_profile());
  }
}

void VadNode::create_camera_info_subscribers(const rclcpp::QoS& camera_info_qos)
{
  camera_info_subs_.resize(num_cameras_);
  for (int32_t i = 0; i < num_cameras_; ++i) {
    auto callback =
        [this, i](const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
          this->camera_info_callback(msg, i);
        };

    camera_info_subs_[i] =
        this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "~/input/camera_info" + std::to_string(i),
            camera_info_qos, callback);
  }
}

}  // namespace autoware::tensorrt_vad

// Register the component with the ROS2 component system
// NOLINTNEXTLINE(readability-identifier-naming,cppcoreguidelines-avoid-non-const-global-variables)
RCLCPP_COMPONENTS_REGISTER_NODE(autoware::tensorrt_vad::VadNode)
