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

namespace autoware::tensorrt_vad
{

VadNode::VadNode(const rclcpp::NodeOptions & options) 
  : Node("vad_node", options), 
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
      declare_parameter<std::vector<double>>("interface_params.image_normalization_param_mean"),
      declare_parameter<std::vector<double>>("interface_params.image_normalization_param_std"),
      declare_parameter<std::vector<double>>("interface_params.vad2base"),
      declare_parameter<std::vector<int64_t>>("interface_params.autoware_to_vad_camera_mapping")
    ),
    tf_buffer_(this->get_clock()),
    trajectory_timestep_(declare_parameter<double>("interface_params.trajectory_timestep", 1.0)),
    frame_started_(false)
{
  // Initialize prev_can_bus from parameters
  std::vector<double> default_can_bus = this->declare_parameter<std::vector<double>>("interface_params.default_can_bus");
  prev_can_bus_.clear();
  for (auto v : default_can_bus) prev_can_bus_.push_back(static_cast<float>(v));

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
  camera_image_subs_.resize(6);
  std::vector<bool> use_raw_cameras = this->declare_parameter<std::vector<bool>>("node_params.use_raw", std::vector<bool>(6, false));
  auto resolve_topic_name = [this](const std::string & query) {
    return this->get_node_topics_interface()->resolve_topic_name(query);
  };
  for (int32_t i = 0; i < 6; ++i) {
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

  // Camera info subscribers
  camera_info_subs_.resize(6);
  for (int32_t i = 0; i < 6; ++i) {
    auto callback =
        [this, i](const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
          this->camera_info_callback(msg, i);
        };

    camera_info_subs_[i] =
        this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "~/input/camera_info" + std::to_string(i),
            camera_info_qos, callback);
  }

  // Odometry subscriber (kinematic state is usually reliable)
  odometry_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/localization/kinematic_state", reliable_qos,
      std::bind(&VadNode::odometry_callback, this, std::placeholders::_1));

  // IMU subscriber (sensor data typically uses best effort)
  imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
      "/sensing/imu/tamagawa/imu_raw", sensor_qos,
      std::bind(&VadNode::imu_callback, this, std::placeholders::_1));

  // TF static subscriber (transient local for persistence)
  auto tf_static_qos = rclcpp::QoS(1).reliability(rclcpp::ReliabilityPolicy::Reliable)
                                     .durability(rclcpp::DurabilityPolicy::TransientLocal);
  tf_static_sub_ = this->create_subscription<tf2_msgs::msg::TFMessage>(
      "/tf_static", tf_static_qos,
      std::bind(&VadNode::tf_static_callback, this, std::placeholders::_1));

  // Load VadConfig
  loadVadConfig();

  // Initialize current frame structure
  current_frame_.images.resize(6);
  current_frame_.camera_infos.resize(6);

  // Initialize VAD model on first complete frame
  if (!vad_model_ptr_) {
    RCLCPP_INFO(this->get_logger(), "Initializing VAD model with first complete frame");
    initializeVadModel();
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
  if (camera_id >= 6) {
    RCLCPP_ERROR(this->get_logger(), "Invalid camera_id: %zu. Expected 0-5", camera_id);
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
    
  check_and_process_frame();

  RCLCPP_DEBUG(this->get_logger(), "Received image from camera %zu", camera_id);
}

void VadNode::camera_info_callback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg, std::size_t camera_id)
{
  // Validate camera_id
  if (camera_id >= 6) {
    RCLCPP_ERROR(this->get_logger(), "Invalid camera_id: %zu. Expected 0-5", camera_id);
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
  check_and_process_frame();

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
  check_and_process_frame();

  RCLCPP_DEBUG(this->get_logger(), "Received odometry data");
}

void VadNode::imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(frame_mutex_);

  // Initialize frame if not started
  if (!frame_started_) {
    current_frame_.stamp = msg->header.stamp;
    frame_started_ = true;
    // Start timeout timer for this frame
    frame_timeout_timer_->reset();
  }

  current_frame_.imu_raw = msg;
  check_and_process_frame();

  RCLCPP_DEBUG(this->get_logger(), "Received IMU data");
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

  check_and_process_frame();

  RCLCPP_DEBUG(this->get_logger(), "Received TF static data");
}

void VadNode::check_and_process_frame()
{
  // Use the built-in is_complete() method to check if we have all required data
  if (current_frame_.is_complete()) {
    
    // Cancel timeout timer since frame is complete
    frame_timeout_timer_->cancel();

    // Execute inference and publish results
    publish(current_frame_);

    // Reset frame for next data collection
    reset_current_frame();
  }
}

void VadNode::reset_current_frame()
{
  current_frame_.images.clear();
  current_frame_.images.resize(6);
  current_frame_.camera_infos.clear();
  current_frame_.camera_infos.resize(6);
  current_frame_.kinematic_state.reset();
  current_frame_.imu_raw.reset();
  current_frame_.tf_static.reset();
  frame_started_ = false;
}

void VadNode::initializeVadModel()
{
  try {
    // Initialize VAD interface and model
    auto tf_buffer_shared = std::shared_ptr<tf2_ros::Buffer>(&tf_buffer_, [](tf2_ros::Buffer*){});
    vad_interface_ptr_ = std::make_unique<VadInterface>(vad_interface_config_, tf_buffer_shared);
    
    // Create RosVadLogger using the logger
    auto ros_logger = std::make_shared<RosVadLogger>(this->get_logger());
    vad_model_ptr_ = std::make_unique<VadModel<RosVadLogger>>(vad_config_, ros_logger);
    
    RCLCPP_INFO(this->get_logger(), "VAD model and interface initialized successfully");
  } catch (const std::exception& e) {
    RCLCPP_ERROR(this->get_logger(), "Failed to initialize VAD model: %s", e.what());
  }
}

void VadNode::loadVadConfig()
{
  // Load VAD config parameters
  vad_config_.plugins_path =
      this->declare_parameter<std::string>("model_params.plugins_path", "");
  vad_config_.warm_up_num =
      this->declare_parameter<int>("model_params.warm_up_num", 20);

  // Load network configurations
  loadNetConfigs();
}

void VadNode::loadNetConfigs()
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

geometry_msgs::msg::Quaternion VadNode::createQuaternionFromYaw(double yaw)
{
  geometry_msgs::msg::Quaternion q{};
  q.x = 0.0;
  q.y = 0.0;
  q.z = std::sin(yaw * 0.5);
  q.w = std::cos(yaw * 0.5);
  return q;
}

void VadNode::publishTrajectory(const std::vector<float> &planning)
{
  auto trajectory_msg =
      std::make_unique<autoware_planning_msgs::msg::Trajectory>();

  // 0秒目の点 (0,0) を追加
  autoware_planning_msgs::msg::TrajectoryPoint initial_point;
  initial_point.pose.position.x = 0.0;
  initial_point.pose.position.y = 0.0;
  initial_point.pose.position.z = 0.0;
  initial_point.pose.orientation = createQuaternionFromYaw(0.0);
  initial_point.longitudinal_velocity_mps = 0.0;
  initial_point.lateral_velocity_mps = 0.0;
  initial_point.acceleration_mps2 = 0.0;
  initial_point.heading_rate_rps = 0.0;
  initial_point.time_from_start.sec = 0;
  initial_point.time_from_start.nanosec = 0;
  trajectory_msg->points.push_back(initial_point);

  for (size_t i = 0; i < planning.size(); i += 2) {
    autoware_planning_msgs::msg::TrajectoryPoint point;

    point.pose.position.x = planning[i + 1];
    point.pose.position.y = -planning[i];
    point.pose.position.z = 0.0;

    if (i + 2 < planning.size()) {
      float ns_dx = planning[i + 2] - planning[i];
      float ns_dy = planning[i + 3] - planning[i + 1];
      float aw_dx = ns_dy;  // Autowareの座標系に変換
      float aw_dy = -ns_dx; // Autowareの座標系に変換
      float yaw = std::atan2(aw_dy, aw_dx);
      point.pose.orientation = createQuaternionFromYaw(yaw);
    }

    point.longitudinal_velocity_mps = 0.0;
    point.lateral_velocity_mps = 0.0;
    point.acceleration_mps2 = 0.0;
    point.heading_rate_rps = 0.0;

    // time_from_startを設定（1秒, 2秒, 3秒, 4秒, 5秒, 6秒）
    size_t point_index = i / 2;
    double time_sec = (point_index + 1) * trajectory_timestep_;
    point.time_from_start.sec = static_cast<int32_t>(time_sec);
    point.time_from_start.nanosec = static_cast<uint32_t>((time_sec - point.time_from_start.sec) * 1e9);

    trajectory_msg->points.push_back(point);
  }

  trajectory_msg->header.stamp = this->now();
  trajectory_msg->header.frame_id = "base_link";

  trajectory_publisher_->publish(std::move(trajectory_msg));
}

autoware_internal_planning_msgs::msg::CandidateTrajectories VadNode::convert_candidate_trajectories(const std::map<int32_t, std::vector<float>> &trajectories_map)
{
  // CandidateTrajectories メッセージを作成
  autoware_internal_planning_msgs::msg::CandidateTrajectories candidate_trajectories_msg;

  // 各コマンドの軌道をCandidateTrajectoryとして追加
  for (const auto& [command_idx, trajectory] : trajectories_map) {
    autoware_internal_planning_msgs::msg::CandidateTrajectory candidate_trajectory;
    
    // ヘッダーを設定
    candidate_trajectory.header.stamp = this->now();
    candidate_trajectory.header.frame_id = "base_link";
    
    // generator_idを設定（ユニークなUUID）
    candidate_trajectory.generator_id = autoware_utils_uuid::generate_uuid();

    // 0秒目の点 (0,0) を追加
    autoware_planning_msgs::msg::TrajectoryPoint initial_point;
    initial_point.pose.position.x = 0.0;
    initial_point.pose.position.y = 0.0;
    initial_point.pose.position.z = 0.0;
    initial_point.pose.orientation = createQuaternionFromYaw(0.0);
    initial_point.longitudinal_velocity_mps = 0.0;
    initial_point.lateral_velocity_mps = 0.0;
    initial_point.acceleration_mps2 = 0.0;
    initial_point.heading_rate_rps = 0.0;
    initial_point.time_from_start.sec = 0;
    initial_point.time_from_start.nanosec = 0;
    candidate_trajectory.points.push_back(initial_point);

    for (size_t i = 0; i < trajectory.size(); i += 2) {
      autoware_planning_msgs::msg::TrajectoryPoint point;

      point.pose.position.x = trajectory[i + 1];
      point.pose.position.y = -trajectory[i];
      point.pose.position.z = 0.0;

      if (i + 2 < trajectory.size()) {
        float ns_dx = trajectory[i + 2] - trajectory[i];
        float ns_dy = trajectory[i + 3] - trajectory[i + 1];
        float aw_dx = ns_dy;  // Autowareの座標系に変換
        float aw_dy = -ns_dx; // Autowareの座標系に変換
        float yaw = std::atan2(aw_dy, aw_dx);
        point.pose.orientation = createQuaternionFromYaw(yaw);
      }

      point.longitudinal_velocity_mps = 0.0;
      point.lateral_velocity_mps = 0.0;
      point.acceleration_mps2 = 0.0;
      point.heading_rate_rps = 0.0;

      // time_from_startを設定（1秒, 2秒, 3秒, 4秒, 5秒, 6秒）
      size_t point_index = i / 2;
      double time_sec = (point_index + 1) * trajectory_timestep_;
      point.time_from_start.sec = static_cast<int32_t>(time_sec);
      point.time_from_start.nanosec = static_cast<uint32_t>((time_sec - point.time_from_start.sec) * 1e9);

      candidate_trajectory.points.push_back(point);
    }

    candidate_trajectories_msg.candidate_trajectories.push_back(candidate_trajectory);

    // 各コマンドのGeneratorInfoを追加
    autoware_internal_planning_msgs::msg::GeneratorInfo generator_info;
    generator_info.generator_id = autoware_utils_uuid::generate_uuid();
    generator_info.generator_name.data = "autoware_tensorrt_vad_cmd_" + std::to_string(command_idx);
    candidate_trajectories_msg.generator_info.push_back(generator_info);
  }

  return candidate_trajectories_msg;
}

void VadNode::publishTrajectories(const autoware_internal_planning_msgs::msg::CandidateTrajectories & candidate_trajectories)
{
  auto candidate_trajectories_msg = std::make_unique<autoware_internal_planning_msgs::msg::CandidateTrajectories>(candidate_trajectories);
  candidate_trajectories_publisher_->publish(std::move(candidate_trajectories_msg));
}

std::optional<autoware_internal_planning_msgs::msg::CandidateTrajectories> VadNode::execute_inference(const VadInputTopicData & vad_topic_data)
{
  if (!vad_interface_ptr_ || !vad_model_ptr_) {
    RCLCPP_ERROR(this->get_logger(), "VAD interface or model not initialized");
    return std::nullopt;
  }

  try {
    // VadInterfaceを通じてVadInputDataに変換
    // scalingされた状態の画像を含む
    const auto vad_input = vad_interface_ptr_->convert_input(vad_topic_data, prev_can_bus_);

    // VadModelで推論実行
    const auto vad_output = vad_model_ptr_->infer(vad_input);

    // VadInterfaceを通じてROS型に変換
    if (vad_output.has_value()) {
      // Update prev_can_bus for next frame
      prev_can_bus_ = vad_input.can_bus_;

      // Publish individual trajectory if available
      if (!vad_output->predicted_trajectory_.empty()) {
        publishTrajectory(vad_output->predicted_trajectory_);
      }

      // Convert candidate trajectories if available
      if (!vad_output->predicted_trajectories_.empty()) {
        auto candidate_trajectories_msg = convert_candidate_trajectories(vad_output->predicted_trajectories_);
        return std::optional<autoware_internal_planning_msgs::msg::CandidateTrajectories>(candidate_trajectories_msg);
      }

      // Return empty message if no trajectories available
      auto candidate_trajectories_msg = autoware_internal_planning_msgs::msg::CandidateTrajectories();
      return std::optional<autoware_internal_planning_msgs::msg::CandidateTrajectories>(candidate_trajectories_msg);
    }
  } catch (const std::exception& e) {
    RCLCPP_ERROR(this->get_logger(), "Error during inference: %s", e.what());
  }

  return std::nullopt;
}

void VadNode::publish(const VadInputTopicData & vad_topic_data)
{
  try {
    // VAD推論を実行
    const auto candidate_trajectories_msg = execute_inference(vad_topic_data);

    // 結果をパブリッシュ
    if (candidate_trajectories_msg.has_value()) {
      publishTrajectories(candidate_trajectories_msg.value());
      RCLCPP_DEBUG(this->get_logger(), "Published candidate trajectories");
    } else {
      RCLCPP_WARN(this->get_logger(), "No valid trajectories to publish");
    }
  } catch (const std::exception& e) {
    RCLCPP_ERROR(this->get_logger(), "Error in publish method: %s", e.what());
  }
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
                "Frame timeout reached. Have %d/%d images, %d/%d camera_infos, kinematic_state: %s, imu_raw: %s, tf_static: %s",
                valid_images, 6, valid_camera_infos, 6,
                current_frame_.kinematic_state ? "yes" : "no",
                current_frame_.imu_raw ? "yes" : "no",
                current_frame_.tf_static ? "yes" : "no");
    
    // Reset frame if incomplete
    reset_current_frame();
  }
  
  // Stop timer until next frame starts
  frame_timeout_timer_->cancel();
}

}  // namespace autoware::tensorrt_vad

// Register the component with the ROS2 component system
// NOLINTNEXTLINE(readability-identifier-naming,cppcoreguidelines-avoid-non-const-global-variables)
RCLCPP_COMPONENTS_REGISTER_NODE(autoware::tensorrt_vad::VadNode)
