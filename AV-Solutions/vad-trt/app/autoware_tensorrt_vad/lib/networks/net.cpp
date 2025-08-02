/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "autoware/tensorrt_vad/networks/net.hpp"

namespace autoware::tensorrt_vad {

// NetworkType utility functions implementation
std::string toString(NetworkType type) {
  switch (type) {
    case NetworkType::BACKBONE:
      return "backbone";
    case NetworkType::HEAD:
      return "head";
    case NetworkType::HEAD_NO_PREV:
      return "head_no_prev";
    default:
      return "unknown";
  }
}

// Backbone NetworkIO設定生成関数
std::vector<autoware::tensorrt_common::NetworkIO> generate_network_io_backbone(const VadConfig& vad_config) {
  int32_t downsampled_image_height = vad_config.target_image_height / vad_config.downsample_factor;
  int32_t downsampled_image_width = vad_config.target_image_width / vad_config.downsample_factor;
  nvinfer1::Dims camera_input_dims{4, {vad_config.num_cameras, 3, vad_config.target_image_height, vad_config.target_image_width}};
  nvinfer1::Dims backbone_output_dims{5, {vad_config.num_cameras, 1, vad_config.bev_feature_dim, downsampled_image_height, downsampled_image_width}};
  std::vector<autoware::tensorrt_common::NetworkIO> backbone_network_io;
  backbone_network_io.emplace_back("img", camera_input_dims);
  backbone_network_io.emplace_back("out.0", backbone_output_dims);
  return backbone_network_io;
}

// Head NetworkIO設定生成関数
std::vector<autoware::tensorrt_common::NetworkIO> generate_network_io_head(const VadConfig& vad_config) {
  int32_t downsampled_image_height = vad_config.target_image_height / vad_config.downsample_factor;
  int32_t downsampled_image_width = vad_config.target_image_width / vad_config.downsample_factor;
  nvinfer1::Dims mlvl_dims{5, {1, vad_config.num_cameras, vad_config.bev_feature_dim, downsampled_image_height, downsampled_image_width}};
  nvinfer1::Dims can_bus_dims{2, {1, vad_config.can_bus_dim}};
  nvinfer1::Dims lidar2img_dims{3, {vad_config.num_cameras, 4, 4}};
  nvinfer1::Dims shift_dims{2, {1, 2}};
  nvinfer1::Dims prev_bev_dims{3, {vad_config.bev_h * vad_config.bev_w, 1, vad_config.bev_feature_dim}};
  nvinfer1::Dims ego_fut_preds_dims{4, {1, vad_config.planning_ego_commands, vad_config.planning_timesteps, 2}};
  nvinfer1::Dims traj_preds_dims{5, {3, 1, vad_config.prediction_num_queries, vad_config.prediction_trajectory_modes, vad_config.prediction_timesteps*2}};
  nvinfer1::Dims traj_cls_dims{4, {3, 1, vad_config.prediction_num_queries, vad_config.prediction_trajectory_modes}};
  nvinfer1::Dims bbox_preds_dims{4, {3, 1, vad_config.prediction_num_queries, vad_config.prediction_bbox_pred_dim}};
  nvinfer1::Dims all_cls_scores_dims{4, {3, 1, vad_config.prediction_num_queries, vad_config.prediction_num_classes}};
  nvinfer1::Dims map_all_cls_scores_dims{4, {3, 1, vad_config.map_num_queries, vad_config.map_num_class}};
  nvinfer1::Dims map_all_pts_preds_dims{5, {3, 1, vad_config.map_num_queries, vad_config.map_points_per_polylines, 2}};
  nvinfer1::Dims map_all_bbox_preds_dims{4, {3, 1, vad_config.map_num_queries, 4}};
  std::vector<autoware::tensorrt_common::NetworkIO> head_network_io;
  head_network_io.emplace_back("mlvl_feats.0", mlvl_dims);
  head_network_io.emplace_back("img_metas.0[can_bus]", can_bus_dims);
  head_network_io.emplace_back("img_metas.0[lidar2img]", lidar2img_dims);
  head_network_io.emplace_back("img_metas.0[shift]", shift_dims);
  head_network_io.emplace_back("prev_bev", prev_bev_dims);
  head_network_io.emplace_back("out.bev_embed", prev_bev_dims);
  head_network_io.emplace_back("out.ego_fut_preds", ego_fut_preds_dims);
  head_network_io.emplace_back("out.all_traj_preds", traj_preds_dims);
  head_network_io.emplace_back("out.all_traj_cls_scores", traj_cls_dims);
  head_network_io.emplace_back("out.all_bbox_preds", bbox_preds_dims);
  head_network_io.emplace_back("out.all_cls_scores", all_cls_scores_dims);
  head_network_io.emplace_back("out.map_all_cls_scores", map_all_cls_scores_dims);
  head_network_io.emplace_back("out.map_all_pts_preds", map_all_pts_preds_dims);
  head_network_io.emplace_back("out.map_all_bbox_preds", map_all_bbox_preds_dims);
  return head_network_io;
}

// Head No Previous NetworkIO設定生成関数
std::vector<autoware::tensorrt_common::NetworkIO> generate_network_io_head_no_prev(const VadConfig& vad_config) {
  int32_t downsampled_image_height = vad_config.target_image_height / vad_config.downsample_factor;
  int32_t downsampled_image_width = vad_config.target_image_width / vad_config.downsample_factor;
  nvinfer1::Dims mlvl_dims{5, {1, vad_config.num_cameras, vad_config.bev_feature_dim, downsampled_image_height, downsampled_image_width}};
  nvinfer1::Dims can_bus_dims{2, {1, vad_config.can_bus_dim}};
  nvinfer1::Dims lidar2img_dims{3, {vad_config.num_cameras, 4, 4}};
  nvinfer1::Dims shift_dims{2, {1, 2}};
  nvinfer1::Dims prev_bev_dims{3, {vad_config.bev_h * vad_config.bev_w, 1, vad_config.bev_feature_dim}};
  nvinfer1::Dims ego_fut_preds_dims{4, {1, vad_config.planning_ego_commands, vad_config.planning_timesteps, 2}};
  nvinfer1::Dims traj_preds_dims{5, {3, 1, vad_config.prediction_num_queries, vad_config.prediction_trajectory_modes, vad_config.prediction_timesteps*2}};
  nvinfer1::Dims traj_cls_dims{4, {3, 1, vad_config.prediction_num_queries, vad_config.prediction_trajectory_modes}};
  nvinfer1::Dims bbox_preds_dims{4, {3, 1, vad_config.prediction_num_queries, vad_config.prediction_bbox_pred_dim}};
  nvinfer1::Dims all_cls_scores_dims{4, {3, 1, vad_config.prediction_num_queries, vad_config.prediction_num_classes}};
  nvinfer1::Dims map_all_cls_scores_dims{4, {3, 1, vad_config.map_num_queries, vad_config.map_num_class}};
  nvinfer1::Dims map_all_pts_preds_dims{5, {3, 1, vad_config.map_num_queries, vad_config.map_points_per_polylines, 2}};
  nvinfer1::Dims map_all_bbox_preds_dims{4, {3, 1, vad_config.map_num_queries, 4}};
  std::vector<autoware::tensorrt_common::NetworkIO> head_no_prev_network_io;
  head_no_prev_network_io.emplace_back("mlvl_feats.0", mlvl_dims);
  head_no_prev_network_io.emplace_back("img_metas.0[can_bus]", can_bus_dims);
  head_no_prev_network_io.emplace_back("img_metas.0[lidar2img]", lidar2img_dims);
  head_no_prev_network_io.emplace_back("img_metas.0[shift]", shift_dims);
  head_no_prev_network_io.emplace_back("out.bev_embed", prev_bev_dims);
  head_no_prev_network_io.emplace_back("out.ego_fut_preds", ego_fut_preds_dims);
  head_no_prev_network_io.emplace_back("out.all_traj_preds", traj_preds_dims);
  head_no_prev_network_io.emplace_back("out.all_traj_cls_scores", traj_cls_dims);
  head_no_prev_network_io.emplace_back("out.all_bbox_preds", bbox_preds_dims);
  head_no_prev_network_io.emplace_back("out.all_cls_scores", all_cls_scores_dims);
  head_no_prev_network_io.emplace_back("out.map_all_cls_scores", map_all_cls_scores_dims);
  head_no_prev_network_io.emplace_back("out.map_all_pts_preds", map_all_pts_preds_dims);
  head_no_prev_network_io.emplace_back("out.map_all_bbox_preds", map_all_bbox_preds_dims);
  return head_no_prev_network_io;
}

// エンジンビルド専用関数
std::unique_ptr<autoware::tensorrt_common::TrtCommon> build_engine(
    const autoware::tensorrt_common::TrtCommonConfig& trt_common_config,
    const std::vector<autoware::tensorrt_common::NetworkIO>& network_io,
    const std::string& engine_name,
    const std::string& plugins_path,
    std::shared_ptr<VadLogger> logger) {
  logger->info("Building " + engine_name + " engine...");
  try {
    auto trt_common = std::make_unique<autoware::tensorrt_common::TrtCommon>(
      trt_common_config, std::make_shared<autoware::tensorrt_common::Profiler>(),
      std::vector<std::string>{plugins_path});
    auto network_io_ptr = std::make_unique<std::vector<autoware::tensorrt_common::NetworkIO>>(network_io);
    if (!trt_common->setup(nullptr, std::move(network_io_ptr))) {
      logger->error("Failed to setup " + engine_name + " TrtCommon");
      return nullptr;
    }
    logger->info(engine_name + " engine built successfully");
    return trt_common;
  } catch (const std::exception& e) {
    logger->error("Exception building " + engine_name + " engine: " + std::string(e.what()));
    return nullptr;
  }
}

// TensorRT初期化API
std::unique_ptr<autoware::tensorrt_common::TrtCommon> init_tensorrt(
    const VadConfig& vad_config,
    const autoware::tensorrt_common::TrtCommonConfig& trt_common_config,
    const std::string& name,
    const std::string& plugins_path,
    std::shared_ptr<VadLogger> logger) {
  logger->info("Initializing TensorRT engine: " + name);
  
  // NetworkIO arrays generation based on name
  std::vector<autoware::tensorrt_common::NetworkIO> network_io;
  if (name == "backbone") {
    network_io = generate_network_io_backbone(vad_config);
  } else if (name == "head") {
    network_io = generate_network_io_head(vad_config);
  } else if (name == "head_no_prev") {
    network_io = generate_network_io_head_no_prev(vad_config);
  } else {
    logger->error("Unknown engine name: " + name);
    return nullptr;
  }

  // Build engine
  auto engine = build_engine(trt_common_config, network_io, name, plugins_path, logger);
  if (!engine) {
    logger->error("Failed to build " + name + " engine");
    return nullptr;
  }

  logger->info(name + " engine initialization completed successfully");
  return engine;
}

// Net class implementation

Net::Net(
  const VadConfig& vad_config,
  const autoware::tensorrt_common::TrtCommonConfig& trt_common_config,
  const std::string& name,
  const std::string& plugins_path,
  std::shared_ptr<VadLogger> logger
) : logger_(logger)
{
  trt_common = init_tensorrt(vad_config, trt_common_config, name, plugins_path, logger_);
  if (!trt_common) {
    logger_->error("Failed to initialize TensorRT engine: " + name);
  }
}

Net::Net(
  const VadConfig& vad_config,
  const autoware::tensorrt_common::TrtCommonConfig& trt_common_config,
  NetworkType network_type,
  const std::string& plugins_path,
  std::shared_ptr<VadLogger> logger
) : logger_(logger)
{
  std::string engine_name = toString(network_type);
  trt_common = init_tensorrt(vad_config, trt_common_config, engine_name, plugins_path, logger_);
  if (!trt_common) {
    logger_->error("Failed to initialize TensorRT engine: " + engine_name);
  }
}

void Net::Enqueue(cudaStream_t stream) {
  trt_common->enqueueV3(stream);
}

Net::~Net() {
    for (auto& pair : bindings) {
        pair.second.reset();
    }
    bindings.clear();
}

// Backbone class implementation

Backbone::Backbone(
  const VadConfig& vad_config,
  const autoware::tensorrt_common::TrtCommonConfig& trt_common_config,
  const std::string& plugins_path,
  std::shared_ptr<VadLogger> logger
) : Net(vad_config, trt_common_config, NetworkType::BACKBONE, plugins_path, logger)
{
}

std::vector<autoware::tensorrt_common::NetworkIO> Backbone::generate_network_io(const VadConfig& vad_config) {
  return generate_network_io_backbone(vad_config);
}

void Backbone::set_input_tensor(TensorMap& ext) {
  int32_t nb = trt_common->getNbIOTensors();

  for (int32_t n = 0; n < nb; n++) {
    std::string name = trt_common->getIOTensorName(n);
    nvinfer1::Dims d = trt_common->getTensorShape(name.c_str());
    nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT;
    
    if (ext.find(name) != ext.end()) {
      // use external memory
      trt_common->setTensorAddress(name.c_str(), ext[name]->ptr);
    } else {
      bindings[name] = std::make_shared<Tensor>(name, d, dtype, logger_);
      trt_common->setTensorAddress(name.c_str(), bindings[name]->ptr);
    }
  }
}

// Head class implementation

Head::Head(
  const VadConfig& vad_config,
  const autoware::tensorrt_common::TrtCommonConfig& trt_common_config,
  NetworkType network_type,
  const std::string& plugins_path,
  std::shared_ptr<VadLogger> logger
) : Net(vad_config, trt_common_config, network_type, plugins_path, logger), network_type_(network_type)
{
}

std::vector<autoware::tensorrt_common::NetworkIO> Head::generate_network_io(const VadConfig& vad_config) {
  switch (network_type_) {
    case NetworkType::HEAD:
      return generate_network_io_head(vad_config);
    case NetworkType::HEAD_NO_PREV:
      return generate_network_io_head_no_prev(vad_config);
    default:
      logger_->error("Unknown network type");
      return {};
  }
}

void Head::set_input_tensor(TensorMap& ext) {
  int32_t nb = trt_common->getNbIOTensors();

  for (int32_t n = 0; n < nb; n++) {
    std::string name = trt_common->getIOTensorName(n);
    nvinfer1::Dims d = trt_common->getTensorShape(name.c_str());
    nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT;
    
    if (ext.find(name) != ext.end()) {
      // use external memory
      trt_common->setTensorAddress(name.c_str(), ext[name]->ptr);
    } else {
      bindings[name] = std::make_shared<Tensor>(name, d, dtype, logger_);
      trt_common->setTensorAddress(name.c_str(), bindings[name]->ptr);
    }
  }
}

} // namespace autoware::tensorrt_vad
