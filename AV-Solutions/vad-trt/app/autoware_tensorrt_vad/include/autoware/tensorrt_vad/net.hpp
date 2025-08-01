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

#ifndef NET_H_
#define NET_H_

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <map>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "tensor.hpp"
#include "ros_vad_logger.hpp"

#include <autoware/tensorrt_common/tensorrt_common.hpp>

namespace autoware::tensorrt_vad {

// NetworkIO configuration parameters
struct VadConfig
{
  int32_t num_cameras;
  int32_t bev_h, bev_w;
  int32_t bev_feature_dim;
  int32_t num_decoder_layers;
  int32_t prediction_num_queries;
  int32_t prediction_num_classes;
  int32_t prediction_bbox_pred_dim;
  int32_t prediction_trajectory_modes;
  int32_t prediction_timesteps;
  int32_t planning_ego_commands;
  int32_t planning_timesteps;
  int32_t can_bus_dim;
  int32_t target_image_width;
  int32_t target_image_height;
  int32_t downsample_factor;
  int32_t map_num_queries;
  int32_t map_num_class;
  int32_t map_points_per_polylines;
};

// Standalone functions for NetworkIO generation and engine building

// Backbone NetworkIO設定生成関数
std::vector<autoware::tensorrt_common::NetworkIO> generate_network_io_backbone(const VadConfig& vad_config);

// Head NetworkIO設定生成関数
std::vector<autoware::tensorrt_common::NetworkIO> generate_network_io_head(const VadConfig& vad_config);

// Head No Previous NetworkIO設定生成関数
std::vector<autoware::tensorrt_common::NetworkIO> generate_network_io_head_no_prev(const VadConfig& vad_config);

// エンジンビルド専用関数
std::unique_ptr<autoware::tensorrt_common::TrtCommon> build_engine(
    const autoware::tensorrt_common::TrtCommonConfig& trt_common_config,
    const std::vector<autoware::tensorrt_common::NetworkIO>& network_io,
    const std::string& engine_name,
    const std::string& plugins_path,
    std::shared_ptr<VadLogger> logger);

// TensorRT初期化API
std::unique_ptr<autoware::tensorrt_common::TrtCommon> init_tensorrt(
    const VadConfig& vad_config,
    const autoware::tensorrt_common::TrtCommonConfig& trt_common_config,
    const std::string& name,
    const std::string& plugins_path,
    std::shared_ptr<VadLogger> logger);

struct Net {
  TensorMap bindings;
  std::unique_ptr<autoware::tensorrt_common::TrtCommon> trt_common;
  std::shared_ptr<VadLogger> logger_;

  Net(
    const VadConfig& vad_config,
    const autoware::tensorrt_common::TrtCommonConfig& trt_common_config,
    const std::string& name,
    const std::string& plugins_path,
    std::shared_ptr<VadLogger> logger
  );

  void set_input_tensor_backbone(TensorMap& ext);
  void set_input_tensor_head(TensorMap& ext);
  void set_input_tensor(TensorMap& ext, const std::string& name);
  void Enqueue(cudaStream_t stream);

  ~Net();
};

} // namespace autoware::tensorrt_vad

#endif // NET_H_
