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

#include "autoware/tensorrt_vad/vad_model.hpp"
#include <cmath>
#include <stdexcept>

namespace autoware::tensorrt_vad {

// Sigmoid activation function
inline float sigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

// Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2)
inline std::vector<float> bbox_cxcywh_to_xyxy(const std::vector<float>& bbox) {
  if (bbox.size() != 4) {
    throw std::invalid_argument("bbox must have 4 elements: cx, cy, w, h");
  }
  
  float cx = bbox[0];
  float cy = bbox[1]; 
  float w = bbox[2];
  float h = bbox[3];
  
  float x1 = cx - 0.5f * w;
  float y1 = cy - 0.5f * h;
  float x2 = cx + 0.5f * w;
  float y2 = cy + 0.5f * h;
  
  return {x1, y1, x2, y2};
}

// Denormalize 2D points using PC range
inline std::vector<float> denormalize_2d_pts(const std::vector<float>& pts, 
                                            const std::vector<float>& pc_range = {-15.0f, -30.0f, -2.0f, 15.0f, 30.0f, 2.0f}) {
  if (pts.size() != 2) {
    throw std::invalid_argument("pts must have 2 elements: x, y");
  }
  if (pc_range.size() != 6) {
    throw std::invalid_argument("pc_range must have 6 elements");
  }
  
  std::vector<float> new_pts = pts;  // Clone the input
  
  // x       = normalized_x * (x_max[3] - x_min[0]) + x_min[0]
  new_pts[0] = pts[0] * (pc_range[3] - pc_range[0]) + pc_range[0];

  // y       = normalized_y * (y_max[4] - y_min[1]) + y_min[1]
  new_pts[1] = pts[1] * (pc_range[4] - pc_range[1]) + pc_range[1];
  
  return new_pts;
}

std::vector<std::vector<std::vector<std::vector<float>>>> postprocess_traj_preds(
    const std::vector<float>& all_traj_preds_flat) {
  const int32_t num_objects = 900;
  const int32_t num_fut_modes = 6;
  const int32_t num_fut_ts = 6;
  const int32_t traj_coords = 2;
  std::vector<std::vector<std::vector<std::vector<float>>>> traj_preds;
  traj_preds.resize(num_objects);
  for (int32_t obj = 0; obj < num_objects; ++obj) {
    traj_preds[obj].resize(num_fut_modes);
    for (int32_t fut_mode = 0; fut_mode < num_fut_modes; ++fut_mode) {
      traj_preds[obj][fut_mode].resize(num_fut_ts);
      for (int32_t ts = 0; ts < num_fut_ts; ++ts) {
        traj_preds[obj][fut_mode][ts].resize(traj_coords);
        int32_t idx_occupied = obj * num_fut_modes * num_fut_ts * traj_coords;
        int32_t idx_flat = idx_occupied + fut_mode * num_fut_ts * traj_coords + ts * traj_coords;
        traj_preds[obj][fut_mode][ts][0] = all_traj_preds_flat[idx_flat];
        traj_preds[obj][fut_mode][ts][1] = all_traj_preds_flat[idx_flat + 1];
      }
    }
  }
  return traj_preds;
}

std::vector<std::vector<float>> postprocess_traj_cls_scores(
    const std::vector<float>& all_traj_cls_scores_flat) {
  const int32_t num_objects = 900;
  const int32_t num_fut_modes = 6;
  std::vector<std::vector<float>> traj_cls_scores;
  traj_cls_scores.resize(num_objects);
  for (int32_t obj = 0; obj < num_objects; ++obj) {
    traj_cls_scores[obj].resize(num_fut_modes);
    for (int32_t fut_mode = 0; fut_mode < num_fut_modes; ++fut_mode) {
      int32_t idx_occupied = obj * num_fut_modes;
      int32_t idx_flat = idx_occupied + fut_mode;
      traj_cls_scores[obj][fut_mode] = all_traj_cls_scores_flat[idx_flat];
    }
  }
  return traj_cls_scores;
}

std::vector<std::vector<float>> postprocess_bbox_preds(
    const std::vector<float>& all_bbox_preds_flat) {
  const int32_t num_objects = 900;
  const int32_t bbox_features = 10;
  std::vector<std::vector<float>> bbox_preds;
  bbox_preds.resize(num_objects);
  for (int32_t obj = 0; obj < num_objects; ++obj) {
    bbox_preds[obj].resize(bbox_features);
    for (int32_t feat = 0; feat < bbox_features; ++feat) {
      int32_t idx_occupied = obj * bbox_features;
      int32_t idx_flat = idx_occupied + feat;
      bbox_preds[obj][feat] = all_bbox_preds_flat[idx_flat];
    }
  }
  return bbox_preds;
}

/**
 * @brief クラス予測のフラットな配列を解析し、スコアの2次元配列に変換する
 * 推論時は最終層（-1番目のlayer）のみを使用
 */
std::vector<std::vector<float>> 
process_class_scores(const std::vector<float>& cls_preds_flat) 
{
    const int32_t num_layers = 3; // 3レイヤーのポイント予測
    const int32_t num_query = 100;
    const int32_t cls_out_channels = 3;
    
    // 最終層（インデックス2）のデータのみを使用
    const int32_t final_layer_idx = num_layers - 1; // = 2
    const int32_t layer_size = num_query * cls_out_channels;
    const int32_t final_layer_offset = final_layer_idx * layer_size;
    
    std::vector<std::vector<float>> cls_scores(num_query, std::vector<float>(cls_out_channels));

    for (int32_t q = 0; q < num_query; ++q) {
        for (int32_t c = 0; c < cls_out_channels; ++c) {
            int32_t flat_idx = final_layer_offset + q * cls_out_channels + c;
            cls_scores[q][c] = sigmoid(cls_preds_flat[flat_idx]);
        }
    }
    return cls_scores;
}

/**
 * @brief 座標予測のフラットな配列を解析し、非正規化された座標の3次元配列に変換する
 * 推論時は最終層（-1番目のlayer）のみを使用
 */
std::vector<std::vector<std::vector<float>>> 
process_points(const std::vector<float>& pts_preds_flat) 
{
    const int32_t num_layers = 3; // 3レイヤーのポイント予測
    const int32_t num_query = 100;
    const int32_t fixed_num_pts = 20;
    const int32_t pts_coords = 2;
    
    // 最終層（インデックス2）のデータのみを使用
    const int32_t final_layer_idx = num_layers - 1; // = 2
    const int32_t layer_size = num_query * fixed_num_pts * pts_coords;
    const int32_t final_layer_offset = final_layer_idx * layer_size;
    
    std::vector<std::vector<std::vector<float>>> pts_preds(num_query, std::vector<std::vector<float>>(fixed_num_pts, std::vector<float>(pts_coords)));

    for (int32_t q = 0; q < num_query; ++q) {
        for (int32_t p = 0; p < fixed_num_pts; ++p) {
            std::vector<float> pt(pts_coords);
            for (int32_t d = 0; d < pts_coords; ++d) {
                int32_t flat_idx = final_layer_offset + q * fixed_num_pts * pts_coords + p * pts_coords + d;
                pt[d] = pts_preds_flat[flat_idx];
            }
            std::vector<float> denormalized_pt = denormalize_2d_pts(pt);
            for (int32_t d = 0; d < pts_coords; ++d) {
                pts_preds[q][p][d] = denormalized_pt[d];
            }
        }
    }
    return pts_preds;
}

/**
 * @brief スコアが最も高い予測を選択し、クラスごとの信頼度閾値でフィルタリングする
 * @param cls_scores クラススコア [num_query, num_classes]
 * @param pts_preds 座標予測 [num_query, num_points, 2]
 * @param class_thresholds クラスごとの信頼度閾値マップ
 */
std::pair<std::vector<std::vector<std::vector<float>>>, std::vector<std::string>>
select_most_confident_predictions(
    const std::vector<std::vector<float>>& cls_scores,
    const std::vector<std::vector<std::vector<float>>>& pts_preds,
    const std::map<std::string, float>& class_thresholds)
{
    const std::vector<std::string> class_names = {"divider", "ped_crossing", "boundary"};
    const int32_t num_query = cls_scores.size();

    std::vector<std::vector<std::vector<float>>> filtered_polylines;
    std::vector<std::string> filtered_types;
    
    for (int32_t q = 0; q < num_query; ++q) {
        auto max_it = std::max_element(cls_scores[q].begin(), cls_scores[q].end());
        float max_score = *max_it;
        int32_t max_class_idx = std::distance(cls_scores[q].begin(), max_it);

        // クラスごとの閾値を適用
        std::string class_name = class_names[max_class_idx];
        auto threshold_it = class_thresholds.find(class_name);
        if (threshold_it != class_thresholds.end() && max_score >= threshold_it->second) {
            filtered_polylines.push_back(pts_preds[q]);
            filtered_types.push_back(class_name);
        }
    }

    return {filtered_polylines, filtered_types};
}

/**
 * @brief モデルの推論結果全体の後処理を行う
 */
std::pair<std::vector<std::vector<std::vector<float>>>, std::vector<std::string>>
postprocess_map_preds(
    const std::vector<float>& map_all_cls_preds_flat,
    const std::vector<float>& map_all_pts_preds_flat,
    const std::map<std::string, float>& class_thresholds) 
{
    // 1. クラススコアを計算する
    auto cls_scores = process_class_scores(map_all_cls_preds_flat);
    
    // 2. 座標を計算する
    auto pts_preds = process_points(map_all_pts_preds_flat);
    
    // 3. 計算結果を基に、クラスごとの信頼度閾値で予測を選択する
    return select_most_confident_predictions(cls_scores, pts_preds, class_thresholds);
}

} // namespace autoware::tensorrt_vad
