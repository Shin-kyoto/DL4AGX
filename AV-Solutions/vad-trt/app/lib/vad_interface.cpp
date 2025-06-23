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

#include "vad_interface.hpp"
#include <sstream>
#include <iomanip>

namespace autoware::tensorrt_vad
{

// Lidar2Cam constructor
Lidar2Cam::Lidar2Cam(const geometry_msgs::msg::Transform& transform,
                     std::function<std::pair<float, float>(float, float)> transform_autoware2vad_xy,
                     std::function<Eigen::Quaternionf(const Eigen::Quaternionf&)> transform_autoware2vad_q)
    : transform_autoware2vad_xy_(std::move(transform_autoware2vad_xy))
    , transform_autoware2vad_q_(std::move(transform_autoware2vad_q)) {
    load_from_tf(transform);
}

void Lidar2Cam::load_from_tf(const geometry_msgs::msg::Transform& transform) {
    // デバッグ用にtransformを保存
    transform_ = transform;
    
    // Autoware座標系での値を取得
    translation_autoware_ = Eigen::Vector3f(
        transform.translation.x,
        transform.translation.y,
        transform.translation.z
    );
    
    q_autoware_ = Eigen::Quaternionf(
        transform.rotation.w,
        transform.rotation.x,
        transform.rotation.y,
        transform.rotation.z
    );
    
    // VAD座標系への変換
    translation_vad_ = transform_translation_to_vad(translation_autoware_);
    q_vad_ = transform_quaternion_to_vad(q_autoware_);
    
    // lidar2cam_rtを構築 (VAD座標系)
    lidar2cam_ = Eigen::Matrix4f::Identity();
    lidar2cam_.block<3, 3>(0, 0) = q_vad_.toRotationMatrix();
    lidar2cam_.block<3, 1>(0, 3) = translation_vad_;
    
    is_valid_ = true;
}

void Lidar2Cam::reset() {
    lidar2cam_ = Eigen::Matrix4f::Identity();
    is_valid_ = false;
}

void Lidar2Cam::log_debug_info(const VadLogger& logger, int camera_id) const {
    if (!is_valid_) {
        logger.warn("Lidar2Cam is not valid for camera_id: " + std::to_string(camera_id));
        return;
    }
    
    logger.info("=== Lidar2Cam Debug Info (Camera ID: " + std::to_string(camera_id) + ") ===");
    
    // 元のtransform情報
    logger.info("Original Transform:");
    logger.info("  Translation: [" + 
               std::to_string(transform_.translation.x) + ", " +
               std::to_string(transform_.translation.y) + ", " +
               std::to_string(transform_.translation.z) + "]");
    logger.info("  Rotation: [" + 
               std::to_string(transform_.rotation.x) + ", " +
               std::to_string(transform_.rotation.y) + ", " +
               std::to_string(transform_.rotation.z) + ", " +
               std::to_string(transform_.rotation.w) + "]");
    
    // Autoware座標系での値
    logger.info("Autoware Coordinates:");
    logger.info("  Translation: [" + 
               std::to_string(translation_autoware_.x()) + ", " +
               std::to_string(translation_autoware_.y()) + ", " +
               std::to_string(translation_autoware_.z()) + "]");
    logger.info("  Quaternion: [" + 
               std::to_string(q_autoware_.x()) + ", " +
               std::to_string(q_autoware_.y()) + ", " +
               std::to_string(q_autoware_.z()) + ", " +
               std::to_string(q_autoware_.w()) + "]");
    
    // VAD座標系での値
    logger.info("VAD Coordinates:");
    logger.info("  Translation: [" + 
               std::to_string(translation_vad_.x()) + ", " +
               std::to_string(translation_vad_.y()) + ", " +
               std::to_string(translation_vad_.z()) + "]");
    logger.info("  Quaternion: [" + 
               std::to_string(q_vad_.x()) + ", " +
               std::to_string(q_vad_.y()) + ", " +
               std::to_string(q_vad_.z()) + ", " +
               std::to_string(q_vad_.w()) + "]");
    
    // 最終的な行列
    logger.info("Final Lidar2Cam Matrix:");
    for (int i = 0; i < 4; ++i) {
        std::stringstream ss;
        ss << "  Row " << i << ": [";
        for (int j = 0; j < 4; ++j) {
            ss << std::fixed << std::setprecision(6) << lidar2cam_(i, j);
            if (j < 3) ss << ", ";
        }
        ss << "]";
        logger.info(ss.str());
    }
    
    logger.info("=== End Lidar2Cam Debug Info ===");
}

Eigen::Vector3f Lidar2Cam::transform_translation_to_vad(const Eigen::Vector3f& translation_autoware) const {
    auto [x_vad, y_vad] = transform_autoware2vad_xy_(translation_autoware.x(), translation_autoware.y());
    return Eigen::Vector3f(x_vad, y_vad, translation_autoware.z());
}

Eigen::Quaternionf Lidar2Cam::transform_quaternion_to_vad(const Eigen::Quaternionf& q_autoware) const {
    return transform_autoware2vad_q_(q_autoware);
}

// CameraProjectionMatrix constructor
CameraProjectionMatrix::CameraProjectionMatrix(const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info) {
    if (!camera_info) {
        is_valid_ = false;
        return;
    }
    
    // デバッグ用にcamera_infoを保存
    camera_info_ = camera_info;
    
    // 4x4投影行列を作成
    projection_matrix_ = create_projection_matrix(camera_info);
    
    is_valid_ = true;
}

void CameraProjectionMatrix::reset() {
    projection_matrix_ = Eigen::Matrix4f::Zero();
    is_valid_ = false;
}

void CameraProjectionMatrix::log_debug_info(const VadLogger& logger, int camera_id) const {
    if (!is_valid_) {
        logger.warn("CameraProjectionMatrix is not valid for camera_id: " + std::to_string(camera_id));
        return;
    }
    
    logger.info("=== CameraProjectionMatrix Debug Info (Camera ID: " + std::to_string(camera_id) + ") ===");
    
    // CameraInfo情報
    logger.info("Camera Info:");
    logger.info("  Width: " + std::to_string(camera_info_->width));
    logger.info("  Height: " + std::to_string(camera_info_->height));
    
    // K行列を抽出して表示
    Eigen::Matrix3f k_matrix;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            k_matrix(i, j) = camera_info_->k[i * 3 + j];
        }
    }
    
    logger.info("K Matrix:");
    for (int i = 0; i < 3; ++i) {
        std::stringstream ss;
        ss << "  Row " << i << ": [";
        for (int j = 0; j < 3; ++j) {
            ss << std::fixed << std::setprecision(6) << k_matrix(i, j);
            if (j < 2) ss << ", ";
        }
        ss << "]";
        logger.info(ss.str());
    }
    
    // 最終的な投影行列
    logger.info("Final Projection Matrix (a.k.a. viewpad):");
    for (int i = 0; i < 4; ++i) {
        std::stringstream ss;
        ss << "  Row " << i << ": [";
        for (int j = 0; j < 4; ++j) {
            ss << std::fixed << std::setprecision(6) << projection_matrix_(i, j);
            if (j < 3) ss << ", ";
        }
        ss << "]";
        logger.info(ss.str());
    }
    
    logger.info("=== End CameraProjectionMatrix Debug Info ===");
}

Eigen::Matrix4f CameraProjectionMatrix::create_projection_matrix(const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info) const {
    // K行列を抽出
    Eigen::Matrix3f k_matrix;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            k_matrix(i, j) = camera_info->k[i * 3 + j];
        }
    }
    
    // 4x4単位行列にK行列を埋め込む（nuScenes viewpadと同様）
    Eigen::Matrix4f projection_matrix = Eigen::Matrix4f::Identity();
    projection_matrix.block<3, 3>(0, 0) = k_matrix;
    return projection_matrix;
}

// Lidar2Img constructor
Lidar2Img::Lidar2Img(const CameraProjectionMatrix& projection_matrix, const Lidar2Cam& lidar2cam) {
    if (!projection_matrix.is_valid() || !lidar2cam.is_valid()) {
        is_valid_ = false;
        return;
    }
    
    // デバッグ用に行列を保存
    projection_matrix_ = projection_matrix.get_matrix();
    lidar2cam_matrix_ = lidar2cam.get_matrix();
    lidar2cam_transpose_ = lidar2cam.get_transpose();
    
    // lidar2img = projection_matrix @ lidar2cam.T を計算
    lidar2img_raw_ = calculate_lidar2img(projection_matrix_, lidar2cam_transpose_);
    lidar2img_ = lidar2img_raw_;
    
    // 初期状態ではスケーリングなし
    lidar2img_scaled_ = lidar2img_raw_;
    lidar2img_scaled_flat_ = matrix_to_flat_vector(lidar2img_scaled_);
    
    is_valid_ = true;
}

void Lidar2Img::apply_scaling(float scale_width, float scale_height) {
    if (!is_valid_) return;
    
    lidar2img_scaled_ = apply_scaling_to_matrix(lidar2img_raw_, scale_width, scale_height);
    lidar2img_ = lidar2img_scaled_;
    lidar2img_scaled_flat_ = matrix_to_flat_vector(lidar2img_scaled_);
}

std::vector<float> Lidar2Img::to_flat_vector() const {
    if (!is_valid_) return std::vector<float>(16, 0.0f);
    
    return lidar2img_scaled_flat_;
}

void Lidar2Img::reset() {
    lidar2img_ = Eigen::Matrix4f::Identity();
    lidar2img_raw_ = Eigen::Matrix4f::Identity();
    lidar2img_scaled_ = Eigen::Matrix4f::Identity();
    lidar2img_scaled_flat_ = std::vector<float>(16, 0.0f);
    is_valid_ = false;
}

void Lidar2Img::log_debug_info(const VadLogger& logger, int camera_id) const {
    if (!is_valid_) {
        logger.warn("Lidar2Img is not valid for camera_id: " + std::to_string(camera_id));
        return;
    }
    
    logger.info("=== Lidar2Img Debug Info (Camera ID: " + std::to_string(camera_id) + ") ===");
    
    // 入力行列
    logger.info("Input Projection Matrix:");
    for (int i = 0; i < 4; ++i) {
        std::stringstream ss;
        ss << "  Row " << i << ": [";
        for (int j = 0; j < 4; ++j) {
            ss << std::fixed << std::setprecision(6) << projection_matrix_(i, j);
            if (j < 3) ss << ", ";
        }
        ss << "]";
        logger.info(ss.str());
    }
    
    logger.info("Input Lidar2Cam Matrix:");
    for (int i = 0; i < 4; ++i) {
        std::stringstream ss;
        ss << "  Row " << i << ": [";
        for (int j = 0; j < 4; ++j) {
            ss << std::fixed << std::setprecision(6) << lidar2cam_matrix_(i, j);
            if (j < 3) ss << ", ";
        }
        ss << "]";
        logger.info(ss.str());
    }
    
    logger.info("Input Lidar2Cam Transpose:");
    for (int i = 0; i < 4; ++i) {
        std::stringstream ss;
        ss << "  Row " << i << ": [";
        for (int j = 0; j < 4; ++j) {
            ss << std::fixed << std::setprecision(6) << lidar2cam_transpose_(i, j);
            if (j < 3) ss << ", ";
        }
        ss << "]";
        logger.info(ss.str());
    }
    
    // 計算結果
    logger.info("Lidar2Img Raw Matrix (before scaling):");
    for (int i = 0; i < 4; ++i) {
        std::stringstream ss;
        ss << "  Row " << i << ": [";
        for (int j = 0; j < 4; ++j) {
            ss << std::fixed << std::setprecision(6) << lidar2img_raw_(i, j);
            if (j < 3) ss << ", ";
        }
        ss << "]";
        logger.info(ss.str());
    }
    
    logger.info("Lidar2Img Scaled Matrix (after scaling):");
    for (int i = 0; i < 4; ++i) {
        std::stringstream ss;
        ss << "  Row " << i << ": [";
        for (int j = 0; j < 4; ++j) {
            ss << std::fixed << std::setprecision(6) << lidar2img_scaled_(i, j);
            if (j < 3) ss << ", ";
        }
        ss << "]";
        logger.info(ss.str());
    }
    
    // フラットベクトル
    std::stringstream ss;
    ss << "Lidar2Img Scaled Flat Vector: [";
    for (size_t i = 0; i < lidar2img_scaled_flat_.size(); ++i) {
        ss << std::fixed << std::setprecision(6) << lidar2img_scaled_flat_[i];
        if (i < lidar2img_scaled_flat_.size() - 1) ss << ", ";
    }
    ss << "]";
    logger.info(ss.str());
    
    logger.info("=== End Lidar2Img Debug Info ===");
}

Eigen::Matrix4f Lidar2Img::calculate_lidar2img(const Eigen::Matrix4f& projection_matrix, 
                                               const Eigen::Matrix4f& lidar2cam_transpose) const {
    return projection_matrix * lidar2cam_transpose;
}

Eigen::Matrix4f Lidar2Img::apply_scaling_to_matrix(const Eigen::Matrix4f& lidar2img, 
                                                   float scale_width, float scale_height) const {
    Eigen::Matrix4f scale_matrix = Eigen::Matrix4f::Identity();
    scale_matrix(0, 0) = scale_width;
    scale_matrix(1, 1) = scale_height;
    
    return scale_matrix * lidar2img;
}

std::vector<float> Lidar2Img::matrix_to_flat_vector(const Eigen::Matrix4f& matrix) const {
    std::vector<float> flat(16);
    int32_t k = 0;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            flat[k++] = matrix(i, j);
        }
    }
    return flat;
}

}  // namespace autoware::tensorrt_vad 
