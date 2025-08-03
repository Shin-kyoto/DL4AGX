#ifndef AUTOWARE_TENSORRT_VAD_DEBUG_BEHIND_TRACKER_HPP_
#define AUTOWARE_TENSORRT_VAD_DEBUG_BEHIND_TRACKER_HPP_

#include <vector>
#include <optional>
#include <eigen3/Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <cmath>
#include <limits>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <chrono>

// BBox構造体のincludeを追加
#include "vad_model.hpp"

namespace autoware::tensorrt_vad
{

class DebugBehindTracker {
private:
  bool is_initialized_;
  Eigen::Vector3f predicted_position_;  // 予測位置（map座標系）
  Eigen::Vector3f last_velocity_;       // 最後の速度（map座標系）
  Eigen::Vector3f smoothed_velocity_;   // 平滑化された速度
  int lost_frames_;                     // 検出を失った連続フレーム数
  int total_tracked_frames_;            // 総追跡フレーム数
  float accumulated_distance_;          // 累積移動距離
  
  // ログファイル関連のメンバ変数（mutableにして、constメソッドからも変更可能にする）
  mutable std::ofstream log_file_;
  mutable std::string log_file_path_;
  
  // 設定パラメータ
  static constexpr float BEHIND_DISTANCE_THRESHOLD = 15.0f;  // 後ろ側判定の最大距離
  static constexpr float BEHIND_X_THRESHOLD = 0.5f;         // VAD座標系でのx軸閾値
  static constexpr float BEHIND_Y_THRESHOLD = -2.0f;        // VAD座標系でのy軸閾値（マイナス側）
  static constexpr float TRACKING_DISTANCE_THRESHOLD_NORMAL = 8.0f; // 通常時の距離閾値（緩和）
  static constexpr float TRACKING_DISTANCE_THRESHOLD_EXTENDED = 15.0f; // 拡張時の距離閾値
  static constexpr float TIMESTEP = 0.1f;                   // フレーム間の時間間隔（秒）
  static constexpr float VELOCITY_ALPHA = 0.7f;             // 速度平滑化係数
  static constexpr float VELOCITY_DECAY_SLOW = 0.98f;       // ゆっくりとした速度減衰係数（直進区間用）
  static constexpr float VELOCITY_DECAY_FAST = 0.95f;       // 速い速度減衰係数（カーブ用・緩和）
  static constexpr int MAX_LOST_FRAMES_NORMAL = 15;         // 通常時の最大許容損失フレーム数（緩和）
  static constexpr int MAX_LOST_FRAMES_EXTENDED = 100;      // 拡張時の最大許容損失フレーム数（10秒）
  static constexpr float STRAIGHT_VELOCITY_THRESHOLD = 2.0f; // 直進判定の速度閾値

public:
  DebugBehindTracker() : is_initialized_(false), predicted_position_(0.0f, 0.0f, 0.0f), 
                        last_velocity_(0.0f, 0.0f, 0.0f),
                        smoothed_velocity_(0.0f, 0.0f, 0.0f), lost_frames_(0),
                        total_tracked_frames_(0), accumulated_distance_(0.0f) {
    initialize_log_file();
  }
  
  ~DebugBehindTracker() {
    if (log_file_.is_open()) {
      log_file_.close();
    }
  }
  
  void update_and_track(
    const std::vector<BBox>& bboxes,
    const Eigen::Matrix4f& base2map_transform);

private:
  // 直進区間判定
  bool is_in_straight_section() const;
  
  // 動的な距離閾値の計算
  float get_dynamic_distance_threshold() const;
  
  // 動的な最大損失フレーム数の計算
  int get_dynamic_max_lost_frames() const;
  
  // 動的な速度平滑化係数の計算
  float get_dynamic_velocity_alpha() const;
  
  // 動的な速度減衰係数の計算
  float get_dynamic_velocity_decay() const;
  // ヘルパー関数：VAD座標系からAutoware座標系への変換（簡易版）
  std::tuple<float, float, float> vad2aw_xyz(float vad_x, float vad_y, float vad_z) const;
  
  // ヘルパー関数：yaw計算
  float calculate_map_yaw(float sin_theta, float cos_theta, const Eigen::Matrix4f& base2map_transform) const;

  // ログファイル関連のメソッド
  void initialize_log_file();
  void log_debug_info(const std::string& message) const;

  void find_initial_behind_bbox(
    const std::vector<BBox>& bboxes,
    const Eigen::Matrix4f& base2map_transform);
  
  void track_bbox(
    const std::vector<BBox>& bboxes,
    const Eigen::Matrix4f& base2map_transform);
};

} // namespace autoware::tensorrt_vad

#endif // AUTOWARE_TENSORRT_VAD_DEBUG_BEHIND_TRACKER_HPP_