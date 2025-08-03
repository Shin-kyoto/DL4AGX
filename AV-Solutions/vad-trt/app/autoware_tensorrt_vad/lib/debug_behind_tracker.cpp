#include "autoware/tensorrt_vad/debug_behind_tracker.hpp"
#include "autoware/tensorrt_vad/vad_model.hpp"
#include <limits>
#include <filesystem>

namespace autoware::tensorrt_vad
{

void DebugBehindTracker::initialize_log_file()
{
  // 現在時刻を取得してファイル名を生成
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  auto tm = *std::localtime(&time_t);
  
  std::stringstream ss;
  ss << "/home/autoware/ghq/github.com/Shin-kyoto/DL4AGX_ns_rosbag/DL4AGX/AV-Solutions/vad-trt/app/autoware_tensorrt_vad/log/"
     << std::setfill('0') << std::setw(2) << (tm.tm_mon + 1)
     << std::setfill('0') << std::setw(2) << tm.tm_mday
     << std::setfill('0') << std::setw(2) << tm.tm_hour
     << std::setfill('0') << std::setw(2) << tm.tm_min
     << std::setfill('0') << std::setw(2) << tm.tm_sec
     << "_tracking_log.txt";
  
  log_file_path_ = ss.str();
  
  // ディレクトリを作成（存在しない場合）
  std::filesystem::create_directories(std::filesystem::path(log_file_path_).parent_path());
  
  // ログファイルを開く
  log_file_.open(log_file_path_, std::ios::out | std::ios::app);
  
  if (log_file_.is_open()) {
    log_debug_info("=== Debug Behind Tracker Log Started ===");
    log_debug_info("Log file: " + log_file_path_);
  }
}

void DebugBehindTracker::log_debug_info(const std::string& message) const
{
  if (log_file_.is_open()) {
    // タイムスタンプを追加
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto tm = *std::localtime(&time_t);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      now.time_since_epoch()) % 1000;
    
    log_file_ << std::setfill('0') << std::setw(2) << tm.tm_hour << ":"
              << std::setfill('0') << std::setw(2) << tm.tm_min << ":"
              << std::setfill('0') << std::setw(2) << tm.tm_sec << "."
              << std::setfill('0') << std::setw(3) << ms.count() << " "
              << message << std::endl;
    log_file_.flush();  // 即座にファイルに書き込み
  }
}

void DebugBehindTracker::update_and_track(
  const std::vector<BBox>& bboxes,
  const Eigen::Matrix4f& base2map_transform)
{
  if (!is_initialized_) {
    // 初期化：真後ろのbboxを探す
    find_initial_behind_bbox(bboxes, base2map_transform);
    return;
  }
  
  // デバッグ：bboxes数を出力
  log_debug_info("=== Update and Track Called ===");
  
  std::stringstream ss;
  ss << "Number of bboxes: " << bboxes.size() << ", Total tracked frames: " << total_tracked_frames_;
  log_debug_info(ss.str());
  
  // 既に初期化済み：近いbboxを探してtrack
  track_bbox(bboxes, base2map_transform);
}

void DebugBehindTracker::find_initial_behind_bbox(
  const std::vector<BBox>& bboxes,
  const Eigen::Matrix4f& base2map_transform)
{
  for (const auto& bbox : bboxes) {
    float vad_x = bbox.bbox[0];
    float vad_y = bbox.bbox[1];
    float vad_z = bbox.bbox[4] + 1.7f;  // z_offset
    
    // ego後ろの条件チェック（VAD座標系）
    if (std::abs(vad_x) < BEHIND_X_THRESHOLD && 
        vad_y < BEHIND_Y_THRESHOLD && 
        std::abs(vad_y) < BEHIND_DISTANCE_THRESHOLD) {
      
      // 座標変換してmap座標系に変換
      auto [aw_x, aw_y, aw_z] = vad2aw_xyz(vad_x, vad_y, vad_z);
      Eigen::Vector4f position_base(aw_x, aw_y, aw_z, 1.0f);
      Eigen::Vector4f position_map = base2map_transform * position_base;
      
      // 初期化
      predicted_position_ = Eigen::Vector3f(position_map.x(), position_map.y(), position_map.z());
      
      // 初期速度を設定
      float v_x = bbox.bbox[8];
      float v_y = bbox.bbox[9];
      auto [aw_vx, aw_vy, aw_vz] = vad2aw_xyz(v_x, v_y, 0.0f);
      Eigen::Vector4f velocity_base(aw_vx, aw_vy, aw_vz, 0.0f);  // 速度なので平行移動成分は0
      Eigen::Vector4f velocity_map = base2map_transform * velocity_base;
      last_velocity_ = Eigen::Vector3f(velocity_map.x(), velocity_map.y(), velocity_map.z());
      smoothed_velocity_ = last_velocity_;
      lost_frames_ = 0;
      total_tracked_frames_ = 0;
      
      is_initialized_ = true;
      
      // デバッグ出力
      log_debug_info("=== Initial Behind BBox Found ===");
      
      std::stringstream ss;
      ss << "VAD position: (" << vad_x << ", " << vad_y << ", " << vad_z << ")";
      log_debug_info(ss.str());
      
      ss.str(""); ss.clear();
      ss << "AW position: (" << aw_x << ", " << aw_y << ", " << aw_z << ")";
      log_debug_info(ss.str());
      
      ss.str(""); ss.clear();
      ss << "Map position: (" << position_map.x() << ", " << position_map.y() << ", " << position_map.z() << ")";
      log_debug_info(ss.str());
      
      ss.str(""); ss.clear();
      ss << "Map velocity: (" << last_velocity_.x() << ", " << last_velocity_.y() << ", " << last_velocity_.z() << ")";
      log_debug_info(ss.str());
      
      ss.str(""); ss.clear();
      ss << "Smoothed velocity: (" << smoothed_velocity_.x() << ", " << smoothed_velocity_.y() << ", " << smoothed_velocity_.z() << ")";
      log_debug_info(ss.str());
      
      log_debug_info("================================");
      
      
      return;  // 最初に見つかったもので初期化完了
    }
  }
}

void DebugBehindTracker::track_bbox(
  const std::vector<BBox>& bboxes,
  const Eigen::Matrix4f& base2map_transform)
{
  // 平滑化された速度で位置を予測
  predicted_position_ += smoothed_velocity_ * TIMESTEP;
  
  // 動的な閾値を取得
  float current_distance_threshold = get_dynamic_distance_threshold();
  int current_max_lost_frames = get_dynamic_max_lost_frames();
  float current_velocity_decay = get_dynamic_velocity_decay();
  float current_velocity_alpha = get_dynamic_velocity_alpha();
  bool is_straight = is_in_straight_section();
  
  // 最も近いbboxを探す
  float min_distance = std::numeric_limits<float>::max();
  std::optional<size_t> best_match_idx = std::nullopt;
  
  for (size_t i = 0; i < bboxes.size(); ++i) {
    const auto& bbox = bboxes[i];
    
    float vad_x = bbox.bbox[0];
    float vad_y = bbox.bbox[1];
    float vad_z = bbox.bbox[4] + 1.7f;
    
    // 座標変換
    auto [aw_x, aw_y, aw_z] = vad2aw_xyz(vad_x, vad_y, vad_z);
    Eigen::Vector4f position_base(aw_x, aw_y, aw_z, 1.0f);
    Eigen::Vector4f position_map = base2map_transform * position_base;
    
    Eigen::Vector3f current_position(position_map.x(), position_map.y(), position_map.z());
    float distance = (current_position - predicted_position_).norm();
    
    if (distance < current_distance_threshold && distance < min_distance) {
      min_distance = distance;
      best_match_idx = i;
    }
  }
  
  if (best_match_idx.has_value()) {
    // マッチするbboxが見つかった場合
    const auto& bbox = bboxes[best_match_idx.value()];
    
    float vad_x = bbox.bbox[0];
    float vad_y = bbox.bbox[1];
    float vad_z = bbox.bbox[4] + 1.7f;
    
    // 座標変換
    auto [aw_x, aw_y, aw_z] = vad2aw_xyz(vad_x, vad_y, vad_z);
    Eigen::Vector4f position_base(aw_x, aw_y, aw_z, 1.0f);
    Eigen::Vector4f position_map = base2map_transform * position_base;
    
    // yaw計算
    float sin_theta = bbox.bbox[6];
    float cos_theta = bbox.bbox[7];
    float map_yaw = calculate_map_yaw(sin_theta, cos_theta, base2map_transform);
    
    // 実際の位置に更新
    Eigen::Vector3f actual_position(position_map.x(), position_map.y(), position_map.z());
    
    // 速度を更新（実際の位置変化から計算）
    Eigen::Vector3f position_change = actual_position - predicted_position_;
    Eigen::Vector3f new_velocity = position_change / TIMESTEP;
    
    // 速度の平滑化
    smoothed_velocity_ = current_velocity_alpha * smoothed_velocity_ + (1.0f - current_velocity_alpha) * new_velocity;
    last_velocity_ = new_velocity;
    
    // 予測位置を実際の位置に更新
    predicted_position_ = actual_position;
    lost_frames_ = 0; // リセット
    total_tracked_frames_++;
    
    // デバッグ出力
    log_debug_info("=== Tracked Behind BBox ===");
    
    std::stringstream ss;
    ss << "VAD position: (" << vad_x << ", " << vad_y << ", " << vad_z << ")";
    log_debug_info(ss.str());
    
    ss.str(""); ss.clear();
    ss << "AW position: (" << aw_x << ", " << aw_y << ", " << aw_z << ")";
    log_debug_info(ss.str());
    
    ss.str(""); ss.clear();
    ss << "Map position: (" << actual_position.x() << ", " << actual_position.y() << ", " << actual_position.z() << ")";
    log_debug_info(ss.str());
    
    // Yaw角の詳細デバッグ情報
    float vad_yaw = std::atan2(sin_theta, cos_theta);
    Eigen::Matrix3f rotation = base2map_transform.block<3,3>(0,0);
    float transform_yaw = std::atan2(rotation(1,0), rotation(0,0));
    
    ss.str(""); ss.clear();
    ss << "Raw sin_theta: " << sin_theta << ", cos_theta: " << cos_theta;
    log_debug_info(ss.str());
    
    ss.str(""); ss.clear();
    ss << "VAD yaw (rad): " << vad_yaw << ", VAD yaw (deg): " << (vad_yaw * 180.0 / M_PI);
    log_debug_info(ss.str());
    
    ss.str(""); ss.clear();
    ss << "Transform yaw (rad): " << transform_yaw << ", Transform yaw (deg): " << (transform_yaw * 180.0 / M_PI);
    log_debug_info(ss.str());
    
    ss.str(""); ss.clear();
    ss << "Map yaw (rad): " << map_yaw << ", Map yaw (deg): " << (map_yaw * 180.0 / M_PI);
    log_debug_info(ss.str());
    
    ss.str(""); ss.clear();
    ss << "Map velocity: (" << last_velocity_.x() << ", " << last_velocity_.y() << ", " << last_velocity_.z() << ")";
    log_debug_info(ss.str());
    
    ss.str(""); ss.clear();
    ss << "Smoothed velocity: (" << smoothed_velocity_.x() << ", " << smoothed_velocity_.y() << ", " << smoothed_velocity_.z() << ")";
    log_debug_info(ss.str());
    
    ss.str(""); ss.clear();
    ss << "Distance to prediction: " << min_distance;
    log_debug_info(ss.str());
    
    ss.str(""); ss.clear();
    ss << "Tracking frames: " << total_tracked_frames_ << ", Lost frames: " << lost_frames_;
    log_debug_info(ss.str());
    
    log_debug_info("==========================");
    
  } else {
    // マッチするbboxが見つからない場合
    lost_frames_++;
    
    if (lost_frames_ > current_max_lost_frames) {
      // トラッキング終了
      log_debug_info("=== Tracking Lost - Too Many Missed Frames ===");
      
      std::stringstream ss;
      ss << "Lost frames: " << lost_frames_ << ", Max allowed: " << current_max_lost_frames;
      log_debug_info(ss.str());
      
      ss.str(""); ss.clear();
      ss << "Straight section: " << (is_straight ? "YES" : "NO") << ", Tracked frames: " << total_tracked_frames_;
      log_debug_info(ss.str());
      
      log_debug_info("===============================================");
      return;
    }
    
    // 速度を減衰
    smoothed_velocity_ *= current_velocity_decay;
    
    // 予測のみで継続
    log_debug_info("=== No Match Found - Prediction Only ===");
    
    std::stringstream ss;
    ss << "Predicted position: (" << predicted_position_.x() << ", " << predicted_position_.y() << ", " << predicted_position_.z() << ")";
    log_debug_info(ss.str());
    
    ss.str(""); ss.clear();
    ss << "Decayed velocity: (" << smoothed_velocity_.x() << ", " << smoothed_velocity_.y() << ", " << smoothed_velocity_.z() << ")";
    log_debug_info(ss.str());
    
    ss.str(""); ss.clear();
    ss << "Distance threshold: " << current_distance_threshold << "m, Velocity decay: " << current_velocity_decay << ", Alpha: " << current_velocity_alpha;
    log_debug_info(ss.str());
    
    ss.str(""); ss.clear();
    ss << "Straight section: " << (is_straight ? "YES" : "NO") << ", Tracked frames: " << total_tracked_frames_;
    log_debug_info(ss.str());
    
    ss.str(""); ss.clear();
    ss << "Lost frames: " << lost_frames_ << "/" << current_max_lost_frames;
    log_debug_info(ss.str());
    
    log_debug_info("=====================================");
  }
}

// 動的メソッドの実装
bool DebugBehindTracker::is_in_straight_section() const
{
  // 速度が十分あり、かつ追跡実績があり、かつ速度の変化が小さい場合に直進と判定
  float speed = smoothed_velocity_.norm();
  float speed_change = (smoothed_velocity_ - last_velocity_).norm();
  
  // より現実的な直進判定：速度変化が小さく、一定速度で移動している場合
  bool has_sufficient_speed = speed > STRAIGHT_VELOCITY_THRESHOLD;
  bool has_tracking_history = total_tracked_frames_ > 30;  // 3秒以上追跡済み（緩和）
  bool has_stable_velocity = speed_change < 2.0f;         // 速度変化が2.0m/s未満（緩和）
  
  bool is_straight = has_sufficient_speed && has_tracking_history && has_stable_velocity;
  
  // デバッグ出力（時々）
  if (total_tracked_frames_ % 20 == 0) {  // 2秒ごと
    log_debug_info("=== Straight Section Check ===");
    
    std::stringstream ss;
    ss << "Speed: " << speed << " (threshold: " << STRAIGHT_VELOCITY_THRESHOLD << ")";
    log_debug_info(ss.str());
    
    ss.str(""); ss.clear();
    ss << "Speed change: " << speed_change << " (threshold: 2.0)";
    log_debug_info(ss.str());
    
    ss.str(""); ss.clear();
    ss << "Tracked frames: " << total_tracked_frames_ << " (threshold: 30)";
    log_debug_info(ss.str());
    
    ss.str(""); ss.clear();
    ss << "Is straight: " << (is_straight ? "YES" : "NO");
    log_debug_info(ss.str());
    
    log_debug_info("==============================");
  }
  
  return is_straight;
}

float DebugBehindTracker::get_dynamic_distance_threshold() const
{
  if (is_in_straight_section()) {
    return TRACKING_DISTANCE_THRESHOLD_EXTENDED;
  }
  return TRACKING_DISTANCE_THRESHOLD_NORMAL;
}

int DebugBehindTracker::get_dynamic_max_lost_frames() const
{
  if (is_in_straight_section()) {
    return MAX_LOST_FRAMES_EXTENDED;
  }
  return MAX_LOST_FRAMES_NORMAL;
}

float DebugBehindTracker::get_dynamic_velocity_alpha() const
{
  if (is_in_straight_section()) {
    return 0.7f;  // 直進時は通常の平滑化
  }
  return 0.8f;    // カーブ時はより強い慣性を保つ
}

float DebugBehindTracker::get_dynamic_velocity_decay() const
{
  if (is_in_straight_section()) {
    return VELOCITY_DECAY_SLOW;
  }
  return VELOCITY_DECAY_FAST;
}

// ヘルパーメソッドの実装
std::tuple<float, float, float> DebugBehindTracker::vad2aw_xyz(float vad_x, float vad_y, float vad_z) const
{
  // 180度回転（VAD座標系はx軸が逆向き）
  float aw_x = -vad_x;
  float aw_y = vad_y;
  float aw_z = vad_z;
  
  return {aw_x, aw_y, aw_z};
}

float DebugBehindTracker::calculate_map_yaw(float sin_theta, float cos_theta, const Eigen::Matrix4f& base2map_transform) const
{
  // VAD座標系での角度（補正なし）
  float vad_yaw = std::atan2(sin_theta, cos_theta);
  
  // 補正なしの方向ベクトル
  float vad_dx = std::cos(vad_yaw);
  float vad_dy = std::sin(vad_yaw);
  
  // 簡易VAD→AW変換
  float aw_dx = -vad_dx;  // 180度回転
  float aw_dy = vad_dy;
  
  // base→map座標変換
  Eigen::Vector3f base_direction(aw_dx, aw_dy, 0.0f);
  Eigen::Vector3f map_direction = base2map_transform.block<3, 3>(0, 0) * base_direction;
  float map_yaw = std::atan2(map_direction.y(), map_direction.x());
  
  // 最後に角度補正を適用（180度 + 90度回転）
  map_yaw += M_PI + M_PI/2;
  
  return map_yaw;
}

} // namespace autoware::tensorrt_vad
