#include <iostream>
#include <vector>
#include "lib/debug_behind_tracker.cpp"  // 直接インクルード

int main() {
    std::cout << "Testing debug logging functionality..." << std::endl;
    
    // DebugBehindTrackerインスタンスを作成
    DebugBehindTracker tracker;
    
    // ダミーのbboxデータを作成
    std::vector<BBox> test_bboxes;
    BBox test_bbox;
    
    // テスト用のbboxパラメータ
    test_bbox.bbox[0] = 5.0f;   // vad_x
    test_bbox.bbox[1] = 0.0f;   // vad_y
    test_bbox.bbox[4] = 2.0f;   // vad_z
    test_bbox.bbox[6] = 0.0f;   // sin_theta
    test_bbox.bbox[7] = 1.0f;   // cos_theta
    test_bbox.score = 0.9f;
    
    test_bboxes.push_back(test_bbox);
    
    // テスト用の変換行列（単位行列）
    Eigen::Matrix4f test_transform = Eigen::Matrix4f::Identity();
    
    std::cout << "Calling find_initial_behind_bbox..." << std::endl;
    tracker.find_initial_behind_bbox(test_bboxes, test_transform);
    
    std::cout << "Calling track_bbox..." << std::endl;
    tracker.track_bbox(test_bboxes, test_transform);
    
    std::cout << "Test completed. Check log directory for output files." << std::endl;
    
    return 0;
}
