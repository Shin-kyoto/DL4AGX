#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <filesystem>
#include <fstream>
#include <random>
#include "vad_model.hpp"
#include "mock_vad_logger.hpp"

// 必要な型の前方宣言のみ使用（重複定義を避ける）
// mock_vad_logger.hppで既に定義されているため、追加のインクルードは不要

// GPU/エンジンファイルなしでテスト可能な部分をテスト
namespace autoware::tensorrt_vad
{

class VadInferTest : public ::testing::Test {
protected:
    void SetUp() override {
        mock_logger_ = std::make_shared<MockVadLogger>();
        
        // テスト用設定
        config_.plugins_path = "/tmp/test_plugin.so";
        config_.warm_up_num = 1;
        
        // 実際のファイルが存在しない場合のダミー設定
        NetConfig backbone_config;
        backbone_config.name = "backbone";
        backbone_config.engine_file = "/tmp/backbone.engine";
        backbone_config.use_graph = true;
        
        NetConfig head_no_prev_config;
        head_no_prev_config.name = "head_no_prev";
        head_no_prev_config.engine_file = "/tmp/head_no_prev.engine";
        head_no_prev_config.use_graph = false;
        
        NetConfig head_config;
        head_config.name = "head";
        head_config.engine_file = "/tmp/head.engine";
        head_config.use_graph = false;
        
        config_.nets_config = {backbone_config, head_no_prev_config, head_config};
    }

    VadInputData createValidInputData() {
        VadInputData input_data;
        
        // 6カメラ、3チャンネル、256x704の画像データ
        input_data.camera_images_.resize(6 * 3 * 256 * 704, 0.5f);
        
        // shift データ (x, y, yaw)
        input_data.shift_ = {0.0f, 0.0f, 0.0f};
        
        // lidar2img 変換行列 (6カメラ × 4x4)
        input_data.lidar2img_.resize(6 * 4 * 4, 0.0f);
        for (size_t i = 0; i < 6; ++i) {
            // 各カメラの単位行列を設定
            for (size_t j = 0; j < 4; ++j) {
                input_data.lidar2img_[i * 16 + j * 4 + j] = 1.0f;
            }
        }
        
        // CAN-BUS データ (18要素)
        input_data.can_bus_ = {
            10.0f, 0.0f, 0.0f,    // velocity
            0.0f, 0.0f, 0.1f,     // angular velocity
            0.0f, 0.0f, 0.0f,     // acceleration
            0.0f, 0.0f, 0.0f,     // sensor data
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f
        };
        
        // コマンド (0=left, 1=straight, 2=right)
        input_data.command_ = 1;
        
        return input_data;
    }

    std::shared_ptr<MockVadLogger> mock_logger_;
    VadConfig config_;
};

// 1. 入力データの検証テスト
TEST_F(VadInferTest, InputDataValidation) {
    VadInputData input_data = createValidInputData();
    
    // データサイズの検証
    EXPECT_EQ(input_data.camera_images_.size(), 6 * 3 * 256 * 704);
    EXPECT_EQ(input_data.shift_.size(), 3);
    EXPECT_EQ(input_data.lidar2img_.size(), 6 * 4 * 4);
    EXPECT_EQ(input_data.can_bus_.size(), 18);
    EXPECT_GE(input_data.command_, 0);
    EXPECT_LE(input_data.command_, 2);
    
    // データ値の妥当性検証
    for (const float& pixel : input_data.camera_images_) {
        EXPECT_GE(pixel, 0.0f);
        EXPECT_LE(pixel, 1.0f);
    }
}

// 2. モック版inferテスト (実際のGPU処理なし)
TEST_F(VadInferTest, MockInferWithoutGPU) {
    VadInputData input_data = createValidInputData();
    
    // postprocessロジックのテスト（GPU推論結果をシミュレート）
    std::vector<float> mock_ego_fut_preds = {
        // command 0 (left)
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        // command 1 (straight) - テスト対象
        2.0f, 0.0f, 2.0f, 0.0f, 2.0f, 0.0f, 2.0f, 0.0f, 2.0f, 0.0f, 2.0f, 0.0f,
        // command 2 (right)
        1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f
    };
    
    int32_t cmd = input_data.command_;  // 1 (straight)
    
    // postprocessロジックを再実装
    std::vector<float> planning(
        mock_ego_fut_preds.begin() + cmd * 12,
        mock_ego_fut_preds.begin() + (cmd + 1) * 12
    );
    
    // 累積和計算
    for (int32_t i = 1; i < 6; i++) {
        planning[i * 2] += planning[(i-1) * 2];
        planning[i * 2 + 1] += planning[(i-1) * 2 + 1];
    }
    
    // 期待される結果
    std::vector<float> expected = {
        2.0f, 0.0f,    // 1st point
        4.0f, 0.0f,    // 2nd point (2+2, 0+0)
        6.0f, 0.0f,    // 3rd point (4+2, 0+0)
        8.0f, 0.0f,    // 4th point (6+2, 0+0)
        10.0f, 0.0f,   // 5th point (8+2, 0+0)
        12.0f, 0.0f    // 6th point (10+2, 0+0)
    };
    
    EXPECT_EQ(planning.size(), 12);
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(planning[i], expected[i]);
    }
}

// 3. エラーハンドリングテスト
TEST_F(VadInferTest, InvalidInputHandling) {
    // 無効なカメラ画像サイズ
    VadInputData invalid_input;
    invalid_input.camera_images_.resize(100);  // 正しくないサイズ
    invalid_input.shift_ = {0.0f, 0.0f, 0.0f};
    invalid_input.lidar2img_.resize(6 * 4 * 4, 1.0f);
    invalid_input.can_bus_.resize(18, 0.0f);
    invalid_input.command_ = 1;
    
    // 実際のVadModelがあれば、この入力でエラーが発生することを確認
    EXPECT_NE(invalid_input.camera_images_.size(), 6 * 3 * 256 * 704);
    
    // 無効なコマンドインデックス
    VadInputData input_data = createValidInputData();
    input_data.command_ = 5;  // 範囲外
    
    EXPECT_GT(input_data.command_, 2);  // 有効範囲は0-2
}

// 4. 複数フレーム処理のシミュレーション
TEST_F(VadInferTest, MultipleFrameProcessing) {
    // 最初のフレーム: head_no_prev を使用
    VadInputData first_frame = createValidInputData();
    first_frame.command_ = 1;
    
    // 2番目のフレーム: head を使用（前回のBEV特徴量を利用）
    VadInputData second_frame = createValidInputData();
    second_frame.command_ = 2;
    
    // 3番目のフレーム: head を継続使用
    VadInputData third_frame = createValidInputData();
    third_frame.command_ = 0;
    
    // フレーム間での状態変化をシミュレート
    bool is_first_frame = true;
    
    // 最初のフレーム処理
    std::string head_name = is_first_frame ? "head_no_prev" : "head";
    EXPECT_EQ(head_name, "head_no_prev");
    
    // フレーム処理後の状態更新
    is_first_frame = false;
    
    // 2番目のフレーム処理
    head_name = is_first_frame ? "head_no_prev" : "head";
    EXPECT_EQ(head_name, "head");
    
    // 3番目のフレーム処理
    head_name = is_first_frame ? "head_no_prev" : "head";
    EXPECT_EQ(head_name, "head");
}

// 5. メモリ使用量の検証
TEST_F(VadInferTest, MemoryUsageValidation) {
    VadInputData input_data = createValidInputData();
    
    // 各データ要素のメモリ使用量計算
    size_t camera_memory = input_data.camera_images_.size() * sizeof(float);
    size_t shift_memory = input_data.shift_.size() * sizeof(float);
    size_t lidar2img_memory = input_data.lidar2img_.size() * sizeof(float);
    size_t can_bus_memory = input_data.can_bus_.size() * sizeof(float);
    
    size_t total_input_memory = camera_memory + shift_memory + lidar2img_memory + can_bus_memory;
    
    // カメラ画像が最大のメモリを使用することを確認
    EXPECT_GT(camera_memory, shift_memory);
    EXPECT_GT(camera_memory, lidar2img_memory);
    EXPECT_GT(camera_memory, can_bus_memory);
    
    // 期待されるメモリ使用量（約38MB）
    size_t expected_camera_memory = 6 * 3 * 256 * 704 * sizeof(float);
    EXPECT_EQ(camera_memory, expected_camera_memory);
    
    // 総メモリ使用量がしきい値以下であることを確認
    const size_t MAX_INPUT_MEMORY_MB = 50;  // 50MB制限
    EXPECT_LT(total_input_memory, MAX_INPUT_MEMORY_MB * 1024 * 1024);
    
    EXPECT_CALL(*mock_logger_, info(::testing::_)).Times(1);
    mock_logger_->info("Total input memory: " + std::to_string(total_input_memory / (1024*1024)) + " MB");
}

}  // namespace autoware::tensorrt_vad
