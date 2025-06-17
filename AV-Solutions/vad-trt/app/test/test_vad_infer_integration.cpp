#include <gtest/gtest.h>
#include <memory>
#include <filesystem>
#include <fstream>
#include <random>
#include <chrono>
#include <dlfcn.h>  // dlopen, dlerror
#include "mock_vad_logger.hpp"
#include "../lib/vad_model.hpp"

// GPU/TensorRTエンジンファイルが利用可能な場合の統合テスト
namespace autoware::tensorrt_vad
{

class VadInferIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        logger_ = std::make_shared<MockVadLogger>();
        
        // 正しいワークスペースルートを設定
        const std::string workspace_root = "/home/autoware/ghq/github.com/Shin-kyoto/DL4AGX/AV-Solutions/vad-trt/app";

        // 環境変数からパスを取得、なければ正しいデフォルトパスを使用
        const char* engine_dir_env = std::getenv("VAD_ENGINE_DIR");
        const char* plugin_path_env = std::getenv("VAD_PLUGIN_PATH");
        
        engine_dir_ = engine_dir_env ? engine_dir_env : workspace_root + "/demo/engines";
        plugin_path_ = plugin_path_env ? plugin_path_env : workspace_root + "/demo/libplugins.so";
        
        // 前提条件のチェック
        bool engines_exist = 
            std::filesystem::exists(engine_dir_ + "/vadv1.extract_img_feat.fp16.engine") &&
            std::filesystem::exists(engine_dir_ + "/vadv1_prev.pts_bbox_head.forward.engine") &&
            std::filesystem::exists(engine_dir_ + "/vadv1.pts_bbox_head.forward.engine");
        bool plugin_exists = std::filesystem::exists(plugin_path_);
        
        int device_count = 0;
        cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
        bool cuda_available = (cuda_status == cudaSuccess && device_count > 0);
        
        integration_test_enabled_ = engines_exist && plugin_exists && cuda_available;

        if (!integration_test_enabled_) {
            std::string reason = "Integration test requirements not met: ";
            if (!engines_exist) reason += "Engine files not found in " + engine_dir_ + ". ";
            if (!plugin_exists) reason += "Plugin not found at " + plugin_path_ + ". ";
            if (!cuda_available) reason += "No CUDA GPU available.";
            GTEST_SKIP() << reason;
        }
    }

    VadConfig createRealConfig() {
        VadConfig config;
        config.plugins_path = plugin_path_;
        config.warm_up_num = 3;
        
        NetConfig backbone_config;
        backbone_config.name = "backbone";
        backbone_config.engine_file = engine_dir_ + "/vadv1.extract_img_feat.fp16.engine";
        backbone_config.use_graph = true;
        
        NetConfig head_no_prev_config;
        head_no_prev_config.name = "head_no_prev";
        head_no_prev_config.engine_file = engine_dir_ + "/vadv1.pts_bbox_head.forward.engine";
        head_no_prev_config.use_graph = false;
        head_no_prev_config.inputs["img_feats"] = {{"net", "backbone"}, {"name", "out.img_feats"}};
        
        NetConfig head_config;
        head_config.name = "head";
        head_config.engine_file = engine_dir_ + "/vadv1_prev.pts_bbox_head.forward.engine";
        head_config.use_graph = false;
        head_config.inputs["img_feats"] = {{"net", "backbone"}, {"name", "out.img_feats"}};
        
        config.nets_config = {backbone_config, head_no_prev_config, head_config};
        return config;
    }

    VadInputData createRealisticInputData() {
        VadInputData input_data;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);

        input_data.camera_images_.resize(6 * 3 * 256 * 704);
        for (auto& pixel : input_data.camera_images_) {
            pixel = dis(gen);
        }
        
        input_data.shift_ = {0.5f, 0.1f, 0.02f};
        
        input_data.lidar2img_.resize(6 * 16);
        for (size_t cam = 0; cam < 6; ++cam) {
            float* matrix = &input_data.lidar2img_[cam * 16];
            matrix[0] = 1000.0f; matrix[1] = 0.0f;    matrix[2] = 352.0f;  matrix[3] = 0.0f;
            matrix[4] = 0.0f;    matrix[5] = 1000.0f; matrix[6] = 128.0f;  matrix[7] = 0.0f;
            matrix[8] = 0.0f;    matrix[9] = 0.0f;    matrix[10] = 1.0f;   matrix[11] = 0.0f;
            matrix[12] = 0.0f;   matrix[13] = 0.0f;   matrix[14] = 0.0f;   matrix[15] = 1.0f;
        }
        
        input_data.can_bus_ = {
            15.5f, 0.2f, -0.1f, 0.0f, 0.0f, 0.05f, 1.2f, -0.3f, 0.0f,
            0.8f, 0.1f, 0.0f, 1.0f, 0.0f, 0.0f, 25.5f, 60.2f, 0.0f
        };
        
        input_data.command_ = 1;
        return input_data;
    }

    std::shared_ptr<MockVadLogger> logger_;
    std::string engine_dir_;
    std::string plugin_path_;
    bool integration_test_enabled_ = false;
};

// 1. モデルが例外を投げずに初期化できることを確認
TEST_F(VadInferIntegrationTest, ModelInitialization) {
    VadConfig config = createRealConfig();
    std::unique_ptr<VadModel<MockVadLogger>> model;
    
    ASSERT_NO_THROW({
        model = std::make_unique<VadModel<MockVadLogger>>(config, logger_);
    }) << "Model initialization failed. Check paths and permissions.";
    
}

// 2. 実際のinfer実行テスト
TEST_F(VadInferIntegrationTest, RealInferExecution) {
    auto model = std::make_unique<VadModel<MockVadLogger>>(createRealConfig(), logger_);
    VadInputData input_data = createRealisticInputData();
    
    // First frame
    auto first_result = model->infer(input_data);
    ASSERT_TRUE(first_result.has_value()) << "First inference failed to return a result.";
    EXPECT_EQ(first_result->predicted_trajectory_.size(), 12);

    // Second frame
    input_data.command_ = 2; // Change command
    auto second_result = model->infer(input_data);
    ASSERT_TRUE(second_result.has_value()) << "Second inference failed to return a result.";
    EXPECT_EQ(second_result->predicted_trajectory_.size(), 12);
    
    // Check if trajectories are different for different commands
    EXPECT_NE(first_result->predicted_trajectory_, second_result->predicted_trajectory_)
        << "Different commands should produce different trajectories.";
}


}  // namespace autoware::tensorrt_vad