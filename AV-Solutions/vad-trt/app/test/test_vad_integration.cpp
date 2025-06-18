#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <dlfcn.h>
#include <yaml-cpp/yaml.h>
#include "mock_vad_logger.hpp"
#include "vad_model.hpp"

namespace autoware::tensorrt_vad
{

// テスト用の設定構造体
struct TestConfig {
    struct {
        struct {
            std::string src_path;
            std::string dst_path;
        } bev_embed;
        struct {
            std::string path;
        } camera_images;
    } test_data;
};

// 設定ファイルからVadConfigとTestConfigを読み込むヘルパー関数
std::pair<VadConfig, TestConfig> loadConfigFromYaml(const std::string& config_path) {
    VadConfig vad_config;
    TestConfig test_config;
    
    try {
        YAML::Node yaml_config = YAML::LoadFile(config_path);
        const auto& test_config_node = yaml_config["test_config"];
        
        vad_config.plugins_path = test_config_node["plugins_path"].as<std::string>();
        vad_config.warm_up_num = test_config_node["warm_up_num"].as<int>();
        
        const auto& nets = test_config_node["nets"];
        for (const auto& net : nets) {
            NetConfig net_config;
            net_config.name = net.second["name"].as<std::string>();
            net_config.engine_file = net.second["engine_file"].as<std::string>();
            net_config.use_graph = net.second["use_graph"].as<bool>();
            
            if (net.second["inputs"]) {
                const auto& inputs = net.second["inputs"];
                for (const auto& input : inputs) {
                    std::map<std::string, std::string> input_map;
                    for (const auto& param : input.second) {
                        input_map[param.first.as<std::string>()] = param.second.as<std::string>();
                    }
                    net_config.inputs[input.first.as<std::string>()] = input_map;
                }
            }
            vad_config.nets_config.push_back(net_config);
        }

        // テストデータの設定を読み込む
        const auto& test_data = test_config_node["test_data"];
        test_config.test_data.bev_embed.src_path = test_data["bev_embed"]["src_path"].as<std::string>();
        test_config.test_data.bev_embed.dst_path = test_data["bev_embed"]["dst_path"].as<std::string>();
        test_config.test_data.camera_images.path = test_data["camera_images"]["path"].as<std::string>();

    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Failed to load config from YAML: " + std::string(e.what()));
    }
    
    return {vad_config, test_config};
}

// 統合テスト用のフィクスチャ
class VadIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        mock_logger_ = std::make_shared<MockVadLogger>();
        auto [vad_config, test_config] = loadConfigFromYaml("../test_config.yaml");
        config_ = vad_config;
        test_config_ = test_config;
    }

    std::shared_ptr<MockVadLogger> mock_logger_;
    VadConfig config_;
    TestConfig test_config_;
};

// VadModelの初期化が、実際のエンジンファイルを用いて成功することを確認するテスト
TEST_F(VadIntegrationTest, ModelInitializationWithRealEngines)
{
    // エラーログが呼ばれないことを期待
    EXPECT_CALL(*mock_logger_, error(testing::_)).Times(0);
    // infoログは何回か呼ばれるはず
    EXPECT_CALL(*mock_logger_, info(testing::_)).Times(testing::AtLeast(1));

    // VadModelのコンストラクタが例外を投げずに完了すればテスト成功
    // これにより、プラグインのロード、エンジンのデシリアライズ、CUDAコンテキストの初期化が
    // 正常に行われることを検証する
    std::unique_ptr<VadModel<MockVadLogger>> model;
    ASSERT_NO_THROW({
        model = std::make_unique<VadModel<MockVadLogger>>(config_, mock_logger_);
    }) << "VadModel initialization failed with real engine files. "
       << "Check if paths are correct and files are not corrupted.";
    
    ASSERT_TRUE(model->initialized_) << "Model should be marked as initialized.";
}

class VadInferIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        logger_ = std::make_shared<MockVadLogger>();
        auto [vad_config, test_config] = loadConfigFromYaml("../test_config.yaml");
        config_ = vad_config;
        test_config_ = test_config;
        
        // 前提条件のチェック
        bool engines_exist = 
            std::filesystem::exists(config_.nets_config[0].engine_file) &&
            std::filesystem::exists(config_.nets_config[1].engine_file) &&
            std::filesystem::exists(config_.nets_config[2].engine_file);
        bool plugin_exists = std::filesystem::exists(config_.plugins_path);
        
        int device_count = 0;
        cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
        bool cuda_available = (cuda_status == cudaSuccess && device_count > 0);
        
        integration_test_enabled_ = engines_exist && plugin_exists && cuda_available;

        if (!integration_test_enabled_) {
            GTEST_SKIP() << "Integration test requirements not met.";
        }

        // bev_embedはビルドディレクトリからコピーする
        const std::string src_bev_path = test_config_.test_data.bev_embed.src_path;
        const std::string dst_bev_path = test_config_.test_data.bev_embed.dst_path;
        if (std::filesystem::exists(src_bev_path)) {
            std::filesystem::copy(src_bev_path, dst_bev_path, std::filesystem::copy_options::overwrite_existing);
        } else {
            GTEST_SKIP() << "bev_embed_frame1.bin not found. Run vad_app first.";
        }
    }

    VadConfig createRealConfig() {
        VadConfig config;
        config.plugins_path = config_.plugins_path;
        config.warm_up_num = config_.warm_up_num;
        
        NetConfig backbone_config;
        backbone_config.name = config_.nets_config[0].name;
        backbone_config.engine_file = config_.nets_config[0].engine_file;
        backbone_config.use_graph = config_.nets_config[0].use_graph;
        
        NetConfig head_no_prev_config;
        head_no_prev_config.name = config_.nets_config[1].name;
        head_no_prev_config.engine_file = config_.nets_config[1].engine_file;
        head_no_prev_config.use_graph = config_.nets_config[1].use_graph;
        head_no_prev_config.inputs["img_feats"] = config_.nets_config[1].inputs["img_feats"];
        
        NetConfig head_config;
        head_config.name = config_.nets_config[2].name;
        head_config.engine_file = config_.nets_config[2].engine_file;
        head_config.use_graph = config_.nets_config[2].use_graph;
        head_config.inputs["img_feats"] = config_.nets_config[2].inputs["img_feats"];
        
        config.nets_config = {backbone_config, head_no_prev_config, head_config};
        return config;
    }

    std::vector<float> loadBevEmbedFromFile(const std::string& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open bev_embed file: " + path);
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::vector<float> data(size / sizeof(float));
        if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
            throw std::runtime_error("Failed to read bev_embed data from file: " + path);
        }
        return data;
    }

    VadInputData createFrame2InputData() {
        VadInputData input_data;
        
        std::ifstream file(test_config_.test_data.camera_images.path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open image data file: " + test_config_.test_data.camera_images.path);
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        size_t expected_elements = 6 * 3 * 384 * 640;
        size_t expected_size_bytes = expected_elements * sizeof(float);
        
        if (static_cast<size_t>(size) != expected_size_bytes) {
            throw std::runtime_error("Image data file has incorrect size. Expected: " + 
                                     std::to_string(expected_size_bytes) + ", Got: " + std::to_string(size));
        }
        
        input_data.camera_images_.resize(expected_elements);
        if (!file.read(reinterpret_cast<char*>(input_data.camera_images_.data()), size)) {
            throw std::runtime_error("Failed to read image data from file: " + test_config_.test_data.camera_images.path);
        }
        
        input_data.shift_ = {0.00224774843081832f, -0.00130402739159763f, 0.0f};
        
        input_data.lidar2img_ = {
            379.27642822f,  388.46176147f,   17.67811966f,    0.00000000f,
              -6.15190125f,  236.73570251f, -371.38888550f,    0.00000000f,
            -0.0121784266084433f, 0.998439431190491f, 0.0545013546943665f, 0.0f,
            9.45881329244003e-05f, -0.354513853788376f, -0.436208814382553f, 1.0f,
            546.047180175781f, -247.578689575195f, -15.7638425827026f, 0.0f,
            152.198425292969f, 128.385726928711f, -495.723114013672f, 0.0f,
            0.843259930610657f, 0.536481976509094f, 0.0331593155860901f, 0.0f,
            0.038077961653471f, -0.331093370914459f, -0.613056242465973f, 1.0f,
            12.5099611282349f, 601.254028320312f, 31.375207901001f, 0.0f,
            -155.273300170898f, 128.329086303711f, -495.084533691406f, 0.0f,
            -0.823839902877808f, 0.565365552902222f, 0.0406133532524109f, 0.0f,
            0.0988269224762917f, -0.349612444639206f, -0.539402663707733f, 1.0f,
            -321.558685302734f, -340.316497802734f, -10.744460105896f, 0.0f,
            -4.18298482894897f, -178.075927734375f, -325.981872558594f, 0.0f,
            -0.00822998490184546f, -0.999196469783783f, -0.0392249822616577f, 0.0f,
            0.000492329010739923f, -0.280932426452637f, -1.01566863059998f, 1.0f,
            -474.615386962891f, 369.318817138672f, 21.3022727966309f, 0.0f,
            -185.020065307617f, -40.9742012023926f, -501.005340576172f, 0.0f,
            -0.947597444057465f, -0.319451540708542f, 0.00308722257614136f, 0.0f,
            -0.208745241165161f, -0.286693632602692f, -0.433616310358047f, 1.0f,
            114.180130004883f, -587.687866210938f, -23.8854846954346f, 0.0f,
            178.267059326172f, -48.9818115234375f, -500.038909912109f, 0.0f,
            0.92411196231842f, -0.382108986377716f, -0.0031280517578125f, 0.0f,
            0.09475177526474f, -0.29829341173172f, -0.460391998291016f, 1.0f
        };
        
        input_data.can_bus_ = {
            3.6282057762146f, -2.17206835746765f, 0.0f,
            -0.962751805782318f, -0.00739681720733643f, -0.00615653395652771f,
            0.270215272903442f, -0.354863733053207f, -0.417147159576416f,
            9.74510097503662f, 0.0086875855922699f, 0.0174264013767242f,
            -0.0592677667737007f, 8.23505878448486f, 0.0f, 0.0f,
            0.0109170079231262f, 0.474326103925705f
        };
        
        input_data.command_ = 2; // Frame 2 was command 2
        return input_data;
    }

    std::shared_ptr<MockVadLogger> logger_;
    VadConfig config_;
    TestConfig test_config_;
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
    
    auto prev_bev_data = loadBevEmbedFromFile("bev_embed_frame1.bin");
    
    auto dummy_input = createFrame2InputData();
    model->infer(dummy_input); 
    
    model->is_first_frame_ = false;

    VadInputData input_data_frame2 = createFrame2InputData();

    auto result = model->infer(input_data_frame2);
    ASSERT_TRUE(result.has_value()) << "Inference failed to return a result.";
    EXPECT_EQ(result->predicted_trajectory_.size(), 12);
    
    const std::vector<float> expected_trajectory = {
        0.04284074902534485f, 4.304573059082031f, 0.1053553968667984f, 8.743409156799316f,
        0.18814155459403992f, 13.293708801269531f, 0.28254231810569763f, 17.906057357788086f,
        0.38268977403640747f, 22.609987258911133f, 0.49197587370872498f, 27.344097137451172f
    };

    ASSERT_EQ(result->predicted_trajectory_.size(), expected_trajectory.size());
    for (size_t i = 0; i < expected_trajectory.size(); ++i) {
        EXPECT_NEAR(result->predicted_trajectory_[i], expected_trajectory[i], 1e-5)
            << "Mismatch at index " << i;
    }
}

}  // namespace autoware::tensorrt_vad
