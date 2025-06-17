#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>
#include <filesystem>
#include "mock_vad_logger.hpp"
#include "vad_model.hpp"

namespace autoware::tensorrt_vad
{

// 統合テスト用のフィクスチャ
class VadIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        mock_logger_ = std::make_shared<MockVadLogger>();

        // ユーザーの指示通り、demoのparam.yamlで使われている実際のモデルパスを使用して設定
        // これにより、ダミーファイルではなく、本物のエンジンとプラグインでテストを実行
        const std::string workspace_root = "/home/autoware/ghq/github.com/Shin-kyoto/DL4AGX/AV-Solutions/vad-trt/app";

        config_.plugins_path = workspace_root + "/demo/libplugins.so";
        config_.warm_up_num = 0; // テストなのでウォームアップは不要

        // 初期化に必要なネットワーク設定
        NetConfig backbone_config;
        backbone_config.name = "backbone";
        backbone_config.engine_file = "/home/autoware/ghq/github.com/Shin-kyoto/DL4AGX/AV-Solutions/vad-trt/app/demo/engines/vadv1.extract_img_feat.fp16.engine";
        backbone_config.use_graph = true;
        
        NetConfig head_no_prev_config;
        head_no_prev_config.name = "head_no_prev";
        head_no_prev_config.engine_file = "/home/autoware/ghq/github.com/Shin-kyoto/DL4AGX/AV-Solutions/vad-trt/app/demo/engines/vadv1.pts_bbox_head.forward.engine";
        head_no_prev_config.use_graph = true;
        // head_no_prevはbackboneの出力に依存するため、入力のマッピング情報を追加
        // このキー("img_feats")と値("out.img_feats")はモデルの実装に依存する
        head_no_prev_config.inputs["img_feats"] = {{"net", "backbone"}, {"name", "out.img_feats"}};

        // `head`は最初の`infer`呼び出し時に遅延ロードされるが、設定自体はコンストラクタに渡す必要がある
        NetConfig head_config;
        head_config.name = "head";
        head_config.engine_file = "/home/autoware/ghq/github.com/Shin-kyoto/DL4AGX/AV-Solutions/vad-trt/app/demo/engines/vadv1_prev.pts_bbox_head.forward.engine";
        head_config.use_graph = true;
        head_config.inputs["img_feats"] = {{"net", "backbone"}, {"name", "out.img_feats"}};

        config_.nets_config = {backbone_config, head_no_prev_config, head_config};
    }

    std::shared_ptr<MockVadLogger> mock_logger_;
    VadConfig config_;
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

}  // namespace autoware::tensorrt_vad
