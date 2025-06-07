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

#include "vad_model.hpp"

#include <chrono>
#include <optional>
#include <string>
#include <iostream>
#include <dlfcn.h>
#include <NvInferRuntime.h>

namespace autoware::tensorrt_vad
{

// Loggerクラス（VadModel内で使用）
class Logger : public nvinfer1::ILogger {
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        // Only print error messages
        if (severity == nvinfer1::ILogger::Severity::kERROR) {
            std::cerr << msg << std::endl;
        }
    }
};

// VadModelクラスの実装
VadModel::VadModel(
    const VadConfig& config)
    : initialized_(false), stream_(nullptr), is_first_frame_(true), config_(config)
{
    // 初期化を実行
    runtime_ = create_runtime();
    
    if (!load_plugin(config.plugins_path)) {
        std::cerr << "Failed to load plugin" << std::endl;
        return;
    }
    
    cudaStreamCreate(&stream_);
    
    nets_ = init_engines(config.nets_config);
    
    printf("[INFO] warm_up=%d\n", config.warm_up_num);
    warm_up(config.warm_up_num);
    
    initialized_ = true;
}

VadModel::~VadModel()
{
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    
    // netsのクリーンアップ
    nets_.clear();
    
    initialized_ = false;
}

std::unique_ptr<nvinfer1::IRuntime, std::function<void(nvinfer1::IRuntime*)>> VadModel::create_runtime() {
    static Logger logger;  // staticで宣言してライフタイムを保証
    auto runtime_deleter = [](nvinfer1::IRuntime *runtime) {};
    std::unique_ptr<nvinfer1::IRuntime, decltype(runtime_deleter)> runtime{
        nvinfer1::createInferRuntime(logger), runtime_deleter};
    return runtime;
}

bool VadModel::load_plugin(const std::string& plugin_dir) {
    void* h_ = dlopen(plugin_dir.c_str(), RTLD_NOW);
    printf("[INFO] loading plugin from: %s\n", plugin_dir.c_str());
    if (!h_) {
        const char* error = dlerror();
        std::cerr << "Failed to load library: " << error << std::endl;
        return false;
    }
    return true;
}

std::unordered_map<std::string, std::shared_ptr<nv::Net>> VadModel::init_engines(
    const std::vector<NetConfig>& nets_config) {
    
    std::unordered_map<std::string, std::shared_ptr<nv::Net>> nets;
    
    for (const auto& engine : nets_config) {
        if (engine.name == "head") {
            continue;  // headは後で初期化
        }
        
        std::string eng_name = engine.name;
        std::string eng_file = engine.engine_file;
        printf("-> engine: %s\n", eng_name.c_str());
        
        std::unordered_map<std::string, std::shared_ptr<nv::Tensor>> external_bindings;
        // reuse memory
        for (const auto& input_pair : engine.inputs) {
            const std::string& k = input_pair.first;
            const auto& ext_map = input_pair.second;      
            std::string ext_net = ext_map.at("net");
            std::string ext_name = ext_map.at("name");
            printf("%s <- %s[%s]\n", k.c_str(), ext_net.c_str(), ext_name.c_str());
            external_bindings[k] = nets[ext_net]->bindings[ext_name];
        }

        nets[eng_name] = std::make_shared<nv::Net>(eng_file, runtime_.get(), external_bindings);

        if (engine.use_graph) {
            nets[eng_name]->EnableCudaGraph(stream_);
        }
    }
    
    return nets;
}

void VadModel::warm_up(int32_t warm_up_num) {
    for(int32_t iw=0; iw < warm_up_num; iw++) {
        nets_["backbone"]->Enqueue(stream_);
        nets_["head_no_prev"]->Enqueue(stream_);
        cudaStreamSynchronize(stream_);
    }
}

std::optional<VadOutputData> VadModel::infer(const VadInputData & vad_input) {
    // 最初のフレームかどうかでheadの名前を変更
    std::string head_name;
    if (is_first_frame_) {
        head_name = "head_no_prev";
    } else {
        head_name = "head";
    }

    // bindingsにload
    load_inputs(vad_input, head_name);

    // backboneとheadをenqueue
    enqueue(head_name);

    // prev_bevを保存
    saved_prev_bev_ = save_prev_bev(head_name);
    // VadOutputDataに出力を変換
    VadOutputData output = postprocess(head_name, vad_input.command_);

    // 最初のフレームなら以下の処理を行う
    if (is_first_frame_) {
        release_network("head_no_prev");
        load_head();
        is_first_frame_ = false;
    }
    
    return output;
}

void VadModel::load_inputs(const VadInputData& vad_input, const std::string& head_name) {
    nets_["backbone"]->bindings["img"]->load(vad_input.camera_images_, stream_);
    nets_[head_name]->bindings["img_metas.0[shift]"]->load(vad_input.shift_, stream_);
    nets_[head_name]->bindings["img_metas.0[lidar2img]"]->load(vad_input.lidar2img_, stream_);
    nets_[head_name]->bindings["img_metas.0[can_bus]"]->load(vad_input.can_bus_, stream_);

    if (head_name == "head") {
        nets_["head"]->bindings["prev_bev"] = saved_prev_bev_;
    }
}

void VadModel::enqueue(const std::string& head_name) {
    nets_["backbone"]->Enqueue(stream_);
    nets_[head_name]->Enqueue(stream_);
}

std::shared_ptr<nv::Tensor> VadModel::save_prev_bev(const std::string& head_name) {
    auto bev_embed = nets_[head_name]->bindings["out.bev_embed"];
    auto prev_bev = std::make_shared<nv::Tensor>("prev_bev", bev_embed->dim, bev_embed->dtype);
    cudaMemcpyAsync(prev_bev->ptr, bev_embed->ptr, bev_embed->nbytes(), 
                    cudaMemcpyDeviceToDevice, stream_);
    return prev_bev;
}

void VadModel::release_network(const std::string& network_name) {
    if (nets_.find(network_name) != nets_.end()) {
        // まずbindingsをクリア
        nets_[network_name]->bindings.clear();
        cudaDeviceSynchronize();
        
        // 次にNetオブジェクトを解放
        nets_[network_name].reset();
        nets_.erase(network_name);
        cudaDeviceSynchronize();
    }
}

/**
 * @brief headネットワークを初期化し、backboneの出力テンソルを再利用する
 * 
 * external_bindingsを使用する理由:
 * - backboneのoutputをheadのinputに使いたい
 * - backboneのoutputのtensorのアドレスを、headのinputのptrに渡す
 * 
 * backbone.output → head.input (ptr共有)
 */
void VadModel::load_head() {
    auto head_engine = std::find_if(config_.nets_config.begin(), config_.nets_config.end(),
        [](const NetConfig& engine) { return engine.name == "head"; });
    
    if (head_engine == config_.nets_config.end()) {
        std::cerr << "Head engine configuration not found" << std::endl;
        return;
    }
    
    std::string eng_file = head_engine->engine_file;
    printf("-> loading head engine: %s\n", eng_file.c_str());
    
    std::unordered_map<std::string, std::shared_ptr<nv::Tensor>> external_bindings;
    for (const auto& input_pair : head_engine->inputs) {
        const std::string& k = input_pair.first;
        const auto& ext_map = input_pair.second;      
        std::string ext_net = ext_map.at("net");
        std::string ext_name = ext_map.at("name");
        printf("%s <- %s[%s]\n", k.c_str(), ext_net.c_str(), ext_name.c_str());
        external_bindings[k] = nets_[ext_net]->bindings[ext_name];
    }

    nets_["head"] = std::make_shared<nv::Net>(eng_file, runtime_.get(), external_bindings);

    if (head_engine->use_graph) {
        nets_["head"]->EnableCudaGraph(stream_);
    }
}

VadOutputData VadModel::postprocess(const std::string& head_name, int32_t cmd) {
    std::vector<float> ego_fut_preds = nets_[head_name]->bindings["out.ego_fut_preds"]->cpu<float>();
    
    // Extract planning for the given command
    std::vector<float> planning(
        ego_fut_preds.begin() + cmd * 12,
        ego_fut_preds.begin() + (cmd + 1) * 12
    );
    
    // cumsum to build trajectory in 3d space
    for (int32_t i = 1; i < 6; i++) {
        planning[i * 2] += planning[(i-1) * 2];
        planning[i * 2 + 1] += planning[(i-1) * 2 + 1];
    }
    
    return VadOutputData{planning};
}
}  // namespace autoware::tensorrt_vad
