
// Copyright 2024-present the vsag project
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

#include "./build_eval_case.h"

#include <algorithm>
#include <filesystem>
#include <utility>

#include "../monitor/duration_monitor.h"
#include "../monitor/memory_peak_monitor.h"
#include "vsag_exception.h"

namespace vsag::eval {

BuildEvalCase::BuildEvalCase(const std::string& dataset_path,
                             const std::string& index_path,
                             vsag::IndexPtr index,
                             EvalConfig config)
    : EvalCase(dataset_path, index_path, index), config_(std::move(config)) {
    this->init_monitors();
}

void
BuildEvalCase::init_monitors() {
    if (config_.enable_memory) {
        auto memory_peak_monitor = std::make_shared<MemoryPeakMonitor>("build");
        this->monitors_.emplace_back(std::move(memory_peak_monitor));
    }
    if (config_.enable_tps) {
        auto duration_monitor = std::make_shared<DurationMonitor>();
        this->monitors_.emplace_back(std::move(duration_monitor));
    }
}

JsonType
BuildEvalCase::Run() {
    this->do_build();
    this->serialize();
    auto result = this->process_result();
    return result;
}
void
BuildEvalCase::do_build() {
    auto base = vsag::Dataset::Make();
    int64_t total_base = this->dataset_ptr_->GetNumberOfBase();
    std::vector<int64_t> ids(total_base);
    std::iota(ids.begin(), ids.end(), 0);
    base->NumElements(total_base)->Dim(this->dataset_ptr_->GetDim())->Ids(ids.data())->Owner(false);
    if (this->dataset_ptr_->GetVectorType() == DENSE_VECTORS) {
        if (this->dataset_ptr_->GetTrainDataType() == vsag::DATATYPE_FLOAT32) {
            base->Float32Vectors((const float*)this->dataset_ptr_->GetTrain());
        } else if (this->dataset_ptr_->GetTrainDataType() == vsag::DATATYPE_INT8) {
            base->Int8Vectors((const int8_t*)this->dataset_ptr_->GetTrain());
        }
    } else {
        base->SparseVectors((const SparseVector*)this->dataset_ptr_->GetTrain());
    }
    for (auto& monitor : monitors_) {
        monitor->Start();
    }
    auto build_index = index_->Build(base);
    if (not build_index.has_value()) {
        throw std::runtime_error(build_index.error().message);
    }
    for (auto& monitor : monitors_) {
        monitor->Record();
        monitor->Stop();
    }
}
void
BuildEvalCase::serialize() {
    std::filesystem::path dir_path(index_path_);
    dir_path = dir_path.parent_path();
    if (!std::filesystem::exists(dir_path)) {
        std::filesystem::create_directories(dir_path);
    }
    std::ofstream outfile(this->index_path_, std::ios::binary);
    this->index_->Serialize(outfile);
}

JsonType
BuildEvalCase::process_result() {
    JsonType result;
    JsonType eval_result;
    for (auto& monitor : this->monitors_) {
        const auto& one_result = monitor->GetResult();
        EvalCase::MergeJsonType(one_result, eval_result);
    }
    result = eval_result;
    result["tps"] = double(this->dataset_ptr_->GetNumberOfBase()) / double(result["duration(s)"]);
    EvalCase::MergeJsonType(this->basic_info_, result);
    result["index_info"] = JsonType::parse(config_.build_param);
    result["action"] = "build";
    result["index"] = config_.index_name;
    // TODO(deming): remove try-catch after implement GetMemoryUsageDetail
    try {
        result["memory_detail(B)"] = this->index_->GetMemoryUsageDetail();
    } catch (std::runtime_error& e) {
        // if GetMemoryUsageDetail not implemented
        logger_->Error(e.what());
    } catch (vsag::VsagException& e) {
        logger_->Error(e.what());
    }
    return result;
}

}  // namespace vsag::eval
