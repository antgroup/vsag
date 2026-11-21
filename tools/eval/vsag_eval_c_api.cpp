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

#include <vsag/options.h>
#include <vsag/vsag_eval_c_api.h>

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "evaluator.h"
#include "nlohmann/json.hpp"
#include "vsag_c_api_internal.h"

using namespace vsag::c_api;

namespace {

constexpr uint64_t kMaxBuildThreads = 200;

vsag::eval::EvalConfig
parse_config(const char* config_json) {
    if (config_json == nullptr || config_json[0] == '\0') {
        return vsag::eval::EvalConfig::FromJson(nlohmann::json::object());
    }
    nlohmann::json json;
    try {
        json = nlohmann::json::parse(config_json);
    } catch (const nlohmann::json::parse_error& e) {
        throw std::invalid_argument(std::string("config_json is not valid JSON: ") + e.what());
    }
    return vsag::eval::EvalConfig::FromJson(json);
}

std::string
require_metric(const char* metric_type) {
    if (metric_type == nullptr) {
        throw std::invalid_argument("metric_type is NULL");
    }
    return metric_type;
}

char*
copy_string(const std::string& value) {
    auto* buffer = static_cast<char*>(std::malloc(value.size() + 1));
    if (buffer == nullptr) {
        throw std::bad_alloc();
    }
    std::memcpy(buffer, value.c_str(), value.size() + 1);
    return buffer;
}

// Restores the process-wide build thread count once the evaluation finished.
class ScopedBuildThreads {
public:
    explicit ScopedBuildThreads(int32_t num_threads)
        : original_(vsag::Options::Instance().num_threads_building()) {
        if (num_threads < 1 || static_cast<uint64_t>(num_threads) > kMaxBuildThreads) {
            throw std::invalid_argument("num_threads_building must be between 1 and 200");
        }
        vsag::Options::Instance().set_num_threads_building(static_cast<uint64_t>(num_threads));
    }

    ~ScopedBuildThreads() {
        vsag::Options::Instance().set_num_threads_building(original_);
    }

    ScopedBuildThreads(const ScopedBuildThreads&) = delete;
    ScopedBuildThreads&
    operator=(const ScopedBuildThreads&) = delete;

private:
    uint64_t original_;
};

}  // namespace

extern "C" {

void
vsag_string_destroy(char* str) {
    std::free(str);
}

vsag_error_t*
vsag_eval_set_num_threads_building(int32_t num_threads, int32_t* out_previous) {
    return guarded([&]() -> vsag_error_t* {
        const auto previous = vsag::Options::Instance().num_threads_building();
        if (num_threads < 1 || static_cast<uint64_t>(num_threads) > kMaxBuildThreads) {
            throw std::invalid_argument("num_threads_building must be between 1 and 200");
        }
        vsag::Options::Instance().set_num_threads_building(static_cast<uint64_t>(num_threads));
        if (out_previous != nullptr) {
            *out_previous = static_cast<int32_t>(previous);
        }
        return nullptr;
    });
}

vsag_error_t*
vsag_eval_build(vsag_index_t* index,
                const vsag_dataset_t* base,
                const char* metric_type,
                const char* config_json,
                char** out_result_json) {
    return guarded([&]() -> vsag_error_t* {
        VSAG_REQUIRE(index != nullptr, "index is NULL");
        VSAG_REQUIRE(out_result_json != nullptr, "out_result_json is NULL");
        *out_result_json = nullptr;
        auto config = parse_config(config_json);
        auto base_view = make_view(base, false);
        auto dataset = vsag::eval::EvalDataset::FromDatasets(
            base_view.dataset, base_view.dataset, nullptr, require_metric(metric_type));
        ScopedBuildThreads build_threads(config.num_threads_building);
        auto result = vsag::eval::EvaluateBuild(index->index, dataset, config);
        *out_result_json = copy_string(result.dump());
        return nullptr;
    });
}

vsag_error_t*
vsag_eval_search(vsag_index_t* index,
                 const vsag_dataset_t* base,
                 const vsag_dataset_t* queries,
                 const int64_t* gt_ids,
                 int64_t gt_k,
                 const char* metric_type,
                 const char* config_json,
                 char** out_result_json) {
    return guarded([&]() -> vsag_error_t* {
        VSAG_REQUIRE(index != nullptr, "index is NULL");
        VSAG_REQUIRE(out_result_json != nullptr, "out_result_json is NULL");
        VSAG_REQUIRE(gt_ids == nullptr || gt_k > 0, "gt_k must be positive when gt_ids is set");
        *out_result_json = nullptr;
        auto config = parse_config(config_json);
        auto query_view = make_view(queries, false);

        vsag::DatasetPtr ground_truth;
        if (gt_ids != nullptr) {
            ground_truth = vsag::Dataset::Make();
            ground_truth->NumElements(query_view.dataset->GetNumElements())
                ->Dim(gt_k)
                ->Ids(gt_ids)
                ->Owner(false);
        }

        vsag::eval::EvalDatasetPtr dataset;
        if (base != nullptr) {
            auto base_view = make_view(base, false);
            dataset = vsag::eval::EvalDataset::FromDatasets(
                base_view.dataset, query_view.dataset, ground_truth, require_metric(metric_type));
        } else {
            dataset = vsag::eval::EvalDataset::FromSearchDatasets(query_view.dataset, ground_truth);
        }
        auto result = vsag::eval::EvaluateSearch(index->index, dataset, config);
        *out_result_json = copy_string(result.dump());
        return nullptr;
    });
}

}  // extern "C"
