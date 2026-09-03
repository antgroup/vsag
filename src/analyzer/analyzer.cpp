
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use hgraph_ file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "analyzer.h"

#include <chrono>
#include <nlohmann/json.hpp>

namespace vsag {

void
AddAnalysisMetadata(JsonType& stats,
                    const std::string& index_type,
                    const std::string& analysis_type,
                    const std::string& status,
                    uint64_t sample_count,
                    int64_t topk) {
    auto metadata = stats["_analysis"];
    for (const auto& [name, value] : stats.GetInnerJson()->items()) {
        if (name != "_analysis" && value.is_object() && value.contains("skipped_reason") &&
            value["skipped_reason"].is_string()) {
            metadata["skipped"][name].SetString(value["skipped_reason"].get<std::string>());
        }
    }
    const bool has_skipped =
        metadata.Contains("skipped") && not metadata["skipped"].GetInnerJson()->empty();
    const auto effective_status = status == "complete" && has_skipped ? "partial" : status;
    metadata["schema_version"].SetInt(1);
    metadata["index_type"].SetString(index_type);
    metadata["analysis_type"].SetString(analysis_type);
    metadata["status"].SetString(effective_status);
    metadata["consistency"].SetString("weak_snapshot");
    if (sample_count > 0) {
        metadata["sample_count"].SetUint64(sample_count);
    }
    if (topk > 0) {
        metadata["topk"].SetInt64(topk);
    }
}

BasicIndexAnalyzer::BasicIndexAnalyzer(const InnerIndexInterface* index, const AnalyzerParam& param)
    : AnalyzerBase(param.allocator, 0), index_(index) {
}

JsonType
BasicIndexAnalyzer::GetStats() {
    JsonType stats;
    std::string status = "complete";
    const auto live_count = index_->GetNumElements();
    stats["total_count"].SetInt64(live_count);
    stats["live_count"].SetInt64(live_count);
    try {
        const auto deleted_count = index_->GetNumberRemoved();
        stats["deleted_count"].SetInt64(deleted_count);
        stats["total_count"].SetInt64(live_count + deleted_count);
    } catch (const VsagException&) {
        status = "partial";
        stats["_analysis"]["skipped"]["deleted_count"].SetString(
            "index does not expose deleted count");
    }
    AddAnalysisMetadata(stats, index_->GetName(), "stats", status);
    return stats;
}

JsonType
BasicIndexAnalyzer::AnalyzeIndexBySearch(const SearchRequest& request) {
    CHECK_ARGUMENT(request.query_ != nullptr, "analysis query cannot be null");
    CHECK_ARGUMENT(request.topk_ > 0, "analysis topk must be greater than 0");
    JsonType stats;
    if (index_->GetNumElements() == 0) {
        AddAnalysisMetadata(stats,
                            index_->GetName(),
                            "search",
                            "not_applicable",
                            request.query_->GetNumElements(),
                            request.topk_);
        stats["_analysis"]["skipped"]["search"].SetString("index is empty");
        return stats;
    }
    const auto begin = std::chrono::steady_clock::now();
    DatasetPtr result;
    try {
        result = index_->SearchWithRequest(request);
    } catch (const VsagException& exception) {
        if (exception.error_.type != ErrorType::UNSUPPORTED_INDEX_OPERATION) {
            throw;
        }
        result =
            index_->KnnSearch(request.query_, request.topk_, request.params_str_, request.filter_);
    }
    const auto elapsed =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    stats["time_cost_query"].SetFloat(static_cast<float>(elapsed));
    stats["result_count"].SetInt64(result == nullptr ? 0 : result->GetDim());
    AddAnalysisMetadata(stats,
                        index_->GetName(),
                        "search",
                        "partial",
                        request.query_->GetNumElements(),
                        request.topk_);
    stats["_analysis"]["skipped"]["recall_query"].SetString("ground truth was not requested");
    return stats;
}

}  // namespace vsag
