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

#include "simq_analyzer.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "typing.h"

namespace vsag {
namespace {

float
calculate_percentile(const std::vector<float>& sorted_values, float percentile) {
    if (sorted_values.empty()) {
        return 0.0F;
    }
    float index = percentile * static_cast<float>(sorted_values.size() - 1);
    auto lower = static_cast<size_t>(std::floor(index));
    auto upper = static_cast<size_t>(std::ceil(index));
    if (lower == upper) {
        return sorted_values[lower];
    }
    float weight = index - static_cast<float>(lower);
    return sorted_values[lower] * (1.0F - weight) + sorted_values[upper] * weight;
}

JsonType
make_doc_vector_count_stats(std::vector<float> values) {
    JsonType json;
    if (values.empty()) {
        json["mean"].SetFloat(0.0F);
        json["p90"].SetFloat(0.0F);
        json["p99"].SetFloat(0.0F);
        json["min"].SetInt(0);
        json["max"].SetInt(0);
        return json;
    }

    std::sort(values.begin(), values.end());
    float sum = std::accumulate(values.begin(), values.end(), 0.0F);
    json["mean"].SetFloat(sum / static_cast<float>(values.size()));
    json["p90"].SetFloat(calculate_percentile(values, 0.90F));
    json["p99"].SetFloat(calculate_percentile(values, 0.99F));
    json["min"].SetInt(static_cast<int64_t>(values.front()));
    json["max"].SetInt(static_cast<int64_t>(values.back()));
    return json;
}

}  // namespace

JsonType
SIMQAnalyzer::GetStats() {
    JsonType stats;
    uint64_t total_count = static_cast<uint64_t>(simq_->GetNumElements());
    stats["total_count"].SetUint64(total_count);

    uint64_t total_token_count = simq_->token_to_doc_.size();
    stats["total_token_count"].SetUint64(total_token_count);

    stats["doc_vector_count_distribution"].SetJson(get_doc_vector_count_distribution());
    return stats;
}

JsonType
SIMQAnalyzer::AnalyzeIndexBySearch(const SearchRequest& /*request*/) {
    JsonType stats;
    return stats;
}

JsonType
SIMQAnalyzer::get_doc_vector_count_distribution() const {
    const auto& token_to_doc = simq_->token_to_doc_;
    uint64_t total_count = simq_->total_count_.load();

    if (token_to_doc.empty() || total_count == 0) {
        return make_doc_vector_count_stats({});
    }

    // Count tokens per doc: the last doc in token_to_doc_ has the max inner_id.
    // Use unordered_map so we don't need to pre-scan for max inner_id, though
    // we know total_count_ bounds it.  For sparse/mixed token distributions
    // after splits, total_count_ is the authoritative doc count.
    std::unordered_map<InnerIdType, uint32_t> doc_token_counts;
    for (InnerIdType doc_id : token_to_doc) {
        ++doc_token_counts[doc_id];
    }

    std::vector<float> values;
    values.reserve(total_count);
    // Iterate over all possible doc IDs to include docs with 0 tokens (should
    // not happen in practice but ensures total_count consistency).
    for (uint64_t i = 0; i < total_count; ++i) {
        auto it = doc_token_counts.find(static_cast<InnerIdType>(i));
        values.push_back(static_cast<float>(it != doc_token_counts.end() ? it->second : 0));
    }

    return make_doc_vector_count_stats(std::move(values));
}

}  // namespace vsag