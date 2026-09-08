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

#include <vsag/vsag.h>

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int64_t DIM = 4;

vsag::DatasetPtr
MakeQuery(std::vector<float>& query_vector) {
    return vsag::Dataset::Make()
        ->NumElements(1)
        ->Dim(DIM)
        ->Float32Vectors(query_vector.data())
        ->Owner(false);
}

}  // namespace

int
main() {
    vsag::init();

    std::vector<int64_t> ids{100, 101, 102, 103};
    std::vector<float> vectors{
        0.0F,
        0.0F,
        0.0F,
        0.0F,
        1.0F,
        0.0F,
        0.0F,
        0.0F,
        0.0F,
        1.0F,
        0.0F,
        0.0F,
        0.0F,
        0.0F,
        1.0F,
        0.0F,
    };
    auto base = vsag::Dataset::Make()
                    ->NumElements(static_cast<int64_t>(ids.size()))
                    ->Dim(DIM)
                    ->Ids(ids.data())
                    ->Float32Vectors(vectors.data())
                    ->Owner(false);

    const std::string create_params = R"({
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 4,
        "index_param": {"base_quantization_type": "fp32"}
    })";
    auto created = vsag::Factory::CreateIndex("brute_force", create_params);
    if (!created.has_value()) {
        std::cerr << "Failed to create index: " << created.error().message << std::endl;
        return 1;
    }
    auto index = created.value();
    auto built = index->Build(base);
    if (!built.has_value()) {
        std::cerr << "Failed to build index: " << built.error().message << std::endl;
        return 1;
    }

    std::vector<float> query_vector{0.1F, 0.1F, 0.0F, 0.0F};
    vsag::SearchRequest request;
    request.query_ = MakeQuery(query_vector);
    request.topk_ = 2;
    request.params_str_ = R"({})";  // Result statistics are enabled by default.

    auto searched = index->SearchWithRequest(request);
    if (!searched.has_value()) {
        std::cerr << "Search failed: " << searched.error().message << std::endl;
        return 1;
    }
    auto result = searched.value();
    if (result->GetDim() != request.topk_) {
        std::cerr << "Unexpected result count" << std::endl;
        return 1;
    }
    std::cout << "nearest id: " << result->GetIds()[0] << std::endl;

    const auto metrics = result->GetSearchMetrics();
    if (!metrics.has_value()) {
        std::cerr << "Typed search statistics are unavailable" << std::endl;
        return 1;
    }
    // APPROXIMATE labels the main search phase, not the accuracy of the distance:
    // this BruteForce scan evaluates exact FP32 distances.
    if (metrics->distance_evaluations != ids.size() or
        metrics->DistanceEvaluations(vsag::DistanceEvaluationPhase::APPROXIMATE) != ids.size() or
        metrics->DistanceEvaluations(vsag::DistanceEvaluationBackend::FP32) != ids.size() or
        not metrics->complete) {
        std::cerr << "Unexpected BruteForce distance accounting" << std::endl;
        return 1;
    }
    std::cout << "distance evaluations: " << metrics->distance_evaluations << '\n'
              << "approximate evaluations: "
              << metrics->DistanceEvaluations(vsag::DistanceEvaluationPhase::APPROXIMATE) << '\n'
              << "fp32 evaluations: "
              << metrics->DistanceEvaluations(vsag::DistanceEvaluationBackend::FP32) << '\n'
              << "legacy JSON: " << result->GetStatistics() << std::endl;

    request.params_str_ = R"({"want_statistics":false})";
    auto without_statistics = index->SearchWithRequest(request);
    if (!without_statistics.has_value()) {
        std::cerr << "Search without statistics failed: " << without_statistics.error().message
                  << std::endl;
        return 1;
    }
    const auto disabled = without_statistics.value();
    if (disabled->GetSearchMetrics().has_value() or disabled->GetStatistics() != "{}" or
        disabled->GetDim() != result->GetDim()) {
        std::cerr << "Statistics suppression contract violated" << std::endl;
        return 1;
    }
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        if (disabled->GetIds()[i] != result->GetIds()[i] or
            disabled->GetDistances()[i] != result->GetDistances()[i]) {
            std::cerr << "Statistics suppression changed search results" << std::endl;
            return 1;
        }
    }
    std::cout << "statistics disabled: typed="
              << (without_statistics.value()->GetSearchMetrics().has_value() ? "present" : "none")
              << ", JSON=" << without_statistics.value()->GetStatistics() << std::endl;

    return 0;
}
