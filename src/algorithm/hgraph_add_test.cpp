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

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <functional>
#include <future>
#include <limits>
#include <new>
#include <vector>

#include "hgraph.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/thread_pool/safe_thread_pool.h"
#include "index/index_impl.h"

namespace {

class ArmableRejectingThreadPool final : public vsag::ThreadPool {
public:
    void
    WaitUntilEmpty() override {
    }

    void
    SetQueueSizeLimit(std::size_t) override {
    }

    void
    SetPoolSize(std::size_t) override {
    }

    std::future<void>
    Enqueue(std::function<void(void)> task) override {
        if (reject_submissions_.load(std::memory_order_acquire)) {
            throw std::bad_alloc();
        }
        task();
        return {};
    }

    void
    SetRejectSubmissions(bool reject) {
        reject_submissions_.store(reject, std::memory_order_release);
    }

private:
    std::atomic<bool> reject_submissions_{false};
};

vsag::DatasetPtr
MakeFloatDataset(std::vector<float>& vectors,
                 std::vector<int64_t>& ids,
                 int64_t dim,
                 int64_t count) {
    auto dataset = vsag::Dataset::Make();
    dataset->NumElements(count)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    return dataset;
}

vsag::IndexCommonParam
MakeCommonParam(int64_t dim) {
    vsag::IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();
    return common_param;
}

}  // namespace

TEST_CASE("HGraph Tune accepts an incomplete source after Add failure",
          "[ut][hgraph][add][hgraph_tune_incomplete_source]") {
    constexpr int64_t dim = 4;
    constexpr int64_t base_count = 2;

    auto rejecting_pool = std::make_shared<ArmableRejectingThreadPool>();
    auto common_param = MakeCommonParam(dim);
    common_param.thread_pool_ = std::make_shared<vsag::SafeThreadPool>(rejecting_pool);
    auto hgraph_json = vsag::JsonType::Parse(R"({
        "base_quantization_type": "sq8",
        "max_degree": 8,
        "ef_construction": 32,
        "build_thread_count": 1,
        "store_raw_vector": true
    })");
    auto index = std::make_shared<vsag::IndexImpl<vsag::HGraph>>(hgraph_json, common_param);

    std::vector<float> base_vectors = {
        0.0F,
        0.0F,
        0.0F,
        0.0F,
        1.0F,
        1.0F,
        1.0F,
        1.0F,
    };
    std::vector<int64_t> base_ids = {10, 20};
    auto base = MakeFloatDataset(base_vectors, base_ids, dim, base_count);
    REQUIRE(index->Build(base).has_value());
    REQUIRE(index->GetNumElements() == base_count);

    std::vector<float> add_vectors = {2.0F, 2.0F, 2.0F, 2.0F};
    std::vector<int64_t> add_ids = {30};
    auto add = MakeFloatDataset(add_vectors, add_ids, dim, 1);

    rejecting_pool->SetRejectSubmissions(true);
    auto add_result = index->Add(add);
    rejecting_pool->SetRejectSubmissions(false);

    REQUIRE_FALSE(add_result.has_value());
    REQUIRE(add_result.error().type == vsag::ErrorType::NO_ENOUGH_MEMORY);
    REQUIRE(index->GetNumElements() == base_count + 1);
    REQUIRE(index->CheckIdExist(add_ids[0]));

    auto tune_result = index->Tune(R"({
        "index_param": {
            "base_quantization_type": "bf16",
            "max_degree": 8,
            "ef_construction": 32
        }
    })");
    REQUIRE(tune_result.has_value());
    CHECK(tune_result.value());
}

TEST_CASE("HGraph search handles an empty route result", "[ut][hgraph][search][nonfinite]") {
    constexpr int64_t dim = 4;
    constexpr int64_t base_count = 64;

    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = vsag::JsonType::Parse(R"({
        "base_quantization_type": "fp32",
        "max_degree": 8,
        "ef_construction": 32,
        "build_thread_count": 1
    })");
    auto index = std::make_shared<vsag::IndexImpl<vsag::HGraph>>(hgraph_json, common_param);

    std::vector<float> base_vectors(base_count * dim);
    std::vector<int64_t> base_ids(base_count);
    for (int64_t i = 0; i < base_count; ++i) {
        base_ids[i] = i;
        for (int64_t j = 0; j < dim; ++j) {
            base_vectors[i * dim + j] = static_cast<float>(i + j);
        }
    }
    auto base = MakeFloatDataset(base_vectors, base_ids, dim, base_count);
    REQUIRE(index->Build(base).has_value());

    std::vector<float> query_vectors(dim, std::numeric_limits<float>::quiet_NaN());
    std::vector<int64_t> query_ids = {0};
    auto query = MakeFloatDataset(query_vectors, query_ids, dim, 1);
    vsag::SearchRequest request;
    request.query_ = query;
    request.topk_ = 10;
    request.params_str_ = R"({"hgraph":{"ef_search":32}})";

    auto result = index->SearchWithRequest(request);
    REQUIRE(result.has_value());
    CHECK(result.value()->GetDim() == 0);
}
