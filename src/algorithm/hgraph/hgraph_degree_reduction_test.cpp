// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <fmt/format.h>

#include <algorithm>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "hgraph.h"
#include "index/index_impl.h"
#include "unittest.h"
#include "vsag/factory.h"

namespace {

std::string
create_parameters(uint32_t max_degree,
                  const std::string& quantization = "fp32",
                  bool store_raw_vector = false,
                  bool support_force_remove = false) {
    return fmt::format(R"({{"dim":8,"dtype":"float32","metric_type":"l2","index_param":{{)"
                       R"("base_quantization_type":"{}","max_degree":{},)"
                       R"("ef_construction":40,"build_thread_count":1,)"
                       R"("store_raw_vector":{},"support_force_remove":{}}}}})",
                       quantization,
                       max_degree,
                       store_raw_vector,
                       support_force_remove);
}

}  // namespace

TEST_CASE("HGraph Tune reduces max degree and changes quantization",
          "[ut][hgraph_degree_reduction]") {
    constexpr int64_t count = 256;
    constexpr int64_t dim = 8;
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 1000);
    std::vector<float> vectors(count * dim);
    for (int64_t i = 0; i < count; ++i) {
        for (int64_t j = 0; j < dim; ++j) {
            vectors[i * dim + j] = static_cast<float>((i * 17 + j * 29) % 257) / 257.0F;
        }
    }
    auto base = vsag::Dataset::Make()
                    ->NumElements(count)
                    ->Dim(dim)
                    ->Ids(ids.data())
                    ->Float32Vectors(vectors.data())
                    ->Owner(false);
    auto query = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(dim)
                     ->Float32Vectors(vectors.data())
                     ->Owner(false);

    auto created = vsag::Factory::CreateIndex("hgraph", create_parameters(16, "fp32", true));
    REQUIRE(created.has_value());
    auto index = std::dynamic_pointer_cast<vsag::IndexImpl<vsag::HGraph>>(created.value());
    REQUIRE(index != nullptr);
    REQUIRE(index->Build(base).has_value());
    REQUIRE(index->KnnSearch(query, 10, R"({"hgraph":{"ef_search":40}})").has_value());

    auto hgraph = std::dynamic_pointer_cast<vsag::HGraph>(index->GetInnerIndex());
    REQUIRE(hgraph != nullptr);
    auto parameter = std::dynamic_pointer_cast<vsag::HGraphParameter>(hgraph->create_param_ptr_);
    REQUIRE(parameter != nullptr);
    auto no_degree = index->Tune(R"({"index_param":{"base_quantization_type":"fp32"}})");
    REQUIRE(no_degree.has_value());
    REQUIRE(no_degree.value());
    REQUIRE(parameter->bottom_graph_param->max_degree_ == 16);

    auto same_degree =
        index->Tune(R"({"index_param":{"base_quantization_type":"fp32","max_degree":16}})");
    REQUIRE(same_degree.has_value());
    REQUIRE(same_degree.value());
    REQUIRE(parameter->bottom_graph_param->max_degree_ == 16);

    auto increased_degree =
        index->Tune(R"({"index_param":{"base_quantization_type":"fp32","max_degree":32}})");
    REQUIRE(increased_degree.has_value());
    REQUIRE_FALSE(increased_degree.value());
    REQUIRE(parameter->bottom_graph_param->max_degree_ == 16);

    auto too_small_degree =
        index->Tune(R"({"index_param":{"base_quantization_type":"fp32","max_degree":3}})");
    REQUIRE(too_small_degree.has_value());
    REQUIRE_FALSE(too_small_degree.value());
    REQUIRE(parameter->bottom_graph_param->max_degree_ == 16);

    auto reduced_and_quantized =
        index->Tune(R"({"index_param":{"base_quantization_type":"sq8","max_degree":8}})", true);
    REQUIRE(reduced_and_quantized.has_value());
    REQUIRE(reduced_and_quantized.value());
    REQUIRE(parameter->bottom_graph_param->max_degree_ == 8);
    REQUIRE(parameter->hierarchical_graph_param->max_degree_ == 4);
    REQUIRE(parameter->base_codes_param->ToJson()["quantization_params"]["type"].GetString() ==
            std::string("sq8"));
    REQUIRE_FALSE(parameter->store_raw_vector);
    const auto memory_detail = index->GetMemoryUsageDetail();
    const auto measured_memory =
        std::accumulate(memory_detail.begin(),
                        memory_detail.end(),
                        uint64_t{sizeof(vsag::HGraph)},
                        [](auto sum, const auto& item) { return sum + item.second; });
    REQUIRE(index->GetMemoryUsage() == measured_memory);
    REQUIRE(index->KnnSearch(query, 10, R"({"hgraph":{"ef_search":40}})").has_value());

    std::stringstream artifact;
    REQUIRE(index->Serialize(artifact).has_value());
    artifact.seekg(0, std::ios::beg);
    auto restored = vsag::Factory::CreateIndex("hgraph", create_parameters(8, "sq8"));
    REQUIRE(restored.has_value());
    REQUIRE(restored.value()->Deserialize(artifact).has_value());
    REQUIRE(restored.value()->KnnSearch(query, 10, R"({"hgraph":{"ef_search":40}})").has_value());

    auto disabled =
        index->Tune(R"({"index_param":{"base_quantization_type":"sq8","max_degree":4}})");
    REQUIRE(disabled.has_value());
    REQUIRE_FALSE(disabled.value());
}

TEST_CASE("HGraph Tune prepares degree reduction again after Add",
          "[ut][hgraph_degree_reduction]") {
    constexpr int64_t count = 128;
    constexpr int64_t dim = 8;
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<float> vectors(count * dim);
    for (int64_t i = 0; i < count; ++i) {
        for (int64_t j = 0; j < dim; ++j) {
            vectors[i * dim + j] = static_cast<float>((i * 11 + j * 23) % 131) / 131.0F;
        }
    }
    auto base = vsag::Dataset::Make()
                    ->NumElements(count)
                    ->Dim(dim)
                    ->Ids(ids.data())
                    ->Float32Vectors(vectors.data())
                    ->Owner(false);
    auto created = vsag::Factory::CreateIndex("hgraph", create_parameters(16));
    REQUIRE(created.has_value());
    auto index = created.value();
    REQUIRE(index->Build(base).has_value());

    auto first_reduction =
        index->Tune(R"({"index_param":{"base_quantization_type":"fp32","max_degree":8}})");
    REQUIRE(first_reduction.has_value());
    REQUIRE(first_reduction.value());

    int64_t added_id = count;
    std::vector<float> added_vector(dim, 0.25F);
    auto added = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(dim)
                     ->Ids(&added_id)
                     ->Float32Vectors(added_vector.data())
                     ->Owner(false);
    REQUIRE(index->Add(added).has_value());

    auto second_reduction =
        index->Tune(R"({"index_param":{"base_quantization_type":"fp32","max_degree":4}})");
    REQUIRE(second_reduction.has_value());
    REQUIRE(second_reduction.value());

    auto query = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(dim)
                     ->Float32Vectors(added_vector.data())
                     ->Owner(false);
    REQUIRE(index->KnnSearch(query, 10, R"({"hgraph":{"ef_search":40}})").has_value());

    std::stringstream artifact;
    REQUIRE(index->Serialize(artifact).has_value());
    artifact.seekg(0, std::ios::beg);
    auto restored = vsag::Factory::CreateIndex("hgraph", create_parameters(4));
    REQUIRE(restored.has_value());
    REQUIRE(restored.value()->Deserialize(artifact).has_value());
    REQUIRE(restored.value()->KnnSearch(query, 10, R"({"hgraph":{"ef_search":40}})").has_value());
}

TEST_CASE("HGraph Tune freeze keeps explicitly requested raw vectors",
          "[ut][hgraph_degree_reduction]") {
    constexpr int64_t count = 64;
    constexpr int64_t dim = 8;
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<float> vectors(count * dim);
    std::iota(vectors.begin(), vectors.end(), 0.0F);
    auto base = vsag::Dataset::Make()
                    ->NumElements(count)
                    ->Dim(dim)
                    ->Ids(ids.data())
                    ->Float32Vectors(vectors.data())
                    ->Owner(false);

    auto created = vsag::Factory::CreateIndex("hgraph", create_parameters(16, "fp32", true));
    REQUIRE(created.has_value());
    auto index = std::dynamic_pointer_cast<vsag::IndexImpl<vsag::HGraph>>(created.value());
    REQUIRE(index != nullptr);
    REQUIRE(index->Build(base).has_value());
    auto hgraph = std::dynamic_pointer_cast<vsag::HGraph>(index->GetInnerIndex());
    REQUIRE(hgraph != nullptr);
    auto parameter = std::dynamic_pointer_cast<vsag::HGraphParameter>(hgraph->create_param_ptr_);
    REQUIRE(parameter != nullptr);

    auto tuned = index->Tune(R"({"index_param":{"base_quantization_type":"sq8","max_degree":8,)"
                             R"("store_raw_vector":true}})",
                             true);
    REQUIRE(tuned.has_value());
    REQUIRE(tuned.value());
    REQUIRE(parameter->store_raw_vector);

    auto query = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(dim)
                     ->Float32Vectors(vectors.data())
                     ->Owner(false);
    REQUIRE(index->KnnSearch(query, 10, R"({"hgraph":{"ef_search":40}})").has_value());

    std::stringstream artifact;
    REQUIRE(index->Serialize(artifact).has_value());
    artifact.seekg(0, std::ios::beg);
    auto restored = vsag::Factory::CreateIndex("hgraph", create_parameters(8, "sq8", true));
    REQUIRE(restored.has_value());
    REQUIRE(restored.value()->Deserialize(artifact).has_value());
    REQUIRE(restored.value()->KnnSearch(query, 10, R"({"hgraph":{"ef_search":40}})").has_value());
}

TEST_CASE("HGraph Tune rejects degree reduction for unsupported graphs",
          "[ut][hgraph_degree_reduction]") {
    auto created = vsag::Factory::CreateIndex("hgraph", create_parameters(16, "fp32", false, true));
    REQUIRE(created.has_value());
    auto tuned = created.value()->Tune(
        R"({"index_param":{"base_quantization_type":"fp32","max_degree":8}})");
    REQUIRE(tuned.has_value());
    REQUIRE_FALSE(tuned.value());
}

TEST_CASE("HGraph Tune prepares degree reduction again after Deserialize and Merge",
          "[ut][hgraph_degree_reduction]") {
    constexpr int64_t count = 64;
    constexpr int64_t dim = 8;
    std::vector<float> vectors(count * 2 * dim);
    for (int64_t i = 0; i < count * 2; ++i) {
        for (int64_t j = 0; j < dim; ++j) {
            vectors[i * dim + j] = static_cast<float>((i * 13 + j * 19) % 137) / 137.0F;
        }
    }
    std::vector<int64_t> first_ids(count);
    std::iota(first_ids.begin(), first_ids.end(), 0);
    auto first_base = vsag::Dataset::Make()
                          ->NumElements(count)
                          ->Dim(dim)
                          ->Ids(first_ids.data())
                          ->Float32Vectors(vectors.data())
                          ->Owner(false);
    auto query = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(dim)
                     ->Float32Vectors(vectors.data())
                     ->Owner(false);

    auto first_created = vsag::Factory::CreateIndex("hgraph", create_parameters(16));
    REQUIRE(first_created.has_value());
    auto first = first_created.value();
    REQUIRE(first->Build(first_base).has_value());
    auto first_tune =
        first->Tune(R"({"index_param":{"base_quantization_type":"fp32","max_degree":8}})");
    REQUIRE(first_tune.has_value());
    REQUIRE(first_tune.value());

    std::stringstream artifact;
    REQUIRE(first->Serialize(artifact).has_value());
    artifact.seekg(0, std::ios::beg);
    auto restored_created = vsag::Factory::CreateIndex("hgraph", create_parameters(8));
    REQUIRE(restored_created.has_value());
    auto restored = restored_created.value();
    REQUIRE(restored->Deserialize(artifact).has_value());
    auto post_deserialize_tune =
        restored->Tune(R"({"index_param":{"base_quantization_type":"fp32","max_degree":4}})");
    REQUIRE(post_deserialize_tune.has_value());
    REQUIRE(post_deserialize_tune.value());
    REQUIRE(restored->KnnSearch(query, 10, R"({"hgraph":{"ef_search":40}})").has_value());

    std::vector<int64_t> second_ids(count);
    std::iota(second_ids.begin(), second_ids.end(), count);
    auto second_base = vsag::Dataset::Make()
                           ->NumElements(count)
                           ->Dim(dim)
                           ->Ids(second_ids.data())
                           ->Float32Vectors(vectors.data() + count * dim)
                           ->Owner(false);
    auto second_created = vsag::Factory::CreateIndex("hgraph", create_parameters(8));
    REQUIRE(second_created.has_value());
    auto second = second_created.value();
    REQUIRE(second->Build(second_base).has_value());

    auto merged = first->Merge({{second, [](int64_t id) { return std::make_tuple(true, id); }}});
    REQUIRE(merged.has_value());
    auto post_merge_tune =
        first->Tune(R"({"index_param":{"base_quantization_type":"fp32","max_degree":4}})");
    REQUIRE(post_merge_tune.has_value());
    REQUIRE(post_merge_tune.value());
    REQUIRE(first->KnnSearch(query, 10, R"({"hgraph":{"ef_search":40}})").has_value());
}
