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
create_parameters(uint32_t max_degree) {
    return fmt::format(R"({{"dim":8,"dtype":"float32","metric_type":"l2","index_param":{{)"
                       R"("base_quantization_type":"fp32","max_degree":{},"ef_construction":40,)"
                       R"("build_thread_count":1}}}})",
                       max_degree);
}

}  // namespace

TEST_CASE("HGraph materializes nested compact max-degree graphs", "[ut][hgraph_degree_reduction]") {
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

    auto created = vsag::Factory::CreateIndex("hgraph", create_parameters(16));
    REQUIRE(created.has_value());
    auto index = std::dynamic_pointer_cast<vsag::IndexImpl<vsag::HGraph>>(created.value());
    REQUIRE(index != nullptr);
    REQUIRE(index->Build(base).has_value());
    REQUIRE(index->KnnSearch(query, 10, R"({"hgraph":{"ef_search":40}})").has_value());

    auto hgraph = std::dynamic_pointer_cast<vsag::HGraph>(index->GetInnerIndex());
    REQUIRE(hgraph != nullptr);
    REQUIRE(hgraph->CanReduceMaxDegree());
    hgraph->PrepareDegreeReduction();

    for (const auto degree : {8U, 4U}) {
        hgraph->ReduceMaxDegree(degree);
        const auto parameter =
            std::dynamic_pointer_cast<vsag::HGraphParameter>(hgraph->create_param_ptr_);
        REQUIRE(parameter != nullptr);
        REQUIRE(parameter->bottom_graph_param->max_degree_ == degree);
        REQUIRE(parameter->hierarchical_graph_param->max_degree_ ==
                std::max<uint32_t>(1, degree / 2));

        std::stringstream artifact;
        REQUIRE(index->Serialize(artifact).has_value());
        artifact.seekg(0, std::ios::beg);
        auto restored = vsag::Factory::CreateIndex("hgraph", create_parameters(degree));
        REQUIRE(restored.has_value());
        REQUIRE(restored.value()->Deserialize(artifact).has_value());
        REQUIRE(
            restored.value()->KnnSearch(query, 10, R"({"hgraph":{"ef_search":40}})").has_value());
    }

    REQUIRE_THROWS(hgraph->ReduceMaxDegree(3));
    REQUIRE_THROWS(hgraph->ReduceMaxDegree(4));
}
