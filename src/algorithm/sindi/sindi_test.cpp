
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

#include "sindi.h"

#include <catch2/catch_test_macros.hpp>

#include "fixtures.h"
#include "impl/allocator/safe_allocator.h"

using namespace vsag;

TEST_CASE("SINDI Basic Test", "[ut][SINDI]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;

    /******************* Prepare Base and Query Dataset *****************/
    uint32_t num_base = 10000;
    int64_t max_dim = 128;
    int64_t max_id = 30000;
    float min_val = 0;
    float max_val = 10;
    int seed_base = 114;
    int seed_query = 514;
    int64_t k = 10;

    std::vector<int64_t> ids(num_base);
    for (int64_t i = 0; i < num_base; ++i) {
        ids[i] = i;
    }

    auto sv_base =
        fixtures::GenerateSparseVectors(num_base, max_dim, max_id, min_val, max_val, seed_base);
    auto base = vsag::Dataset::Make();
    base->NumElements(num_base)->SparseVectors(sv_base.data())->Ids(ids.data())->Owner(false);

    auto param_str = R"({
        "use_reorder": true,
        "query_prune_ratio": 1,
        "doc_prune_ratio": 1,
        "term_prune_ratio": 1,
        "window_size": 1000
    })";

    vsag::JsonType param_json = vsag::JsonType::parse(param_str);
    auto index_param = std::make_shared<vsag::SINDIParameters>();
    index_param->FromJson(param_json);
    auto index = std::make_unique<SINDI>(index_param, common_param);
    auto another_index = std::make_unique<SINDI>(index_param, common_param);

    auto build_res = index->Build(base);
    REQUIRE(build_res.size() == 0);
    REQUIRE(index->GetNumElements() == num_base);

    auto dir = fixtures::TempDir("serialize");
    auto path = dir.GenerateRandomFile();
    std::ofstream outfile(path, std::ios::out | std::ios::binary);
    IOStreamWriter writer(outfile);
    index->Serialize(writer);
    outfile.close();

    std::ifstream infile(path, std::ios::in | std::ios::binary);
    IOStreamReader reader(infile);
    another_index->Deserialize(reader);
    infile.close();
    REQUIRE(another_index->GetNumElements() == num_base);

    std::string search_param_str = R"(
    {
        "n_candidate": 20
    }
    )";

    auto query = vsag::Dataset::Make();

    for (int i = 0; i < num_base; ++i) {
        query->NumElements(1)->SparseVectors(sv_base.data() + i)->Owner(false);

        auto result = index->KnnSearch(query, k, search_param_str, nullptr);
        auto another_result = another_index->KnnSearch(query, k, search_param_str, nullptr);

        for (int j = 0; j < k; j++) {
            REQUIRE(result->GetIds()[j] == another_result->GetIds()[j]);
        }
        REQUIRE(result->GetIds()[0] == ids[i]);
    }

    for (auto& item : sv_base) {
        delete[] item.vals_;
        delete[] item.ids_;
    }
}
