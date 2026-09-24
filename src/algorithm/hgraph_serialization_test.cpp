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

#include <catch2/catch_test_macros.hpp>
#include <set>
#include <string>
#include <vector>

#include "vsag/vsag.h"

TEST_CASE("HGraph restores labels within logical vector bounds", "[ut][hgraph][label-bounds]") {
    for (const auto* metric : {"cosine", "l2"}) {
        for (const int64_t dim : {3, 4}) {
            for (const bool duplicates : {false, true}) {
                for (const bool equal_vectors : {false, true}) {
                    for (const auto& ids : {std::vector<int64_t>{0, 10, 20},
                                            std::vector<int64_t>{10, 0, 20},
                                            std::vector<int64_t>{10, 20, 0},
                                            std::vector<int64_t>{10, 20, 30}}) {
                        CAPTURE(metric, dim, duplicates, equal_vectors, ids);
                        const auto config = std::string(R"({"dtype":"float32","metric_type":")") +
                                            metric + R"(","dim":)" + std::to_string(dim) +
                                            R"(,"index_param":{"base_quantization_type":"fp32",
                                                "max_degree":8,"ef_construction":32,
                                                "build_thread_count":1,"support_tomb_stone":false,
                                                "support_duplicate":)" +
                                            (duplicates ? "true" : "false") + "}}";
                        auto index = vsag::Factory::CreateIndex("hgraph", config).value();
                        std::vector<float> vectors(3 * dim, 0.0F);
                        vectors[0] = 1.0F;
                        vectors[dim + 1] = 1.0F;
                        vectors[2 * dim + (equal_vectors ? 0 : 2)] = 1.0F;
                        auto base = vsag::Dataset::Make()
                                        ->NumElements(3)
                                        ->Dim(dim)
                                        ->Ids(ids.data())
                                        ->Float32Vectors(vectors.data())
                                        ->Owner(false);
                        REQUIRE(index->Build(base).has_value());
                        const auto check = [&](const vsag::IndexPtr& current) {
                            REQUIRE(current->GetNumElements() == 3);
                            auto query = vsag::Dataset::Make()
                                             ->NumElements(1)
                                             ->Dim(dim)
                                             ->Float32Vectors(vectors.data())
                                             ->Owner(false);
                            auto result =
                                current->KnnSearch(query, 3, R"({"hgraph":{"ef_search":32}})");
                            REQUIRE(result.has_value());
                            const auto count = result.value()->GetDim();
                            std::set<int64_t> found(result.value()->GetIds(),
                                                    result.value()->GetIds() + count);
                            REQUIRE(found.size() == 3);
                            for (uint64_t i = 0; i < ids.size(); ++i) {
                                REQUIRE(found.count(ids[i]) == 1);
                            }
                            REQUIRE_FALSE(current->CheckIdExist(99));
                            if (ids.back() == 30) {
                                REQUIRE_FALSE(current->CheckIdExist(0));
                            }
                            for (uint64_t i = 0; i < ids.size(); ++i) {
                                REQUIRE(current->CheckIdExist(ids[i]));
                                auto distance =
                                    current->CalDistanceById(vectors.data() + i * dim, &ids[i], 1);
                                REQUIRE(distance.has_value());
                                REQUIRE(distance.value()->GetDistances()[0] == 0.0F);
                            }
                        };
                        const auto roundtrip = [&]() {
                            auto payload = index->Serialize();
                            REQUIRE(payload.has_value());
                            auto restored = vsag::Factory::CreateIndex("hgraph", config).value();
                            REQUIRE(restored->Deserialize(payload.value()).has_value());
                            index = restored;
                        };
                        check(index);
                        roundtrip();
                        check(index);
                    }
                }
            }
        }
    }
}
