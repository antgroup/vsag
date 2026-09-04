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

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <string>
#include <vector>

namespace {

constexpr int64_t DIM = 8;

vsag::DatasetPtr
MakeDataset(int64_t first_id,
            int64_t count,
            std::vector<int64_t>& ids,
            std::vector<float>& vectors) {
    ids.resize(static_cast<uint64_t>(count));
    vectors.resize(static_cast<uint64_t>(count * DIM));
    for (int64_t row = 0; row < count; ++row) {
        ids[static_cast<uint64_t>(row)] = first_id + row;
        for (int64_t col = 0; col < DIM; ++col) {
            vectors[static_cast<uint64_t>(row * DIM + col)] =
                static_cast<float>((first_id + row) * 0.1 + col);
        }
    }
    return vsag::Dataset::Make()
        ->NumElements(count)
        ->Dim(DIM)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
}

vsag::DatasetPtr
MakeQuery(const float* vector) {
    return vsag::Dataset::Make()->NumElements(1)->Dim(DIM)->Float32Vectors(vector)->Owner(false);
}

std::string
BuildParameters(const std::string& index_name) {
    if (index_name == "brute_force") {
        return R"({
            "dtype": "float32",
            "metric_type": "l2",
            "dim": 8,
            "index_param": {
                "base_quantization_type": "fp32",
                "store_raw_vector": true
            }
        })";
    }
    if (index_name == "warp") {
        return R"({
            "dtype": "float32",
            "metric_type": "l2",
            "dim": 8,
            "index_param": {
                "base_quantization_type": "fp32",
                "base_io_type": "memory_io"
            }
        })";
    }
    if (index_name == "lazy_hgraph") {
        return R"({
            "dtype": "float32",
            "metric_type": "l2",
            "dim": 8,
            "lazy_hgraph": {
                "transition_threshold": 4,
                "hgraph": {
                    "base_quantization_type": "fp32",
                    "max_degree": 8,
                    "ef_construction": 32,
                    "build_thread_count": 1,
                    "support_force_remove": true
                }
            }
        })";
    }
    return R"({
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 8,
        "index_param": {
            "base_quantization_type": "fp32",
            "base_io_type": "memory_io",
            "max_degree": 8,
            "ef_construction": 32,
            "build_thread_count": 1
        }
    })";
}

const char*
SearchParameters() {
    return R"({"hgraph":{"ef_search":32}})";
}

}  // namespace

TEST_CASE("Lite exposes the retained index factories", "[ft][lite]") {
    for (const auto* index_name : {"hgraph", "lazy_hgraph", "brute_force", "warp"}) {
        INFO(index_name);
        auto created = vsag::Factory::CreateIndex(index_name, BuildParameters(index_name));
        REQUIRE(created.has_value());
    }
}

#ifdef VSAG_LITE
TEST_CASE("Lite excludes unsupported index factories", "[ft][lite]") {
    for (const auto* index_name : {"ivf", "pyramid", "sindi", "sindi_v2", "simq"}) {
        INFO(index_name);
        auto created = vsag::Factory::CreateIndex(index_name, R"({"dtype":"float32"})");
        REQUIRE_FALSE(created.has_value());
    }
}
#endif

TEST_CASE("Lite supports CRUD, search, and marked-deletion persistence", "[ft][lite]") {
    auto created = vsag::Factory::CreateIndex("lazy_hgraph", BuildParameters("lazy_hgraph"));
    REQUIRE(created.has_value());
    auto index = created.value();

    std::vector<int64_t> ids;
    std::vector<float> vectors;
    auto dataset = MakeDataset(100, 6, ids, vectors);
    auto add_result = index->Add(dataset);
    REQUIRE(add_result.has_value());
    REQUIRE(add_result.value().empty());
    REQUIRE(index->GetNumElements() == 6);

    auto update_id = index->UpdateId(100, 900);
    REQUIRE(update_id.has_value());
    REQUIRE(update_id.value());

    std::vector<int64_t> replacement_ids{900};
    std::vector<float> replacement(static_cast<uint64_t>(DIM), -3.0F);
    auto replacement_dataset = vsag::Dataset::Make()
                                   ->NumElements(1)
                                   ->Dim(DIM)
                                   ->Ids(replacement_ids.data())
                                   ->Float32Vectors(replacement.data())
                                   ->Owner(false);
    auto update_vector = index->UpdateVector(900, replacement_dataset, true);
    REQUIRE(update_vector.has_value());
    REQUIRE(update_vector.value());

    auto remove_result = index->Remove(101, vsag::RemoveMode::MARK_REMOVE);
    REQUIRE(remove_result.has_value());
    REQUIRE(remove_result.value() == 1);
    REQUIRE(index->GetNumElements() == 5);

    auto binary = index->Serialize();
    REQUIRE(binary.has_value());
    auto restored = vsag::Factory::CreateIndex("lazy_hgraph", BuildParameters("lazy_hgraph"));
    REQUIRE(restored.has_value());
    REQUIRE(restored.value()->Deserialize(binary.value()).has_value());
    REQUIRE(restored.value()->GetNumElements() == 5);
    REQUIRE_FALSE(restored.value()->CheckIdExist(101));

    auto result =
        restored.value()->KnnSearch(MakeQuery(vectors.data() + DIM), 5, SearchParameters());
    REQUIRE(result.has_value());
    for (int64_t i = 0; i < result.value()->GetDim(); ++i) {
        REQUIRE(result.value()->GetIds()[i] != 101);
    }

    auto updated_result =
        restored.value()->KnnSearch(MakeQuery(replacement.data()), 1, SearchParameters());
    REQUIRE(updated_result.has_value());
    REQUIRE(updated_result.value()->GetIds()[0] == 900);
}
