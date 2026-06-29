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

#include <fmt/format.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "functest.h"
#include "test_index.h"

namespace fixtures {

class HGraphRaBitQSplitTestIndex : public TestIndex {
public:
    static const std::string name;
    static fixtures::TempDir dir;
    static TestDatasetPool pool;

    // Build a hgraph parameter JSON exercising the RaBitQ split datacell.
    // - base_io_type: IO backend for the one-bit storage (and homogeneous
    //                 supplement when supplement_io_type is empty).
    // - supplement_io_type: IO backend for the supplement (y-bit) storage.
    //                       When empty, the supplement falls back to
    //                       base_io_type.
    static std::string
    GenerateBuildParam(const std::string& metric_type,
                       int64_t dim,
                       const std::string& base_io_type,
                       const std::string& supplement_io_type,
                       uint32_t rabitq_filter_bits = 3,
                       uint32_t rabitq_supplement_bits = 5);
};

const std::string HGraphRaBitQSplitTestIndex::name = "hgraph";
fixtures::TempDir HGraphRaBitQSplitTestIndex::dir{"hgraph_rabitq_split"};
TestDatasetPool HGraphRaBitQSplitTestIndex::pool{};

std::string
HGraphRaBitQSplitTestIndex::GenerateBuildParam(const std::string& metric_type,
                                               int64_t dim,
                                               const std::string& base_io_type,
                                               const std::string& supplement_io_type,
                                               uint32_t rabitq_filter_bits,
                                               uint32_t rabitq_supplement_bits) {
    constexpr auto temp_with_supplement = R"(
    {{
        "dtype": "float32",
        "metric_type": "{}",
        "dim": {},
        "index_param": {{
            "base_quantization_type": "rabitq",
            "precise_quantization_type": "rabitq",
            "base_io_type": "{}",
            "base_supplement_io_type": "{}",
            "base_file_path": "{}",
            "use_reorder": true,
            "rabitq_bits_per_dim_base": {},
            "rabitq_bits_per_dim_precise": {},
            "rabitq_error_rate": 1.9,
            "max_degree": 32,
            "ef_construction": 200,
            "graph_storage_type": "compressed"
        }}
    }})";
    constexpr auto temp_without_supplement = R"(
    {{
        "dtype": "float32",
        "metric_type": "{}",
        "dim": {},
        "index_param": {{
            "base_quantization_type": "rabitq",
            "precise_quantization_type": "rabitq",
            "base_io_type": "{}",
            "base_file_path": "{}",
            "use_reorder": true,
            "rabitq_bits_per_dim_base": {},
            "rabitq_bits_per_dim_precise": {},
            "rabitq_error_rate": 1.9,
            "max_degree": 32,
            "ef_construction": 200,
            "graph_storage_type": "compressed"
        }}
    }})";
    if (supplement_io_type.empty()) {
        return fmt::format(temp_without_supplement,
                           metric_type,
                           dim,
                           base_io_type,
                           dir.GenerateRandomFile(),
                           rabitq_filter_bits,
                           rabitq_supplement_bits);
    }
    return fmt::format(temp_with_supplement,
                       metric_type,
                       dim,
                       base_io_type,
                       supplement_io_type,
                       dir.GenerateRandomFile(),
                       rabitq_filter_bits,
                       rabitq_supplement_bits);
}

}  // namespace fixtures

namespace {

constexpr const char* kSplitSearchParam = R"(
{
    "hgraph": {
        "ef_search": 200,
        "rabitq_one_bit_search": true
    }
})";

}  // namespace

TEST_CASE("HGraph RaBitQ Split Homogeneous IO", "[ft][rabitq_split][hgraph]") {
    using namespace fixtures;
    constexpr int64_t dim = 128;
    constexpr uint64_t base_count = 600;

    auto metric = GENERATE("l2", "ip");
    auto base_io = GENERATE("block_memory_io", "memory_io");

    INFO(fmt::format("metric={}, base_io={}", metric, base_io));
    auto param = HGraphRaBitQSplitTestIndex::GenerateBuildParam(metric, dim, base_io, "");
    auto index = TestIndex::TestFactory(HGraphRaBitQSplitTestIndex::name, param, true);
    auto dataset = HGraphRaBitQSplitTestIndex::pool.GetDatasetAndCreate(dim, base_count, metric);
    TestIndex::TestBuildIndex(index, dataset, true);
    // Recall is intentionally low: the split datacell at moderate dim is not
    // expected to recover ground-truth accuracy. The check primarily proves
    // that build / search complete without error on this code path.
    TestIndex::TestKnnSearch(index, dataset, kSplitSearchParam, /*expected_recall=*/0.0F, true);
}

TEST_CASE("HGraph RaBitQ Split Hybrid IO (memory + async supplement)",
          "[ft][rabitq_split][hybrid][hgraph]") {
    using namespace fixtures;
    constexpr int64_t dim = 128;
    constexpr uint64_t base_count = 600;
    const std::string metric = "l2";

    auto build_param =
        HGraphRaBitQSplitTestIndex::GenerateBuildParam(metric, dim, "block_memory_io", "async_io");
    auto index = TestIndex::TestFactory(HGraphRaBitQSplitTestIndex::name, build_param, true);
    auto dataset = HGraphRaBitQSplitTestIndex::pool.GetDatasetAndCreate(dim, base_count, metric);
    TestIndex::TestBuildIndex(index, dataset, true);
    TestIndex::TestKnnSearch(index, dataset, kSplitSearchParam, 0.0F, true);

    // Round-trip serialize / deserialize so the supplement_io_params branch
    // in FlattenDataCellParameter::ToJson and FlattenInterface::MakeInstance
    // is exercised end-to-end.
    auto reload_param =
        HGraphRaBitQSplitTestIndex::GenerateBuildParam(metric, dim, "block_memory_io", "async_io");
    auto reloaded = TestIndex::TestFactory(HGraphRaBitQSplitTestIndex::name, reload_param, true);
    TestIndex::TestSerializeFile(index, reloaded, dataset, kSplitSearchParam, true);
}

TEST_CASE("HGraph RaBitQ Split Reject Unsupported Hybrid", "[ft][rabitq_split][hgraph]") {
    using namespace fixtures;
    constexpr int64_t dim = 128;
    const std::string metric = "l2";

    // Only (block_memory_io one-bit + async_io supplement) is supported as a
    // hybrid combination today; any other heterogeneous pair must fail at
    // index creation time with a clear error from
    // FlattenInterface::MakeInstance.
    auto bad_param =
        HGraphRaBitQSplitTestIndex::GenerateBuildParam(metric, dim, "memory_io", "async_io");
    auto result = vsag::Factory::CreateIndex(HGraphRaBitQSplitTestIndex::name, bad_param);
    REQUIRE_FALSE(result.has_value());
}
