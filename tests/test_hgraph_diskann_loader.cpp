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

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "algorithm/hgraph_diskann_loader.h"
#include "fixtures/test_dataset_pool.h"
#include "fixtures/test_logger.h"
#include "fixtures/test_reader.h"
#include "impl/allocator/default_allocator.h"
#include "test_index.h"
#include "typing.h"
#include "vsag/vsag.h"

namespace fixtures {

class HGraphDiskANNLoaderTestIndex : public fixtures::TestIndex {
public:
    static TestDatasetPool pool;

    // Generate HGraphDiskANNLoader build parameters string from DiskANN format
    static std::string
    GenerateHGraphDiskANNLoaderBuildParametersString(const std::string& metric_type,
                                                     int64_t dim,
                                                     int64_t max_degree = 16,
                                                     int64_t pq_dims = 32);

    // Generate DiskANN build parameters string for creating source index
    static std::string
    GenerateDiskANNBuildParametersString(const std::string& metric_type,
                                         int64_t dim,
                                         int64_t max_degree = 16,
                                         int64_t ef_construction = 200,
                                         int64_t pq_dims = 32);

    // Search parameter template for HGraph
    static constexpr auto search_param_template = R"(
        {{
            "hgraph": {{
                "ef_search": 200
            }}
        }}
    )";

    static constexpr uint64_t base_count = 1000;
};

TestDatasetPool HGraphDiskANNLoaderTestIndex::pool{};

std::string
HGraphDiskANNLoaderTestIndex::GenerateHGraphDiskANNLoaderBuildParametersString(
    const std::string& metric_type, int64_t dim, int64_t max_degree, int64_t pq_dims) {
    constexpr auto build_parameter_json = R"(
        {{
            "dtype": "float32",
            "metric_type": "{}",
            "dim": {},
            "diskann": {{
                "max_degree": {},
                "pq_dims": {}
            }}
        }}
    )";
    return fmt::format(build_parameter_json, metric_type, dim, max_degree, pq_dims);
}

std::string
HGraphDiskANNLoaderTestIndex::GenerateDiskANNBuildParametersString(const std::string& metric_type,
                                                                   int64_t dim,
                                                                   int64_t max_degree,
                                                                   int64_t ef_construction,
                                                                   int64_t pq_dims) {
    constexpr auto build_parameter_json = R"(
        {{
            "dtype": "float32",
            "metric_type": "{}",
            "dim": {},
            "diskann": {{
                "max_degree": {},
                "ef_construction": {},
                "pq_dims": {},
                "pq_sample_rate": 0.5,
                "use_pq_search": true
            }}
        }}
    )";
    return fmt::format(
        build_parameter_json, metric_type, dim, max_degree, ef_construction, pq_dims);
}

}  // namespace fixtures

// ============================================================================
// Test 1: Factory Creation Test
// ============================================================================
TEST_CASE_METHOD(fixtures::HGraphDiskANNLoaderTestIndex,
                 "hgraph_diskann_loader factory test",
                 "[ft][hgraph_diskann_loader]") {
    // Use dims that can be divided by common pq_dims values
    const std::vector<int> dims = {64, 128, 256};
    auto metric_type = GENERATE("l2", "ip");
    const std::string name = "hgraph_diskann_loader";

    for (auto dim : dims) {
        // Use dim/4 as pq_dims to ensure it divides dim evenly
        auto param =
            GenerateHGraphDiskANNLoaderBuildParametersString(metric_type, dim, 16, dim / 4);
        auto index = TestFactory(name, param, true);
        REQUIRE(index != nullptr);
        REQUIRE(index->GetIndexType() == vsag::IndexType::HGRAPH_DISKANN_LOADER);
    }
}

// ============================================================================
// Test 2: Load from DiskANN BinarySet and Search Test
// ============================================================================
TEST_CASE_METHOD(fixtures::HGraphDiskANNLoaderTestIndex,
                 "hgraph_diskann_loader load from diskann binaryset",
                 "[ft][hgraph_diskann_loader]") {
    auto dims = fixtures::get_common_used_dims(1);
    auto metric_type = GENERATE("l2", "ip");
    const std::string diskann_name = "diskann";
    const std::string loader_name = "hgraph_diskann_loader";

    for (auto dim : dims) {
        // Step 1: Create and build DiskANN index
        auto diskann_param =
            GenerateDiskANNBuildParametersString(metric_type, dim, 16, 200, dim / 4);
        auto diskann_index = TestFactory(diskann_name, diskann_param, true);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
        TestBuildIndex(diskann_index, dataset, true);

        // Step 2: Serialize DiskANN index to BinarySet
        auto serialize_result = diskann_index->Serialize();
        REQUIRE(serialize_result.has_value());
        auto binary_set = serialize_result.value();

        // Step 3: Create HGraphDiskANNLoader and deserialize from BinarySet
        auto loader_param =
            GenerateHGraphDiskANNLoaderBuildParametersString(metric_type, dim, 16, dim / 4);
        auto loader_index = TestFactory(loader_name, loader_param, true);
        auto deserialize_result = loader_index->Deserialize(binary_set);
        REQUIRE(deserialize_result.has_value());

        // Step 4: Verify the number of elements
        REQUIRE(loader_index->GetNumElements() == base_count);

        // Step 5: Test KnnSearch
        auto search_param = fmt::format(search_param_template);
        TestKnnSearch(loader_index, dataset, search_param, 0.95, true);
    }
}

// ============================================================================
// Test 3: KnnSearch Recall Test with Different Parameters
// ============================================================================
TEST_CASE_METHOD(fixtures::HGraphDiskANNLoaderTestIndex,
                 "hgraph_diskann_loader knn search recall",
                 "[ft][hgraph_diskann_loader]") {
    const std::vector<int> dims = {64, 128, 256};
    auto metric_type = GENERATE("l2", "ip");
    const std::string diskann_name = "diskann";
    const std::string loader_name = "hgraph_diskann_loader";

    for (auto dim : dims) {
        // Create and build DiskANN index
        auto diskann_param =
            GenerateDiskANNBuildParametersString(metric_type, dim, 16, 200, dim / 4);
        auto diskann_index = TestFactory(diskann_name, diskann_param, true);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
        TestBuildIndex(diskann_index, dataset, true);

        // Serialize and load
        auto serialize_result = diskann_index->Serialize();
        REQUIRE(serialize_result.has_value());
        auto binary_set = serialize_result.value();
        auto loader_param =
            GenerateHGraphDiskANNLoaderBuildParametersString(metric_type, dim, 16, dim / 4);
        auto loader_index = TestFactory(loader_name, loader_param, true);
        auto deserialize_result = loader_index->Deserialize(binary_set);
        REQUIRE(deserialize_result.has_value());

        // Test KnnSearch with higher recall threshold
        auto search_param = fmt::format(search_param_template);
        TestKnnSearch(loader_index, dataset, search_param, 0.99, true);
    }
}

// ============================================================================
// Test 4: RangeSearch Test
// ============================================================================
TEST_CASE_METHOD(fixtures::HGraphDiskANNLoaderTestIndex,
                 "hgraph_diskann_loader range search",
                 "[ft][hgraph_diskann_loader]") {
    const std::vector<int> dims = {64, 128};
    auto metric_type = GENERATE("l2", "ip");
    const std::string diskann_name = "diskann";
    const std::string loader_name = "hgraph_diskann_loader";

    for (auto dim : dims) {
        // Create and build DiskANN index
        auto diskann_param =
            GenerateDiskANNBuildParametersString(metric_type, dim, 16, 200, dim / 4);
        auto diskann_index = TestFactory(diskann_name, diskann_param, true);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
        TestBuildIndex(diskann_index, dataset, true);

        // Serialize and load
        auto serialize_result1 = diskann_index->Serialize();
        REQUIRE(serialize_result1.has_value());
        auto binary_set = serialize_result1.value();
        auto loader_param =
            GenerateHGraphDiskANNLoaderBuildParametersString(metric_type, dim, 16, dim / 4);
        auto loader_index = TestFactory(loader_name, loader_param, true);
        auto deserialize_result1 = loader_index->Deserialize(binary_set);
        REQUIRE(deserialize_result1.has_value());

        // Test RangeSearch
        auto search_param = fmt::format(search_param_template);
        TestRangeSearch(loader_index, dataset, search_param, 0.95, 10, true);
        TestRangeSearch(loader_index, dataset, search_param, 0.49, 5, true);
    }
}

// ============================================================================
// Test 5: FilterSearch Test
// ============================================================================
TEST_CASE_METHOD(fixtures::HGraphDiskANNLoaderTestIndex,
                 "hgraph_diskann_loader filter search",
                 "[ft][hgraph_diskann_loader]") {
    const std::vector<int> dims = {64, 128};
    auto metric_type = GENERATE("l2", "ip");
    const std::string diskann_name = "diskann";
    const std::string loader_name = "hgraph_diskann_loader";

    for (auto dim : dims) {
        // Create and build DiskANN index
        auto diskann_param =
            GenerateDiskANNBuildParametersString(metric_type, dim, 16, 200, dim / 4);
        auto diskann_index = TestFactory(diskann_name, diskann_param, true);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
        TestBuildIndex(diskann_index, dataset, true);

        // Serialize and load
        auto serialize_result2 = diskann_index->Serialize();
        REQUIRE(serialize_result2.has_value());
        auto binary_set = serialize_result2.value();
        auto loader_param =
            GenerateHGraphDiskANNLoaderBuildParametersString(metric_type, dim, 16, dim / 4);
        auto loader_index = TestFactory(loader_name, loader_param, true);
        auto deserialize_result2 = loader_index->Deserialize(binary_set);
        REQUIRE(deserialize_result2.has_value());

        // Test FilterSearch
        auto search_param = fmt::format(search_param_template);
        TestFilterSearch(loader_index, dataset, search_param, 0.95, true);
    }
}

// ============================================================================
// Test 6: ReaderSet Loading Test
// ============================================================================
TEST_CASE_METHOD(fixtures::HGraphDiskANNLoaderTestIndex,
                 "hgraph_diskann_loader load from readerset",
                 "[ft][hgraph_diskann_loader]") {
    const std::vector<int> dims = {64, 128};
    auto metric_type = GENERATE("l2", "ip");
    const std::string diskann_name = "diskann";
    const std::string loader_name = "hgraph_diskann_loader";

    for (auto dim : dims) {
        // Create and build DiskANN index
        auto diskann_param =
            GenerateDiskANNBuildParametersString(metric_type, dim, 16, 200, dim / 4);
        auto diskann_index = TestFactory(diskann_name, diskann_param, true);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
        TestBuildIndex(diskann_index, dataset, true);

        // Serialize to BinarySet
        auto binary_set = diskann_index->Serialize().value();

        // Convert BinarySet to ReaderSet
        vsag::ReaderSet reader_set;
        for (const auto& key : binary_set.GetKeys()) {
            reader_set.Set(key, std::make_shared<fixtures::TestReader>(binary_set.Get(key)));
        }

        // Create HGraphDiskANNLoader and deserialize from ReaderSet
        auto loader_param =
            GenerateHGraphDiskANNLoaderBuildParametersString(metric_type, dim, 16, dim / 4);
        auto loader_index = TestFactory(loader_name, loader_param, true);
        auto deserialize_result = loader_index->Deserialize(reader_set);
        REQUIRE(deserialize_result.has_value());

        // Verify the number of elements
        REQUIRE(loader_index->GetNumElements() == base_count);

        // Test KnnSearch
        auto search_param = fmt::format(search_param_template);
        TestKnnSearch(loader_index, dataset, search_param, 0.95, true);
    }
}

// ============================================================================
// Test 7: Different Parameter Configurations Test
// ============================================================================
TEST_CASE_METHOD(fixtures::HGraphDiskANNLoaderTestIndex,
                 "hgraph_diskann_loader different parameters",
                 "[ft][hgraph_diskann_loader]") {
    auto dim = 128;
    auto metric_type = GENERATE("l2", "ip");
    auto max_degree = GENERATE(16, 32, 64);
    const std::string diskann_name = "diskann";
    const std::string loader_name = "hgraph_diskann_loader";

    // Create and build DiskANN index
    auto diskann_param =
        GenerateDiskANNBuildParametersString(metric_type, dim, max_degree, 200, dim / 4);
    auto diskann_index = TestFactory(diskann_name, diskann_param, true);
    auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
    TestBuildIndex(diskann_index, dataset, true);

    // Serialize and load
    auto serialize_result = diskann_index->Serialize();
    REQUIRE(serialize_result.has_value());
    auto binary_set = serialize_result.value();
    auto loader_param =
        GenerateHGraphDiskANNLoaderBuildParametersString(metric_type, dim, max_degree, dim / 4);
    auto loader_index = TestFactory(loader_name, loader_param, true);
    auto deserialize_result = loader_index->Deserialize(binary_set);
    REQUIRE(deserialize_result.has_value());

    // Test KnnSearch
    auto search_param = fmt::format(search_param_template);
    TestKnnSearch(loader_index, dataset, search_param, 0.95, true);
}

// ============================================================================
// Test 8: Special Dimensions Test
// ============================================================================
TEST_CASE_METHOD(fixtures::HGraphDiskANNLoaderTestIndex,
                 "hgraph_diskann_loader special dimensions",
                 "[ft][hgraph_diskann_loader]") {
    // Test with various dimensions including larger ones
    const std::vector<int> dims = {64, 128, 256, 512};
    auto metric_type = "l2";
    const std::string diskann_name = "diskann";
    const std::string loader_name = "hgraph_diskann_loader";

    for (auto dim : dims) {
        // Create and build DiskANN index
        auto diskann_param =
            GenerateDiskANNBuildParametersString(metric_type, dim, 16, 200, dim / 4);
        auto diskann_index = TestFactory(diskann_name, diskann_param, true);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
        TestBuildIndex(diskann_index, dataset, true);

        // Serialize and load
        auto serialize_result4 = diskann_index->Serialize();
        REQUIRE(serialize_result4.has_value());
        auto binary_set = serialize_result4.value();
        auto loader_param =
            GenerateHGraphDiskANNLoaderBuildParametersString(metric_type, dim, 16, dim / 4);
        auto loader_index = TestFactory(loader_name, loader_param, true);
        auto deserialize_result4 = loader_index->Deserialize(binary_set);
        REQUIRE(deserialize_result4.has_value());

        // Test KnnSearch
        auto search_param = fmt::format(search_param_template);
        TestKnnSearch(loader_index, dataset, search_param, 0.95, true);
    }
}

// ============================================================================
// Test 9: CalcDistanceById Test
// ============================================================================
TEST_CASE_METHOD(fixtures::HGraphDiskANNLoaderTestIndex,
                 "hgraph_diskann_loader calc distance by id",
                 "[ft][hgraph_diskann_loader]") {
    const std::vector<int> dims = {64, 128};
    auto metric_type = GENERATE("l2", "ip");
    const std::string diskann_name = "diskann";
    const std::string loader_name = "hgraph_diskann_loader";

    for (auto dim : dims) {
        // Create and build DiskANN index
        auto diskann_param =
            GenerateDiskANNBuildParametersString(metric_type, dim, 16, 200, dim / 4);
        auto diskann_index = TestFactory(diskann_name, diskann_param, true);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
        TestBuildIndex(diskann_index, dataset, true);

        // Serialize and load
        auto serialize_result5 = diskann_index->Serialize();
        REQUIRE(serialize_result5.has_value());
        auto binary_set = serialize_result5.value();
        auto loader_param =
            GenerateHGraphDiskANNLoaderBuildParametersString(metric_type, dim, 16, dim / 4);
        auto loader_index = TestFactory(loader_name, loader_param, true);
        auto deserialize_result5 = loader_index->Deserialize(binary_set);
        REQUIRE(deserialize_result5.has_value());

        // Test CalcDistanceById
        TestCalcDistanceById(loader_index, dataset, 1e-4, true);
    }
}

// ============================================================================
// Test 10: GetNumElements Test
// ============================================================================
TEST_CASE_METHOD(fixtures::HGraphDiskANNLoaderTestIndex,
                 "hgraph_diskann_loader get num elements",
                 "[ft][hgraph_diskann_loader]") {
    const std::vector<int> dims = {64, 128};
    auto metric_type = GENERATE("l2", "ip");
    const std::string diskann_name = "diskann";
    const std::string loader_name = "hgraph_diskann_loader";

    for (auto dim : dims) {
        // Create and build DiskANN index
        auto diskann_param =
            GenerateDiskANNBuildParametersString(metric_type, dim, 16, 200, dim / 4);
        auto diskann_index = TestFactory(diskann_name, diskann_param, true);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
        TestBuildIndex(diskann_index, dataset, true);

        // Verify DiskANN num elements
        REQUIRE(diskann_index->GetNumElements() == base_count);

        // Serialize and load
        auto serialize_result6 = diskann_index->Serialize();
        REQUIRE(serialize_result6.has_value());
        auto binary_set = serialize_result6.value();
        auto loader_param =
            GenerateHGraphDiskANNLoaderBuildParametersString(metric_type, dim, 16, dim / 4);
        auto loader_index = TestFactory(loader_name, loader_param, true);
        auto deserialize_result6 = loader_index->Deserialize(binary_set);
        REQUIRE(deserialize_result6.has_value());

        // Verify HGraphDiskANNLoader num elements matches DiskANN
        REQUIRE(loader_index->GetNumElements() == diskann_index->GetNumElements());
        REQUIRE(loader_index->GetNumElements() == base_count);
    }
}