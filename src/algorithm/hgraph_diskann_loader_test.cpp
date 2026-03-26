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

#include "hgraph_diskann_loader.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <fstream>
#include <memory>
#include <sstream>

#include "fixtures.h"
#include "hgraph_parameter.h"
#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
#include "math_utils.h"
#include "pq.h"
#include "quantization/product_quantization/product_quantizer.h"

using namespace vsag;

// ============================================================================
// Helper Functions
// ============================================================================

// Generate DiskANN format PQ data
void
GenerateDiskANNPQData(const std::vector<float>& vectors,
                      uint32_t dim,
                      uint32_t num_pq_chunks,
                      std::stringstream& pq_pivots_stream,
                      std::stringstream& compressed_stream) {
    uint64_t num_vectors = vectors.size() / dim;
    uint32_t num_centers = 256;
    uint32_t max_k_means_reps = 12;

    int result = ::diskann::generate_pq_pivots(vectors.data(),
                                               num_vectors,
                                               dim,
                                               num_centers,
                                               num_pq_chunks,
                                               max_k_means_reps,
                                               pq_pivots_stream,
                                               false);

    REQUIRE(result == 0);
    pq_pivots_stream.seekg(0, std::ios::beg);

    std::vector<uint64_t> skip_locs;
    result = ::diskann::generate_pq_data_from_pivots<float>(vectors.data(),
                                                            num_vectors,
                                                            dim,
                                                            skip_locs,
                                                            num_centers,
                                                            num_pq_chunks,
                                                            pq_pivots_stream,
                                                            compressed_stream,
                                                            false,
                                                            nullptr,
                                                            false);

    REQUIRE(result == 0);
    pq_pivots_stream.seekg(0, std::ios::beg);
    compressed_stream.seekg(0, std::ios::beg);
}

// Create HGraphDiskANNLoader for PQ testing
std::shared_ptr<HGraphDiskANNLoader>
CreatePQTestLoader(uint32_t dim,
                   uint32_t pq_dim,
                   MetricType metric,
                   std::shared_ptr<Allocator>& out_allocator) {
    out_allocator = SafeAllocator::FactoryDefaultAllocator();

    auto param = std::make_shared<HGraphParameter>();
    std::string param_str = fmt::format(R"({{
        "type": "hgraph_diskann_loader",
        "base_codes": {{
            "codes_type": "flatten_codes",
            "io_params": {{ "type": "memory_io" }},
            "quantization_params": {{ "type": "pq", "pq_dim": {} }}
        }},
        "graph": {{
            "graph_storage_type": "flat",
            "io_params": {{ "type": "memory_io" }},
            "max_degree": 26
        }},
        "precise_codes": {{
            "codes_type": "flatten_codes",
            "io_params": {{ "type": "memory_io" }},
            "quantization_params": {{ "type": "fp32" }}
        }},
        "use_reorder": false
    }})",
                                        pq_dim);
    param->FromString(param_str);

    IndexCommonParam common_param;
    common_param.metric_ = metric;
    common_param.dim_ = dim;
    common_param.allocator_ = out_allocator;

    return std::make_shared<HGraphDiskANNLoader>(param, common_param);
}

// Verify that codes match byte-by-byte
void
VerifyCodesMatch(FlattenInterfacePtr& flatten_codes,
                 const std::vector<uint8_t>& expected_codes,
                 uint32_t num_vectors,
                 uint32_t pq_dim) {
    for (uint32_t i = 0; i < num_vectors; ++i) {
        bool need_release = false;
        const uint8_t* vsag_codes =
            flatten_codes->GetCodesById(static_cast<InnerIdType>(i), need_release);
        REQUIRE(vsag_codes != nullptr);

        const uint8_t* original_codes = expected_codes.data() + i * pq_dim;
        for (uint32_t j = 0; j < pq_dim; ++j) {
            REQUIRE(vsag_codes[j] == original_codes[j]);
        }

        if (need_release) {
            flatten_codes->Release(vsag_codes);
        }
    }
}

// Read original DiskANN codes from stream
std::vector<uint8_t>
ReadDiskANNCodes(std::stringstream& compressed_stream, uint32_t num_vectors, uint32_t pq_dim) {
    compressed_stream.seekg(0, std::ios::beg);
    int32_t file_npts = 0, file_nchunks = 0;
    compressed_stream.read(reinterpret_cast<char*>(&file_npts), sizeof(int32_t));
    compressed_stream.read(reinterpret_cast<char*>(&file_nchunks), sizeof(int32_t));

    REQUIRE(file_npts == static_cast<int32_t>(num_vectors));
    REQUIRE(file_nchunks == static_cast<int32_t>(pq_dim));

    std::vector<uint8_t> codes(num_vectors * pq_dim);
    compressed_stream.read(reinterpret_cast<char*>(codes.data()), num_vectors * pq_dim);
    return codes;
}

// ============================================================================
// Graph Helper Functions
// ============================================================================

// Generate a test graph with specified topology
std::vector<std::vector<uint32_t>>
GenerateTestGraph(uint64_t num_nodes, uint64_t max_degree, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> dist(0, static_cast<uint32_t>(num_nodes - 1));
    uint64_t effective_max_degree = std::min(max_degree, num_nodes > 0 ? num_nodes - 1 : 0);
    std::uniform_int_distribution<uint32_t> degree_dist(
        1, static_cast<uint32_t>(effective_max_degree));

    std::vector<std::vector<uint32_t>> neighbors(num_nodes);

    for (uint64_t i = 0; i < num_nodes; ++i) {
        uint32_t degree = degree_dist(rng);
        std::unordered_set<uint32_t> neighbor_set;

        while (neighbor_set.size() < degree) {
            uint32_t neighbor = dist(rng);
            if (neighbor != i) {
                neighbor_set.insert(neighbor);
            }
        }

        neighbors[i].assign(neighbor_set.begin(), neighbor_set.end());
    }

    return neighbors;
}

// Serialize graph to DiskANN format
void
SerializeDiskANNGraph(const std::vector<std::vector<uint32_t>>& neighbors,
                      uint32_t entry_point,
                      uint64_t num_frozen_pts,
                      std::stringstream& out) {
    uint64_t num_nodes = neighbors.size();
    uint32_t max_observed_degree = 0;

    uint64_t index_size = 24;
    for (const auto& node_neighbors : neighbors) {
        uint32_t degree = static_cast<uint32_t>(node_neighbors.size());
        index_size += sizeof(uint32_t) * (1 + degree);
        max_observed_degree = std::max(max_observed_degree, degree);
    }

    out.write(reinterpret_cast<const char*>(&index_size), sizeof(index_size));
    out.write(reinterpret_cast<const char*>(&max_observed_degree), sizeof(max_observed_degree));
    out.write(reinterpret_cast<const char*>(&entry_point), sizeof(entry_point));
    out.write(reinterpret_cast<const char*>(&num_frozen_pts), sizeof(num_frozen_pts));

    for (const auto& node_neighbors : neighbors) {
        uint32_t GK = static_cast<uint32_t>(node_neighbors.size());
        out.write(reinterpret_cast<const char*>(&GK), sizeof(GK));
        if (GK > 0) {
            out.write(reinterpret_cast<const char*>(node_neighbors.data()), GK * sizeof(uint32_t));
        }
    }
}

// Create loader for graph testing
std::shared_ptr<HGraphDiskANNLoader>
CreateGraphTestLoader(uint64_t max_degree, std::shared_ptr<Allocator>& out_allocator) {
    out_allocator = SafeAllocator::FactoryDefaultAllocator();

    auto param = std::make_shared<HGraphParameter>();
    std::string param_str = fmt::format(R"({{
        "type": "hgraph_diskann_loader",
        "base_codes": {{
            "codes_type": "flatten_codes",
            "io_params": {{ "type": "memory_io" }},
            "quantization_params": {{ "type": "fp32" }}
        }},
        "graph": {{
            "graph_storage_type": "flat",
            "io_params": {{ "type": "memory_io" }},
            "max_degree": {}
        }},
        "precise_codes": {{
            "codes_type": "flatten_codes",
            "io_params": {{ "type": "memory_io" }},
            "quantization_params": {{ "type": "fp32" }}
        }},
        "use_reorder": false
    }})",
                                        max_degree);
    param->FromString(param_str);

    IndexCommonParam common_param;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common_param.dim_ = 128;
    common_param.allocator_ = out_allocator;

    return std::make_shared<HGraphDiskANNLoader>(param, common_param);
}

// Compare graphs for equality
bool
CompareGraphsEqual(const std::vector<std::vector<uint32_t>>& expected,
                   const std::vector<std::vector<InnerIdType>>& actual) {
    if (expected.size() != actual.size()) {
        return false;
    }

    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i].size() != actual[i].size()) {
            return false;
        }

        if (expected[i].empty()) {
            continue;
        }

        std::unordered_set<uint32_t> expected_set(expected[i].begin(), expected[i].end());
        std::unordered_set<uint32_t> actual_set;
        for (const auto& id : actual[i]) {
            actual_set.insert(static_cast<uint32_t>(id));
        }
        if (expected_set != actual_set) {
            return false;
        }
    }

    return true;
}

// Read HGraph structure
std::vector<std::vector<InnerIdType>>
ReadHGraphStructure(const GraphInterfacePtr& graph, uint64_t num_nodes) {
    std::vector<std::vector<InnerIdType>> result(num_nodes);
    auto safe_allocator = SafeAllocator::FactoryDefaultAllocator();

    for (uint64_t i = 0; i < num_nodes; ++i) {
        uint32_t neighbor_size = graph->GetNeighborSize(static_cast<InnerIdType>(i));
        if (neighbor_size > 0) {
            Vector<InnerIdType> temp_neighbors(safe_allocator.get());
            graph->GetNeighbors(static_cast<InnerIdType>(i), temp_neighbors);
            result[i].reserve(temp_neighbors.size());
            for (auto& n : temp_neighbors) {
                result[i].push_back(n);
            }
        }
    }

    return result;
}

// ============================================================================
// Precise Vectors Helper Functions
// ============================================================================

// Create DiskANN layout file for precise vectors testing
std::stringstream
CreateDiskANNLayoutFile(const std::vector<float>& vectors,
                        const std::vector<std::vector<uint32_t>>& graph,
                        uint32_t max_degree,
                        uint32_t entry_point) {
    std::stringstream layout_stream;
    uint64_t num_points = graph.size();
    uint64_t dim = vectors.size() / num_points;

    uint64_t max_node_len = dim * sizeof(float) + (max_degree * 1 + 1) * sizeof(uint32_t);
    uint64_t sector_len =
        std::max(static_cast<uint64_t>(4096), (max_node_len + 4096 - 1) & ~(4096 - 1));
    uint64_t nnodes_per_sector = sector_len / max_node_len;
    uint64_t n_sectors = (num_points + nnodes_per_sector - 1) / nnodes_per_sector;

    std::vector<char> sector_buf(sector_len, 0);
    std::vector<char> node_buf(max_node_len, 0);

    // Write Sector 0: Metadata
    layout_stream.write(sector_buf.data(), sector_len);

    // Write data sectors
    for (uint64_t sector = 0; sector < n_sectors; ++sector) {
        std::fill(sector_buf.begin(), sector_buf.end(), 0);

        for (uint64_t sector_node_id = 0; sector_node_id < nnodes_per_sector; ++sector_node_id) {
            uint64_t node_id = sector * nnodes_per_sector + sector_node_id;
            if (node_id >= num_points) {
                break;
            }

            std::fill(node_buf.begin(), node_buf.end(), 0);

            const float* vec_data = vectors.data() + node_id * dim;
            std::memcpy(node_buf.data(), vec_data, dim * sizeof(float));

            uint32_t num_neighbors = static_cast<uint32_t>(graph[node_id].size());
            num_neighbors = std::min(num_neighbors, max_degree);
            std::memcpy(node_buf.data() + dim * sizeof(float), &num_neighbors, sizeof(uint32_t));

            if (num_neighbors > 0) {
                std::memcpy(node_buf.data() + dim * sizeof(float) + sizeof(uint32_t),
                            graph[node_id].data(),
                            num_neighbors * sizeof(uint32_t));
            }

            char* sector_node_buf = sector_buf.data() + (sector_node_id * max_node_len);
            std::memcpy(sector_node_buf, node_buf.data(), max_node_len);
        }

        layout_stream.write(sector_buf.data(), sector_len);
    }

    layout_stream.seekg(0, std::ios::beg);
    return layout_stream;
}

// Create loader with reorder enabled
std::shared_ptr<HGraphDiskANNLoader>
CreateReorderTestLoader(uint64_t max_degree,
                        uint64_t dim,
                        std::shared_ptr<Allocator>& out_allocator) {
    out_allocator = SafeAllocator::FactoryDefaultAllocator();

    auto param = std::make_shared<HGraphParameter>();
    std::string param_str = fmt::format(R"({{
        "type": "hgraph_diskann_loader",
        "base_codes": {{
            "codes_type": "flatten_codes",
            "io_params": {{ "type": "memory_io" }},
            "quantization_params": {{ "type": "fp32" }}
        }},
        "graph": {{
            "graph_storage_type": "flat",
            "io_params": {{ "type": "memory_io" }},
            "max_degree": {}
        }},
        "precise_codes": {{
            "codes_type": "flatten_codes",
            "io_params": {{ "type": "memory_io" }},
            "quantization_params": {{ "type": "fp32" }}
        }},
        "use_reorder": true
    }})",
                                        max_degree);
    param->FromString(param_str);

    IndexCommonParam common_param;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common_param.dim_ = dim;
    common_param.allocator_ = out_allocator;

    auto loader = std::make_shared<HGraphDiskANNLoader>(param, common_param);
    loader->use_reorder_ = true;
    return loader;
}

// Get vector from precise codes
std::vector<float>
GetVectorFromPreciseCodes(FlattenInterfacePtr& precise_codes, InnerIdType inner_id, uint32_t dim) {
    bool need_release = false;
    const uint8_t* codes = precise_codes->GetCodesById(inner_id, need_release);
    REQUIRE(codes != nullptr);

    std::vector<float> vec(dim);
    bool decode_result = precise_codes->Decode(codes, vec.data());
    REQUIRE(decode_result);

    if (need_release) {
        precise_codes->Release(codes);
    }

    return vec;
}

// Compute L2 distance
float
ComputeL2Distance(const float* vec1, const float* vec2, uint32_t dim) {
    float dist = 0.0f;
    for (uint32_t d = 0; d < dim; ++d) {
        float diff = vec1[d] - vec2[d];
        dist += diff * diff;
    }
    return std::sqrt(dist);
}

// Verify all precise vectors match
void
VerifyPreciseVectors(FlattenInterfacePtr& precise_codes,
                     const std::vector<float>& original_vectors,
                     uint32_t num_points,
                     uint32_t dim) {
    for (uint32_t i = 0; i < num_points; ++i) {
        auto loaded_vector = GetVectorFromPreciseCodes(precise_codes, i, dim);
        const float* original_vector = original_vectors.data() + i * dim;
        float l2_dist = ComputeL2Distance(loaded_vector.data(), original_vector, dim);
        REQUIRE(l2_dist == 0.0f);
    }
}

// ============================================================================
// PQ Distance Consistency Tests
// ============================================================================

TEST_CASE("HGraphDiskANNLoader PQ Distance Consistency", "[ut][hgraph_diskann_loader]") {
    auto dim = GENERATE(64u, 128u);
    auto pq_dim = GENERATE(8u, 16u);
    auto metric_type = GENERATE(MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_IP);

    const uint32_t num_vectors = 100;
    const uint32_t num_queries = 10;

    INFO("Testing with dim=" << dim << ", pq_dim=" << pq_dim
                             << ", metric=" << static_cast<int>(metric_type));

    // Generate test data
    auto vectors = fixtures::generate_vectors(num_vectors, dim, true, 42);
    auto queries = fixtures::generate_vectors(num_queries, dim, true, 123);

    // Generate DiskANN PQ data
    std::stringstream pq_pivots_stream;
    std::stringstream compressed_stream;
    GenerateDiskANNPQData(vectors, dim, pq_dim, pq_pivots_stream, compressed_stream);

    // Read original codes
    auto diskann_codes = ReadDiskANNCodes(compressed_stream, num_vectors, pq_dim);

    // Reset streams
    pq_pivots_stream.seekg(0, std::ios::beg);
    compressed_stream.seekg(0, std::ios::beg);

    // Create DiskANN reference implementation
    ::diskann::FixedChunkPQTable diskann_pq_table;
    diskann_pq_table.load_pq_centroid_bin(pq_pivots_stream, pq_dim);

    // Create VSAG loader
    std::shared_ptr<Allocator> allocator;
    auto loader = CreatePQTestLoader(dim, pq_dim, metric_type, allocator);

    pq_pivots_stream.seekg(0, std::ios::beg);
    compressed_stream.seekg(0, std::ios::beg);
    loader->load_diskann_pq_data(pq_pivots_stream, compressed_stream, num_vectors);

    // Verify codes
    auto flatten_codes = loader->get_basic_flatten_codes();
    REQUIRE(flatten_codes != nullptr);
    VerifyCodesMatch(flatten_codes, diskann_codes, num_vectors, pq_dim);

    // Verify distances
    void* quantizer_ptr = flatten_codes->GetQuantizer();
    REQUIRE(quantizer_ptr != nullptr);

    for (uint32_t q = 0; q < num_queries; ++q) {
        const float* query_vec = queries.data() + q * dim;

        std::vector<float> diskann_query_copy(query_vec, query_vec + dim);
        diskann_pq_table.preprocess_query(diskann_query_copy.data());

        for (uint32_t i = 0; i < num_vectors; ++i) {
            const uint8_t* diskann_code = diskann_codes.data() + i * pq_dim;

            bool need_release = false;
            const uint8_t* vsag_code =
                flatten_codes->GetCodesById(static_cast<InnerIdType>(i), need_release);
            REQUIRE(vsag_code != nullptr);

            float vsag_dist;
            float expected_dist;

            if (metric_type == MetricType::METRIC_TYPE_L2SQR) {
                auto* vsag_pq =
                    static_cast<ProductQuantizer<MetricType::METRIC_TYPE_L2SQR>*>(quantizer_ptr);
                auto computer = vsag_pq->FactoryComputer();
                vsag_pq->ProcessQuery(query_vec, *computer);
                vsag_dist = vsag_pq->ComputeDist(*computer, vsag_code);

                float diskann_dist = diskann_pq_table.l2_distance(
                    diskann_query_copy.data(), const_cast<uint8_t*>(diskann_code));
                expected_dist = diskann_dist;
            } else {
                auto* vsag_pq =
                    static_cast<ProductQuantizer<MetricType::METRIC_TYPE_IP>*>(quantizer_ptr);
                auto computer = vsag_pq->FactoryComputer();
                vsag_pq->ProcessQuery(query_vec, *computer);
                vsag_dist = vsag_pq->ComputeDist(*computer, vsag_code);

                float diskann_ip_dist = diskann_pq_table.inner_product(
                    diskann_query_copy.data(), const_cast<uint8_t*>(diskann_code));
                // Convert: DiskANN returns -ip, VSAG returns 1-ip
                expected_dist = 1.0f + diskann_ip_dist;
            }

            if (need_release) {
                flatten_codes->Release(vsag_code);
            }

            REQUIRE(fixtures::dist_t(expected_dist) == vsag_dist);
        }
    }
}

TEST_CASE("HGraphDiskANNLoader PQ Codebook and Dimensions", "[ut][hgraph_diskann_loader]") {
    auto dim = GENERATE(64u, 128u, 256u);
    auto pq_dim = GENERATE(8u, 16u, 32u);

    // Skip invalid combinations
    if (dim % pq_dim != 0) {
        return;
    }

    const uint32_t num_vectors = 50;

    INFO("Testing codebook with dim=" << dim << ", pq_dim=" << pq_dim);

    auto vectors = fixtures::generate_vectors(num_vectors, dim, true, 42);

    std::stringstream pq_pivots_stream;
    std::stringstream compressed_stream;
    GenerateDiskANNPQData(vectors, dim, pq_dim, pq_pivots_stream, compressed_stream);

    std::shared_ptr<Allocator> allocator;
    auto loader = CreatePQTestLoader(dim, pq_dim, MetricType::METRIC_TYPE_L2SQR, allocator);

    pq_pivots_stream.seekg(0, std::ios::beg);
    compressed_stream.seekg(0, std::ios::beg);
    loader->load_diskann_pq_data(pq_pivots_stream, compressed_stream, num_vectors);

    auto flatten_codes = loader->get_basic_flatten_codes();
    REQUIRE(flatten_codes != nullptr);

    void* quantizer_ptr = flatten_codes->GetQuantizer();
    REQUIRE(quantizer_ptr != nullptr);

    auto* vsag_pq = static_cast<ProductQuantizer<MetricType::METRIC_TYPE_L2SQR>*>(quantizer_ptr);

    // Verify dimensions
    REQUIRE(vsag_pq->pq_dim_ == static_cast<int64_t>(pq_dim));
    REQUIRE(vsag_pq->subspace_dim_ == static_cast<int64_t>(dim / pq_dim));

    // Verify codebook size
    size_t expected_codebook_size = pq_dim * 256 * (dim / pq_dim);
    REQUIRE(vsag_pq->codebooks_.size() == expected_codebook_size);

    // Verify decoding works
    auto diskann_codes = ReadDiskANNCodes(compressed_stream, num_vectors, pq_dim);

    for (uint32_t i = 0; i < num_vectors; ++i) {
        bool need_release = false;
        const uint8_t* codes =
            flatten_codes->GetCodesById(static_cast<InnerIdType>(i), need_release);
        REQUIRE(codes != nullptr);

        std::vector<float> decoded_vec(dim);
        REQUIRE(flatten_codes->Decode(codes, decoded_vec.data()));

        if (need_release) {
            flatten_codes->Release(codes);
        }

        // Check quantization error is reasonable
        const float* original_vec = vectors.data() + i * dim;
        float error_sum = 0.0f;
        for (uint32_t d = 0; d < dim; ++d) {
            error_sum += std::abs(original_vec[d] - decoded_vec[d]);
        }
        REQUIRE(error_sum / dim < 0.5f);
    }
}

// ============================================================================
// Graph Structure Tests
// ============================================================================

TEST_CASE("HGraphDiskANNLoader Graph Structure Conversion", "[ut][hgraph_diskann_loader]") {
    auto num_nodes = GENERATE(10u, 50u, 100u);
    auto max_degree = GENERATE(8u, 16u, 32u);
    const uint32_t entry_point = 0;

    INFO("Testing graph with " << num_nodes << " nodes, max_degree=" << max_degree);

    auto neighbors = GenerateTestGraph(num_nodes, max_degree, 42);

    std::stringstream graph_stream;
    SerializeDiskANNGraph(neighbors, entry_point, 0, graph_stream);

    std::shared_ptr<Allocator> allocator;
    auto loader = CreateGraphTestLoader(max_degree, allocator);
    graph_stream.seekg(0, std::ios::beg);
    loader->load_diskann_graph(graph_stream, num_nodes);

    auto hgraph = loader->get_bottom_graph();
    REQUIRE(hgraph != nullptr);
    REQUIRE(hgraph->TotalCount() == static_cast<InnerIdType>(num_nodes));
    REQUIRE(loader->get_entry_point_id() == static_cast<InnerIdType>(entry_point));

    // Verify structure
    auto hgraph_structure = ReadHGraphStructure(hgraph, num_nodes);
    REQUIRE(CompareGraphsEqual(neighbors, hgraph_structure));
}

TEST_CASE("HGraphDiskANNLoader Graph Topologies", "[ut][hgraph_diskann_loader]") {
    const uint64_t num_nodes = 50;
    const uint64_t max_degree = 64;             // Increased to accommodate star topology
    auto topology_type = GENERATE(0, 1, 2, 3);  // 0=chain, 1=ring, 2=star, 3=complete

    std::vector<std::vector<uint32_t>> neighbors(num_nodes);

    switch (topology_type) {
        case 0: {  // Chain
            for (uint64_t i = 0; i < num_nodes; ++i) {
                if (i > 0)
                    neighbors[i].push_back(static_cast<uint32_t>(i - 1));
                if (i < num_nodes - 1)
                    neighbors[i].push_back(static_cast<uint32_t>(i + 1));
            }
            break;
        }
        case 1: {  // Ring
            for (uint64_t i = 0; i < num_nodes; ++i) {
                neighbors[i].push_back(static_cast<uint32_t>((i + num_nodes - 1) % num_nodes));
                neighbors[i].push_back(static_cast<uint32_t>((i + 1) % num_nodes));
            }
            break;
        }
        case 2: {  // Star
            for (uint64_t i = 1; i < num_nodes; ++i) {
                neighbors[0].push_back(static_cast<uint32_t>(i));
                neighbors[i].push_back(0);
            }
            break;
        }
        case 3: {  // Complete (smaller for practicality)
            for (uint64_t i = 0; i < std::min(num_nodes, 10ul); ++i) {
                for (uint64_t j = 0; j < std::min(num_nodes, 10ul); ++j) {
                    if (i != j) {
                        neighbors[i].push_back(static_cast<uint32_t>(j));
                    }
                }
            }
            break;
        }
    }

    INFO("Testing topology type " << topology_type);

    std::stringstream graph_stream;
    SerializeDiskANNGraph(neighbors, 0, 0, graph_stream);

    std::shared_ptr<Allocator> allocator;
    auto loader = CreateGraphTestLoader(max_degree, allocator);
    graph_stream.seekg(0, std::ios::beg);
    loader->load_diskann_graph(graph_stream, num_nodes);

    auto hgraph = loader->get_bottom_graph();
    REQUIRE(hgraph != nullptr);

    // Verify neighbor counts
    for (uint64_t i = 0; i < num_nodes; ++i) {
        REQUIRE(hgraph->GetNeighborSize(static_cast<InnerIdType>(i)) == neighbors[i].size());
    }
}

TEST_CASE("HGraphDiskANNLoader Graph Degree Truncation", "[ut][hgraph_diskann_loader]") {
    const uint64_t num_nodes = 50;
    auto actual_degree = GENERATE(32u, 64u);
    auto loader_degree = GENERATE(8u, 16u);

    INFO("Testing truncation: actual=" << actual_degree << ", loader=" << loader_degree);

    auto neighbors = GenerateTestGraph(num_nodes, actual_degree, 42);

    std::stringstream graph_stream;
    SerializeDiskANNGraph(neighbors, 0, 0, graph_stream);

    std::shared_ptr<Allocator> allocator;
    auto loader = CreateGraphTestLoader(loader_degree, allocator);
    graph_stream.seekg(0, std::ios::beg);
    loader->load_diskann_graph(graph_stream, num_nodes);

    auto hgraph = loader->get_bottom_graph();
    REQUIRE(hgraph != nullptr);

    // Verify truncation
    for (uint64_t i = 0; i < num_nodes; ++i) {
        REQUIRE(hgraph->GetNeighborSize(static_cast<InnerIdType>(i)) <= loader_degree);
    }
}

TEST_CASE("HGraphDiskANNLoader Graph Edge Cases", "[ut][hgraph_diskann_loader]") {
    auto edge_case = GENERATE(0, 1, 2, 3);  // 0=empty, 1=single, 2=zero_degree, 3=frozen

    std::vector<std::vector<uint32_t>> neighbors;
    uint32_t entry_point = 0;
    uint64_t num_frozen = 0;
    uint64_t expected_nodes = 0;
    uint64_t max_degree = 16;

    switch (edge_case) {
        case 0: {  // Empty graph
            expected_nodes = 0;
            break;
        }
        case 1: {  // Single node
            neighbors.resize(1);
            expected_nodes = 1;
            break;
        }
        case 2: {  // Zero degree nodes
            neighbors = GenerateTestGraph(20, max_degree, 42);
            neighbors[5].clear();
            neighbors[10].clear();
            neighbors[15].clear();
            expected_nodes = 20;
            break;
        }
        case 3: {  // Frozen points
            num_frozen = 5;
            expected_nodes = 50 + num_frozen;
            neighbors = GenerateTestGraph(expected_nodes, max_degree, 42);
            break;
        }
    }

    INFO("Testing edge case " << edge_case);

    std::stringstream graph_stream;
    SerializeDiskANNGraph(neighbors, entry_point, num_frozen, graph_stream);

    std::shared_ptr<Allocator> allocator;
    auto loader = CreateGraphTestLoader(max_degree, allocator);
    graph_stream.seekg(0, std::ios::beg);
    loader->load_diskann_graph(graph_stream, expected_nodes);

    auto hgraph = loader->get_bottom_graph();
    REQUIRE(hgraph != nullptr);
    REQUIRE(hgraph->TotalCount() == static_cast<InnerIdType>(expected_nodes));

    if (edge_case == 2) {
        // Verify zero-degree nodes
        REQUIRE(hgraph->GetNeighborSize(5) == 0);
        REQUIRE(hgraph->GetNeighborSize(10) == 0);
        REQUIRE(hgraph->GetNeighborSize(15) == 0);
    }

    if (edge_case == 1) {
        REQUIRE(hgraph->GetNeighborSize(0) == 0);
        REQUIRE(loader->get_entry_point_id() == 0);
    }
}

// ============================================================================
// Precise Vectors Tests
// ============================================================================

TEST_CASE("HGraphDiskANNLoader Precise Vectors Loading",
          "[ut][hgraph_diskann_loader][precise_vectors]") {
    auto dim = GENERATE(32u, 64u, 128u, 256u);
    auto num_points = GENERATE(1u, 50u, 100u);
    const uint32_t max_degree = 16;

    INFO("Testing precise vectors with dim=" << dim << ", num_points=" << num_points);

    auto vectors = fixtures::generate_vectors(num_points, dim, true, 42);

    // Create simple chain graph
    std::vector<std::vector<uint32_t>> graph(num_points);
    for (uint32_t i = 0; i < num_points; ++i) {
        if (i > 0)
            graph[i].push_back(i - 1);
        if (i < num_points - 1)
            graph[i].push_back(i + 1);
    }

    auto layout_stream = CreateDiskANNLayoutFile(vectors, graph, max_degree, 0);

    std::shared_ptr<Allocator> allocator;
    auto loader = CreateReorderTestLoader(max_degree, dim, allocator);

    layout_stream.seekg(0, std::ios::beg);
    loader->load_diskann_precise_vectors(layout_stream, num_points, dim, max_degree);

    auto precise_codes = loader->get_high_precise_codes();
    REQUIRE(precise_codes != nullptr);

    VerifyPreciseVectors(precise_codes, vectors, num_points, dim);
}

TEST_CASE("HGraphDiskANNLoader Precise Vectors Sector Boundary",
          "[ut][hgraph_diskann_loader][precise_vectors]") {
    const uint32_t dim = 64;
    const uint32_t max_degree = 16;

    uint64_t max_node_len = dim * sizeof(float) + (max_degree + 1) * sizeof(uint32_t);
    uint64_t sector_len = 4096;
    uint64_t nnodes_per_sector = sector_len / max_node_len;
    const uint32_t num_points = static_cast<uint32_t>(nnodes_per_sector * 3 + 5);

    INFO("Testing sector boundary with " << num_points << " points");

    auto vectors = fixtures::generate_vectors(num_points, dim, true, 42);

    std::vector<std::vector<uint32_t>> graph(num_points);
    for (uint32_t i = 0; i < num_points; ++i) {
        if (i > 0)
            graph[i].push_back(i - 1);
        if (i < num_points - 1)
            graph[i].push_back(i + 1);
    }

    auto layout_stream = CreateDiskANNLayoutFile(vectors, graph, max_degree, 0);

    std::shared_ptr<Allocator> allocator;
    auto loader = CreateReorderTestLoader(max_degree, dim, allocator);

    layout_stream.seekg(0, std::ios::beg);
    loader->load_diskann_precise_vectors(layout_stream, num_points, dim, max_degree);

    auto precise_codes = loader->get_high_precise_codes();
    REQUIRE(precise_codes != nullptr);

    // Verify boundary nodes
    std::vector<uint32_t> boundary_nodes = {0,
                                            static_cast<uint32_t>(nnodes_per_sector - 1),
                                            static_cast<uint32_t>(nnodes_per_sector),
                                            static_cast<uint32_t>(nnodes_per_sector * 2 - 1),
                                            static_cast<uint32_t>(nnodes_per_sector * 2),
                                            num_points - 1};

    for (auto i : boundary_nodes) {
        if (i < num_points) {
            auto loaded_vector = GetVectorFromPreciseCodes(precise_codes, i, dim);
            const float* original_vector = vectors.data() + i * dim;
            REQUIRE(ComputeL2Distance(loaded_vector.data(), original_vector, dim) == 0.0f);
        }
    }
}

TEST_CASE("HGraphDiskANNLoader Precise Vectors Large Degree",
          "[ut][hgraph_diskann_loader][precise_vectors]") {
    const uint32_t dim = 128;
    const uint32_t num_points = 100;
    const uint32_t max_degree = 64;

    auto vectors = fixtures::generate_vectors(num_points, dim, true, 42);

    std::vector<std::vector<uint32_t>> graph(num_points);
    for (uint32_t i = 0; i < num_points; ++i) {
        for (uint32_t j = 1; j <= max_degree / 2 && j < num_points; ++j) {
            graph[i].push_back((i + j) % num_points);
        }
    }

    auto layout_stream = CreateDiskANNLayoutFile(vectors, graph, max_degree, 0);

    std::shared_ptr<Allocator> allocator;
    auto loader = CreateReorderTestLoader(max_degree, dim, allocator);

    layout_stream.seekg(0, std::ios::beg);
    loader->load_diskann_precise_vectors(layout_stream, num_points, dim, max_degree);

    auto precise_codes = loader->get_high_precise_codes();
    REQUIRE(precise_codes != nullptr);

    VerifyPreciseVectors(precise_codes, vectors, num_points, dim);
}
