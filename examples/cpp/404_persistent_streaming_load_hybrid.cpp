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

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "vsag/options.h"

namespace {

constexpr int64_t K_DIM = 128;
constexpr int64_t K_NUM_VECTORS = 500;
constexpr int64_t K_TOP_K = 10;
constexpr uint64_t K_DB_HEADER_SIZE = 4096;
constexpr uint64_t K_DB_FOOTER_SIZE = 128;
constexpr const char* K_DB_FILE_PATH = "/tmp/vsag-streaming-load-hybrid.index";

// Build parameters describe the index algorithm and encoding only. They no longer need
// placement keys such as base_io_type, precise_io_type, or their file paths.
// Those keys belong to Index::Load(), where callers choose the serving layout.
const char* hybrid_hgraph_build_parameters = R"(
{
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "rabitq",
        "precise_quantization_type": "sq8",
        "use_reorder": true,
        "rabitq_bits_per_dim_base": 3,
        "rabitq_bits_per_dim_query": 32,
        "rabitq_error_rate": 1.9,
        "max_degree": 32,
        "ef_construction": 200,
        "graph_storage_type": "compressed"
    }
}
)";

// Load parameters choose where each serialized component is placed for serving.
// This example keeps RaBitQ base codes in memory and serves SQ8 reorder codes
// through a read_func reader. The build parameters stay independent from the
// final memory/disk placement.

const char* hgraph_search_parameters = R"(
{
    "hgraph": {
        "ef_search": 200,
        "rabitq_one_bit_search": true
    }
}
)";

void
cleanup_hybrid_files() {
    std::remove(K_DB_FILE_PATH);
}

vsag::DatasetPtr
make_base_dataset(std::vector<int64_t>& ids, std::vector<float>& vectors) {
    ids.resize(K_NUM_VECTORS);
    vectors.resize(K_NUM_VECTORS * K_DIM);

    std::mt19937 rng(47);
    std::uniform_real_distribution<float> dist;
    for (int64_t i = 0; i < K_NUM_VECTORS; ++i) {
        ids[i] = i;
    }
    for (auto& value : vectors) {
        value = dist(rng);
    }

    return vsag::Dataset::Make()
        ->NumElements(K_NUM_VECTORS)
        ->Dim(K_DIM)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
}

vsag::DatasetPtr
make_query_dataset(std::vector<float>& query_vector) {
    query_vector.resize(K_DIM);
    std::mt19937 rng(101);
    std::uniform_real_distribution<float> dist;
    for (auto& value : query_vector) {
        value = dist(rng);
    }

    return vsag::Dataset::Make()
        ->NumElements(1)
        ->Dim(K_DIM)
        ->Float32Vectors(query_vector.data())
        ->Owner(false);
}

template <typename T>
void
check_result(const tl::expected<T, vsag::Error>& result, const std::string& action) {
    if (!result.has_value()) {
        std::cerr << action << " failed: " << result.error().message << std::endl;
        std::abort();
    }
}

void
print_results(const vsag::DatasetPtr& result) {
    std::cout << "results:" << std::endl;
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        std::cout << "  " << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
    }
}

}  // namespace

int
main() {
    vsag::Options::Instance().set_block_size_limit(2UL * 1024 * 1024);
    cleanup_hybrid_files();

    std::vector<int64_t> ids;
    std::vector<float> vectors;
    auto base = make_base_dataset(ids, vectors);

    auto index = vsag::Factory::CreateIndex("hgraph", hybrid_hgraph_build_parameters).value();
    check_result(index->Build(base), "Build");

    {
        std::ofstream out(K_DB_FILE_PATH, std::ios::binary);
        std::vector<char> db_header(K_DB_HEADER_SIZE, 'H');
        std::vector<char> db_footer(K_DB_FOOTER_SIZE, 'F');
        out.write(db_header.data(), static_cast<std::streamsize>(db_header.size()));
        check_result(index->SerializeStreaming(out), "SerializeStreaming");
        out.write(db_footer.data(), static_cast<std::streamsize>(db_footer.size()));
    }

    auto metadata_file = std::ifstream(K_DB_FILE_PATH, std::ios::binary);
    metadata_file.seekg(static_cast<std::streamoff>(K_DB_HEADER_SIZE), std::ios::beg);
    auto metadata_result = vsag::Index::GetStreamingMetadata(metadata_file);
    check_result(metadata_result, "GetStreamingMetadata");

    const vsag::StreamingBlockLayout* high_precision_codes = nullptr;
    for (const auto& block : metadata_result.value().blocks) {
        if (block.name == "high_precision_codes") {
            high_precision_codes = &block;
            break;
        }
    }
    if (high_precision_codes == nullptr) {
        std::cerr << "high_precision_codes block is missing" << std::endl;
        std::abort();
    }

    auto db_reader_file = std::make_shared<std::ifstream>(K_DB_FILE_PATH, std::ios::binary);
    auto db_reader_mutex = std::make_shared<std::mutex>();
    auto db_read_func = [db_reader_file, db_reader_mutex](
                            uint64_t offset, uint64_t len, void* dest) {
        std::lock_guard<std::mutex> lock(*db_reader_mutex);
        db_reader_file->clear();
        db_reader_file->seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        db_reader_file->read(static_cast<char*>(dest), static_cast<std::streamsize>(len));
        if (db_reader_file->fail()) {
            throw std::runtime_error("failed to read wrapped DB index file");
        }
    };

    const uint64_t high_precision_codes_offset =
        K_DB_HEADER_SIZE + high_precision_codes->payload_offset;
    auto precise_reader = vsag::Factory::CreateReadFuncReader(
        db_read_func, high_precision_codes_offset, high_precision_codes->payload_size);

    vsag::LoadParameters load_parameters;
    load_parameters.Set("base_io_type", "block_memory_io")
        .Set("precise_io_type", "reader_io")
        .SetReader("precise_reader", precise_reader);

    vsag::IndexPtr loaded_index;
    {
        std::ifstream in(K_DB_FILE_PATH, std::ios::binary);
        in.seekg(static_cast<std::streamoff>(K_DB_HEADER_SIZE), std::ios::beg);
        auto load_result = vsag::Index::Load(in, load_parameters);
        check_result(load_result, "Load");
        loaded_index = load_result.value();
    }

    std::vector<float> query_vector;
    auto query = make_query_dataset(query_vector);
    auto result = loaded_index->KnnSearch(query, K_TOP_K, hgraph_search_parameters);
    check_result(result, "KnnSearch");

    std::cout << "After streaming Load(), hybrid HGraph contains: "
              << loaded_index->GetNumElements() << std::endl;
    std::cout << "Wrapped DB index file: " << K_DB_FILE_PATH << std::endl;
    std::cout << "Load parameters: " << load_parameters.Dump() << std::endl;
    std::cout << "high_precision_codes payload offset in DB file: " << high_precision_codes_offset
              << std::endl;
    print_results(result.value());
    return 0;
}
