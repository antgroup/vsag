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

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

namespace {

constexpr uint64_t DIM = 128;
constexpr uint64_t LEAF_COUNT = 9;
constexpr uint64_t DEFAULT_DATASET_MIB = 400;
constexpr uint64_t BYTES_PER_MIB = 1024 * 1024;

struct Measurement {
    std::string name;
    uint64_t total_rss_bytes;
    uint64_t tracked_bytes;
    uint64_t reported_bytes;
    std::vector<std::pair<std::string, uint64_t>> reported_detail;
};

class TrackingAllocator : public vsag::Allocator {
public:
    std::string
    Name() override {
        return "memory-comparison-allocator";
    }

    void*
    Allocate(uint64_t size) override {
        auto* pointer = std::malloc(size);
        if (pointer == nullptr) {
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        allocations_[pointer] = size;
        current_bytes_ += size;
        peak_bytes_ = std::max(peak_bytes_, current_bytes_);
        return pointer;
    }

    void
    Deallocate(void* pointer) override {
        if (pointer == nullptr) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto allocation = allocations_.find(pointer);
            if (allocation != allocations_.end()) {
                current_bytes_ -= allocation->second;
                allocations_.erase(allocation);
            }
        }
        std::free(pointer);
    }

    void*
    Reallocate(void* pointer, uint64_t size) override {
        std::lock_guard<std::mutex> lock(mutex_);
        uint64_t old_size = 0;
        auto allocation = allocations_.find(pointer);
        if (allocation != allocations_.end()) {
            old_size = allocation->second;
        }
        auto* new_pointer = std::realloc(pointer, size);
        if (new_pointer == nullptr) {
            return nullptr;
        }
        allocations_.erase(pointer);
        allocations_[new_pointer] = size;
        current_bytes_ = current_bytes_ - old_size + size;
        peak_bytes_ = std::max(peak_bytes_, current_bytes_);
        return new_pointer;
    }

    uint64_t
    CurrentBytes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return current_bytes_;
    }

private:
    mutable std::mutex mutex_;
    std::unordered_map<void*, uint64_t> allocations_;
    uint64_t current_bytes_{0};
    uint64_t peak_bytes_{0};
};

double
to_mib(uint64_t bytes) {
    return static_cast<double>(bytes) / static_cast<double>(BYTES_PER_MIB);
}

uint64_t
get_current_rss() {
#if defined(__APPLE__)
    mach_task_basic_info_data_t info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(
            mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info), &count) !=
        KERN_SUCCESS) {
        return 0;
    }
    return info.resident_size;
#elif defined(__linux__)
    std::ifstream statm("/proc/self/statm");
    uint64_t total_pages = 0;
    uint64_t resident_pages = 0;
    statm >> total_pages >> resident_pages;
    return resident_pages * static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
#else
    return 0;
#endif
}

Measurement
build_and_measure(const std::string& name,
                  const std::string& parameters,
                  const vsag::DatasetPtr& base) {
    TrackingAllocator allocator;
    const uint64_t baseline_rss = get_current_rss();
    vsag::Resource resource(&allocator, nullptr);
    vsag::Engine engine(&resource);
    const uint64_t baseline_bytes = allocator.CurrentBytes();
    auto create_result = engine.CreateIndex(name, parameters);
    if (not create_result.has_value()) {
        std::cerr << "Failed to create " << name << ": " << create_result.error().message
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }

    auto index = create_result.value();
    auto build_result = index->Build(base);
    if (not build_result.has_value()) {
        std::cerr << "Failed to build " << name << ": " << build_result.error().message
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }

    const uint64_t built_rss = get_current_rss();
    Measurement result{name,
                       built_rss > baseline_rss ? built_rss - baseline_rss : 0,
                       allocator.CurrentBytes() - baseline_bytes,
                       index->GetMemoryUsage(),
                       {}};
    const auto detail = index->GetMemoryUsageDetail();
    result.reported_detail.assign(detail.begin(), detail.end());
    std::sort(result.reported_detail.begin(),
              result.reported_detail.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });
    index.reset();
    engine.Shutdown();
    return result;
}

}  // namespace

int
main(int argc, char** argv) {
    // Keep internal block pools from reserving large default chunks that dominate small trial runs.
    vsag::Options::Instance().set_block_size_limit(2 * BYTES_PER_MIB);
    if (argc < 2 || (std::string(argv[1]) != "hgraph" && std::string(argv[1]) != "pyramid")) {
        std::cerr << "Usage:\n  " << argv[0] << " <hgraph|pyramid> [dataset MiB]\n  " << argv[0]
                  << " <hgraph|pyramid> synthetic <count> <dim>\n  " << argv[0]
                  << " <hgraph|pyramid> fbin <base.fbin>" << std::endl;
        return EXIT_FAILURE;
    }
    const std::string index_name = argv[1];
    uint64_t dim = DIM;
    uint64_t num_vectors = 0;
    std::vector<float> vectors;
    if (argc == 4 && std::string(argv[2]) == "fbin") {
        std::ifstream input(argv[3], std::ios::binary);
        uint32_t file_num_vectors = 0;
        uint32_t file_dim = 0;
        input.read(reinterpret_cast<char*>(&file_num_vectors), sizeof(file_num_vectors));
        input.read(reinterpret_cast<char*>(&file_dim), sizeof(file_dim));
        if (not input || file_num_vectors == 0 || file_dim == 0) {
            std::cerr << "Invalid fbin header: " << argv[3] << std::endl;
            return EXIT_FAILURE;
        }
        num_vectors = file_num_vectors;
        dim = file_dim;
        vectors.resize(num_vectors * dim);
        input.read(reinterpret_cast<char*>(vectors.data()),
                   static_cast<std::streamsize>(vectors.size() * sizeof(float)));
        if (not input) {
            std::cerr << "Incomplete fbin data: " << argv[3] << std::endl;
            return EXIT_FAILURE;
        }
    } else {
        if (argc == 5 && std::string(argv[2]) == "synthetic") {
            num_vectors = std::stoull(argv[3]);
            dim = std::stoull(argv[4]);
        } else {
            const uint64_t dataset_mib = argc > 2 ? std::stoull(argv[2]) : DEFAULT_DATASET_MIB;
            const uint64_t vector_bytes = dataset_mib * BYTES_PER_MIB;
            num_vectors = vector_bytes / (dim * sizeof(float));
        }
        vectors.resize(num_vectors * dim);
        for (uint64_t i = 0; i < num_vectors; ++i) {
            for (uint64_t j = 0; j < dim; ++j) {
                uint64_t value = (i * dim + j) + 0x9E3779B97F4A7C15ULL;
                value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
                value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
                value ^= value >> 31U;
                vectors[i * dim + j] =
                    static_cast<float>(value & 0xFFFFFFU) / static_cast<float>(0x1000000U);
            }
        }
    }
    const uint64_t actual_vector_bytes = vectors.size() * sizeof(float);

    std::vector<int64_t> ids(num_vectors);
    std::vector<std::string> paths(num_vectors);
    const std::array<std::string, LEAF_COUNT> leaf_paths = {
        "leaf-0", "leaf-1", "leaf-2", "leaf-3", "leaf-4", "leaf-5", "leaf-6", "leaf-7", "leaf-8"};

    for (uint64_t i = 0; i < num_vectors; ++i) {
        ids[i] = static_cast<int64_t>(i);
        paths[i] = leaf_paths[i % LEAF_COUNT];
    }

    auto hgraph_base = vsag::Dataset::Make();
    hgraph_base->NumElements(static_cast<int64_t>(num_vectors))
        ->Dim(static_cast<int64_t>(dim))
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);

    auto pyramid_base = vsag::Dataset::Make();
    pyramid_base->NumElements(static_cast<int64_t>(num_vectors))
        ->Dim(static_cast<int64_t>(dim))
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Paths(paths.data())
        ->Owner(false);

    std::string hgraph_parameters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "index_param": {
            "base_quantization_type": "fp32",
            "max_degree": 64,
            "ef_construction": 200,
            "alpha": 1.2,
            "use_reorder": false,
            "use_reverse_edges": true,
            "support_force_remove": true,
            "store_raw_vector": true,
            "base_io_type": "block_memory_io",
            "build_thread_count": 1
        }
    }
    )";

    std::string pyramid_parameters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "index_param": {
            "base_quantization_type": "fp32",
            "max_degree": 32,
            "ef_construction": 400,
            "alpha": 1.2,
            "graph_type": "nsw",
            "no_build_levels": [0],
            "use_reorder": false,
            "store_raw_vector": true,
            "index_min_size": 0,
            "build_thread_count": 1
        }
    }
    )";

    const std::string default_dim = "\"dim\": 128";
    const std::string actual_dim = "\"dim\": " + std::to_string(dim);
    hgraph_parameters.replace(hgraph_parameters.find(default_dim), default_dim.size(), actual_dim);
    pyramid_parameters.replace(
        pyramid_parameters.find(default_dim), default_dim.size(), actual_dim);

    std::cout << "Dataset: " << num_vectors << " x " << dim << " float32 vectors (" << std::fixed
              << std::setprecision(2) << to_mib(actual_vector_bytes) << " MiB)" << std::endl;
    if (index_name == "pyramid") {
        std::cout << "Pyramid layout: one root (graph disabled) + " << LEAF_COUNT << " leaf graphs"
                  << std::endl;
    }

    const auto& parameters = index_name == "hgraph" ? hgraph_parameters : pyramid_parameters;
    const auto& base = index_name == "hgraph" ? hgraph_base : pyramid_base;
    const auto measurement = build_and_measure(index_name, parameters, base);

    std::cout << "\n" << measurement.name << " memory after Build():\n";
    std::cout << "  Total process RSS increase: " << to_mib(measurement.total_rss_bytes)
              << " MiB\n";
    std::cout << "  VSAG allocator subset:      " << to_mib(measurement.tracked_bytes) << " MiB\n";
    std::cout << "  Index::GetMemoryUsage():    " << to_mib(measurement.reported_bytes) << " MiB\n";
    std::cout << "  Total RSS bytes/vector:     "
              << static_cast<double>(measurement.total_rss_bytes) / static_cast<double>(num_vectors)
              << "\n";

    if (not measurement.reported_detail.empty()) {
        uint64_t detail_total = 0;
        std::cout << "\n  Index::GetMemoryUsageDetail():\n";
        for (const auto& [component, bytes] : measurement.reported_detail) {
            detail_total += bytes;
            std::cout << "    " << std::left << std::setw(24) << component << std::right
                      << std::setw(10) << to_mib(bytes) << " MiB\n";
        }
        std::cout << "    " << std::left << std::setw(24) << "detail total" << std::right
                  << std::setw(10) << to_mib(detail_total) << " MiB\n";
    }

    return 0;
}
