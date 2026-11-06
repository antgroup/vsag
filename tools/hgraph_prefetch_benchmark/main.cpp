// Reproducible, synthetic HGraph search workload for prefetch experiments.
// Build manually against the release library; keep the index file across variants.
#include <vsag/vsag.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr int64_t kDim = 128;
constexpr int64_t kQueries = 16384;

void
FillData(int64_t count, std::vector<float>& vectors, std::vector<int64_t>& ids) {
    std::mt19937 random(1979);
    std::normal_distribution<float> center(0.0F, 1.0F);
    std::normal_distribution<float> noise(0.0F, 0.3F);
    std::vector<float> centers(256 * kDim);
    for (auto& value : centers) {
        value = center(random);
    }
    vectors.resize(count * kDim);
    ids.resize(count);
    for (int64_t i = 0; i < count; ++i) {
        ids[i] = i;
        for (int64_t d = 0; d < kDim; ++d) {
            vectors[i * kDim + d] = centers[(i % 256) * kDim + d] + noise(random);
        }
    }
}

void
FillQueries(const std::vector<float>& vectors, int64_t count, std::vector<float>& queries) {
    std::mt19937 random(2026);
    std::normal_distribution<float> noise(0.0F, 0.15F);
    queries.resize(kQueries * kDim);
    for (int64_t i = 0; i < kQueries; ++i) {
        const int64_t source = (i * 97 + 31) % count;
        for (int64_t d = 0; d < kDim; ++d) {
            queries[i * kDim + d] = vectors[source * kDim + d] + noise(random);
        }
    }
}

uint64_t
Search(const vsag::IndexPtr& index,
       const std::vector<float>& queries,
       int64_t id,
       const std::string& search_parameters) {
    auto query = vsag::Dataset::Make();
    query->NumElements(1)
        ->Dim(kDim)
        ->Float32Vectors(const_cast<float*>(queries.data() + id * kDim))
        ->Owner(false);
    auto result = index->KnnSearch(query, 10, search_parameters);
    if (not result.has_value()) {
        std::cerr << result.error().message << std::endl;
        std::abort();
    }
    uint64_t hash = 0;
    for (int64_t i = 0; i < result.value()->GetDim(); ++i) {
        hash +=
            static_cast<uint64_t>(result.value()->GetIds()[i] + 1) * static_cast<uint64_t>(i + 1);
    }
    return hash;
}
}  // namespace

int
main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "usage: benchmark build|bench sq8|fp32|rabitq INDEX N THREADS ROUNDS\n";
        return 2;
    }
    const std::string mode = argv[1];
    const std::string quantization = argv[2];
    const std::string index_path = argv[3];
    const int64_t count = std::stoll(argv[4]);
    const int threads = std::stoi(argv[5]);
    const int rounds = std::stoi(argv[6]);
    if (count < kQueries or threads <= 0 or rounds <= 0) {
        return 2;
    }
    vsag::init();
    vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
    vsag::Engine engine(&resource);
    std::string parameters =
        "{\"dtype\":\"float32\",\"metric_type\":\"l2\",\"dim\":128,\"index_param\":"
        "{\"base_quantization_type\":\"" +
        quantization + "\",\"max_degree\":32,\"ef_construction\":100}}";
    if (quantization == "rabitq") {
        parameters = R"({"dtype":"float32","metric_type":"l2","dim":128,"index_param":{)"
                     R"("base_quantization_type":"rabitq","precise_quantization_type":"rabitq",)"
                     R"("base_io_type":"block_memory_io","use_reorder":true,)"
                     R"("rabitq_bits_per_dim_base":3,"rabitq_bits_per_dim_precise":5,)"
                     R"("rabitq_error_rate":1.9,"max_degree":32,"ef_construction":100,)"
                     R"("graph_storage_type":"compressed"}})";
    }
    const std::string search_parameters =
        quantization == "rabitq" ? R"({"hgraph":{"ef_search":100,"rabitq_one_bit_search":true}})"
                                 : R"({"hgraph":{"ef_search":100}})";
    auto created = engine.CreateIndex("hgraph", parameters);
    if (not created.has_value()) {
        std::cerr << created.error().message << std::endl;
        return 1;
    }
    auto index = created.value();
    std::vector<float> vectors;
    std::vector<int64_t> ids;
    FillData(count, vectors, ids);
    if (mode == "build") {
        auto base = vsag::Dataset::Make();
        base->NumElements(count)
            ->Dim(kDim)
            ->Ids(ids.data())
            ->Float32Vectors(vectors.data())
            ->Owner(false);
        auto built = index->Build(base);
        if (not built.has_value()) {
            std::cerr << built.error().message << std::endl;
            return 1;
        }
        std::ofstream output(index_path, std::ios::binary);
        auto saved = index->Serialize(output);
        if (not saved.has_value()) {
            std::cerr << saved.error().message << std::endl;
            return 1;
        }
        std::cout << "built " << count << " vectors\n";
        return 0;
    }
    if (mode != "bench") {
        return 2;
    }
    std::ifstream input(index_path, std::ios::binary);
    auto loaded = index->Deserialize(input);
    if (not loaded.has_value()) {
        std::cerr << loaded.error().message << std::endl;
        return 1;
    }
    std::vector<float> queries;
    FillQueries(vectors, count, queries);
    for (int64_t i = 0; i < 512; ++i) {
        Search(index, queries, i, search_parameters);
    }
    for (int round = 0; round < rounds; ++round) {
        std::vector<uint64_t> checksums(threads);
        std::vector<std::thread> workers;
        const auto start = std::chrono::steady_clock::now();
        for (int thread_id = 0; thread_id < threads; ++thread_id) {
            workers.emplace_back([&, thread_id]() {
                uint64_t local = 0;
                for (int64_t i = thread_id; i < kQueries; i += threads) {
                    local += Search(index, queries, i, search_parameters);
                }
                checksums[thread_id] = local;
            });
        }
        for (auto& worker : workers) {
            worker.join();
        }
        const auto end = std::chrono::steady_clock::now();
        uint64_t checksum = 0;
        for (auto value : checksums) {
            checksum += value;
        }
        const double seconds = std::chrono::duration<double>(end - start).count();
        std::cout << "{\"quantization\":\"" << quantization << "\",\"threads\":" << threads
                  << ",\"round\":" << round << ",\"qps\":" << kQueries / seconds
                  << ",\"checksum\":" << checksum << "}\n";
    }
    return 0;
}
