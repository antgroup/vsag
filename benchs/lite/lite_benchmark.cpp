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
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

double
Millis(Clock::duration value) {
    return std::chrono::duration<double, std::milli>(value).count();
}

int64_t
PositiveArg(int argc, char** argv, int position, int64_t fallback) {
    if (position >= argc) {
        return fallback;
    }
    const auto value = std::strtoll(argv[position], nullptr, 10);
    return value > 0 ? value : fallback;
}

std::string
BuildParameters(int64_t dim) {
    return std::string(R"({"dtype":"float32","metric_type":"l2","dim":)") + std::to_string(dim) +
           R"(,"index_param":{"base_quantization_type":"fp32","max_degree":16,)"
           R"("ef_construction":100,"build_thread_count":4}})";
}

std::vector<int64_t>
ExactTopK(
    const float* query, const std::vector<float>& base, int64_t count, int64_t dim, int64_t k) {
    std::vector<std::pair<float, int64_t>> distances;
    distances.reserve(static_cast<size_t>(count));
    for (int64_t row = 0; row < count; ++row) {
        float distance = 0.0F;
        for (int64_t col = 0; col < dim; ++col) {
            const auto delta = query[col] - base[static_cast<size_t>(row * dim + col)];
            distance += delta * delta;
        }
        distances.emplace_back(distance, row);
    }
    std::partial_sort(distances.begin(), distances.begin() + k, distances.end());
    std::vector<int64_t> result(static_cast<size_t>(k));
    for (int64_t i = 0; i < k; ++i) {
        result[static_cast<size_t>(i)] = distances[static_cast<size_t>(i)].second;
    }
    return result;
}

double
Percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const auto raw = std::ceil(fraction * static_cast<double>(values.size())) - 1.0;
    const auto position = std::min(static_cast<size_t>(raw), values.size() - 1);
    return values[position];
}
}  // namespace

int
main(int argc, char** argv) {
    try {
        const auto count = PositiveArg(argc, argv, 1, 10000);
        const auto dim = PositiveArg(argc, argv, 2, 64);
        const auto query_count = PositiveArg(argc, argv, 3, 100);
        constexpr int64_t K = 10;
        if (count < K) {
            throw std::invalid_argument("count must be at least 10");
        }

        vsag::init();
        std::mt19937 generator(42);
        std::normal_distribution<float> distribution;
        std::vector<float> base(static_cast<size_t>(count * dim));
        std::generate(base.begin(), base.end(), [&]() { return distribution(generator); });
        std::vector<int64_t> ids(static_cast<size_t>(count));
        std::iota(ids.begin(), ids.end(), 0);
        auto dataset = vsag::Dataset::Make()
                           ->NumElements(count)
                           ->Dim(dim)
                           ->Ids(ids.data())
                           ->Float32Vectors(base.data())
                           ->Owner(false);

        auto created = vsag::Factory::CreateIndex("hgraph", BuildParameters(dim));
        if (!created.has_value()) {
            throw std::runtime_error(created.error().message);
        }
        const auto build_start = Clock::now();
        auto built = created.value()->Build(dataset);
        const auto build_end = Clock::now();
        if (!built.has_value()) {
            throw std::runtime_error(built.error().message);
        }

        const auto serialize_start = Clock::now();
        auto binary = created.value()->Serialize();
        const auto serialize_end = Clock::now();
        if (!binary.has_value()) {
            throw std::runtime_error(binary.error().message);
        }
        uint64_t index_bytes = 0;
        for (const auto& key : binary.value().GetKeys()) {
            index_bytes += binary.value().Get(key).size;
        }

        auto restored = vsag::Factory::CreateIndex("hgraph", BuildParameters(dim));
        const auto deserialize_start = Clock::now();
        auto loaded = restored.value()->Deserialize(binary.value());
        const auto deserialize_end = Clock::now();
        if (!loaded.has_value()) {
            throw std::runtime_error(loaded.error().message);
        }

        std::vector<double> latencies;
        int64_t hits = 0;
        for (int64_t query_id = 0; query_id < query_count; ++query_id) {
            const auto row = query_id * count / query_count;
            const auto* query = base.data() + row * dim;
            auto query_data =
                vsag::Dataset::Make()->NumElements(1)->Dim(dim)->Float32Vectors(query)->Owner(
                    false);
            const auto start = Clock::now();
            auto result =
                restored.value()->KnnSearch(query_data, K, R"({"hgraph":{"ef_search":100}})");
            latencies.push_back(Millis(Clock::now() - start));
            if (!result.has_value()) {
                throw std::runtime_error(result.error().message);
            }
            const auto exact = ExactTopK(query, base, count, dim, K);
            const std::unordered_set<int64_t> truth(exact.begin(), exact.end());
            for (int64_t i = 0; i < K; ++i) {
                hits += truth.count(result.value()->GetIds()[i]);
            }
        }
        const auto search_seconds =
            std::accumulate(latencies.begin(), latencies.end(), 0.0) / 1000.0;
#ifdef VSAG_LITE
        constexpr const char* VARIANT = "lite";
#else
        constexpr const char* VARIANT = "full";
#endif
        std::cout << "{\"variant\":\"" << VARIANT << "\",\"count\":" << count
                  << ",\"dimension\":" << dim << ",\"queries\":" << query_count
                  << ",\"build_ms\":" << Millis(build_end - build_start)
                  << ",\"serialize_ms\":" << Millis(serialize_end - serialize_start)
                  << ",\"deserialize_ms\":" << Millis(deserialize_end - deserialize_start)
                  << ",\"index_bytes\":" << index_bytes
                  << ",\"qps\":" << static_cast<double>(query_count) / search_seconds
                  << ",\"p50_ms\":" << Percentile(latencies, 0.50)
                  << ",\"p99_ms\":" << Percentile(latencies, 0.99) << ",\"recall_at_10\":"
                  << static_cast<double>(hits) / static_cast<double>(query_count * K) << "}\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "benchmark failed: " << exception.what() << '\n';
        return 1;
    }
}
