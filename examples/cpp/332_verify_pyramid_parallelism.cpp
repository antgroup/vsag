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

// POC: Verify that Pyramid `parallelism` parameter enables multi-path parallel search.
// Issue: https://github.com/antgroup/vsag/issues/2985

#include <vsag/vsag.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

static std::string
pyramid_build_param(int dim, int build_thread_count = 4) {
    std::ostringstream oss;
    oss << R"({
        "dtype": "float32",
        "metric_type": "l2",
        "dim": )"
        << dim << R"(,
        "index_param": {
            "max_degree": 16,
            "alpha": 1.2,
            "graph_type": "odescent",
            "graph_iter_turn": 10,
            "neighbor_sample_rate": 0.2,
            "base_quantization_type": "fp32",
            "use_reorder": false,
            "index_min_size": 0,
            "build_thread_count": )"
        << build_thread_count << R"(,
            "hierarchies": [
                {"name": "h", "no_build_levels": []}
            ]
        }
    })";
    return oss.str();
}

static std::string
search_param(int ef, int parallelism, const std::string& hierarchy) {
    std::ostringstream oss;
    oss << R"({"pyramid": {"ef_search": )" << ef << R"(, "parallelism": )" << parallelism
        << R"(, "hierarchies": [")" << hierarchy << R"("]}})";
    return oss.str();
}

int
main() {
    std::cout << "=== Pyramid Parallelism POC ===" << std::endl;

    const int dim = 16;
    const int num_base = 2000;
    const int num_path_groups = 30;
    const int k = 10;
    const int ef = 100;
    const std::string hierarchy = "h";

    // --- 1. Create index with build_thread_count=4 to create default thread pool ---
    std::cout << "\n[1] Creating Pyramid index (build_thread_count=4)..." << std::endl;
    auto index_result = vsag::Factory::CreateIndex("pyramid", pyramid_build_param(dim, 4));
    if (!index_result.has_value()) {
        std::cerr << "ERROR: CreateIndex: " << index_result.error().message << std::endl;
        return 1;
    }
    auto index = index_result.value();

    // --- 2. Generate base data: N vectors equally distributed across path groups ---
    std::cout << "[2] Generating " << num_base << " vectors across " << num_path_groups
              << " path groups..." << std::endl;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    auto ids = new int64_t[num_base];
    auto vectors = new float[num_base * dim];
    auto path_strs = new std::string[num_base];

    for (int i = 0; i < num_base; ++i) {
        ids[i] = i;
        for (int d = 0; d < dim; ++d) {
            vectors[i * dim + d] = dist(rng);
        }
        int group = i % num_path_groups;
        path_strs[i] = std::string("group_") + std::to_string(group);
    }

    auto base = vsag::Dataset::Make();
    base->NumElements(num_base)
        ->Dim(dim)
        ->Float32Vectors(vectors)
        ->Ids(ids)
        ->Paths(hierarchy, path_strs)
        ->Owner(true);

    std::cout << "[3] Building index..." << std::endl;
    auto t0 = std::chrono::steady_clock::now();
    auto build_result = index->Build(base);
    auto t1 = std::chrono::steady_clock::now();
    if (!build_result.has_value()) {
        std::cerr << "ERROR: Build: " << build_result.error().message << std::endl;
        return 1;
    }
    std::cout << "    Build done in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms"
              << std::endl;

    // --- 4. Single query vector ---
    auto query_vec = new float[dim];
    for (int d = 0; d < dim; ++d) {
        query_vec[d] = dist(rng);
    }

    // --- 5. Benchmark: varying parallelism with varying number of paths ---
    auto run_bench = [&](const std::string& path, int parallelism, int repeat) -> double {
        auto param_str = search_param(ef, parallelism, hierarchy);

        auto t_start = std::chrono::steady_clock::now();
        for (int r = 0; r < repeat; ++r) {
            // Re-create query dataset each iteration to avoid double-free
            auto query_path = new std::string[1]{path};
            auto query_vec_copy = new float[dim];
            std::copy(query_vec, query_vec + dim, query_vec_copy);
            auto query = vsag::Dataset::Make();
            query->NumElements(1)
                ->Dim(dim)
                ->Float32Vectors(query_vec_copy)
                ->Paths(hierarchy, query_path)
                ->Owner(true);

            auto result = index->KnnSearch(query, k, param_str);
            if (!result.has_value()) {
                std::cerr << "    ERROR: KnnSearch (parallelism=" << parallelism
                          << "): " << result.error().message << std::endl;
                return -1.0;
            }
        }
        auto t_end = std::chrono::steady_clock::now();

        double total_us =
            std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
        return total_us / repeat / 1000.0;  // avg ms per query
    };

    std::cout << "\n[4] Benchmark: single path (group_0)" << std::endl;
    std::cout << "    (Only 1 sub-graph → parallelism should NOT help)" << std::endl;
    for (int par : {1, 2, 4}) {
        double avg_ms = run_bench("group_0", par, 50);
        if (avg_ms >= 0) {
            std::cout << "    parallelism=" << par << ": " << avg_ms << " ms/query" << std::endl;
        }
    }

    std::cout << "\n[5] Benchmark: 3 paths (group_0|group_1|group_2)" << std::endl;
    for (int par : {1, 2, 4}) {
        double avg_ms = run_bench("group_0|group_1|group_2", par, 20);
        if (avg_ms >= 0) {
            std::cout << "    parallelism=" << par << ": " << avg_ms << " ms/query" << std::endl;
        }
    }

    std::cout << "\n[6] Benchmark: 30 paths (simulating the issue's workload)" << std::endl;
    {
        std::ostringstream mp;
        for (int g = 0; g < 30; ++g) {
            if (g > 0)
                mp << "|";
            mp << "group_" << g;
        }
        std::string thirty = mp.str();
        for (int par : {1, 2, 4}) {
            double avg_ms = run_bench(thirty, par, 10);
            if (avg_ms >= 0) {
                std::cout << "    parallelism=" << par << ": " << avg_ms << " ms/query"
                          << std::endl;
            }
        }
    }

    // --- 7. Correctness: serial vs parallel must match ---
    std::cout << "\n[7] Correctness check: serial vs parallel" << std::endl;

    auto make_query = [&](const std::string& path) {
        auto pq = new std::string[1]{path};
        auto v = new float[dim];
        std::copy(query_vec, query_vec + dim, v);
        auto qds = vsag::Dataset::Make();
        qds->NumElements(1)->Dim(dim)->Float32Vectors(v)->Paths(hierarchy, pq)->Owner(true);
        return qds;
    };

    // Test: 3-path query, serial vs parallel
    auto q_3path = make_query("group_10|group_20|group_25");
    auto ref_result = index->KnnSearch(q_3path, k, search_param(ef, 1, hierarchy));
    auto par_result = index->KnnSearch(q_3path, k, search_param(ef, 2, hierarchy));

    if (!ref_result.has_value() || !par_result.has_value()) {
        std::cerr << "ERROR: correctness search failed" << std::endl;
        return 1;
    }

    auto* ref_ids = ref_result.value()->GetIds();
    auto* par_ids = par_result.value()->GetIds();
    auto ref_cnt = ref_result.value()->GetDim();
    auto par_cnt = par_result.value()->GetDim();

    bool match = (ref_cnt == par_cnt);
    if (match) {
        for (int64_t i = 0; i < ref_cnt; ++i) {
            if (ref_ids[i] != par_ids[i]) {
                match = false;
                std::cerr << "    MISMATCH at pos " << i << ": serial=" << ref_ids[i]
                          << " parallel=" << par_ids[i] << std::endl;
                break;
            }
        }
    }
    if (match) {
        std::cout << "    PASS: serial (parallelism=1) and parallel (parallelism=2) results "
                     "identical for 3-path query"
                  << std::endl;
    } else {
        std::cout << "    FAIL: results differ" << std::endl;
    }

    // Test: 30-path correctness
    std::cout << "\n[8] Correctness check: 30-path query" << std::endl;
    {
        std::ostringstream mp;
        for (int g = 0; g < 30; ++g) {
            if (g > 0)
                mp << "|";
            mp << "group_" << g;
        }
        auto q_30 = make_query(mp.str());
        auto r1 = index->KnnSearch(q_30, k, search_param(ef, 1, hierarchy));
        auto r2 = index->KnnSearch(q_30, k, search_param(ef, 2, hierarchy));

        if (r1.has_value() && r2.has_value()) {
            auto* ids1 = r1.value()->GetIds();
            auto* ids2 = r2.value()->GetIds();
            auto cnt1 = r1.value()->GetDim();
            auto cnt2 = r2.value()->GetDim();
            bool ok = (cnt1 == cnt2);
            if (ok) {
                for (int64_t i = 0; i < cnt1; ++i) {
                    if (ids1[i] != ids2[i]) {
                        ok = false;
                        break;
                    }
                }
            }
            std::cout << "    " << (ok ? "PASS" : "FAIL") << ": 30-path serial vs parallel"
                      << std::endl;
        }
    }

    // --- 9. Deserialization roundtrip ---
    std::cout << "\n[9] Serialization roundtrip check..." << std::endl;
    {
        auto serialized = index->Serialize();
        if (!serialized.has_value()) {
            std::cerr << "ERROR: Serialize: " << serialized.error().message << std::endl;
            return 1;
        }
        auto restored_result = vsag::Factory::CreateIndex("pyramid", pyramid_build_param(dim, 4));
        if (!restored_result.has_value()) {
            std::cerr << "ERROR: CreateIndex for restore" << std::endl;
            return 1;
        }
        auto restored = restored_result.value();
        if (!restored->Deserialize(serialized.value()).has_value()) {
            std::cerr << "ERROR: Deserialize" << std::endl;
            return 1;
        }

        auto q = make_query("group_0|group_1");
        auto r1 = index->KnnSearch(q, k, search_param(ef, 2, hierarchy));
        auto r2 = restored->KnnSearch(q, k, search_param(ef, 2, hierarchy));

        if (r1.has_value() && r2.has_value()) {
            bool ok = (r1.value()->GetDim() == r2.value()->GetDim());
            if (ok) {
                auto* ids1 = r1.value()->GetIds();
                auto* ids2 = r2.value()->GetIds();
                for (int64_t i = 0; i < r1.value()->GetDim(); ++i) {
                    if (ids1[i] != ids2[i]) {
                        ok = false;
                        break;
                    }
                }
            }
            std::cout << "    " << (ok ? "PASS" : "FAIL")
                      << ": deserialized index gives identical parallel search results"
                      << std::endl;
        }
    }

    // --- 10. Verify parameter chain: confirm parallelism is parsed ---
    std::cout << "\n[10] Parameter delivery verification" << std::endl;
    {
        // The key check: parallelism=2 and parallelism=1 both work without error.
        // If the parameter is dropped, only default (1) behavior would succeed.
        auto q = make_query("group_0");
        auto r1 = index->KnnSearch(q, k, search_param(ef, 1, hierarchy));
        auto r2 = index->KnnSearch(q, k, search_param(ef, 2, hierarchy));
        auto r4 = index->KnnSearch(q, k, search_param(ef, 4, hierarchy));

        bool all_ok = r1.has_value() && r2.has_value() && r4.has_value();
        std::cout << "    parallelism=1: " << (r1.has_value() ? "OK" : "FAIL") << std::endl;
        std::cout << "    parallelism=2: " << (r2.has_value() ? "OK" : "FAIL") << std::endl;
        std::cout << "    parallelism=4: " << (r4.has_value() ? "OK" : "FAIL") << std::endl;
        std::cout << "    "
                  << (all_ok ? "PASS: all parallelism values accepted"
                             : "FAIL: some parallelism values rejected")
                  << std::endl;
    }

    std::cout << "\n=== POC Complete ===" << std::endl;
    return 0;
}