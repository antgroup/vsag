// SPDX-License-Identifier: Apache-2.0
// Copyright 2026-present the vsag project
// Linux-only benchmark support; see README.md for scope and invocation.
#include <sys/resource.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

#include "vsag/vsag.h"

int
main(int argc, char** argv) {
    if (argc != 3)
        return 2;
    vsag::Options::Instance().set_block_size_limit(128ULL * 1024 * 1024);
    if (std::string(argv[1]) == "create") {
        constexpr int64_t count = 1024, dim = 4, extra_size = 131073;
        auto index = vsag::Factory::CreateIndex("hgraph", R"({"dtype":"float32",
          "metric_type":"l2","dim":4,"extra_info_size":131073,"index_param":{
          "base_quantization_type":"fp32","max_degree":16,"ef_construction":32,
          "build_thread_count":1}})");
        if (!index) {
            std::cerr << index.error().message;
            return 1;
        }
        std::vector<int64_t> ids(count);
        std::iota(ids.begin(), ids.end(), 0);
        std::vector<float> vectors(count * dim);
        for (uint64_t i = 0; i < vectors.size(); ++i) vectors[i] = float(i % 103);
        std::vector<char> extra(count * extra_size, 'x');
        auto base = vsag::Dataset::Make();
        base->NumElements(count)
            ->Dim(dim)
            ->Ids(ids.data())
            ->Float32Vectors(vectors.data())
            ->ExtraInfos(extra.data())
            ->Owner(false);
        auto built = index.value()->Build(base);
        if (!built) {
            std::cerr << built.error().message;
            return 1;
        }
        std::ofstream output(argv[2], std::ios::binary);
        auto written = index.value()->SerializeStreaming(output);
        if (!written) {
            std::cerr << written.error().message;
            return 1;
        }
        std::cout << "artifact_bytes=" << output.tellp() << '\n';
        return 0;
    }
    std::ifstream input(argv[2], std::ios::binary);
    rusage before{}, after{};
    getrusage(RUSAGE_SELF, &before);
    auto start = std::chrono::steady_clock::now();
    auto index = vsag::Index::Load(input, "{}");
    auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    getrusage(RUSAGE_SELF, &after);
    if (!index) {
        std::cerr << index.error().message;
        return 1;
    }
    std::cout << "load_seconds=" << seconds << " elements=" << index.value()->GetNumElements()
              << " peak_rss_delta_kib=" << after.ru_maxrss - before.ru_maxrss
              << " output_bytes=" << (after.ru_oublock - before.ru_oublock) * 512 << '\n';
}
