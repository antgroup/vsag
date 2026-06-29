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

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "test_dataset_pool.h"
#include "vsag/options.h"
#include "vsag/vsag.h"

namespace fixtures {

struct BlockSizeGuard {
    uint64_t orig_;
    explicit BlockSizeGuard(uint64_t new_size)
        : orig_(vsag::Options::Instance().block_size_limit()) {
        vsag::Options::Instance().set_block_size_limit(new_size);
    }
    ~BlockSizeGuard() {
        vsag::Options::Instance().set_block_size_limit(orig_);
    }
    BlockSizeGuard(const BlockSizeGuard&) = delete;
    BlockSizeGuard&
    operator=(const BlockSizeGuard&) = delete;
};

struct TestMatrix {
    std::vector<std::string> metrics;
    std::vector<int> dims;
    std::vector<std::pair<std::string, float>> quantizers;
    uint64_t base_count{1000};
    uint64_t block_size{1024 * 1024 * 2};

    struct Iteration {
        std::string metric;
        int dim;
        std::string quantizer;
        float recall;
    };

    std::vector<Iteration>
    Expand() const {
        std::vector<Iteration> result;
        result.reserve(metrics.size() * dims.size() * quantizers.size());
        for (const auto& metric : metrics)
            for (auto dim : dims)
                for (const auto& [q, recall] : quantizers)
                    result.push_back({metric, dim, q, recall});
        return result;
    }

    std::vector<Iteration>
    ExpandDimQuantizer() const {
        std::vector<Iteration> result;
        if (metrics.empty())
            return result;
        const auto& metric = metrics[0];
        for (auto dim : dims)
            for (const auto& [q, recall] : quantizers) result.push_back({metric, dim, q, recall});
        return result;
    }

    std::vector<Iteration>
    ExpandMetricQuantizer() const {
        std::vector<Iteration> result;
        if (dims.empty())
            return result;
        int dim = dims[0];
        for (const auto& metric : metrics)
            for (const auto& [q, recall] : quantizers) result.push_back({metric, dim, q, recall});
        return result;
    }

    void
    ForEach(TestDatasetPool& pool, const std::function<void(const Iteration&)>& fn) const {
        BlockSizeGuard guard(block_size);
        for (const auto& iter : Expand()) {
            pool.GetDatasetAndCreate(iter.dim, base_count, iter.metric);
            fn(iter);
        }
    }

    void
    ForEachDimQuantizer(TestDatasetPool& pool,
                        const std::function<void(const Iteration&)>& fn) const {
        BlockSizeGuard guard(block_size);
        for (const auto& iter : ExpandDimQuantizer()) {
            pool.GetDatasetAndCreate(iter.dim, base_count, iter.metric);
            fn(iter);
        }
    }
};

}  // namespace fixtures
