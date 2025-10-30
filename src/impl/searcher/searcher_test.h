
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

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "algorithm/hnswlib/hnswalg.h"
#include "algorithm/hnswlib/space_l2.h"
#include "datacell/flatten_datacell.h"
#include "fixtures.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/basic_optimizer.h"
#include "io/memory_io.h"
#include "quantization/fp32_quantizer.h"
#include "quantization/scalar_quantization/sq4_uniform_quantizer.h"
#include "test_logger.h"
#include "utils/visited_list.h"

namespace vsag {

class AdaptGraphDataCell : public GraphInterface {
public:
    AdaptGraphDataCell(std::shared_ptr<hnswlib::HierarchicalNSW> alg_hnsw) : alg_hnsw_(alg_hnsw){};

    void
    InsertNeighborsById(InnerIdType id, const Vector<InnerIdType>& neighbor_ids) override {
        return;
    };

    void
    Resize(InnerIdType new_size) override {
        return;
    };

    void
    GetNeighbors(InnerIdType id, Vector<InnerIdType>& neighbor_ids) const override {
        alg_hnsw_->getNeighborsInternalId(id, neighbor_ids);
    }

    uint32_t
    GetNeighborSize(InnerIdType id) const override {
        int* data = (int*)alg_hnsw_->get_linklist0(id);
        return alg_hnsw_->getListCount((hnswlib::linklistsizeint*)data);
    }

    void
    Prefetch(InnerIdType id, InnerIdType neighbor_i) override {
        int* data = (int*)alg_hnsw_->get_linklist0(id);
        vsag::Prefetch(data + neighbor_i + 1);
    }

    InnerIdType
    MaximumDegree() const override {
        return alg_hnsw_->getMaxDegree();
    }

private:
    std::shared_ptr<hnswlib::HierarchicalNSW> alg_hnsw_;
};

}  // namespace vsag
