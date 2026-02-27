
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
#include <queue>
#include <string>

#include "impl/filter/filter_headers.h"
#include "index/iterator_filter.h"
#include "space_interface.h"
#include "storage/stream_reader.h"
#include "typing.h"
#include "vsag/dataset.h"
#include "vsag/errors.h"
#include "vsag/expected.hpp"
#include "vsag/iterator_context.h"

namespace hnswlib {

using LabelType = vsag::LabelType;
using InnerIdType = vsag::InnerIdType;

template <typename dist_t>
class AlgorithmInterface {
public:
    virtual bool
    addPoint(const void* datapoint, LabelType label) = 0;

    virtual std::priority_queue<std::pair<dist_t, LabelType>>
    searchKnn(const void* query_data,
              uint64_t k,
              uint64_t ef,
              const vsag::FilterPtr is_id_allowed = nullptr,
              float skip_ratio = 0.9f,
              vsag::Allocator* allocator = nullptr,
              vsag::IteratorFilterContext* iter_ctx = nullptr,
              bool is_last_filter = false) const = 0;

    virtual std::priority_queue<std::pair<dist_t, LabelType>>
    searchRange(const void* query_data,
                float radius,
                uint64_t ef,
                const vsag::FilterPtr is_id_allowed = nullptr) const = 0;

    // Return k nearest neighbor in the order of closer fist
    virtual std::vector<std::pair<dist_t, LabelType>>
    searchKnnCloserFirst(const void* query_data,
                         uint64_t k,
                         uint64_t ef,
                         const vsag::FilterPtr& is_id_allowed = nullptr) const;

    virtual void
    saveIndex(StreamWriter& writer) = 0;

    virtual uint64_t
    getMaxElements() = 0;

    virtual float
    getDistanceByLabel(LabelType label, const void* data_point) = 0;

    virtual float
    getDistanceByInternalId(uint32_t internal_id, const void* data_point) {
        return 0;
    }

    virtual float
    getSelfDistanceByInternalId(uint32_t internal_id) {
        return 0;
    }

    virtual tl::expected<vsag::DatasetPtr, vsag::Error>
    getBatchDistanceByLabel(const int64_t* ids, const void* data_point, int64_t count) = 0;

    virtual std::pair<int64_t, int64_t>
    getMinAndMaxId() = 0;

    virtual const float*
    getDataByLabel(LabelType label) const = 0;

    virtual void
    copyDataByLabel(LabelType label, void* data_point) = 0;

    virtual std::priority_queue<std::pair<float, LabelType>>
    bruteForce(const void* data_point,
               int64_t k,
               const vsag::FilterPtr is_id_allowed = nullptr) const = 0;

    virtual void
    resizeIndex(uint64_t new_max_elements) = 0;

    virtual uint64_t
    calcSerializeSize() = 0;

    virtual void
    loadIndex(StreamReader& reader, SpaceInterface* s, uint64_t max_elements_i = 0) = 0;

    virtual uint64_t
    getCurrentElementCount() = 0;

    virtual uint64_t
    getDeletedCount() = 0;

    virtual vsag::PGUnorderedMap<LabelType, InnerIdType>
    getDeletedElements() = 0;

    virtual bool
    isValidLabel(LabelType label) = 0;

    virtual bool
    isTombLabel(LabelType label) {
        return false;
    };

    virtual uint32_t
    getInternalId(LabelType label) {
        return 0;
    }

    virtual void
    getNeighborsInternalId(uint32_t internal_id, vsag::Vector<InnerIdType>& neighbor_ids) {
        return;
    }

    virtual bool
    init_memory_space() = 0;

    virtual uint64_t
    estimateMemory(uint64_t num_elements) {
        return 0;
    }

    virtual void
    setImmutable() {
        throw std::runtime_error("Index doesn't support set immutable");
    }

    virtual ~AlgorithmInterface() {
    }
};

template class AlgorithmInterface<float>;

}  // namespace hnswlib
