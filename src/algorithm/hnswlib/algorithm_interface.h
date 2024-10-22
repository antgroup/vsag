
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

#include "base_filter_functor.h"
#include "space_interface.h"
#include "typing.h"

namespace hnswlib {

using LabelType = vsag::LabelType;

template <typename dist_t>
class AlgorithmInterface {
public:
    virtual bool
    addPoint(const void* datapoint, LabelType label) = 0;

    virtual std::priority_queue<std::pair<dist_t, LabelType>>
    searchKnn(const void*,
              size_t,
              size_t,
              vsag::BaseFilterFunctor* isIdAllowed = nullptr) const = 0;

    virtual std::priority_queue<std::pair<dist_t, LabelType>>
    searchRange(const void*,
                float,
                size_t,
                vsag::BaseFilterFunctor* isIdAllowed = nullptr) const = 0;

    // Return k nearest neighbor in the order of closer fist
    virtual std::vector<std::pair<dist_t, LabelType>>
    searchKnnCloserFirst(const void* query_data,
                         size_t k,
                         size_t ef,
                         vsag::BaseFilterFunctor* isIdAllowed = nullptr) const;

    virtual void
    saveIndex(const std::string& location) = 0;

    virtual void
    saveIndex(void* d) = 0;

    virtual void
    saveIndex(std::ostream& out_stream) = 0;

    virtual size_t
    getMaxElements() = 0;

    virtual float
    getDistanceByLabel(LabelType label, const void* data_point) = 0;

    virtual const float*
    getDataByLabel(LabelType label) const = 0;

    virtual std::priority_queue<std::pair<float, LabelType>>
    bruteForce(const void* data_point, int64_t k) = 0;

    virtual void
    resizeIndex(size_t new_max_elements) = 0;

    virtual size_t
    calcSerializeSize() = 0;

    virtual void
    loadIndex(std::function<void(uint64_t, uint64_t, void*)> read_func,
              SpaceInterface* s,
              size_t max_elements_i = 0) = 0;

    virtual void
    loadIndex(std::istream& in_stream, SpaceInterface* s, size_t max_elements_i = 0) = 0;

    virtual size_t
    getCurrentElementCount() = 0;

    virtual size_t
    getDeletedCount() = 0;

    virtual bool
    isValidLabel(LabelType label) = 0;

    virtual bool
    init_memory_space() = 0;

    virtual ~AlgorithmInterface() {
    }
};

template class AlgorithmInterface<float>;

}  // namespace hnswlib
