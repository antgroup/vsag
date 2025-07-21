
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

#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <mutex>

#include "impl/allocator/default_allocator.h"
#include "storage/stream_writer.h"
namespace vsag {

extern void*
allocate(size_t size);

extern void
deallocate(void* p);

extern void*
reallocate(void* p, size_t size);

}  // namespace vsag

namespace hnswlib {
typedef unsigned short int vl_type;

class VisitedList {
public:
    vl_type curV{0};
    vl_type* mass{nullptr};
    uint64_t numelements{0};

    VisitedList(uint64_t numelements1, vsag::Allocator* allocator) : allocator_(allocator) {
        curV = -1;
        numelements = numelements1;
    }

    void
    reset() {
        if (not mass) {
            mass = (vl_type*)allocator_->Allocate(numelements * sizeof(vl_type));
        }
        curV++;
        if (curV == 0) {
            memset(mass, 0, sizeof(vl_type) * numelements);
            curV++;
        }
    }

    ~VisitedList() {
        allocator_->Deallocate(mass);
    }

    vsag::Allocator* allocator_;
};

using VisitedListPtr = std::shared_ptr<VisitedList>;

///////////////////////////////////////////////////////////
//
// Class for multi-threaded pool-management of VisitedLists
//
/////////////////////////////////////////////////////////

class VisitedListPool {
public:
    VisitedListPool(uint64_t max_element_count, vsag::Allocator* allocator)
        : allocator_(allocator), pool_(allocator), max_element_count_(max_element_count) {
    }

    VisitedListPtr
    getFreeVisitedList() {
        VisitedListPtr rez;
        {
            std::unique_lock<std::mutex> lock(poolguard_);
            if (not pool_.empty()) {
                rez = pool_.front();
                pool_.pop_front();
            } else {
                rez = std::make_shared<VisitedList>(max_element_count_, allocator_);
            }
        }
        rez->reset();
        return rez;
    }

    void
    releaseVisitedList(VisitedListPtr vl) {
        std::unique_lock<std::mutex> lock(poolguard_);
        pool_.push_back(vl);
    }

private:
    vsag::Deque<VisitedListPtr> pool_;
    std::mutex poolguard_;
    uint64_t max_element_count_;
    vsag::Allocator* allocator_;
};

}  // namespace hnswlib
