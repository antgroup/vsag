
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
#include <mutex>

#include "../../default_allocator.h"
#include "stream_writer.h"

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
    vl_type curV;
    vl_type* mass;
    unsigned int numelements;

    VisitedList(int numelements1, vsag::Allocator* allocator) : allocator_(allocator) {
        curV = -1;
        numelements = numelements1;
        mass = (vl_type*)allocator_->Allocate(numelements * sizeof(vl_type));
    }

    void
    reset() {
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

///////////////////////////////////////////////////////////
//
// Class for multi-threaded pool-management of VisitedLists
//
/////////////////////////////////////////////////////////

class VisitedListPool {
    std::deque<VisitedList*> pool;
    std::mutex poolguard;
    int numelements;

public:
    VisitedListPool(int initmaxpools, int numelements1, vsag::Allocator* allocator)
        : allocator_(allocator) {
        numelements = numelements1;
        for (int i = 0; i < initmaxpools; i++)
            pool.push_front(new VisitedList(numelements, allocator_));
    }

    VisitedList*
    getFreeVisitedList() {
        VisitedList* rez;
        {
            std::unique_lock<std::mutex> lock(poolguard);
            if (pool.size() > 0) {
                rez = pool.front();
                pool.pop_front();
            } else {
                rez = new VisitedList(numelements, allocator_);
            }
        }
        rez->reset();
        return rez;
    }

    void
    releaseVisitedList(VisitedList* vl) {
        std::unique_lock<std::mutex> lock(poolguard);
        pool.push_front(vl);
    }

    ~VisitedListPool() {
        while (pool.size()) {
            VisitedList* rez = pool.front();
            pool.pop_front();
            delete rez;
        }
    }

private:
    vsag::Allocator* allocator_;
};

}  // namespace hnswlib
