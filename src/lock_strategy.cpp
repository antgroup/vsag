
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

#include "lock_strategy.h"

namespace vsag {

PointsMutex::PointsMutex(uint32_t element_num, Allocator* allocator)
    : allocator_(allocator),
      neighbors_mutex_(element_num, nullptr, allocator),
      element_num_(element_num) {
    for (int i = 0; i < element_num_; ++i) {
        neighbors_mutex_[i] = std::make_shared<std::shared_mutex>();
    }
}

void
PointsMutex::SharedLock(uint32_t i) {
    neighbors_mutex_[i]->lock_shared();
}

void
PointsMutex::SharedUnlock(uint32_t i) {
    neighbors_mutex_[i]->unlock_shared();
}

void
PointsMutex::Lock(uint32_t i) {
    neighbors_mutex_[i]->lock();
}

void
PointsMutex::Unlock(uint32_t i) {
    neighbors_mutex_[i]->unlock();
}

void
PointsMutex::Resize(uint32_t new_element_num) {
    neighbors_mutex_.resize(new_element_num);
    if (new_element_num > element_num_) {
        for (auto i = element_num_; i < new_element_num; ++i) {
            neighbors_mutex_[i] = AllocateShared<std::shared_mutex>(allocator_);
        }
    }
    element_num_ = new_element_num;
}

}  // namespace vsag
