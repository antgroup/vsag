
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
    : allocator_(allocator), mutex_blocks_(allocator) {
    this->Resize(element_num);
}

void
PointsMutex::MutexBlockDeleter::operator()(MutexBlock* block) const {
    allocator->Delete(block);
}

void
PointsMutex::SharedLock(uint32_t i) {
    GetMutex(i).lock_shared();
}

void
PointsMutex::SharedUnlock(uint32_t i) {
    GetMutex(i).unlock_shared();
}

void
PointsMutex::Lock(uint32_t i) {
    GetMutex(i).lock();
}

void
PointsMutex::Unlock(uint32_t i) {
    GetMutex(i).unlock();
}

void
PointsMutex::Resize(uint32_t new_element_num) {
    const auto required_blocks =
        (static_cast<uint64_t>(new_element_num) + kMutexesPerBlock - 1) / kMutexesPerBlock;
    if (new_element_num > element_num_) {
        if (required_blocks > mutex_blocks_.size()) {
            mutex_blocks_.reserve(required_blocks);
            while (mutex_blocks_.size() < required_blocks) {
                mutex_blocks_.emplace_back(allocator_->New<MutexBlock>(),
                                           MutexBlockDeleter{allocator_});
            }
        }
    } else if (new_element_num < element_num_) {
        while (mutex_blocks_.size() > required_blocks) {
            mutex_blocks_.pop_back();
        }
    }
    element_num_ = new_element_num;
}

uint64_t
PointsMutex::GetMemoryUsage() {
    return mutex_blocks_.size() * sizeof(MutexBlock) +
           mutex_blocks_.capacity() * sizeof(MutexBlockPtr);
}

}  // namespace vsag
