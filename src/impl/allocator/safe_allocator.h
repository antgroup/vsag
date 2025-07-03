
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

#include <memory>
#include <new>

#include "default_allocator.h"
#include "vsag/allocator.h"

namespace vsag {

class SafeAllocator : public Allocator {
public:
    static std::shared_ptr<Allocator>
    FactoryDefaultAllocator() {
        return std::make_shared<SafeAllocator>(new DefaultAllocator(), true);
    }

public:
    explicit SafeAllocator(Allocator* raw_allocator) : SafeAllocator(raw_allocator, false){};

    explicit SafeAllocator(Allocator* raw_allocator, bool owned)
        : raw_allocator_(raw_allocator), owned_(owned) {
    }

    explicit SafeAllocator(const std::shared_ptr<Allocator>& raw_allocator)
        : raw_allocator_shared_(raw_allocator), raw_allocator_(raw_allocator.get()) {
    }

    std::string
    Name() override {
        return raw_allocator_->Name() + "_safewrapper";
    }

    void*
    Allocate(size_t size) override {
        auto ret = raw_allocator_->Allocate(size);
        if (not ret) {
            throw std::bad_alloc();
        }
        return ret;
    }

    void
    Deallocate(void* p) override {
        raw_allocator_->Deallocate(p);
    }

    void*
    Reallocate(void* p, size_t size) override {
        auto ret = raw_allocator_->Reallocate(p, size);
        if (not ret) {
            throw std::bad_alloc();
        }
        return ret;
    }
    Allocator*
    GetRawAllocator() {
        return raw_allocator_;
    }

public:
    ~SafeAllocator() override {
        if (owned_) {
            delete raw_allocator_;
        }
    }

private:
    Allocator* const raw_allocator_{nullptr};

    std::shared_ptr<Allocator> const raw_allocator_shared_;

    bool owned_{false};
};

}  // namespace vsag
