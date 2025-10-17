
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

#include "noncontinuous_allocator.h"

namespace vsag {
class Allocator;

template <typename IOTmpl>
class IOArray {
public:
    static constexpr bool InMemory = IOTmpl::InMemory;

public:
    template <typename... Args>
    explicit IOArray(Allocator* allocator, Args&&... args) : allocator_(allocator) {
        non_contiguous_allocator_ = std::make_unique<NonContiguousAllocator>(allocator);
        if constexpr (InMemory) {
            io_create_func_ = [allocator, args...]() {
                return std::make_shared<IOTmpl>(std::forward<Args>(args)..., allocator);
            };
        } else {
            io_create_func_ =
                [non_contiguous_allocator = non_contiguous_allocator_.get(), allocator, args...]() {
                    return std::make_shared<IOTmpl>(
                        non_contiguous_allocator, allocator, std::forward<Args>(args)...);
                };
        }
    }

    IOTmpl&
    operator[](int64_t index) {
        return *datas_[index];
    }

    const IOTmpl&
    operator[](int64_t index) const {
        return *datas_[index];
    }

    IOTmpl&
    At(int64_t index) {
        if (index >= datas_.size()) {
            throw std::out_of_range("IOArray index out of range");
        }
        return *datas_[index];
    }

    const IOTmpl&
    At(int64_t index) const {
        if (index >= datas_.size()) {
            throw std::out_of_range("IOArray index out of range");
        }
        return *datas_[index];
    }

    void
    Resize(int64_t size) {
        auto cur_size = datas_.size();
        this->data_.resize(size);
        for (int64_t i = cur_size; i < size; i++) {
            datas_[i] = this->io_create_func_();
        }
    }

private:
    Allocator* const allocator_{nullptr};

    Vector<std::shared_ptr<IOTmpl>> datas_;

    std::unique_ptr<NonContiguousAllocator> non_contiguous_allocator_{nullptr};

    std::function<std::shared_ptr<IOTmpl>()> io_create_func_;
};
}  // namespace vsag
