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
#include <memory>

#include "vsag/allocator.h"

namespace vsag {

class Page {
public:
    explicit Page(Allocator* allocator) : allocator_(allocator) {
        data_ = static_cast<uint8_t*>(allocator_->Allocate(Page::DEFAULT_PAGE_SIZE));
    }

    ~Page() {
        if (data_ != nullptr) {
            allocator_->Deallocate(data_);
        }
    }

    Page(const Page&) = delete;
    Page&
    operator=(const Page&) = delete;

    [[nodiscard]] uint8_t*
    Data() {
        return data_;
    }
    [[nodiscard]] const uint8_t*
    Data() const {
        return data_;
    }

    static constexpr uint64_t DEFAULT_PAGE_SIZE = 128 * 1024;

private:
    uint8_t* data_{nullptr};
    Allocator* allocator_{nullptr};
};

using PagePtr = std::shared_ptr<Page>;

}  // namespace vsag
