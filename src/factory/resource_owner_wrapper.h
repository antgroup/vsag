
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

#include "vsag/resource.h"

namespace vsag {
class ResourceOwnerWrapper : public Resource {
public:
    explicit ResourceOwnerWrapper(Resource* resource, bool owned = false)
        : Resource(nullptr, nullptr), resource_(resource), owned_(owned) {
    }

    std::shared_ptr<Allocator>
    GetAllocator() const override {
        return resource_->GetAllocator();
    }

    std::shared_ptr<ThreadPool>
    GetThreadPool() const override {
        return resource_->GetThreadPool();
    }

    ~ResourceOwnerWrapper() override {
        if (owned_) {
            delete resource_;
        }
    }

private:
    Resource* resource_{nullptr};
    bool owned_{false};
};
}  // namespace vsag
