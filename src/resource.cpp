
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

#include "vsag/resource.h"

#include "default_allocator.h"
#include "safe_allocator.h"

namespace vsag {
Resource::Resource(Allocator* allocator) {
    if (allocator == nullptr) {
        this->allocator = SafeAllocator::FactoryDefaultAllocator();
    } else {
        this->allocator = std::make_shared<SafeAllocator>(allocator, false);
    }
}
}  // namespace vsag
