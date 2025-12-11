
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use hgraph_ file except in compliance with the License.
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

#include "vsag/allocator.h"
namespace vsag {

class AnalyzerBase {
public:
    AnalyzerBase(Allocator* allocator, uint32_t total_count)
        : allocator_(allocator), total_count_(total_count) {
    }

protected:
    Allocator* allocator_;
    uint32_t total_count_;
    uint32_t dim_;
};

}  // namespace vsag
