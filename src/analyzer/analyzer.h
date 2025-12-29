
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

#include <memory>

#include "algorithm/hgraph.h"
#include "algorithm/inner_index_interface.h"
#include "utils/pointer_define.h"
#include "vsag/allocator.h"
namespace vsag {

DEFINE_POINTER(AnalyzerBase);

class AnalyzerBase {
public:
    AnalyzerBase(Allocator* allocator, uint32_t total_count)
        : allocator_(allocator), total_count_(total_count) {
    }

    virtual JsonType
    GetStats() = 0;

    virtual ~AnalyzerBase() = default;

    virtual JsonType
    AnalyzeIndexBySearch(const SearchRequest& request) = 0;

protected:
    Allocator* allocator_;
    uint32_t total_count_;
    uint32_t dim_;
};

struct AnalyzerParam {
public:
    AnalyzerParam(Allocator* allocator) : allocator(allocator) {
    }

public:
    Allocator* allocator;
    int64_t topk{100};
    uint64_t base_sample_size{10};
    std::string search_params;
};

AnalyzerBasePtr
CreateAnalyzer(const InnerIndexInterface* index, const AnalyzerParam& param);

}  // namespace vsag
