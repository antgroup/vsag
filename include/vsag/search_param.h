
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

#include <vsag/allocator.h>
#include <vsag/filter.h>
#include <vsag/iterator_context.h>

#include <memory>

namespace vsag {

struct SearchParam {
public:
    SearchParam(bool iter_filter_flag,
                const std::string& parameter,
                FilterPtr flt,
                Allocator* alloc)
        : is_iter_filter(iter_filter_flag),
          is_last_search(false),
          parameters(parameter),
          filter(flt),
          allocator(alloc) {
    }

    SearchParam(bool iter_filter_flag,
                const std::string& parameter,
                FilterPtr flt,
                Allocator* alloc,
                IteratorContext* ctx,
                bool last_search_flag)
        : is_iter_filter(iter_filter_flag),
          is_last_search(last_search_flag),
          parameters(parameter),
          filter(flt),
          allocator(alloc),
          iter_ctx(ctx) {
    }

public:
    bool is_iter_filter{false};
    bool is_last_search{false};
    const std::string& parameters;
    FilterPtr filter{nullptr};
    Allocator* allocator{nullptr};
    IteratorContext* iter_ctx{nullptr};
};

};  // namespace vsag