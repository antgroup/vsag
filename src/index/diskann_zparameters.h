
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

#include <distance.h>

#include <string>

#include "index_common_param.h"

namespace vsag {

struct DiskannParameters {
public:
    static DiskannParameters
    FromJson(const JsonType& diskann_param_obj, const IndexCommonParam& index_common_param);

public:
    // require vars
    int64_t dim{-1};
    diskann::Metric metric{diskann::Metric::L2};
    int64_t max_degree{-1};
    int64_t ef_construction{-1};
    int64_t pq_dims{-1};
    float pq_sample_rate{.0f};

    // optional vars with default value
    bool use_preload = false;
    bool use_reference = true;
    bool use_opq = false;
    bool use_bsa = false;
    bool use_async_io = false;

    // use new construction method
    std::string graph_type = "vamana";
    float alpha = 1.2;
    int64_t turn = 40;
    float sample_rate = 0.3;

private:
    DiskannParameters() = default;
};

struct DiskannSearchParameters {
public:
    static DiskannSearchParameters
    FromJson(const std::string& json_string);

public:
    // required vars
    int64_t ef_search{-1};
    uint64_t beam_search{0};
    int64_t io_limit{-1};

    // optional vars with default value
    bool use_reorder = false;
    bool use_async_io = false;

private:
    DiskannSearchParameters() = default;
};

}  // namespace vsag
