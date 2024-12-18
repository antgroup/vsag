
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

#include "pyramid_zparameters.h"

#include "diskann_zparameters.h"
#include "hnsw.h"
#include "hnsw_zparameters.h"

namespace vsag {

PyramidParameters
PyramidParameters::FromJson(JsonType& pyramid_param_obj,
                            const IndexCommonParam& index_common_param) {
    PyramidParameters obj;
    CHECK_ARGUMENT(
        pyramid_param_obj.contains(PYRAMID_PARAMETER_SUBINDEX_TYPE),
        fmt::format(
            "parameters[{}] must contains {}", INDEX_PYRAMID, PYRAMID_PARAMETER_SUBINDEX_TYPE));
    if (pyramid_param_obj[PYRAMID_PARAMETER_SUBINDEX_TYPE] == INDEX_HNSW) {
        auto hnsw_param_obj =
            HnswParameters::FromJson(pyramid_param_obj[INDEX_PARAM], index_common_param);
        obj.index_builder = [hnsw_param_obj, index_common_param]() {
            auto index = std::make_shared<HNSW>(hnsw_param_obj, index_common_param);
            if (auto result = index->InitMemorySpace(); not result.has_value()) {
                throw std::runtime_error("create hnsw failed in pyramid");
            }
            return index;
        };
    }
    return obj;
}

}  // namespace vsag
