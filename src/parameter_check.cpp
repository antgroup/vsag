
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

#include <stdexcept>

#include "factory/resource_owner_wrapper.h"
#include "index/diskann_zparameters.h"
#include "index/hnsw_zparameters.h"
#include "utils/util_functions.h"
#include "vsag/constants.h"
#include "vsag/errors.h"
#include "vsag/expected.hpp"
#include "vsag/index.h"
#include "vsag_exception.h"

namespace vsag {

tl::expected<bool, Error>
check_diskann_hnsw_build_parameters(const std::string& json_string) {
    JsonType parsed_params = JsonType::Parse(json_string);
    std::shared_ptr<vsag::Resource> resource =
        std::make_shared<vsag::ResourceOwnerWrapper>(new vsag::Resource(), true);

    IndexCommonParam index_common_params;
    try {
        index_common_params = IndexCommonParam::CheckAndCreate(parsed_params, resource);
    } catch (const VsagException& e) {
        return tl::unexpected<Error>(e.error_);
    }

    if (not parsed_params.Contains(INDEX_HNSW)) {
        LOG_ERROR_AND_RETURNS(ErrorType::INVALID_ARGUMENT,
                              fmt::format("parameters must contains {}", INDEX_HNSW));
    }

    if (not parsed_params.Contains(INDEX_DISKANN)) {
        LOG_ERROR_AND_RETURNS(ErrorType::INVALID_ARGUMENT,
                              fmt::format("parameters must contains {}", INDEX_DISKANN));
    }
    if (auto ret =
            try_parse_parameters<HnswParameters>(parsed_params[INDEX_HNSW], index_common_params);
        not ret.has_value()) {
        return tl::unexpected(ret.error());
    }
    if (auto ret = try_parse_parameters<DiskannParameters>(parsed_params[INDEX_DISKANN],
                                                           index_common_params);
        not ret.has_value()) {
        return tl::unexpected(ret.error());
    }
    return true;
}

tl::expected<bool, Error>
check_diskann_hnsw_search_parameters(const std::string& json_string) {
    if (auto ret = try_parse_parameters<HnswSearchParameters>(json_string); not ret.has_value()) {
        return tl::unexpected(ret.error());
    }
    if (auto ret = try_parse_parameters<DiskannSearchParameters>(json_string);
        not ret.has_value()) {
        return tl::unexpected(ret.error());
    }
    return true;
}

}  // namespace vsag
