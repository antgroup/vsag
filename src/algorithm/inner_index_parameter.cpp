
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

#include "inner_index_parameter.h"

#include "common.h"
#include "data_cell/extra_info_datacell_parameter.h"
#include "data_cell/flatten_datacell_parameter.h"
#include "inner_string_params.h"
#include "logger.h"
#include "vsag/constants.h"

namespace vsag {

void
InnerIndexParameter::FromJson(const JsonType& json) {
    if (json.contains(USE_REORDER_KEY)) {
        this->use_reorder = json[USE_REORDER_KEY];
    }

    if (json.contains(USE_ATTRIBUTE_FILTER_KEY)) {
        this->use_attribute_filter = json[USE_ATTRIBUTE_FILTER_KEY];
    }

    if (json.contains(BUILD_THREAD_COUNT_KEY)) {
        this->build_thread_count = json[BUILD_THREAD_COUNT_KEY];
    }

    if (this->use_reorder) {
        CHECK_ARGUMENT(
            json.contains(PRECISE_CODES_KEY),
            fmt::format("ivf parameters must contains {} when enable reorder", PRECISE_CODES_KEY));
        this->precise_codes_param = std::make_shared<FlattenDataCellParameter>();
        this->precise_codes_param->FromJson(json[PRECISE_CODES_KEY]);
    }

    if (json.contains(STORE_RAW_VECTOR_KEY)) {
        this->store_raw_vector = json[STORE_RAW_VECTOR_KEY];
    }

    if (this->store_raw_vector) {
        this->raw_vector_param = std::make_shared<FlattenDataCellParameter>();
        this->raw_vector_param->FromJson(json[RAW_VECTOR_KEY]);
    }

    if (json.contains(EXTRA_INFO_KEY)) {
        this->extra_info_param = std::make_shared<ExtraInfoDataCellParameter>();
        this->extra_info_param->FromJson(json[EXTRA_INFO_KEY]);
    }
}

JsonType
InnerIndexParameter::ToJson() const {
    JsonType json;
    json[USE_REORDER_KEY] = this->use_reorder;
    json[BUILD_THREAD_COUNT_KEY] = this->build_thread_count;
    json[USE_ATTRIBUTE_FILTER_KEY] = this->use_attribute_filter;
    if (use_reorder) {
        json[PRECISE_CODES_KEY] = this->precise_codes_param->ToJson();
    }
    json[STORE_RAW_VECTOR_KEY] = this->store_raw_vector;
    if (this->store_raw_vector) {
        json[RAW_VECTOR_KEY] = this->raw_vector_param->ToJson();
    }
    if (this->extra_info_param) {
        json[EXTRA_INFO_KEY] = this->extra_info_param->ToJson();
    }

    return json;
}
bool
InnerIndexParameter::CheckCompatibility(const ParamPtr& other) const {
    auto inner_index_param = std::dynamic_pointer_cast<InnerIndexParameter>(other);
    if (not inner_index_param) {
        logger::error(
            "InnerIndexParameter::CheckCompatibility: other parameter is not InnerIndexParameter");
        return false;
    }
    if (this->use_reorder != inner_index_param->use_reorder) {
        logger::error("InnerIndexParameter::CheckCompatibility: use_reorder mismatch");
        return false;
    }
    if (this->use_reorder) {
        if (not this->precise_codes_param->CheckCompatibility(
                inner_index_param->precise_codes_param)) {
            logger::error("InnerIndexParameter::CheckCompatibility: precise_codes_param mismatch");
            return false;
        }
    }

    if (this->use_attribute_filter != inner_index_param->use_attribute_filter) {
        logger::error("InnerIndexParameter::CheckCompatibility: use_attribute_filter mismatch");
        return false;
    }

    return true;
}
}  // namespace vsag
