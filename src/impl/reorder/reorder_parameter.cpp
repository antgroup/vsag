
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

#include "reorder_parameter.h"

#include <fmt/format.h>

#include "flatten_reorder_parameter.h"
#include "inner_string_params.h"
#include "pqr_reorder_parameter.h"
namespace vsag {

ReorderParameterPtr
CreateReorderParam(const JsonType& json) {
    std::string reorder_type = FLATTEN_REORDER;
    if (json.Contains(REORDER_TYPE)) {
        reorder_type = json[REORDER_TYPE].GetString();
    }
    ReorderParameterPtr param = nullptr;
    if (reorder_type == PQR_REORDER) {
        param = std::make_shared<PqrReorderParameter>();
    } else if (reorder_type == FLATTEN_REORDER) {
        param = std::make_shared<FlattenReorderParameter>();
    } else {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "unknown reorder type: " + reorder_type);
    }
    param->FromJson(json);
    return param;
}

}  // namespace vsag
