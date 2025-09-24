
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

#include "flatten_interface_parameter.h"

#include "flatten_datacell_parameter.h"
#include "inner_string_params.h"
#include "sparse_vector_datacell_parameter.h"

namespace vsag {

FlattenInterfaceParamPtr
CreateFlattenParam(const JsonType& json) {
    FlattenInterfaceParamPtr param = nullptr;
    if (json.Contains(CODES_TYPE_KEY) && json[CODES_TYPE_KEY].GetString() == SPARSE_CODES) {
        param = std::make_shared<SparseVectorDataCellParameter>();
    } else {
        param = std::make_shared<FlattenDataCellParameter>();
    }
    param->FromJson(json);
    return param;
}

}  // namespace vsag
