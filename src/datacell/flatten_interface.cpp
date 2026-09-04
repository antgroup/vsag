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

#include "flatten_interface.h"

#include "flatten_interface_factory.h"
#include "index_common_param.h"
#include "inner_string_params.h"

namespace vsag {

IndexCommonParam
FlattenInterface::ExportCommonParam() {
    throw VsagException(ErrorType::INTERNAL_ERROR, "ExportCommonParam is not implemented");
}

FlattenInterfacePtr
FlattenInterface::MakeInstance(const FlattenInterfaceParamPtr& param,
                               const IndexCommonParam& common_param) {
    if (param->name == MULTI_VECTOR_DATA_CELL) {
        return MakeMultiVectorDataCell(param, common_param);
    }
    if (param->name == SPARSE_VECTOR_DATA_CELL) {
        return MakeSparseVectorDataCell(param, common_param);
    }
    return MakeFlattenDataCell(param, common_param);
}

}  // namespace vsag
