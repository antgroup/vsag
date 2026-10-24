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

#include <type_traits>

#include "datacell/flatten_interface_parameter.h"
#include "datacell/sq_local_split_datacell.h"
#include "io/common/io_type_dispatch.h"

namespace vsag {

template <MetricType metric>
FlattenInterfacePtr
MakeSQLocalSplitDataCell(const FlattenInterfaceParamPtr& param,
                         const IndexCommonParam& common_param) {
    CHECK_ARGUMENT(param->io_parameter != nullptr, "sq_local_split requires an io parameter");
    const auto base_kind = param->io_parameter->Kind();
    const auto supplement_kind = param->supplement_io_parameter == nullptr
                                     ? base_kind
                                     : param->supplement_io_parameter->Kind();
    CHECK_ARGUMENT(base_kind == supplement_kind,
                   "sq_local_split requires the same io kind for the filter and the supplement "
                   "storage");

    auto create = [&](auto tag) -> FlattenInterfacePtr {
        using IO = typename decltype(tag)::Type;
        if constexpr (std::is_void_v<IO> or std::is_same_v<IO, UringIO> or
                      std::is_same_v<IO, ReaderIO>) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                "sq_local_split does not support this io type");
        } else {
            return std::make_shared<SQLocalSplitDataCell<metric, IO, IO>>(
                param->quantizer_parameter,
                param->io_parameter,
                param->supplement_io_parameter,
                common_param);
        }
    };
    return VisitIOKind(base_kind, create);
}

}  // namespace vsag
