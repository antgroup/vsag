
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

#include "extra_info_interface.h"

#include <fmt/format.h>

#include "extra_info_datacell.h"
#include "inner_string_params.h"
#include "io/io_headers.h"
#include "quantization/quantizer_headers.h"

namespace vsag {
template <typename IOTemp>
static ExtraInfoInterfacePtr
make_instance(const ExtraInfoDataCellParamPtr& param, const IndexCommonParam& common_param) {
    auto& io_param = param->io_parameter;
    return std::make_shared<ExtraInfoDataCell<IOTemp>>(io_param, common_param);
}

ExtraInfoInterfacePtr
ExtraInfoInterface::MakeInstance(const ExtraInfoDataCellParamPtr& param,
                                 const IndexCommonParam& common_param) {
    auto io_type_name = param->io_parameter->GetTypeName();
    if (io_type_name == IO_TYPE_VALUE_BLOCK_MEMORY_IO) {
        return make_instance<MemoryBlockIO>(param, common_param);
    }
    throw VsagException(ErrorType::INVALID_ARGUMENT,
                        fmt::format("Extra Info not support {} IO type", io_type_name));
}

}  // namespace vsag
