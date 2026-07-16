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

#include "bucket_interface.h"

#include "bucket_interface_factory.h"
#include "inner_string_params.h"

namespace vsag {

BucketInterfacePtr
BucketInterface::MakeInstance(const BucketDataCellParamPtr& param,
                              const IndexCommonParam& common_param) {
    if (!param || !param->io_parameter || !param->quantizer_parameter) {
        return nullptr;
    }
    auto io_type_name = param->io_parameter->GetTypeName();
    if (io_type_name == IO_TYPE_VALUE_BLOCK_MEMORY_IO) {
        return MakeMemoryBlockBucketDataCell(param, common_param);
    }
    if (io_type_name == IO_TYPE_VALUE_MEMORY_IO) {
        return MakeMemoryBucketDataCell(param, common_param);
    }
    if (io_type_name == IO_TYPE_VALUE_MMAP_IO) {
        return MakeMMapBucketDataCell(param, common_param);
    }
    if (io_type_name == IO_TYPE_VALUE_ASYNC_IO) {
        return MakeAsyncBucketDataCell(param, common_param);
    }
    if (io_type_name == IO_TYPE_VALUE_BUFFER_IO) {
        return MakeBufferBucketDataCell(param, common_param);
    }
    return nullptr;
}
}  // namespace vsag
