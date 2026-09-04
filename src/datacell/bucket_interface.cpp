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
#include "bucket_interface_factory_impl.h"
#include "inner_string_params.h"
#include "io/common/io_type_dispatch.h"

namespace vsag {

BucketInterfacePtr
BucketInterface::MakeInstance(const BucketDataCellParamPtr& param,
                              const IndexCommonParam& common_param) {
    if (!param || !param->io_parameter || !param->quantizer_parameter) {
        return nullptr;
    }
    return VisitIOKind(param->io_parameter->Kind(), [&](auto tag) -> BucketInterfacePtr {
        using IO = typename decltype(tag)::Type;
        if constexpr (std::is_same_v<IO, MemoryBlockIO>) {
            return MakeMemoryBlockBucketDataCell(param, common_param);
        } else if constexpr (std::is_same_v<IO, MemoryIO>) {
            return MakeMemoryBucketDataCell(param, common_param);
        } else if constexpr (std::is_same_v<IO, MMapIO>) {
            return MakeMMapBucketDataCell(param, common_param);
        } else if constexpr (std::is_same_v<IO, AsyncIO>) {
            return MakeAsyncBucketDataCell(param, common_param);
        } else if constexpr (std::is_same_v<IO, UringIO>) {
            return MakeUringBucketDataCell(param, common_param);
        } else if constexpr (std::is_same_v<IO, BufferIO>) {
            return MakeBufferBucketDataCell(param, common_param);
        } else if constexpr (std::is_same_v<IO, ReaderIO>) {
            return MakeReaderBucketDataCell(param, common_param);
        } else {
            return nullptr;
        }
    });
}
}  // namespace vsag
