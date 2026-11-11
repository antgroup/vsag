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

#include "flatten_interface_factory.h"
#include "index_common_param.h"
#include "inner_string_params.h"
#include "io/io_headers.h"

namespace vsag {

// Defined and explicitly instantiated per (metric, IO) in generated shard translation units
// (see flatten_datacell_factory_shard.cpp.in) to bound per-TU compiler memory.
template <MetricType metric, typename IOTmpl>
FlattenInterfacePtr
MakeFlattenDataCellShard(const FlattenInterfaceParamPtr& param,
                         const IndexCommonParam& common_param);

namespace {

template <MetricType metric>
FlattenInterfacePtr
DispatchByIO(const FlattenInterfaceParamPtr& param, const IndexCommonParam& common_param) {
    const auto& io_type = param->io_parameter->GetTypeName();
    if (io_type == IO_TYPE_VALUE_BLOCK_MEMORY_IO) {
        return MakeFlattenDataCellShard<metric, MemoryBlockIO>(param, common_param);
    }
    if (io_type == IO_TYPE_VALUE_MEMORY_IO) {
        return MakeFlattenDataCellShard<metric, MemoryIO>(param, common_param);
    }
    if (io_type == IO_TYPE_VALUE_BUFFER_IO) {
        return MakeFlattenDataCellShard<metric, BufferIO>(param, common_param);
    }
    if (io_type == IO_TYPE_VALUE_ASYNC_IO) {
        return MakeFlattenDataCellShard<metric, AsyncIO>(param, common_param);
    }
    if (io_type == IO_TYPE_VALUE_URING_IO) {
        return MakeFlattenDataCellShard<metric, UringIO>(param, common_param);
    }
    if (io_type == IO_TYPE_VALUE_MMAP_IO) {
        return MakeFlattenDataCellShard<metric, MMapIO>(param, common_param);
    }
    if (io_type == IO_TYPE_VALUE_READER_IO) {
        return MakeFlattenDataCellShard<metric, ReaderIO>(param, common_param);
    }
    if (io_type == IO_TYPE_VALUE_USER_DEFINED_IO) {
        return MakeFlattenDataCellShard<metric, UserDefinedIO>(param, common_param);
    }
    return nullptr;
}

}  // namespace

FlattenInterfacePtr
MakeFlattenDataCell(const FlattenInterfaceParamPtr& param, const IndexCommonParam& common_param) {
    if (common_param.metric_ == MetricType::METRIC_TYPE_L2SQR) {
        return DispatchByIO<MetricType::METRIC_TYPE_L2SQR>(param, common_param);
    }
    if (common_param.metric_ == MetricType::METRIC_TYPE_IP) {
        return DispatchByIO<MetricType::METRIC_TYPE_IP>(param, common_param);
    }
    if (common_param.metric_ == MetricType::METRIC_TYPE_COSINE) {
        return DispatchByIO<MetricType::METRIC_TYPE_COSINE>(param, common_param);
    }
    return nullptr;
}

}  // namespace vsag
