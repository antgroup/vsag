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
#include "inner_string_params.h"
#include "io/io_headers.h"
#include "quantization/sparse_quantization/sparse_quantizer.h"
#include "sparse_vector_datacell.h"

namespace vsag {

template <typename IOTmpl>
FlattenInterfacePtr
make_sparse_vector_data_cell_instance(const FlattenInterfaceParamPtr& param,
                                      const IndexCommonParam& common_param) {
    if (param->name != SPARSE_VECTOR_DATA_CELL) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("Unknown flatten interface name: {}", param->name));
    }
    const auto& quantization = param->quantizer_parameter->GetTypeName();
    if (common_param.data_type_ == DataTypes::DATA_TYPE_INT8) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            fmt::format("INT8 data type does not support {} quantization", quantization));
    }
    if (common_param.data_type_ == DataTypes::DATA_TYPE_FP16 ||
        common_param.data_type_ == DataTypes::DATA_TYPE_BF16) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            fmt::format("FP16/BF16 data type does not support {} quantization", quantization));
    }
    if (common_param.metric_ != MetricType::METRIC_TYPE_IP) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("Sparse quantization only supports IP metric, got {}",
                                        static_cast<int>(common_param.metric_)));
    }
    if (quantization != QUANTIZATION_TYPE_VALUE_SPARSE) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("Unsupported quantization type: {}", quantization));
    }
    return std::make_shared<
        SparseVectorDataCell<SparseQuantizer<MetricType::METRIC_TYPE_IP>, IOTmpl>>(
        param->quantizer_parameter, param->io_parameter, common_param);
}

FlattenInterfacePtr
MakeSparseVectorDataCell(const FlattenInterfaceParamPtr& param,
                         const IndexCommonParam& common_param) {
    const auto& io_type = param->io_parameter->GetTypeName();
    if (io_type == IO_TYPE_VALUE_BLOCK_MEMORY_IO) {
        return make_sparse_vector_data_cell_instance<MemoryBlockIO>(param, common_param);
    }
    if (io_type == IO_TYPE_VALUE_MEMORY_IO) {
        return make_sparse_vector_data_cell_instance<MemoryIO>(param, common_param);
    }
    if (io_type == IO_TYPE_VALUE_BUFFER_IO) {
        return make_sparse_vector_data_cell_instance<BufferIO>(param, common_param);
    }
    if (io_type == IO_TYPE_VALUE_ASYNC_IO) {
        return make_sparse_vector_data_cell_instance<AsyncIO>(param, common_param);
    }
    if (io_type == IO_TYPE_VALUE_URING_IO) {
        return make_sparse_vector_data_cell_instance<UringIO>(param, common_param);
    }
    if (io_type == IO_TYPE_VALUE_MMAP_IO) {
        return make_sparse_vector_data_cell_instance<MMapIO>(param, common_param);
    }
    if (io_type == IO_TYPE_VALUE_READER_IO) {
        return make_sparse_vector_data_cell_instance<ReaderIO>(param, common_param);
    }
    return nullptr;
}

}  // namespace vsag
