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

#include "flatten_datacell_factory_impl.h"
#include "quantization/binary_quantizer.h"

namespace vsag {

namespace {

template <typename IOTmpl>
FlattenInterfacePtr
make_binary_flatten_data_cell(const FlattenInterfaceParamPtr& param,
                              const IndexCommonParam& common_param) {
    return MakeFlattenDataCellInstance<BinaryQuantizer, IOTmpl>(param, common_param);
}

}  // namespace

// Binary vectors compared with the Hamming metric are stored only by the exact binary quantizer;
// every IO type supported by the dense flatten cells is available.
FlattenInterfacePtr
make_flatten_data_cell_hamming(const FlattenInterfaceParamPtr& param,
                               const IndexCommonParam& common_param) {
    CHECK_ARGUMENT(common_param.data_type_ == DataTypes::DATA_TYPE_BINARY,
                   fmt::format("metric {} requires dtype {}, got {}",
                               METRIC_HAMMING,
                               DATATYPE_BINARY,
                               ToString(common_param.data_type_)));
    const auto quantization = param->quantizer_parameter->GetTypeName();
    CHECK_ARGUMENT(quantization == QUANTIZATION_TYPE_VALUE_BINARY,
                   fmt::format("dtype {} stores packed bits exactly and only supports {} "
                               "quantization, got {}",
                               DATATYPE_BINARY,
                               QUANTIZATION_TYPE_VALUE_BINARY,
                               quantization));

    const auto& io_type = param->io_parameter->GetTypeName();
    if (io_type == IO_TYPE_VALUE_BLOCK_MEMORY_IO) {
        return make_binary_flatten_data_cell<MemoryBlockIO>(param, common_param);
    }
    if (io_type == IO_TYPE_VALUE_MEMORY_IO) {
        return make_binary_flatten_data_cell<MemoryIO>(param, common_param);
    }
    if (io_type == IO_TYPE_VALUE_BUFFER_IO) {
        return make_binary_flatten_data_cell<BufferIO>(param, common_param);
    }
    if (io_type == IO_TYPE_VALUE_ASYNC_IO) {
        return make_binary_flatten_data_cell<AsyncIO>(param, common_param);
    }
    if (io_type == IO_TYPE_VALUE_URING_IO) {
        return make_binary_flatten_data_cell<UringIO>(param, common_param);
    }
    if (io_type == IO_TYPE_VALUE_MMAP_IO) {
        return make_binary_flatten_data_cell<MMapIO>(param, common_param);
    }
    if (io_type == IO_TYPE_VALUE_READER_IO) {
        return make_binary_flatten_data_cell<ReaderIO>(param, common_param);
    }
    return nullptr;
}

}  // namespace vsag
