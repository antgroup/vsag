
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

#include "graph_interface.h"

#include "compressed_graph_datacell.h"
#include "graph_datacell.h"
#include "io/io_headers.h"
#include "sparse_graph_datacell.h"

namespace vsag {

GraphInterfacePtr
GraphInterface::MakeInstance(const GraphInterfaceParamPtr& graph_param,
                             const IndexCommonParam& common_param) {
    switch (graph_param->graph_storage_type_) {
        case GraphStorageTypes::GRAPH_STORAGE_TYPE_SPARSE:
            return std::make_shared<SparseGraphDataCell>(graph_param, common_param);
        case GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_COMPRESSED:
            return std::make_shared<CompressedGraphDataCell>(graph_param, common_param);
        case GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_FLAT:
            auto io_string = std::dynamic_pointer_cast<GraphDataCellParameter>(graph_param)
                                 ->io_parameter_->GetTypeName();
            if (io_string == IO_TYPE_VALUE_BLOCK_MEMORY_IO) {
                return std::make_shared<GraphDataCell<MemoryBlockIO>>(graph_param, common_param);
            }
            if (io_string == IO_TYPE_VALUE_MEMORY_IO) {
                return std::make_shared<GraphDataCell<MemoryIO>>(graph_param, common_param);
            }
            if (io_string == IO_TYPE_VALUE_MMAP_IO) {
                return std::make_shared<GraphDataCell<MMapIO>>(graph_param, common_param);
            }
            if (io_string == IO_TYPE_VALUE_BUFFER_IO) {
                return std::make_shared<GraphDataCell<BufferIO>>(graph_param, common_param);
            }
            if (io_string == IO_TYPE_VALUE_ASYNC_IO) {
                return std::make_shared<GraphDataCell<AsyncIO>>(graph_param, common_param);
            }
            if (io_string == IO_TYPE_VALUE_READER_IO) {
                return std::make_shared<GraphDataCell<ReaderIO>>(graph_param, common_param);
            }
            return nullptr;
    }
    return nullptr;
}
}  // namespace vsag
