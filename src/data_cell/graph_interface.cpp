
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

#include "graph_datacell.h"
#include "io/io_headers.h"
#include "sparse_graph_datacell.h"

namespace vsag {

GraphInterfacePtr
GraphInterface::MakeInstance(const GraphInterfaceParamPtr& param,
                             const IndexCommonParam& common_param,
                             bool is_sparse) {
    if (is_sparse) {
        return std::make_shared<SparseGraphDataCell>(param, common_param);
    }

    auto io_string =
        std::dynamic_pointer_cast<GraphDataCellParameter>(param)->io_parameter_->GetTypeName();

    if (io_string == IO_TYPE_VALUE_BLOCK_MEMORY_IO) {
        return std::make_shared<GraphDataCell<MemoryBlockIO>>(param, common_param);
    }

    if (io_string == IO_TYPE_VALUE_MEMORY_IO) {
        return std::make_shared<GraphDataCell<MemoryIO>>(param, common_param);
    }

    return nullptr;
}
}  // namespace vsag
