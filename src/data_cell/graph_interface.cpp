
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
GraphInterface::MakeInstance(const JsonType& graph_interface_param,
                             const IndexCommonParam& common_param,
                             bool is_sparse) {
    CHECK_ARGUMENT(graph_interface_param.contains(GRAPH_PARAMS_KEY),
                   fmt::format("graph interface parameters must contains {}", GRAPH_PARAMS_KEY));
    const auto& graph_param = graph_interface_param[GRAPH_PARAMS_KEY];
    if (is_sparse) {
        return std::make_shared<SparseGraphDataCell>(graph_param, common_param);
    }

    CHECK_ARGUMENT(graph_interface_param.contains(IO_TYPE_KEY),
                   fmt::format("graph interface parameters must contains {}", IO_TYPE_KEY));
    std::string io_string = graph_interface_param[IO_TYPE_KEY];

    CHECK_ARGUMENT(graph_interface_param.contains(IO_PARAMS_KEY),
                   fmt::format("graph interface parameters must contains {}", IO_PARAMS_KEY));
    const auto& io_param = graph_interface_param[IO_PARAMS_KEY];

    if (io_string == IO_TYPE_VALUE_BLOCK_MEMORY_IO) {
        return std::make_shared<GraphDataCell<MemoryBlockIO, false>>(
            graph_param, io_param, common_param);
    } else if (io_string == IO_TYPE_VALUE_MEMORY_IO) {
        return std::make_shared<GraphDataCell<MemoryIO, false>>(
            graph_param, io_param, common_param);
    }
    return nullptr;
}
}  // namespace vsag
