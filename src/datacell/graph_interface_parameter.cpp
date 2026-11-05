
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

#include "graph_interface_parameter.h"

#include "compressed_graph_datacell_parameter.h"
#include "graph_datacell_parameter.h"
#include "sparse_graph_datacell_parameter.h"

namespace vsag {
GraphInterfaceParamPtr
GraphInterfaceParameter::GetGraphParameterByJson(GraphStorageTypes graph_type,
                                                 const JsonType& json) {
    GraphInterfaceParamPtr param{nullptr};
    switch (graph_type) {
        case GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_FLAT:
            param = std::make_shared<GraphDataCellParameter>();
            break;
        case GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_COMPRESSED:
            param = std::make_shared<CompressedGraphDatacellParameter>();
            break;
        case GraphStorageTypes::GRAPH_STORAGE_TYPE_SPARSE:
            param = std::make_shared<SparseGraphDatacellParameter>();
            break;
    }
    param->FromJson(json);
    return param;
}
}  // namespace vsag
