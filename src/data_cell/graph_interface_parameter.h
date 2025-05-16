
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

#pragma once

#include "parameter.h"

namespace vsag {

class GraphInterfaceParameter;
using GraphInterfaceParamPtr = std::shared_ptr<GraphInterfaceParameter>;

enum class GraphStorageTypes {
    GRAPH_STORAGE_TYPE_FLAT = 0,
    GRAPH_STORAGE_TYPE_COMPRESSED = 1,
    GRAPH_STORAGE_TYPE_SPARSE = 2
};

class GraphInterfaceParameter : public Parameter {
public:
    static GraphInterfaceParamPtr
    GetGraphParameterByJson(GraphStorageTypes graph_type, const JsonType& json);

public:
    GraphStorageTypes graph_storage_type_{GraphStorageTypes::GRAPH_STORAGE_TYPE_FLAT};

protected:
    explicit GraphInterfaceParameter(GraphStorageTypes graph_type)
        : graph_storage_type_(graph_type){};
};

}  // namespace vsag
