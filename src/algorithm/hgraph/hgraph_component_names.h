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

namespace vsag {

/// Component names recorded in the chunked layout. The writer
/// (hgraph_serialize.cpp) and the parallel reader
/// (hgraph_parallel_deserialize.cpp) must agree on every name, so they are
/// declared once here: adding or renaming a component in only one of the two
/// would otherwise surface as a load-time error instead of a compile error.
constexpr const char* COMPONENT_LABEL_TABLE = "label_table";
constexpr const char* COMPONENT_CODE_SLOT_MAP = "code_slot_map";
constexpr const char* COMPONENT_BASE_CODES = "base_codes";
constexpr const char* COMPONENT_BOTTOM_GRAPH = "bottom_graph";
constexpr const char* COMPONENT_PRECISE_CODES = "precise_codes";
constexpr const char* COMPONENT_ROUTE_GRAPHS = "route_graphs";
constexpr const char* COMPONENT_EXTRA_INFOS = "extra_infos";
constexpr const char* COMPONENT_ATTR_FILTER = "attr_filter";
constexpr const char* COMPONENT_RAW_VECTOR = "raw_vector";
constexpr const char* COMPONENT_MCI_CLIQUES = "mci_cliques";
constexpr const char* COMPONENT_CONJUGATE_GRAPH = "conjugate_graph";

}  // namespace vsag
