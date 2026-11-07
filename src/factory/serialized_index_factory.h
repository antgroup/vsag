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

#include <string>

#include "json_types.h"
#include "vsag/index.h"

namespace vsag {

/**
 * @brief Create an empty index from the parameters recorded by a serialized index.
 *
 * Footers record basic_info.index_param as the output of the index parameter ToJson(), which is
 * not the external layout accepted by Factory::CreateIndex. This rebuilds the parameter object
 * from that recorded layout, the same way Index::Load does for streaming files.
 *
 * @param index_name factory name of the index type, e.g. "hgraph".
 * @param common_param_json external common keys (dtype, metric_type, dim, extra_info_size),
 *        validated exactly like Factory::CreateIndex validates them.
 * @param index_param the parsed basic_info.index_param object.
 * @throws VsagException for unsupported index types and invalid parameters.
 */
IndexPtr
create_index_from_serialized_param(const std::string& index_name,
                                   JsonType& common_param_json,
                                   const JsonType& index_param);

}  // namespace vsag
