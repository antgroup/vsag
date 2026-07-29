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

#include <cstdint>
#include <string>

#include "data_type.h"
#include "json_types.h"
#include "vsag/dataset.h"

namespace vsag {

class StreamReader;
class StreamWriter;

namespace index_utils {

DatasetPtr
MakeEmptyResult(const std::string& stats_json = "");

void
WriteIndexFooter(StreamWriter& writer, const JsonType& basic_info);

bool
ReadIndexFooter(StreamReader& reader, JsonType& basic_info);

void
ValidateSearchQuery(DataTypes data_type, int64_t dim, const DatasetPtr& query);

void
ValidateKnnArgs(DataTypes data_type, int64_t dim, const DatasetPtr& query, int64_t k);

void
ValidateRangeArgs(
    DataTypes data_type, int64_t dim, const DatasetPtr& query, float radius, int64_t limited_size);

}  // namespace index_utils

}  // namespace vsag
