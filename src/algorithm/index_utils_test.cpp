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

#include "index_utils.h"

#include <sstream>

#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "unittest.h"

using namespace vsag;

TEST_CASE("Index utilities validate search arguments", "[ut][index_utils]") {
    auto query = Dataset::Make()->Dim(4)->NumElements(1);

    REQUIRE_NOTHROW(index_utils::ValidateKnnArgs(DataTypes::DATA_TYPE_FLOAT, 4, query, 1));
    REQUIRE_THROWS(index_utils::ValidateKnnArgs(DataTypes::DATA_TYPE_FLOAT, 3, query, 1));
    REQUIRE_THROWS(index_utils::ValidateKnnArgs(DataTypes::DATA_TYPE_FLOAT, 4, query, 0));
    REQUIRE_NOTHROW(index_utils::ValidateRangeArgs(DataTypes::DATA_TYPE_FLOAT, 4, query, 0.0F, -1));
    REQUIRE_THROWS(index_utils::ValidateRangeArgs(DataTypes::DATA_TYPE_FLOAT, 4, query, -1.0F, -1));
    REQUIRE_THROWS(index_utils::ValidateRangeArgs(DataTypes::DATA_TYPE_FLOAT, 4, query, 0.0F, 0));
}

TEST_CASE("Index utilities accept sparse query dimensions", "[ut][index_utils]") {
    auto query = Dataset::Make()->Dim(8)->NumElements(1);

    REQUIRE_NOTHROW(index_utils::ValidateSearchQuery(DataTypes::DATA_TYPE_SPARSE, 4, query));
}

TEST_CASE("Index utilities build empty results", "[ut][index_utils]") {
    auto default_result = index_utils::MakeEmptyResult();

    REQUIRE(default_result->GetNumElements() == 1);
    REQUIRE(default_result->GetStatistics().empty());

    auto result = index_utils::MakeEmptyResult("{\"visited\": 0}");

    REQUIRE(result->GetNumElements() == 1);
    REQUIRE(result->GetStatistics() == "{\"visited\": 0}");
}

TEST_CASE("Index utilities round-trip index footers", "[ut][index_utils]") {
    std::stringstream stream;
    IOStreamWriter writer(stream);
    JsonType basic_info;
    basic_info["dim"].SetInt(4);
    index_utils::WriteIndexFooter(writer, basic_info);

    IOStreamReader reader(stream);
    JsonType parsed_info;
    REQUIRE(index_utils::ReadIndexFooter(reader, parsed_info));
    REQUIRE(parsed_info["dim"].GetInt() == 4);
}

TEST_CASE("Index utilities detect missing index footers", "[ut][index_utils]") {
    std::stringstream stream;
    IOStreamReader reader(stream);
    JsonType basic_info;

    REQUIRE_FALSE(index_utils::ReadIndexFooter(reader, basic_info));
}
