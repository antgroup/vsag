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

#include <cstring>
#include <sstream>

#include "serialization.h"
#include "stream_reader.h"
#include "unittest.h"

namespace {
constexpr uint64_t FOOTER_MIN_SIZE = 36;
}  // namespace

TEST_CASE("Footer Parse rejects streams shorter than minimum footer size", "[ut][footer]") {
    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
    ss.str(std::string(FOOTER_MIN_SIZE - 1, 'a'));
    vsag::IOStreamReader reader(ss);
    REQUIRE(vsag::Footer::Parse(reader) == nullptr);
}

TEST_CASE("Footer Parse rejects mismatched metadata_string_length", "[ut][footer]") {
    const uint64_t footer_length = 44;
    std::string data(footer_length, '\0');
    std::memcpy(&data[0], "vsag0000", 8);
    const uint64_t wrong_metadata_length = 16;
    std::memcpy(&data[8], &wrong_metadata_length, sizeof(wrong_metadata_length));
    std::memcpy(&data[footer_length - 16], &footer_length, sizeof(footer_length));
    std::memcpy(&data[footer_length - 8], "0000gasv", 8);
    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
    ss.str(data);
    vsag::IOStreamReader reader(ss);
    REQUIRE(vsag::Footer::Parse(reader) == nullptr);
}
