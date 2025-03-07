
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

#include "extra_info_interface_test.h"

#include <catch2/catch_template_test_macros.hpp>
#include <fstream>
#include <iostream>

#include "fixtures.h"
#include "simd/simd.h"

namespace vsag {
void
ExtraInfoInterfaceTest::BasicTest(uint64_t base_count) {
    // prepare
    int64_t query_count = 100;
    uint64_t extra_info_size = extra_info_->ExtraInfoSize();
    auto extra_infos = fixtures::generate_extra_infos(base_count, extra_info_size);

    // test InsertExtraInfo and BatchInsertExtraInfo
    auto old_count = extra_info_->TotalCount();
    InnerIdType first_one = base_count + old_count;
    InnerIdType last_one = base_count + old_count - 1;
    extra_info_->InsertExtraInfo(extra_infos.data());
    extra_info_->BatchInsertExtraInfo(extra_infos.data() + extra_info_size, base_count - 2);
    extra_info_->BatchInsertExtraInfo(
        extra_infos.data() + (base_count - 1) * extra_info_size, 1, &last_one);
    REQUIRE(extra_info_->TotalCount() == base_count + old_count);
}
void
ExtraInfoInterfaceTest::TestSerializeAndDeserialize(ExtraInfoInterfacePtr other) {
    fixtures::TempDir dir("extra_info");
    auto path = dir.GenerateRandomFile();
    std::ofstream outfile(path.c_str(), std::ios::binary);
    IOStreamWriter writer(outfile);
    this->extra_info_->Serialize(writer);
    outfile.close();

    std::ifstream infile(path.c_str(), std::ios::binary);
    IOStreamReader reader(infile);
    other->Deserialize(reader);

    auto total_count = other->TotalCount();
    REQUIRE(total_count == this->extra_info_->TotalCount());
    REQUIRE(other->ExtraInfoSize() == this->extra_info_->ExtraInfoSize());

    infile.close();
}
}  // namespace vsag
