
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

#include "basic_io.h"

#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <memory>

#include "fixtures.h"

class WrongIO : public vsag::BasicIO<WrongIO> {};

TEST_CASE("wrong io", "[ut][basic io]") {
    auto io = std::make_shared<WrongIO>();
    std::vector<uint8_t> data(100);
    bool release;
    fixtures::TempDir dirname("TestWrongIO");
    std::string filename = dirname.GenerateRandomFile();
    std::ofstream outfile(filename.c_str(), std::ios::binary);
    IOStreamWriter writer(outfile);
    REQUIRE_THROWS(io->Serialize(writer));
    outfile.close();

    std::ifstream infile(filename.c_str(), std::ios::binary);
    IOStreamReader reader(infile);
    REQUIRE_THROWS(io->Read(1, 0, data.data()));
    REQUIRE_THROWS(io->Write(data.data(), 1, 0));
    REQUIRE_THROWS(io->Read(1, 0, release));
    REQUIRE_THROWS(io->Deserialize(reader));
    REQUIRE_THROWS(io->Prefetch(1, 0));
    REQUIRE_THROWS(io->MultiRead(data.data(), nullptr, nullptr, 1));

    infile.close();
}
