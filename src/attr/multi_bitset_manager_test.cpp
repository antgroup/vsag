
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

#include "multi_bitset_manager.h"

#include <catch2/catch_test_macros.hpp>

#include "fixtures.h"
#include "impl/allocator/safe_allocator.h"

using namespace vsag;

TEST_CASE("MultiBitsetManager Basic Test", "[ut][MultiBitsetManager]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto manager = std::make_unique<MultiBitsetManager>(allocator.get());

    REQUIRE(manager->GetOneBitset(100) == nullptr);

    manager->SetNewCount(100);
    REQUIRE(manager->GetOneBitset(100) == nullptr);

    manager->SetNewCount(50);
    REQUIRE(manager->GetOneBitset(100) == nullptr);

    manager->InsertValue(100, 10, true);
    REQUIRE(manager->GetOneBitset(100) != nullptr);

    auto* ptr = manager->GetOneBitset(100);
    REQUIRE(ptr->Test(10) == true);
    REQUIRE(ptr->Test(9) == false);

    auto tmp_dir = fixtures::TempDir("multi_bitset_manager");
    auto filepath = tmp_dir.GenerateRandomFile();
    std::ofstream ofile(filepath, std::ios::binary);
    IOStreamWriter writer(ofile);
    manager->Serialize(writer);
    ofile.close();

    auto manager2 = std::make_unique<MultiBitsetManager>(allocator.get());
    std::ifstream ifile(filepath, std::ios::binary);
    IOStreamReader reader(ifile);
    manager2->Deserialize(reader);
    ifile.close();

    REQUIRE(manager2->GetOneBitset(100) != nullptr);

    auto* ptr2 = manager2->GetOneBitset(100);
    REQUIRE(ptr2->Test(10) == true);
    REQUIRE(ptr2->Test(9) == false);
}
