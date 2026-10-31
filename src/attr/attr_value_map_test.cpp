
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

#include "attr/attr_value_map.h"

#include <catch2/catch_all.hpp>
#include <new>
#include <sstream>
#include <unordered_set>

#include "impl/allocator/default_allocator.h"
#include "impl/allocator/safe_allocator.h"
#include "storage/serialization_template_test.h"
#include "unittest.h"

using namespace vsag;

template <class T>
T
GetRandomValue() {
    if constexpr (std::is_integral<T>::value) {
        return random() % std::numeric_limits<T>::max();
    } else if constexpr (std::is_same_v<T, std::string>) {
        return "abc";
    }
}

template <class T>
void
TestAttrValueMap() {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto type = GENERATE(ComputableBitsetType::SparseBitset, ComputableBitsetType::FastBitset);
    AttrValueMap map(allocator.get(), type);
    T value = GetRandomValue<T>();
    InnerIdType id = random() % 10 + 1;

    map.Insert(value, id);
    auto manager = map.GetBitsetByValue(value);
    REQUIRE(manager != nullptr);
    REQUIRE(manager->GetOneBitset(0)->Test(id) == true);
    REQUIRE(nullptr == map.GetBitsetByValue(999));

    map.Insert(value, id, 3);
    manager = map.GetBitsetByValue(value);
    REQUIRE(manager != nullptr);
    REQUIRE(manager->GetOneBitset(2) == nullptr);
    REQUIRE(manager->GetOneBitset(3)->Test(id) == true);

    AttrValueMap map2(allocator.get(), type);
    test_serializion(map, map2);

    manager = map2.GetBitsetByValue(value);
    REQUIRE(manager != nullptr);
    REQUIRE(manager->GetOneBitset(2) == nullptr);
    REQUIRE(manager->GetOneBitset(3)->Test(id) == true);
    REQUIRE(nullptr == map2.GetBitsetByValue(999));
}

TEST_CASE("AttrValueMap Basic Test", "[ut][AttrValueMap]") {
    TestAttrValueMap<int64_t>();
    TestAttrValueMap<int32_t>();
    TestAttrValueMap<int16_t>();
    TestAttrValueMap<int8_t>();
    TestAttrValueMap<uint64_t>();
    TestAttrValueMap<uint32_t>();
    TestAttrValueMap<uint16_t>();
    TestAttrValueMap<uint8_t>();
    TestAttrValueMap<std::string>();
}

namespace {

class FailingAttrAllocator : public DefaultAllocator {
public:
    void*
    Allocate(uint64_t size) override {
        if (++calls == fail_at) {
            throw std::bad_alloc();
        }
        auto* pointer = DefaultAllocator::Allocate(size);
        live.insert(pointer);
        return pointer;
    }

    void
    Deallocate(void* pointer) override {
        live.erase(pointer);
        DefaultAllocator::Deallocate(pointer);
    }

    uint64_t calls{0};
    uint64_t fail_at{0};
    std::unordered_set<void*> live;
};

}  // namespace

TEST_CASE("AttrValueMap retains ownership on insertion failure", "[ut][AttrValueMap]") {
    const bool string_key = GENERATE(false, true);
    const bool replace = GENERATE(false, true);
    const auto type =
        GENERATE(ComputableBitsetType::SparseBitset, ComputableBitsetType::FastBitset);
    auto source_allocator = SafeAllocator::FactoryDefaultAllocator();
    AttrValueMap source(source_allocator.get(), type);
    if (string_key) {
        source.Insert(std::string("key"), 7);
    } else {
        source.Insert(int64_t{42}, 7);
    }
    std::stringstream output;
    IOStreamWriter writer(output);
    source.Serialize(writer);
    const auto bytes = output.str();
    uint64_t allocations = 0;
    for (uint64_t failure = 0; failure <= allocations + 1; ++failure) {
        FailingAttrAllocator allocator;
        bool threw = false;
        {
            AttrValueMap restored(&allocator, type);
            if (replace) {
                std::istringstream initial_input(bytes);
                IOStreamReader initial_reader(initial_input);
                restored.Deserialize(initial_reader);
            }
            allocator.calls = 0;
            allocator.fail_at = failure;
            std::istringstream input(bytes);
            IOStreamReader reader(input);
            try {
                restored.Deserialize(reader);
            } catch (const std::bad_alloc&) {
                threw = true;
            }
            allocator.fail_at = 0;
            if (failure == 0) {
                allocations = allocator.calls;
                REQUIRE(allocations > 0);
            }
            REQUIRE(threw == (failure > 0 && failure <= allocations));
        }
        CAPTURE(string_key, replace, type, failure, allocations);
        REQUIRE(allocator.live.empty());
        if (failure == allocations + 1) {
            break;
        }
    }
}
