
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

#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <memory>

#include "basic_io.h"
#include "fixtures.h"
#include "storage/serialization_template_test.h"

using namespace vsag;

template <typename T>
void
TestBasicReadWrite(BasicIO<T>& io) {
    std::vector<uint64_t> counts = {200, 500};
    std::vector<uint64_t> max_lengths = {2, 20, 37, 64, 128, 260, 999};
    for (auto count : counts) {
        for (auto max_length : max_lengths) {
            auto vecs = fixtures::GenTestItems(count, max_length);
            std::vector<uint64_t> offs, sizes;
            uint64_t total_size = 0;
            for (auto& item : vecs) {
                io.Write(item.data_, item.length_, item.start_);
                offs.emplace_back(item.start_);
                sizes.emplace_back(item.length_);
                total_size += item.length_;
            }
            std::vector<uint8_t> datas(total_size);
            io.MultiRead(datas.data(), sizes.data(), offs.data(), count);
            uint8_t* cur = datas.data();
            for (auto& item : vecs) {
                std::vector<uint8_t> data(item.length_);
                io.Read(item.length_, item.start_, data.data());
                REQUIRE(memcmp(item.data_, data.data(), item.length_) == 0);

                bool need_release = false;
                const auto* ptr = io.Read(item.length_, item.start_, need_release);
                REQUIRE(memcmp(item.data_, ptr, item.length_) == 0);
                if (need_release) {
                    io.Release(ptr);
                }

                REQUIRE(memcmp(item.data_, cur, item.length_) == 0);
                cur += item.length_;
            }
        }
    }
}

template <typename T>
void
TestSerializeAndDeserialize(BasicIO<T>& wio, BasicIO<T>& rio) {
    std::vector<uint64_t> counts = {200, 500};
    std::vector<uint64_t> max_lengths = {2, 20, 37, 64, 128, 260, 999};
    srandom(time(nullptr));
    fixtures::TempDir dirname("TestSerializeAndDeserialize");
    for (auto count : counts) {
        for (auto max_length : max_lengths) {
            auto vecs = fixtures::GenTestItems(count, max_length);
            for (auto& item : vecs) {
                wio.Write(item.data_, item.length_, item.start_);
            }
            test_serializion(wio, rio);

            for (auto& item : vecs) {
                std::vector<uint8_t> data(item.length_);
                rio.Read(item.length_, item.start_, data.data());
                bool need_release = false;
                const auto* ptr = rio.Read(item.length_, item.start_, need_release);
                REQUIRE(memcmp(data.data(), ptr, item.length_) == 0);
                REQUIRE(memcmp(data.data(), item.data_, item.length_) == 0);
                if (need_release) {
                    rio.Release(ptr);
                }
            }
        }
    }
}

template <typename T>
void
TestDistIOWrongInit(Allocator* allocator) {
    if (T::InMemory) {
        return;
    }
    auto dirname = fixtures::TempDir("TestDistIOWrongInit");
    auto func = [&]() { auto io = std::make_unique<T>(dirname.path, allocator); };
    REQUIRE_THROWS(func());
}
