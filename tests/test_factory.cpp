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

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <string>

#include "vsag/vsag.h"

namespace {

static_assert(static_cast<int>(vsag::IndexType::HGRAPH) == 2);
static_assert(static_cast<int>(vsag::IndexType::IVF) == 3);
static_assert(static_cast<int>(vsag::IndexType::PYRAMID) == 4);
static_assert(static_cast<int>(vsag::IndexType::BRUTEFORCE) == 5);
static_assert(static_cast<int>(vsag::IndexType::SPARSE) == 6);
static_assert(static_cast<int>(vsag::IndexType::SINDI) == 7);
static_assert(static_cast<int>(vsag::IndexType::WARP) == 8);
static_assert(static_cast<int>(vsag::IndexType::LAZY_HGRAPH) == 9);
static_assert(static_cast<int>(vsag::IndexType::SIMQ) == 10);

constexpr const char* VALID_COMMON_PARAMETERS = R"(
{
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 4
}
)";

}  // namespace

#ifndef VSAG_MOCKIMPL_TEST

TEST_CASE("Factory rejects removed index names", "[ft][factory][unsupported_index]") {
    constexpr std::array<const char*, 8> removed_names = {
        "hnsw", "HNSW", "Hnsw", "fresh_hnsw", "FRESH_HNSW", "diskann", "DiskANN", "DISKANN"};

    for (const auto* name : removed_names) {
        auto index = vsag::Factory::CreateIndex(name, VALID_COMMON_PARAMETERS);
        INFO(name);
        REQUIRE_FALSE(index.has_value());
        REQUIRE(index.error().type == vsag::ErrorType::UNSUPPORTED_INDEX);
        REQUIRE(index.error().message.find("unsupported") != std::string::npos);
    }
}

TEST_CASE("Factory still creates a supported index", "[ft][factory][supported_index]") {
    auto index = vsag::Factory::CreateIndex("brute_force", R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 4,
        "index_param": {
            "base_quantization_type": "fp32"
        }
    }
    )");

    REQUIRE(index.has_value());
    REQUIRE(index.value()->GetIndexType() == vsag::IndexType::BRUTEFORCE);
}

#else

TEST_CASE("Mock factory creates an index", "[ft][factory]") {
    auto index = vsag::Factory::CreateIndex("brute_force", VALID_COMMON_PARAMETERS);
    REQUIRE(index.has_value());
}

#endif
