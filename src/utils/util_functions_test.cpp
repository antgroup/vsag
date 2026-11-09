
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

#include "util_functions.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <unordered_set>
#include <vector>

#include "impl/allocator/default_allocator.h"
#include "unittest.h"

using namespace vsag;

TEST_CASE("UtilFunctions Basic", "[ut][UtilFunctions]") {
    SECTION("split string") {
        auto parts = split_string("a,,b", ',');
        REQUIRE(parts.size() == 3);
        REQUIRE(parts[0] == "a");
        REQUIRE(parts[1].empty());
        REQUIRE(parts[2] == "b");
    }

    SECTION("align up") {
        REQUIRE(align_up(7, 4) == 8);
        REQUIRE(align_up(8, 4) == 8);
    }

    SECTION("approx zero") {
        REQUIRE(is_approx_zero(1e-6F));
        REQUIRE_FALSE(is_approx_zero(1e-3F));
    }

    SECTION("next multiple of power of two") {
        REQUIRE(next_multiple_of_power_of_two(5, 2) == 8);
        REQUIRE(next_multiple_of_power_of_two(16, 4) == 16);
        REQUIRE_THROWS(next_multiple_of_power_of_two(1, 64));
        REQUIRE_THROWS(next_multiple_of_power_of_two(std::numeric_limits<uint64_t>::max(), 1));
    }

    SECTION("select k numbers") {
        auto selected = select_k_numbers(100, 10);
        REQUIRE(selected.size() == 10);
        std::unordered_set<int> dedup(selected.begin(), selected.end());
        REQUIRE(dedup.size() == 10);
        for (int v : selected) {
            REQUIRE(v >= 0);
            REQUIRE(v < 100);
        }
        REQUIRE_THROWS(select_k_numbers(10, 0));
        REQUIRE_THROWS(select_k_numbers(10, 11));
    }

    SECTION("string stream equality") {
        std::stringstream s1;
        std::stringstream s2;
        s1 << "abc";
        s2 << "abc";
        REQUIRE(check_equal_on_string_stream(s1, s2));

        std::stringstream s3;
        std::stringstream s4;
        s3 << "abc";
        s4 << "abd";
        REQUIRE_FALSE(check_equal_on_string_stream(s3, s4));
    }

    SECTION("base64 roundtrip") {
        std::string input = "vsag-base64-roundtrip";
        std::string encoded = base64_encode(input);
        std::string decoded = base64_decode(encoded);
        REQUIRE(decoded == input);
    }

    SECTION("create fast dataset") {
        auto allocator = std::make_shared<DefaultAllocator>();
        auto [dataset, dists, ids] = create_fast_dataset(4, allocator.get());
        REQUIRE(dataset != nullptr);
        REQUIRE(dataset->GetDim() == 4);
        REQUIRE(dataset->GetNumElements() == 1);
        REQUIRE(dists != nullptr);
        REQUIRE(ids != nullptr);
    }

    SECTION("current time string") {
        auto t = get_current_time();
        REQUIRE_FALSE(t.empty());
    }
}

TEST_CASE("get_vectors and set_dataset carry binary rows", "[ut][UtilFunctions]") {
    const int64_t dim = 13;  // two bytes per row
    std::vector<uint8_t> bytes = {1, 2, 3, 4, 5, 6};
    auto base = vsag::Dataset::Make();
    base->NumElements(3)->Dim(dim)->BinaryVectors(bytes.data())->Owner(false);
    void* vectors = nullptr;
    uint64_t row_size = 0;
    get_vectors(DataTypes::DATA_TYPE_BINARY, dim, base, &vectors, &row_size);
    REQUIRE(vectors == bytes.data());
    REQUIRE(row_size == 2);

    auto view = vsag::Dataset::Make();
    set_dataset(DataTypes::DATA_TYPE_BINARY, dim, view, bytes.data(), 3);
    REQUIRE(view->GetBinaryVectors() == bytes.data());
    REQUIRE(view->GetFloat32Vectors() == nullptr);
    REQUIRE(view->GetDim() == dim);
    REQUIRE(view->GetNumElements() == 3);
}

TEST_CASE("sample_train_data honors a fixed random seed", "[ut][UtilFunctions]") {
    constexpr int64_t dim = 3;
    constexpr int64_t count = 2048;
    constexpr int64_t sample_count = 512;
    std::vector<float> vectors(count * dim);
    std::iota(vectors.begin(), vectors.end(), 0.0F);
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 0);
    auto data = Dataset::Make();
    data->NumElements(count)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    auto allocator = std::make_shared<DefaultAllocator>();

    auto first = sample_train_data(data, count, dim, sample_count, allocator.get(), 0x52425131U);
    auto second = sample_train_data(data, count, dim, sample_count, allocator.get(), 0x52425131U);

    REQUIRE(first->GetNumElements() == sample_count);
    REQUIRE(second->GetNumElements() == sample_count);
    REQUIRE(std::equal(first->GetIds(), first->GetIds() + sample_count, second->GetIds()));
    REQUIRE(std::equal(first->GetFloat32Vectors(),
                       first->GetFloat32Vectors() + sample_count * dim,
                       second->GetFloat32Vectors()));
}

TEST_CASE("sample_train_data honors configured counts above the default", "[ut][UtilFunctions]") {
    constexpr int64_t dim = 1;
    constexpr int64_t count = 65538;
    constexpr int64_t sample_count = 65537;
    std::vector<float> vectors(count);
    std::iota(vectors.begin(), vectors.end(), 0.0F);
    auto data = Dataset::Make();
    data->NumElements(count)->Dim(dim)->Float32Vectors(vectors.data())->Owner(false);
    auto allocator = std::make_shared<DefaultAllocator>();

    auto sampled = sample_train_data(data, count, dim, sample_count, allocator.get(), 0x52425131U);

    REQUIRE(sampled->GetNumElements() == sample_count);
}

TEST_CASE("sample_train_data uses all vectors for an unbounded count", "[ut][UtilFunctions]") {
    constexpr int64_t dim = 2;
    constexpr int64_t count = 1024;
    std::vector<float> vectors(count * dim);
    auto data = Dataset::Make();
    data->NumElements(count)->Dim(dim)->Float32Vectors(vectors.data())->Owner(false);
    auto allocator = std::make_shared<DefaultAllocator>();

    auto sampled =
        sample_train_data(data, count, dim, std::numeric_limits<int64_t>::max(), allocator.get());

    REQUIRE(sampled == data);
    REQUIRE(sampled->GetNumElements() == count);
}

TEST_CASE("sample_train_data keeps the element type of dense rows", "[ut][UtilFunctions]") {
    // Rows are sampled with their own byte width; binary rows hold ceil(dim / 8) bytes.
    constexpr int64_t count = 1500;
    constexpr int64_t sample_count = 600;
    constexpr int64_t dim = 13;
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 100);
    auto allocator = std::make_shared<DefaultAllocator>();

    // Every row starts with its own index, so a sampled row can be traced back to its source.
    auto check_rows = [&](const DatasetPtr& sampled,
                          const uint8_t* source,
                          uint64_t row_bytes,
                          const uint8_t* sampled_rows) {
        REQUIRE(sampled->GetNumElements() == sample_count);
        REQUIRE(sampled->GetDim() == dim);
        for (int64_t i = 0; i < sample_count; ++i) {
            const auto source_row = sampled->GetIds()[i] - 100;
            REQUIRE(std::equal(sampled_rows + i * row_bytes,
                               sampled_rows + (i + 1) * row_bytes,
                               source + source_row * row_bytes));
        }
    };

    SECTION("binary") {
        const uint64_t row_bytes = (dim + 7) / 8;
        std::vector<uint8_t> rows(count * row_bytes);
        for (int64_t i = 0; i < count; ++i) {
            rows[i * row_bytes] = static_cast<uint8_t>(i);
            rows[i * row_bytes + 1] = static_cast<uint8_t>(i >> 8);
        }
        auto data = Dataset::Make();
        data->NumElements(count)
            ->Dim(dim)
            ->Ids(ids.data())
            ->BinaryVectors(rows.data())
            ->Owner(false);
        auto sampled = sample_train_data(data, count, dim, sample_count, allocator.get(), 7);
        REQUIRE(sampled->GetFloat32Vectors() == nullptr);
        REQUIRE(sampled->GetBinaryVectors() != nullptr);
        check_rows(sampled, rows.data(), row_bytes, sampled->GetBinaryVectors());
    }

    SECTION("int8") {
        std::vector<int8_t> rows(count * dim);
        for (int64_t i = 0; i < count; ++i) {
            rows[i * dim] = static_cast<int8_t>(i % 127);
            rows[i * dim + 1] = static_cast<int8_t>(i / 127);
        }
        auto data = Dataset::Make();
        data->NumElements(count)->Dim(dim)->Ids(ids.data())->Int8Vectors(rows.data())->Owner(false);
        auto sampled = sample_train_data(data, count, dim, sample_count, allocator.get(), 7);
        REQUIRE(sampled->GetInt8Vectors() != nullptr);
        check_rows(sampled,
                   reinterpret_cast<const uint8_t*>(rows.data()),
                   dim,
                   reinterpret_cast<const uint8_t*>(sampled->GetInt8Vectors()));
    }

    SECTION("float16") {
        std::vector<uint16_t> rows(count * dim);
        for (int64_t i = 0; i < count; ++i) {
            rows[i * dim] = static_cast<uint16_t>(i);
        }
        auto data = Dataset::Make();
        data->NumElements(count)
            ->Dim(dim)
            ->Ids(ids.data())
            ->Float16Vectors(rows.data())
            ->Owner(false);
        auto sampled = sample_train_data(data, count, dim, sample_count, allocator.get(), 7);
        REQUIRE(sampled->GetFloat16Vectors() != nullptr);
        check_rows(sampled,
                   reinterpret_cast<const uint8_t*>(rows.data()),
                   dim * sizeof(uint16_t),
                   reinterpret_cast<const uint8_t*>(sampled->GetFloat16Vectors()));
    }

    SECTION("datasets without dense vectors are rejected") {
        auto data = Dataset::Make();
        data->NumElements(count)->Dim(dim)->Ids(ids.data())->Owner(false);
        REQUIRE_THROWS(sample_train_data(data, count, dim, sample_count, allocator.get(), 7));
    }
}
