
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
#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>
#include <unordered_set>
#include <vector>

#include "impl/allocator/default_allocator.h"
#include "unittest.h"
#include "vsag_exception.h"

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

TEST_CASE("Training sampling preserves configured half input", "[ut][UtilFunctions][half_input]") {
    const auto dtype = GENERATE(DataTypes::DATA_TYPE_FP16, DataTypes::DATA_TYPE_BF16);
    const int64_t count = GENERATE(512, 513);
    constexpr int64_t dim = 3;
    std::vector<uint16_t> bits(count * dim);
    std::vector<int64_t> ids(count);
    std::vector<float> floats(count * dim, -1.0F);
    for (int64_t i = 0; i < count; ++i) {
        ids[i] = i + 1000;
        for (int64_t d = 0; d < dim; ++d) {
            bits[i * dim + d] = static_cast<uint16_t>(i * dim + d);
        }
    }
    auto base = Dataset::Make()
                    ->NumElements(count)
                    ->Dim(dim)
                    ->Ids(ids.data())
                    ->Float16Vectors(bits.data())
                    ->Float32Vectors(floats.data())
                    ->Owner(false);
    DefaultAllocator allocator;
    auto sampled = sample_train_data(base, count, dim, 512, &allocator, 42, dtype);
    REQUIRE(sampled->GetNumElements() == 512);
    REQUIRE(sampled->GetFloat16Vectors() != nullptr);
    if (count == 512) {
        REQUIRE(sampled == base);
    } else {
        REQUIRE(sampled->GetFloat32Vectors() == nullptr);
    }
    for (int64_t i = 0; i < 512; ++i) {
        const auto original = sampled->GetIds()[i] - 1000;
        REQUIRE(original >= 0);
        REQUIRE(original < count);
        for (int64_t d = 0; d < dim; ++d) {
            REQUIRE(sampled->GetFloat16Vectors()[i * dim + d] == bits[original * dim + d]);
        }
    }
}

TEST_CASE("set_dataset preserves 64-bit element counts", "[ut][UtilFunctions][half_input]") {
    const int64_t count = static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) + 1;
    for (const auto type : {DataTypes::DATA_TYPE_FLOAT,
                            DataTypes::DATA_TYPE_INT8,
                            DataTypes::DATA_TYPE_FP16,
                            DataTypes::DATA_TYPE_BF16}) {
        auto data = Dataset::Make();
        set_dataset(type, 3, data, nullptr, count);
        REQUIRE(data->GetNumElements() == count);
        REQUIRE(data->GetDim() == 3);
    }
}

namespace {
// Count allocator API requests, not implementation-dependent libc allocations.
class SamplingReviewAllocator : public DefaultAllocator {
public:
    void*
    Allocate(uint64_t size) override {
        requests.push_back(size);
        if (reject_all or (fail_after_rows and saw_rows)) {
            throw std::bad_alloc();
        }
        auto* ptr = DefaultAllocator::Allocate(size);
        live.insert(ptr);
        saw_rows = saw_rows or size == row_bytes;
        return ptr;
    }

    void
    Deallocate(void* ptr) override {
        if (ptr == nullptr) {
            return;
        }
        if (live.erase(ptr) != 1) {
            invalid_free = true;
            return;
        }
        DefaultAllocator::Deallocate(ptr);
    }

    void*
    Reallocate(void* ptr, uint64_t size) override {
        auto* result = Allocate(size);
        // These tests do not require reallocating a populated vector. Refuse to
        // conceal such a change in allocator usage behind a lossy copy.
        if (ptr != nullptr) {
            Deallocate(result);
            throw std::bad_alloc();
        }
        return result;
    }

    uint64_t row_bytes{0};
    bool fail_after_rows{false};
    bool saw_rows{false};
    bool reject_all{false};
    bool invalid_free{false};
    std::vector<uint64_t> requests;
    std::unordered_set<void*> live;
};

template <typename T>
void
CheckSamplingReview(DataTypes dtype, bool with_ids, bool default_allocator) {
    constexpr int64_t count = 513, dim = 13, sampled_count = 512;
    std::vector<T> source(count * dim);
    std::vector<int64_t> ids(count);
    for (int64_t i = 0; i < count; ++i) {
        ids[i] = 1000 + i;
        for (int64_t d = 0; d < dim; ++d) {
            source[i * dim + d] = static_cast<T>((i + 3 * d) % 97);
        }
    }
    auto base = Dataset::Make();
    set_dataset(dtype, dim, base, source.data(), count);
    base->Owner(false);
    if (with_ids) {
        base->Ids(ids.data());
    }
    SamplingReviewAllocator allocator;
    allocator.row_bytes = sampled_count * dim * sizeof(T);
    auto sampled = sample_train_data(
        base, count, dim, sampled_count, default_allocator ? nullptr : &allocator, 42, dtype);
    REQUIRE(sampled != base);
    REQUIRE(sampled->GetNumElements() == sampled_count);
    REQUIRE(sampled->GetDim() == dim);
    // The default allocator used inside sample_train_data has left scope. These
    // reads AND reset exercise its required lifetime (run with ASAN as well).
    void* sampled_bytes = nullptr;
    uint64_t row_bytes = 0;
    get_vectors(dtype, dim, sampled, &sampled_bytes, &row_bytes);
    REQUIRE(sampled_bytes != nullptr);
    REQUIRE(row_bytes == dim * sizeof(T));
    REQUIRE((sampled->GetIds() != nullptr) == with_ids);
    std::unordered_set<int64_t> selected;
    for (int64_t i = 0; i < sampled_count; ++i) {
        const auto* row = static_cast<const uint8_t*>(sampled_bytes) + i * row_bytes;
        if (with_ids) {
            const auto original = sampled->GetIds()[i] - 1000;
            REQUIRE(original >= 0);
            REQUIRE(original < count);
            REQUIRE(selected.insert(original).second);
            REQUIRE(std::memcmp(row, source.data() + original * dim, row_bytes) == 0);
        } else {
            bool found = false;
            for (int64_t original = 0; original < count; ++original) {
                found = found or std::memcmp(row, source.data() + original * dim, row_bytes) == 0;
            }
            REQUIRE(found);
        }
    }
    sampled.reset();
    if (not default_allocator) {
        // dim=13 ensures row bytes differ from the reservoir/ID allocation.
        REQUIRE(std::count(allocator.requests.begin(),
                           allocator.requests.end(),
                           allocator.row_bytes) == 1);
        REQUIRE(allocator.live.empty());
        REQUIRE_FALSE(allocator.invalid_free);
    }
}
}  // namespace

TEST_CASE("Training samples retain default allocator and allocate rows once",
          "[ut][UtilFunctions][maintenance_review][sampling_lifetime]") {
    const bool with_ids = GENERATE(false, true);
    const bool default_allocator = GENERATE(false, true);
    CheckSamplingReview<float>(DataTypes::DATA_TYPE_FLOAT, with_ids, default_allocator);
    CheckSamplingReview<int8_t>(DataTypes::DATA_TYPE_INT8, with_ids, default_allocator);
    CheckSamplingReview<uint16_t>(DataTypes::DATA_TYPE_FP16, with_ids, default_allocator);
    CheckSamplingReview<uint16_t>(DataTypes::DATA_TYPE_BF16, with_ids, default_allocator);
}

TEST_CASE("Training sample allocation failure releases owned rows",
          "[ut][UtilFunctions][maintenance_review][sampling_failure]") {
    constexpr int64_t count = 513, dim = 13;
    std::vector<float> source(count * dim, 1.0F);
    std::vector<int64_t> ids(count, 7);
    auto base = Dataset::Make()
                    ->NumElements(count)
                    ->Dim(dim)
                    ->Float32Vectors(source.data())
                    ->Ids(ids.data())
                    ->Owner(false);
    SamplingReviewAllocator allocator;
    allocator.row_bytes = 512 * dim * sizeof(float);
    // Reservoir allocation succeeds. Fail the first allocator request following
    // row allocation: with IDs this is the ID buffer, regardless of call count.
    allocator.fail_after_rows = true;
    REQUIRE_THROWS_AS(sample_train_data(base, count, dim, 512, &allocator, 42), std::bad_alloc);
    REQUIRE(allocator.saw_rows);
    REQUIRE(allocator.live.empty());
    REQUIRE_FALSE(allocator.invalid_free);
}

TEST_CASE("Training sampling rejects overflowing spans before allocation or reads",
          "[ut][UtilFunctions][maintenance_review][sampling_overflow]") {
    const auto dtype = GENERATE(DataTypes::DATA_TYPE_FLOAT,
                                DataTypes::DATA_TYPE_INT8,
                                DataTypes::DATA_TYPE_FP16,
                                DataTypes::DATA_TYPE_BF16);
    const uint64_t element_bytes = dtype == DataTypes::DATA_TYPE_FLOAT  ? sizeof(float)
                                   : dtype == DataTypes::DATA_TYPE_INT8 ? sizeof(int8_t)
                                                                        : sizeof(uint16_t);
    // Correctly aligned dummy pointers, deliberately NOT real input spans.
    float dummy_float = 0;
    int8_t dummy_int8 = 0;
    uint16_t dummy_half = 0;
    void* dummy = dtype == DataTypes::DATA_TYPE_FLOAT  ? static_cast<void*>(&dummy_float)
                  : dtype == DataTypes::DATA_TYPE_INT8 ? static_cast<void*>(&dummy_int8)
                                                       : static_cast<void*>(&dummy_half);
    const auto check = [&](int64_t count, int64_t dim, int64_t samples) {
        auto base = Dataset::Make();
        set_dataset(dtype, dim, base, dummy, count);
        base->Owner(false);
        SamplingReviewAllocator allocator;
        // A missing preflight check must fail safely rather than attempt huge
        // allocations or touch a dummy source; bad_alloc is NOT an accepted result.
        allocator.reject_all = true;
        bool rejected = false;
        try {
            auto sampled = sample_train_data(base, count, dim, samples, &allocator, 42, dtype);
        } catch (const VsagException& error) {
            REQUIRE(error.error_.type == ErrorType::INVALID_ARGUMENT);
            rejected = true;
        }
        REQUIRE(rejected);
        REQUIRE(allocator.requests.empty());
    };
    check(513, 0, 512);
    check(513, -1, 512);
    check(513, std::numeric_limits<int64_t>::max(), 512);
    // Source span overflows while the 512-row sampled span still fits.
    const auto dim = std::numeric_limits<size_t>::max() / (513 * element_bytes) + 1;
    REQUIRE(dim <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
    REQUIRE(dim <= std::numeric_limits<size_t>::max() / (512 * element_bytes));
    check(513, static_cast<int64_t>(dim), 512);
    if (dtype == DataTypes::DATA_TYPE_INT8) {
        check(std::numeric_limits<int64_t>::max(), 1, std::numeric_limits<int64_t>::max() - 1);
    }
}

TEST_CASE("Training sampling rejects unsupported dtype before allocation",
          "[ut][UtilFunctions][maintenance_review][sampling_dtype]") {
    const auto dtype = GENERATE(DataTypes::DATA_TYPE_SPARSE, static_cast<DataTypes>(999));
    float dummy = 0;
    auto base = Dataset::Make()->Owner(false)->NumElements(513)->Dim(1)->Float32Vectors(&dummy);
    SamplingReviewAllocator allocator;
    allocator.reject_all = true;
    bool rejected = false;
    try {
        auto sampled = sample_train_data(base, 513, 1, 512, &allocator, 42, dtype);
    } catch (const VsagException& error) {
        REQUIRE(error.error_.type == ErrorType::INVALID_ARGUMENT);
        rejected = true;
    }
    REQUIRE(rejected);
    REQUIRE(allocator.requests.empty());
}
