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

#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>
#include <type_traits>
#include <vector>

#include "flatten_datacell_parameter.h"
#include "impl/allocator/safe_allocator.h"
#include "quantization/scalar_quantization/half_precision_traits.h"
#include "rabitq_split_datacell.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "unittest.h"
#include "utils/half_input.h"

namespace vsag {
namespace {

class SplitHalfCountingAllocator : public Allocator {
public:
    std::string
    Name() override {
        return "SplitHalfCountingAllocator";
    }

    void*
    Allocate(uint64_t size) override {
        ++allocation_count;
        allocated_bytes += size;
        return delegate_->Allocate(size);
    }

    void
    Deallocate(void* pointer) override {
        delegate_->Deallocate(pointer);
    }

    void*
    Reallocate(void* pointer, uint64_t size) override {
        ++allocation_count;
        allocated_bytes += size;
        return delegate_->Reallocate(pointer, size);
    }

    uint64_t allocation_count{0};
    uint64_t allocated_bytes{0};

private:
    std::shared_ptr<Allocator> delegate_ = SafeAllocator::FactoryDefaultAllocator();
};

template <typename Format>
void
CheckSplitHalfInput(MetricType metric, bool mrle) {
    constexpr InnerIdType train_count = 128;
    constexpr InnerIdType insert_count = 5;
    const uint64_t dim = mrle ? 65 : 64;
    constexpr auto input_type =
        std::is_same_v<Format, FP16Format> ? DataTypes::DATA_TYPE_FP16 : DataTypes::DATA_TYPE_BF16;
    CAPTURE(input_type, metric, mrle, dim);

    auto json = JsonType::Parse(R"(
        {
            "codes_type": "rabitq_split",
            "io_params": {"type": "memory_io"},
            "quantization_params": {
                "type": "rabitq",
                "rabitq_version": "split",
                "rabitq_bits_per_dim_query": 32,
                "rabitq_bits_per_dim_base": 8,
                "rabitq_bits_per_dim_filter": 3,
                "use_fht": true
            }
        }
    )");
    if (mrle) {
        json["quantization_params"]["type"].SetString("tq");
        json["quantization_params"]["tq_chain"].SetString("mrle, rabitq");
        json["quantization_params"]["mrle_dim"].SetUint64(64);
    }
    auto param = std::make_shared<FlattenDataCellParameter>();
    param->FromJson(json);
    IndexCommonParam common;
    common.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    common.dim_ = dim;
    common.metric_ = metric;
    common.data_type_ = input_type;
    auto float_common = common;
    float_common.data_type_ = DataTypes::DATA_TYPE_FLOAT;

    // Reference inputs are expanded from the actual half bits, not the source
    // floats. Keep every row nonzero, but include signed zeros and subnormals.
    const auto source = fixtures::generate_vectors(train_count, dim, false, 193);
    std::vector<uint16_t> bits(source.size());
    std::vector<float> decoded(source.size());
    for (uint64_t i = 0; i < source.size(); ++i) {
        bits[i] = Format::FloatToHalf(source[i]);
    }
    bits[0] = 0x0000;
    bits[1] = 0x8000;
    bits[2] = 0x0001;
    bits[3] = 0x8001;
    for (uint64_t i = 0; i < bits.size(); ++i) {
        if constexpr (std::is_same_v<Format, FP16Format>) {
            decoded[i] = half_input::FP16ToFloat(bits[i]);
        } else {
            decoded[i] = Format::HalfToFloat(bits[i]);
        }
    }
    REQUIRE(decoded[2] > 0.0F);
    REQUIRE(decoded[3] < 0.0F);
    REQUIRE(std::signbit(decoded[1]));

    auto actual = FlattenInterface::MakeInstance(param, common);
    auto expected = FlattenInterface::MakeInstance(param, float_common);
    REQUIRE(actual != nullptr);
    REQUIRE(expected != nullptr);
    // Train independently to exercise the half boundary, including the MRLE
    // transform's constructor dtype rather than copying the FLOAT model over.
    actual->Train(bits.data(), train_count);
    expected->Train(decoded.data(), train_count);
    actual->Resize(16);
    expected->Resize(16);
    std::array<InnerIdType, insert_count> ids = {9, 2, 13, 5, 0};
    actual->BatchInsertVector(bits.data(), insert_count, ids.data());
    expected->BatchInsertVector(decoded.data(), insert_count, ids.data());

    auto check = [&](const FlattenInterfacePtr& half_cell, const FlattenInterfacePtr& float_cell) {
        auto half_split = std::dynamic_pointer_cast<RaBitQSplitDataCellInterface>(half_cell);
        auto float_split = std::dynamic_pointer_cast<RaBitQSplitDataCellInterface>(float_cell);
        REQUIRE(half_split != nullptr);
        REQUIRE(float_split != nullptr);
        REQUIRE(half_split->FusedFilterBits() == 3);
        REQUIRE(half_split->FusedSupplementBits() == 5);
        REQUIRE(half_split->OneBitCodeSize() == float_split->OneBitCodeSize());
        REQUIRE(half_split->SupplementCodeSize() == float_split->SupplementCodeSize());
        std::vector<uint8_t> half_filter(half_split->OneBitCodeSize());
        std::vector<uint8_t> float_filter(float_split->OneBitCodeSize());
        std::vector<uint8_t> half_supplement(half_split->SupplementCodeSize());
        std::vector<uint8_t> float_supplement(float_split->SupplementCodeSize());
        for (const auto id : ids) {
            CAPTURE(id);
            REQUIRE(half_split->CopySplitCodes(id, half_filter.data(), half_supplement.data()));
            REQUIRE(float_split->CopySplitCodes(id, float_filter.data(), float_supplement.data()));
            REQUIRE(half_filter == float_filter);
            REQUIRE(half_supplement == float_supplement);
        }
        auto close = [](float value, float reference) {
            REQUIRE(std::isfinite(value));
            REQUIRE(std::isfinite(reference));
            REQUIRE(std::abs(value - reference) <=
                    2e-5F * std::max({1.0F, std::abs(value), std::abs(reference)}));
        };
        for (const uint64_t row : {0ULL, 17ULL, 91ULL}) {
            CAPTURE(row);
            auto half_computer = half_cell->FactoryComputer(bits.data() + row * dim);
            auto float_computer = float_cell->FactoryComputer(decoded.data() + row * dim);
            std::array<float, insert_count> distances{}, reference_distances{};
            half_cell->Query(distances.data(), half_computer, ids.data(), insert_count, nullptr);
            float_cell->Query(
                reference_distances.data(), float_computer, ids.data(), insert_count, nullptr);
            for (InnerIdType i = 0; i < insert_count; ++i) {
                CAPTURE(i);
                close(distances[i], reference_distances[i]);
            }
            // The typed input adapter must not replace the native split interface.
            std::array<float, insert_count> coarse{}, reference_coarse{};
            std::array<float, insert_count> bounds{}, reference_bounds{};
            std::array<float, insert_count> ips{}, reference_ips{};
            half_split->QueryWithDistanceLowerBoundAndFilterIP(
                coarse.data(), bounds.data(), ips.data(), half_computer, ids.data(), insert_count);
            float_split->QueryWithDistanceLowerBoundAndFilterIP(reference_coarse.data(),
                                                                reference_bounds.data(),
                                                                reference_ips.data(),
                                                                float_computer,
                                                                ids.data(),
                                                                insert_count);
            for (InnerIdType i = 0; i < insert_count; ++i) {
                CAPTURE(i);
                close(coarse[i], reference_coarse[i]);
                close(bounds[i], reference_bounds[i]);
                close(ips[i], reference_ips[i]);
            }
            half_split->QueryWithFilterIPHint(
                distances.data(), ips.data(), half_computer, ids.data(), insert_count);
            float_split->QueryWithFilterIPHint(reference_distances.data(),
                                               reference_ips.data(),
                                               float_computer,
                                               ids.data(),
                                               insert_count);
            for (InnerIdType i = 0; i < insert_count; ++i) {
                close(distances[i], reference_distances[i]);
            }
        }
    };
    auto check_raw = [&](const FlattenInterfacePtr& half_cell,
                         const FlattenInterfacePtr& float_cell,
                         uint64_t updated_row) {
        REQUIRE_FALSE(half_cell->CompareRawVectorWithId(nullptr, ids[0]));
        for (InnerIdType i = 0; i < insert_count; ++i) {
            const uint64_t row = i == 1 ? updated_row : i;
            CAPTURE(i, row);
            REQUIRE(half_cell->CompareRawVectorWithId(bits.data() + row * dim, ids[i]));
            REQUIRE(float_cell->CompareRawVectorWithId(decoded.data() + row * dim, ids[i]));
            const auto wrong_row = (row + 37) % train_count;
            const auto reference =
                float_cell->CompareRawVectorWithId(decoded.data() + wrong_row * dim, ids[i]);
            REQUIRE_FALSE(reference);
            REQUIRE(half_cell->CompareRawVectorWithId(bits.data() + wrong_row * dim, ids[i]) ==
                    reference);
        }
    };
    check(actual, expected);
    check_raw(actual, expected, 1);
    REQUIRE(actual->UpdateVector(bits.data() + 23 * dim, ids[1]));
    REQUIRE(expected->UpdateVector(decoded.data() + 23 * dim, ids[1]));
    REQUIRE_FALSE(actual->UpdateVector(bits.data(), 16));
    REQUIRE_FALSE(expected->UpdateVector(decoded.data(), 16));
    check(actual, expected);
    check_raw(actual, expected, 23);

    auto reload = [&](const FlattenInterfacePtr& cell, const IndexCommonParam& common_param) {
        std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
        IOStreamWriter writer(stream);
        cell->Serialize(writer);
        stream.seekg(0);
        IOStreamReader reader(stream);
        auto restored = FlattenInterface::MakeInstance(param, common_param);
        restored->Deserialize(reader);
        return restored;
    };
    auto restored = reload(actual, common);
    auto restored_float = reload(expected, float_common);
    check(restored, restored_float);
    check_raw(restored, restored_float, 23);
    // Restored cells must retain the caller's dtype for later writes as well.
    REQUIRE(restored->UpdateVector(bits.data() + 47 * dim, ids[1]));
    REQUIRE(restored_float->UpdateVector(decoded.data() + 47 * dim, ids[1]));
    check(restored, restored_float);
    check_raw(restored, restored_float, 47);
}

template <typename Format>
void
CheckSplitHalfBatchAllocations(bool explicit_ids) {
    constexpr InnerIdType train_count = 128;
    constexpr InnerIdType count = 5;
    constexpr uint64_t dim = 64;
    constexpr uint64_t row_bytes = dim * sizeof(float);
    constexpr auto input_type =
        std::is_same_v<Format, FP16Format> ? DataTypes::DATA_TYPE_FP16 : DataTypes::DATA_TYPE_BF16;
    CAPTURE(input_type, explicit_ids);
    auto param = std::make_shared<FlattenDataCellParameter>();
    param->FromJson(JsonType::Parse(R"(
        {
            "codes_type": "rabitq_split",
            "io_params": {"type": "memory_io"},
            "quantization_params": {
                "type": "rabitq",
                "rabitq_version": "split",
                "rabitq_bits_per_dim_query": 32,
                "rabitq_bits_per_dim_base": 8,
                "rabitq_bits_per_dim_filter": 3,
                "use_fht": true
            }
        }
    )"));
    const auto source = fixtures::generate_vectors(train_count, dim, false, 193);
    std::vector<uint16_t> bits(source.size());
    std::vector<float> decoded(source.size());
    for (uint64_t i = 0; i < source.size(); ++i) {
        bits[i] = Format::FloatToHalf(source[i]);
        if constexpr (std::is_same_v<Format, FP16Format>) {
            decoded[i] = half_input::FP16ToFloat(bits[i]);
        } else {
            decoded[i] = Format::HalfToFloat(bits[i]);
        }
    }
    // Compare half batch/individual and native FLOAT batch/individual. Count only
    // insertion allocations: model training and capacity growth are out of scope.
    std::array<std::shared_ptr<SplitHalfCountingAllocator>, 4> allocators;
    std::array<FlattenInterfacePtr, 4> cells;
    for (size_t i = 0; i < cells.size(); ++i) {
        allocators[i] = std::make_shared<SplitHalfCountingAllocator>();
        IndexCommonParam common;
        common.allocator_ = allocators[i];
        common.dim_ = dim;
        common.metric_ = MetricType::METRIC_TYPE_L2SQR;
        common.data_type_ = i < 2 ? input_type : DataTypes::DATA_TYPE_FLOAT;
        cells[i] = FlattenInterface::MakeInstance(param, common);
        REQUIRE(cells[i] != nullptr);
        cells[i]->Train(i < 2 ? static_cast<const void*>(bits.data()) : decoded.data(),
                        train_count);
        cells[i]->Resize(32);
        allocators[i]->allocation_count = 0;
        allocators[i]->allocated_bytes = 0;
    }
    std::array<InnerIdType, count> ids = explicit_ids
                                             ? std::array<InnerIdType, count>{9, 2, 13, 5, 0}
                                             : std::array<InnerIdType, count>{0, 1, 2, 3, 4};
    cells[0]->BatchInsertVector(bits.data(), count, explicit_ids ? ids.data() : nullptr);
    cells[2]->BatchInsertVector(decoded.data(), count, explicit_ids ? ids.data() : nullptr);
    for (InnerIdType i = 0; i < count; ++i) {
        if (explicit_ids) {
            cells[1]->InsertVector(bits.data() + i * dim, ids[i]);
            cells[3]->InsertVector(decoded.data() + i * dim, ids[i]);
        } else {
            cells[1]->InsertVector(bits.data() + i * dim);
            cells[3]->InsertVector(decoded.data() + i * dim);
        }
    }
    REQUIRE(allocators[2]->allocation_count == allocators[3]->allocation_count);
    REQUIRE(allocators[2]->allocated_bytes == allocators[3]->allocated_bytes);
    REQUIRE(allocators[0]->allocation_count == allocators[2]->allocation_count + 1);
    REQUIRE(allocators[0]->allocated_bytes == allocators[2]->allocated_bytes + row_bytes);
    REQUIRE(allocators[1]->allocation_count == allocators[3]->allocation_count + count);
    REQUIRE(allocators[1]->allocated_bytes == allocators[3]->allocated_bytes + count * row_bytes);
    const InnerIdType expected_count = explicit_ids ? 14 : count;
    for (size_t i = 0; i < cells.size(); ++i) {
        REQUIRE(cells[i]->TotalCount() == expected_count);
        // Empty batches must neither allocate nor change the high-water count.
        const auto allocations_before = allocators[i]->allocation_count;
        const auto bytes_before = allocators[i]->allocated_bytes;
        cells[i]->BatchInsertVector(nullptr, 0, nullptr);
        REQUIRE(cells[i]->TotalCount() == expected_count);
        REQUIRE(allocators[i]->allocation_count == allocations_before);
        REQUIRE(allocators[i]->allocated_bytes == bytes_before);
    }
    for (size_t i = 0; i < cells.size(); ++i) {
        cells[i]->InsertVector(i < 2 ? static_cast<const void*>(bits.data() + count * dim)
                                     : decoded.data() + count * dim);
        REQUIRE(cells[i]->TotalCount() == expected_count + 1);
    }
    auto reference = std::dynamic_pointer_cast<RaBitQSplitDataCellInterface>(cells[3]);
    REQUIRE(reference != nullptr);
    std::vector<uint8_t> filter(reference->OneBitCodeSize());
    std::vector<uint8_t> supplement(reference->SupplementCodeSize());
    std::vector<uint8_t> actual_filter(filter.size());
    std::vector<uint8_t> actual_supplement(supplement.size());
    std::vector<InnerIdType> checked_ids(ids.begin(), ids.end());
    checked_ids.push_back(expected_count);
    for (const auto id : checked_ids) {
        CAPTURE(id);
        REQUIRE(reference->CopySplitCodes(id, filter.data(), supplement.data()));
        for (const auto& cell : cells) {
            auto split = std::dynamic_pointer_cast<RaBitQSplitDataCellInterface>(cell);
            REQUIRE(split != nullptr);
            REQUIRE(split->CopySplitCodes(id, actual_filter.data(), actual_supplement.data()));
            REQUIRE(actual_filter == filter);
            REQUIRE(actual_supplement == supplement);
        }
    }
}

}  // namespace

TEST_CASE("RaBitQ half batch reuses one row allocation", "[ut][RaBitQSplitHalfInput]") {
    const bool explicit_ids = GENERATE(false, true);
    CheckSplitHalfBatchAllocations<FP16Format>(explicit_ids);
    CheckSplitHalfBatchAllocations<BF16Format>(explicit_ids);
}

TEST_CASE("RaBitQ split3+5 half input matches decoded FLOAT", "[ut][RaBitQSplitHalfInput]") {
    const auto metric = GENERATE(MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_IP);
    CheckSplitHalfInput<FP16Format>(metric, false);
    CheckSplitHalfInput<BF16Format>(metric, false);
}

TEST_CASE("RaBitQ MRLE split3+5 half input matches decoded FLOAT", "[ut][RaBitQSplitHalfInput]") {
    CheckSplitHalfInput<FP16Format>(MetricType::METRIC_TYPE_L2SQR, true);
    CheckSplitHalfInput<BF16Format>(MetricType::METRIC_TYPE_L2SQR, true);
}

}  // namespace vsag
