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

#include "sq_local_split_datacell.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "flatten_datacell_parameter.h"
#include "flatten_interface.h"
#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
#include "inner_string_params.h"
#include "unittest.h"

using namespace vsag;

namespace {

std::vector<float>
random_vectors(int64_t count, int64_t dim, uint32_t seed) {
    std::vector<float> values(static_cast<size_t>(count * dim));
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    for (auto& value : values) {
        value = dist(rng);
    }
    return values;
}

FlattenInterfacePtr
make_datacell(const std::shared_ptr<Allocator>& allocator,
              IndexCommonParam& common_param,
              uint64_t dim,
              const std::string& io_type = "memory_io") {
    auto param_json = JsonType::Parse(
        std::string(R"({"codes_type":"sq_local_split","io_params":{"type":")") + io_type +
        R"("},"quantization_params":{"type":"sq_local_split",)"
        R"("sq_local_bits_per_dim_base":3,"sq_local_bits_per_dim_precise":5}})");
    auto param = std::make_shared<FlattenDataCellParameter>();
    param->FromJson(param_json);
    common_param.allocator_ = allocator;
    common_param.dim_ = static_cast<int64_t>(dim);
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common_param.data_type_ = DataTypes::DATA_TYPE_FLOAT;
    return FlattenInterface::MakeInstance(param, common_param);
}

}  // namespace

TEST_CASE("sq_local_split datacell reports the split storage contract",
          "[ut][SQLocalSplitDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    constexpr uint64_t dim = 32;
    auto flatten = make_datacell(allocator, common_param, dim);

    REQUIRE(flatten != nullptr);
    REQUIRE(flatten->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_SQ_LOCAL_SPLIT);
    REQUIRE(flatten->SupportSplitCodeStorage());
    REQUIRE(flatten->GetMetricType() == MetricType::METRIC_TYPE_L2SQR);
    REQUIRE(flatten->InMemory());
    REQUIRE(flatten->code_size_ > dim);

    auto optimized = std::dynamic_pointer_cast<FlattenOptimizedBuildInterface>(flatten);
    REQUIRE(optimized != nullptr);
    REQUIRE_FALSE(optimized->IsOptimizedBuildActive());
}

TEST_CASE("sq_local_split datacell build modes produce identical distances",
          "[ut][SQLocalSplitDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    constexpr uint64_t dim = 32;
    constexpr InnerIdType count = 64;
    const auto vectors = random_vectors(count, static_cast<int64_t>(dim), 42);
    const auto queries = random_vectors(5, static_cast<int64_t>(dim), 7);

    // Reference: direct insert into the packed split layout.
    auto direct = make_datacell(allocator, common_param, dim);
    direct->Train(vectors.data(), count);
    direct->Resize(count);
    for (InnerIdType i = 0; i < count; ++i) {
        direct->InsertVector(vectors.data() + static_cast<uint64_t>(i) * dim, i);
    }

    // Optimized build: temporary byte-per-dimension layout, packed in FinalizeOptimizedBuild.
    IndexCommonParam optimized_common_param;
    auto optimized = std::dynamic_pointer_cast<FlattenOptimizedBuildInterface>(
        make_datacell(allocator, optimized_common_param, dim));
    REQUIRE(optimized != nullptr);
    auto optimized_flatten = std::dynamic_pointer_cast<FlattenInterface>(optimized);
    REQUIRE(optimized_flatten != nullptr);
    optimized_flatten->Train(vectors.data(), count);
    optimized_flatten->Resize(count);
    FlattenOptimizedBuildContext context;
    context.thread_count = 1;
    REQUIRE(optimized->BeginOptimizedBuild(context));
    for (InnerIdType i = 0; i < count; ++i) {
        optimized_flatten->InsertVector(vectors.data() + static_cast<uint64_t>(i) * dim, i);
    }

    const std::vector<InnerIdType> ids{0, 1, 7, 33, 63};

    // Distances in optimized-build mode use the stored code of the query id (this is what the
    // graph build does), so they must match the packed layout's id-to-id distances.
    {
        auto build_computer = optimized->FactoryComputerForBuild(nullptr, ids[0]);
        std::vector<float> expected(ids.size());
        std::vector<float> actual(ids.size());
        direct->QueryById(
            expected.data(), ids[0], ids.data(), static_cast<InnerIdType>(ids.size()));
        optimized_flatten->Query(
            actual.data(), build_computer, ids.data(), static_cast<InnerIdType>(ids.size()));
        for (size_t i = 0; i < ids.size(); ++i) {
            REQUIRE(actual[i] == Catch::Approx(expected[i]).epsilon(1e-4));
        }
    }

    optimized->FinalizeOptimizedBuild();
    REQUIRE_FALSE(optimized->IsOptimizedBuildActive());

    // After finalize the packed records must behave exactly like the direct-insert layout.
    for (uint64_t q = 0; q < queries.size() / dim; ++q) {
        const float* query = queries.data() + q * dim;
        auto reference_computer = direct->FactoryComputer(query);
        auto computer = optimized_flatten->FactoryComputer(query);
        std::vector<float> expected(ids.size());
        std::vector<float> actual(ids.size());
        direct->Query(
            expected.data(), reference_computer, ids.data(), static_cast<InnerIdType>(ids.size()));
        optimized_flatten->Query(
            actual.data(), computer, ids.data(), static_cast<InnerIdType>(ids.size()));
        for (size_t i = 0; i < ids.size(); ++i) {
            REQUIRE(actual[i] == Catch::Approx(expected[i]).epsilon(1e-4));
        }
    }

    // Id-to-id distances and the batched QueryById path agree with each other.
    std::vector<float> by_pair(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        by_pair[i] = direct->ComputePairVectors(0, ids[i]);
    }
    std::vector<float> by_id(ids.size());
    direct->QueryById(by_id.data(), 0, ids.data(), static_cast<InnerIdType>(ids.size()));
    for (size_t i = 0; i < ids.size(); ++i) {
        // Self distances cancel to ~1e-3, so compare with an absolute floor as well.
        REQUIRE(std::abs(by_id[i] - by_pair[i]) <= 1e-4F * std::max(1.0F, std::abs(by_pair[i])));
    }
}

TEST_CASE("sq_local_split datacell serializes its split records", "[ut][SQLocalSplitDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    constexpr uint64_t dim = 24;
    constexpr InnerIdType count = 32;
    const auto vectors = random_vectors(count, static_cast<int64_t>(dim), 9);
    const auto queries = random_vectors(4, static_cast<int64_t>(dim), 11);

    auto source = make_datacell(allocator, common_param, dim);
    source->Train(vectors.data(), count);
    source->Resize(count);
    for (InnerIdType i = 0; i < count; ++i) {
        source->InsertVector(vectors.data() + static_cast<uint64_t>(i) * dim, i);
    }

    std::stringstream stream;
    IOStreamWriter writer(stream);
    source->Serialize(writer);

    IndexCommonParam restored_param;
    auto restored = make_datacell(allocator, restored_param, dim);
    stream.seekg(0, std::ios::beg);
    IOStreamReader reader(stream);
    REQUIRE_NOTHROW(restored->Deserialize(reader));
    REQUIRE(restored->TotalCount() == count);

    const std::vector<InnerIdType> ids{0, 5, 17, 31};
    for (uint64_t q = 0; q < queries.size() / dim; ++q) {
        const float* query = queries.data() + q * dim;
        auto source_computer = source->FactoryComputer(query);
        auto restored_computer = restored->FactoryComputer(query);
        std::vector<float> expected(ids.size());
        std::vector<float> actual(ids.size());
        source->Query(
            expected.data(), source_computer, ids.data(), static_cast<InnerIdType>(ids.size()));
        restored->Query(
            actual.data(), restored_computer, ids.data(), static_cast<InnerIdType>(ids.size()));
        for (size_t i = 0; i < ids.size(); ++i) {
            REQUIRE(actual[i] == Catch::Approx(expected[i]).epsilon(1e-5));
        }
    }

    // The own-code accessor returns a packed record with the datacell code size.
    bool need_release = false;
    const uint8_t* codes = restored->GetCodesById(3, need_release);
    REQUIRE(codes != nullptr);
    REQUIRE(source->code_size_ == restored->code_size_);
    restored->Release(codes);
}
