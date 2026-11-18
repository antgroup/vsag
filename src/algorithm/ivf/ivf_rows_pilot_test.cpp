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

#include <catch2/catch_approx.hpp>
#include <cmath>
#include <numeric>
#include <sstream>

#include "impl/allocator/safe_allocator.h"
#include "ivf.h"
#include "simd/bf16_simd.h"
#include "simd/fp16_simd.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "unittest.h"

namespace {
class PrivateRowsIVF final : public vsag::IVF {
public:
    PrivateRowsIVF(const vsag::IVFParameterPtr& p, const vsag::IndexCommonParam& c)
        : IVF(p, c, PrivateTrainingRowsPilot{}) {
    }
};

float
ExpandHalf(uint16_t bits, bool bf16) {
    // Independent mathematical decoding, not the production half-input helper.
    if (bf16) {
        const int exponent = (bits >> 7) & 255;
        const float mantissa = static_cast<float>(bits & 127);
        const float magnitude = exponent == 0
                                    ? std::ldexp(mantissa, -133)
                                    : std::ldexp(1.0F + mantissa / 128.0F, exponent - 127);
        return bits & 0x8000 ? -magnitude : magnitude;
    }
    const int exponent = (bits >> 10) & 31;
    const float mantissa = static_cast<float>(bits & 1023);
    const float magnitude = exponent == 0 ? std::ldexp(mantissa, -24)
                                          : std::ldexp(1.0F + mantissa / 1024.0F, exponent - 15);
    return bits & 0x8000 ? -magnitude : magnitude;
}
}  // namespace

TEST_CASE("Private nearest SQ8 TrainingRows native lifecycle", "[ut][IVFRowsPilot]") {
    const bool bf16 = GENERATE(false, true);
    const auto metric = GENERATE(vsag::MetricType::METRIC_TYPE_L2SQR,
                                 vsag::MetricType::METRIC_TYPE_IP,
                                 vsag::MetricType::METRIC_TYPE_COSINE);
    const auto quantizer = std::string("sq8");
    const bool residual = GENERATE(false, true);
    CAPTURE(bf16, metric, quantizer);
    const int64_t dim = GENERATE(7, 33);
    constexpr int64_t count = 48;
    std::vector<uint16_t> half(count * dim);
    std::vector<float> rounded(count * dim);
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 0);
    for (uint64_t i = 0; i < half.size(); ++i) {
        const auto value = std::sin(static_cast<float>(i * 13 + 7)) * 0.75F;
        half[i] = bf16 ? vsag::generic::FloatToBF16(value) : vsag::generic::FloatToFP16(value);
        rounded[i] = ExpandHalf(half[i], bf16);
    }
    auto params = std::make_shared<vsag::IVFParameter>();
    params->FromString(fmt::format(R"({{
        "type":"ivf", "build_thread_count":1,
        "buckets_params":{{"io_params":{{"type":"block_memory_io"}},
            "quantization_params":{{"type":"{}"}}, "buckets_count":1, "use_residual":{}}},
        "partition_strategy":{{"partition_strategy_type":"ivf", "ivf_train_type":"kmeans",
            "use_route_graph":false}}, "use_reorder":false, "buckets_per_data":1,
        "use_attribute_filter":false
    }})",
                                   quantizer,
                                   residual ? "true" : "false"));
    params->ivf_partition_strategy_parameter->use_route_graph = false;
    vsag::IndexCommonParam common;
    common.dim_ = dim;
    common.metric_ = metric;
    common.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();
    common.data_type_ = bf16 ? vsag::DataTypes::DATA_TYPE_BF16 : vsag::DataTypes::DATA_TYPE_FP16;
    PrivateRowsIVF actual(params, common);
    common.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    vsag::IVF reference(params, common);
    auto dataset = [&](bool typed, int64_t first, int64_t rows) {
        auto data = vsag::Dataset::Make()
                        ->Dim(dim)
                        ->NumElements(rows)
                        ->Ids(ids.data() + first)
                        ->Owner(false);
        if (typed) {
            data->Float16Vectors(half.data() + first * dim);
        } else {
            data->Float32Vectors(rounded.data() + first * dim);
        }
        return data;
    };
    REQUIRE(actual.Build(dataset(true, 0, 32)).empty());
    REQUIRE(reference.Build(dataset(false, 0, 32)).empty());
    REQUIRE(actual.Add(dataset(true, 32, 16)).empty());
    REQUIRE(reference.Add(dataset(false, 32, 16)).empty());
    std::vector<uint8_t> ca(4096), cb(4096);
    for (int64_t id = 0; id < count; ++id) {
        std::fill(ca.begin(), ca.end(), 0);
        std::fill(cb.begin(), cb.end(), 0);
        actual.GetCodeByInnerId(id, ca.data());
        reference.GetCodeByInnerId(id, cb.data());
        REQUIRE(ca == cb);
    }

    auto compare = [&](const vsag::DatasetPtr& a, const vsag::DatasetPtr& b) {
        REQUIRE(a->GetNumElements() == b->GetNumElements());
        REQUIRE(a->GetDim() == b->GetDim());
        for (int64_t i = 0; i < a->GetNumElements() * a->GetDim(); ++i) {
            REQUIRE(a->GetIds()[i] == b->GetIds()[i]);
            REQUIRE(a->GetDistances()[i] == Catch::Approx(b->GetDistances()[i]).margin(1e-5));
        }
    };
    const std::string search = R"({"ivf":{"scan_buckets_count":1}})";
    compare(actual.KnnSearch(dataset(true, 5, 1), 10, search, nullptr),
            reference.KnnSearch(dataset(false, 5, 1), 10, search, nullptr));
    compare(actual.KnnSearch(dataset(true, 5, 3), 10, search, nullptr),
            reference.KnnSearch(dataset(false, 5, 3), 10, search, nullptr));
    compare(actual.RangeSearch(dataset(true, 5, 1), 100.0F, search, nullptr, 48),
            reference.RangeSearch(dataset(false, 5, 1), 100.0F, search, nullptr, 48));
    // IVF does not implement vector update. Keep the existing capability rather
    // than silently introducing a different update contract for half inputs.
    REQUIRE_THROWS(actual.UpdateVector(5, dataset(true, 5, 1), false));
    REQUIRE_THROWS(reference.UpdateVector(5, dataset(false, 5, 1), false));
    std::stringstream serialized;
    vsag::IOStreamWriter writer(serialized);
    actual.Serialize(writer);
    common.data_type_ = bf16 ? vsag::DataTypes::DATA_TYPE_BF16 : vsag::DataTypes::DATA_TYPE_FP16;
    PrivateRowsIVF restored(params, common);
    vsag::IOStreamReader reader(serialized);
    restored.Deserialize(reader);
    compare(restored.KnnSearch(dataset(true, 5, 1), 10, search, nullptr),
            reference.KnnSearch(dataset(false, 5, 1), 10, search, nullptr));
}
