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
#include <nlohmann/json.hpp>
#include <numeric>
#include <set>
#include <sstream>

#include "impl/allocator/safe_allocator.h"
#include "ivf.h"
#include "simd/bf16_simd.h"
#include "simd/fp16_simd.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "unittest.h"
#include "vsag/factory.h"
#include "vsag_exception.h"

namespace {
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

TEST_CASE("IVF actual half inputs match independently rounded FLOAT", "[ut][IVFHalfInput]") {
    const bool bf16 = GENERATE(false, true);
    const auto metric = GENERATE(vsag::MetricType::METRIC_TYPE_L2SQR,
                                 vsag::MetricType::METRIC_TYPE_IP,
                                 vsag::MetricType::METRIC_TYPE_COSINE);
    const auto quantizer =
        GENERATE(std::string("fp32"), std::string("sq8"), std::string("fp16"), std::string("bf16"));
    CAPTURE(bf16, metric, quantizer);
    constexpr int64_t dim = 16;
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
            "quantization_params":{{"type":"{}"}}, "buckets_count":1, "use_residual":false}},
        "partition_strategy":{{"partition_strategy_type":"ivf", "ivf_train_type":"random",
            "use_route_graph":false}}, "use_reorder":false, "buckets_per_data":1,
        "use_attribute_filter":false
    }})",
                                   quantizer));
    params->ivf_partition_strategy_parameter->use_route_graph = false;
    vsag::IndexCommonParam common;
    common.dim_ = dim;
    common.metric_ = metric;
    common.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();
    common.data_type_ = bf16 ? vsag::DataTypes::DATA_TYPE_BF16 : vsag::DataTypes::DATA_TYPE_FP16;
    vsag::IVF actual(params, common);
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
    auto compare = [&](const vsag::DatasetPtr& a, const vsag::DatasetPtr& b) {
        REQUIRE(a->GetNumElements() == b->GetNumElements());
        REQUIRE(a->GetDim() == b->GetDim());
        for (int64_t i = 0; i < a->GetNumElements() * a->GetDim(); ++i) {
            REQUIRE(a->GetIds()[i] == b->GetIds()[i]);
            REQUIRE(a->GetDistances()[i] == Catch::Approx(b->GetDistances()[i]).margin(1e-5));
        }
    };
    auto compare_distance_api = [&](vsag::IVF& index) {
        vsag::InnerIndexInterface& native_api = index;
        vsag::InnerIndexInterface& float_api = reference;
        for (const bool precise : {false, true}) {
            REQUIRE(native_api.CalcDistanceById(dataset(true, 5, 1), 7, precise) ==
                    Catch::Approx(float_api.CalcDistanceById(dataset(false, 5, 1), 7, precise))
                        .margin(1e-5));
            auto require_missing_id = [&](vsag::InnerIndexInterface& api, bool typed) {
                try {
                    api.CalcDistanceById(dataset(typed, 5, 1), 999, precise);
                    FAIL("missing ID must raise INVALID_ARGUMENT");
                } catch (const vsag::VsagException& error) {
                    REQUIRE(error.error_.type == vsag::ErrorType::INVALID_ARGUMENT);
                }
            };
            require_missing_id(native_api, true);
            require_missing_id(float_api, false);
            std::vector<int64_t> selected{0, 7, 999, 0, 7, 999, 0, 7, 999};
            REQUIRE_THROWS(
                native_api.CalcDistancesById(dataset(true, 5, 3), selected.data(), 3, precise, -1));
            REQUIRE_THROWS(
                float_api.CalcDistancesById(dataset(false, 5, 3), selected.data(), 3, precise, -1));
            for (const int64_t topk : {-1, 2}) {
                const auto actual_distances = native_api.CalcDistancesById(
                    dataset(true, 5, 1), selected.data(), 3, precise, topk);
                const auto reference_distances = float_api.CalcDistancesById(
                    dataset(false, 5, 1), selected.data(), 3, precise, topk);
                REQUIRE(actual_distances->GetNumElements() == 1);
                REQUIRE(actual_distances->GetDim() == reference_distances->GetDim());
                REQUIRE(actual_distances->GetDim() == (topk == -1 ? 3 : topk));
                REQUIRE(actual_distances->GetValids() != nullptr);
                REQUIRE(reference_distances->GetValids() != nullptr);
                for (int64_t i = 0; i < actual_distances->GetDim(); ++i) {
                    const bool valid = i != 2;
                    REQUIRE(actual_distances->GetValids()[i] == valid);
                    REQUIRE(reference_distances->GetValids()[i] == valid);
                    if (valid) {
                        REQUIRE(actual_distances->GetDistances()[i] ==
                                Catch::Approx(reference_distances->GetDistances()[i]).margin(1e-5));
                    } else {
                        REQUIRE(std::isnan(actual_distances->GetDistances()[i]));
                        REQUIRE(std::isnan(reference_distances->GetDistances()[i]));
                    }
                }
            }
        }
        REQUIRE_THROWS(native_api.CalcDistanceById(dataset(false, 5, 1), 7));
        REQUIRE_THROWS(native_api.CalcDistanceById(dataset(true, 5, 3), 7));
    };
    compare_distance_api(actual);
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
    vsag::IVF restored(params, common);
    vsag::IOStreamReader reader(serialized);
    restored.Deserialize(reader);
    compare_distance_api(restored);
    compare(restored.KnnSearch(dataset(true, 5, 1), 10, search, nullptr),
            reference.KnnSearch(dataset(false, 5, 1), 10, search, nullptr));
}

TEST_CASE("IVF half route-only batching spans block boundaries", "[ut][IVFHalfInput][route]") {
    // route_buckets_only decodes typed queries in bounded blocks, so the interesting
    // query counts sit around the block size: 1 (no batching), exactly one block,
    // one past it, and several blocks. The random trainer draws centroids from
    // std::random_device, so a separately built FLOAT index would train different
    // centroids and could not be compared; instead this pins one trained index and
    // requires the batched route to reproduce the per-query route exactly.
    const bool bf16 = GENERATE(false, true);
    const auto metric =
        GENERATE(vsag::MetricType::METRIC_TYPE_L2SQR, vsag::MetricType::METRIC_TYPE_IP);
    const int64_t query_count = GENERATE(1, 3, 255, 256, 257, 600);
    CAPTURE(bf16, metric, query_count);
    constexpr int64_t dim = 8;
    constexpr int64_t buckets_count = 4;
    constexpr int64_t scan_buckets = 2;
    constexpr int64_t base_count = 64;

    std::vector<uint16_t> base16(base_count * dim);
    for (uint64_t i = 0; i < base16.size(); ++i) {
        const auto value = std::cos(static_cast<float>(i * 5 + 3)) * 0.5F;
        base16[i] = bf16 ? vsag::generic::FloatToBF16(value) : vsag::generic::FloatToFP16(value);
    }
    std::vector<uint16_t> query16(query_count * dim);
    for (uint64_t i = 0; i < query16.size(); ++i) {
        const auto value = std::sin(static_cast<float>(i * 11 + 1)) * 0.25F;
        query16[i] = bf16 ? vsag::generic::FloatToBF16(value) : vsag::generic::FloatToFP16(value);
    }
    std::vector<int64_t> base_ids(base_count);
    std::iota(base_ids.begin(), base_ids.end(), 0);
    std::vector<int64_t> query_ids(query_count);
    std::iota(query_ids.begin(), query_ids.end(), 0);

    auto params = std::make_shared<vsag::IVFParameter>();
    params->FromString(fmt::format(R"({{
        "type":"ivf", "build_thread_count":1,
        "buckets_params":{{"io_params":{{"type":"block_memory_io"}},
            "quantization_params":{{"type":"fp32"}}, "buckets_count":{}, "use_residual":false}},
        "partition_strategy":{{"partition_strategy_type":"ivf", "ivf_train_type":"random",
            "use_route_graph":false}}, "use_reorder":false, "buckets_per_data":1,
        "use_attribute_filter":false
    }})",
                                   buckets_count));
    params->ivf_partition_strategy_parameter->use_route_graph = false;
    vsag::IndexCommonParam common;
    common.dim_ = dim;
    common.metric_ = metric;
    common.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();
    common.data_type_ = bf16 ? vsag::DataTypes::DATA_TYPE_BF16 : vsag::DataTypes::DATA_TYPE_FP16;
    vsag::IVF index(params, common);

    auto typed_base = vsag::Dataset::Make()
                          ->Dim(dim)
                          ->NumElements(base_count)
                          ->Ids(base_ids.data())
                          ->Float16Vectors(base16.data())
                          ->Owner(false);
    REQUIRE(index.Build(typed_base).empty());

    auto typed_queries = [&](int64_t first, int64_t rows) {
        return vsag::Dataset::Make()
            ->Dim(dim)
            ->NumElements(rows)
            ->Ids(query_ids.data() + first)
            ->Float16Vectors(query16.data() + first * dim)
            ->Owner(false);
    };
    // Bucket-only routing returns the selected bucket and its distance for every query,
    // so this is the routing decision itself rather than a downstream search result.
    const auto route_params = fmt::format(
        R"({{"ivf":{{"scan_buckets_count":{},"disable_bucket_scan":true}}}})", scan_buckets);

    const auto batched = index.KnnSearch(typed_queries(0, query_count), 1, route_params, nullptr);
    REQUIRE(batched->GetNumElements() == query_count);
    REQUIRE(batched->GetDim() == scan_buckets);

    // Each query routed alone must land on the same buckets at the same distances as it
    // does inside the batched call. A block of typed queries that is decoded with a
    // shifted or stale row buffer would show up here.
    for (int64_t q = 0; q < query_count; ++q) {
        const auto single = index.KnnSearch(typed_queries(q, 1), 1, route_params, nullptr);
        REQUIRE(single->GetNumElements() == 1);
        REQUIRE(single->GetDim() == scan_buckets);
        for (int64_t j = 0; j < scan_buckets; ++j) {
            const auto batched_slot = q * scan_buckets + j;
            REQUIRE(single->GetIds()[j] == batched->GetIds()[batched_slot]);
            REQUIRE(single->GetDistances()[j] ==
                    Catch::Approx(batched->GetDistances()[batched_slot]).margin(1e-5));
        }
    }

    // A multi-query route must not leave the shared decode buffer dirty for a later call.
    const auto repeat = index.KnnSearch(typed_queries(0, query_count), 1, route_params, nullptr);
    REQUIRE(repeat->GetDim() == batched->GetDim());
    for (int64_t i = 0; i < query_count * scan_buckets; ++i) {
        REQUIRE(repeat->GetIds()[i] == batched->GetIds()[i]);
        REQUIRE(repeat->GetDistances()[i] ==
                Catch::Approx(batched->GetDistances()[i]).margin(1e-5));
    }
}

TEST_CASE("IVF flat reorder consumes FP32 rows for half indexes", "[ut][IVFHalfFlatReorder]") {
    const bool bf16 = GENERATE(false, true);
    const auto dtype = bf16 ? "bfloat16" : "float16";
    constexpr int64_t dim = 7;
    constexpr int64_t count = 270;
    std::vector<uint16_t> half(count * dim);
    std::vector<float> rounded(count * dim);
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 0);
    for (int64_t i = 0; i < count * dim; ++i) {
        const float v = std::sin(static_cast<float>(i * 17 + 3)) * 0.75F;
        half[i] = bf16 ? vsag::generic::FloatToBF16(v) : vsag::generic::FloatToFP16(v);
        rounded[i] = ExpandHalf(half[i], bf16);
    }
    const auto config = [](const char* type) {
        return fmt::format(R"({{
            "dtype":"{}", "metric_type":"l2", "dim":7,
            "index_param":{{"base_quantization_type":"sq8", "buckets_count":1,
                "ivf_train_type":"random", "use_route_graph":false, "use_reorder":true,
                "precise_quantization_type":"fp32", "precise_codes_layout":"flat"}}
        }})",
                           type);
    };
    const auto dataset = [&](bool typed, int64_t first, int64_t rows) {
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
    auto actual = vsag::Factory::CreateIndex("ivf", config(dtype));
    auto reference = vsag::Factory::CreateIndex("ivf", config("float32"));
    REQUIRE(actual.has_value());
    REQUIRE(reference.has_value());
    for (const bool typed : {false, true}) {
        const auto index = typed ? actual.value() : reference.value();
        auto built = index->Build(dataset(typed, 0, 13));
        REQUIRE(built.has_value());
        REQUIRE(built.value().empty());
        auto added = index->Add(dataset(typed, 13, count - 13));
        REQUIRE(added.has_value());
        REQUIRE(added.value().empty());
        REQUIRE(index->GetNumElements() == count);
    }
    const auto verify = [&](const auto& index) {
        const std::string search = R"({"ivf":{"scan_buckets_count":1}})";
        for (const auto id : {int64_t{0}, int64_t{13}, count - 2, count - 1}) {
            const auto query = dataset(true, id, 1);
            const auto self = index->CalcDistanceById(query, id, true);
            REQUIRE(self.has_value());
            REQUIRE(self.value() == 0.0F);
            const auto expected = reference.value()->KnnSearch(dataset(false, id, 1), 5, search);
            const auto got = index->KnnSearch(query, 5, search);
            REQUIRE(expected.has_value());
            REQUIRE(got.has_value());
            REQUIRE(got.value()->GetDim() == 5);
            for (int64_t j = 0; j < 5; ++j) {
                REQUIRE(got.value()->GetIds()[j] == expected.value()->GetIds()[j]);
                REQUIRE(got.value()->GetDistances()[j] ==
                        Catch::Approx(expected.value()->GetDistances()[j]).margin(1e-6));
            }
        }
    };
    verify(actual.value());
    auto artifact = actual.value()->Serialize();
    REQUIRE(artifact.has_value());
    auto restored = vsag::Factory::CreateIndex("ivf", config(dtype));
    REQUIRE(restored.has_value());
    REQUIRE(restored.value()->Deserialize(artifact.value()).has_value());
    verify(restored.value());
}

TEST_CASE("IVF balanced_kmeans half inputs build floor and ceil buckets",
          "[ut][IVFHalfInput][balanced]") {
    const bool bf16 = GENERATE(false, true);
    const std::string metric = GENERATE("l2", "ip", "cosine");
    // With residual codes the decoded rows also train them (TrainRows), for every metric.
    const auto [quantizer, residual] = GENERATE(std::make_pair(std::string("fp32"), false),
                                                std::make_pair(std::string("sq8"), true));
    CAPTURE(bf16, metric, quantizer, residual);
    constexpr int64_t dim = 16;
    constexpr int64_t buckets_count = 37;
    constexpr int64_t base_count = 1000;  // 37 * 27 + 1
    constexpr int64_t added_count = 100;
    std::vector<uint16_t> half((base_count + added_count) * dim);
    for (uint64_t i = 0; i < half.size(); ++i) {
        const auto value = std::sin(static_cast<float>(i * 13 + 5)) * 0.75F;
        half[i] = bf16 ? vsag::generic::FloatToBF16(value) : vsag::generic::FloatToFP16(value);
    }
    std::vector<int64_t> ids(base_count + added_count);
    std::iota(ids.begin(), ids.end(), 0);
    const auto config = fmt::format(R"({{
        "dtype":"{}", "metric_type":"{}", "dim":{},
        "index_param":{{"base_quantization_type":"{}", "buckets_count":{},
            "ivf_train_type":"balanced_kmeans", "use_residual":{}, "use_route_graph":false}}
    }})",
                                    bf16 ? "bfloat16" : "float16",
                                    metric,
                                    dim,
                                    quantizer,
                                    buckets_count,
                                    residual);
    auto index = vsag::Factory::CreateIndex("ivf", config);
    REQUIRE(index.has_value());
    const auto rows = [&](int64_t first, int64_t count) {
        return vsag::Dataset::Make()
            ->Dim(dim)
            ->NumElements(count)
            ->Ids(ids.data() + first)
            ->Float16Vectors(half.data() + first * dim)
            ->Owner(false);
    };
    // The trainer reads the decoded sample, and the Build is placed in one assignment.
    auto built = index.value()->Build(rows(0, base_count));
    REQUIRE(built.has_value());
    REQUIRE(built.value().empty());
    auto stats = nlohmann::json::parse(index.value()->GetStats());
    REQUIRE(stats["bucket_count"].get<int64_t>() == buckets_count);
    REQUIRE(stats["bucket_num"]["min"].get<float>() == base_count / buckets_count);
    REQUIRE(stats["bucket_num"]["max"].get<float>() == base_count / buckets_count + 1);
    // Add routes every vector to its nearest bucket, in blocks, as for any trainer.
    auto added = index.value()->Add(rows(base_count, added_count));
    REQUIRE(added.has_value());
    REQUIRE(added.value().empty());
    REQUIRE(index.value()->GetNumElements() == base_count + added_count);
    // Every vector is reachable through its bucket.
    const auto search = fmt::format(R"({{"ivf":{{"scan_buckets_count":{}}}}})", buckets_count);
    auto all = index.value()->KnnSearch(rows(7, 1), base_count + added_count, search);
    REQUIRE(all.has_value());
    REQUIRE(all.value()->GetDim() == base_count + added_count);
    const std::set<int64_t> seen(all.value()->GetIds(),
                                 all.value()->GetIds() + all.value()->GetDim());
    REQUIRE(seen.size() == static_cast<uint64_t>(base_count + added_count));
}

TEST_CASE("IVF balanced_kmeans half inputs train residual codes on the Build buckets",
          "[ut][IVFHalfInput][balanced]") {
    // As the FP32 test: 300 zeros and 100 vectors at 100, in two buckets of 200, so 100
    // zeros share a bucket with the far vectors, 50 below its centroid in every dimension.
    // Residual codes trained on nearest-bucket residuals cover only [0, 50] and turned
    // those zeros into 50s. Both values are exact in FP16 and BF16.
    const bool bf16 = GENERATE(false, true);
    CAPTURE(bf16);
    constexpr int64_t dim = 8;
    constexpr int64_t zeros = 300;
    constexpr int64_t far = 100;
    const auto encode = [&](float value) {
        return bf16 ? vsag::generic::FloatToBF16(value) : vsag::generic::FloatToFP16(value);
    };
    std::vector<uint16_t> half((zeros + far) * dim, encode(0.0F));
    std::fill(half.begin() + zeros * dim, half.end(), encode(100.0F));
    std::vector<int64_t> ids(zeros + far);
    std::iota(ids.begin(), ids.end(), 0);
    const auto config = fmt::format(R"({{
        "dtype":"{}", "metric_type":"l2", "dim":{},
        "index_param":{{"base_quantization_type":"sq8", "buckets_count":2,
            "ivf_train_type":"balanced_kmeans", "use_residual":true, "use_route_graph":false}}
    }})",
                                    bf16 ? "bfloat16" : "float16",
                                    dim);
    auto index = vsag::Factory::CreateIndex("ivf", config);
    REQUIRE(index.has_value());
    auto base = vsag::Dataset::Make()
                    ->Dim(dim)
                    ->NumElements(zeros + far)
                    ->Ids(ids.data())
                    ->Float16Vectors(half.data())
                    ->Owner(false);
    REQUIRE(index.value()->Build(base).has_value());
    auto stats = nlohmann::json::parse(index.value()->GetStats());
    REQUIRE(stats["bucket_num"]["min"].get<float>() == 200);
    REQUIRE(stats["bucket_num"]["max"].get<float>() == 200);

    std::vector<uint16_t> origin(dim, encode(0.0F));
    auto query = vsag::Dataset::Make()
                     ->Dim(dim)
                     ->NumElements(1)
                     ->Float16Vectors(origin.data())
                     ->Owner(false);
    auto result = index.value()->KnnSearch(query, zeros, R"({"ivf":{"scan_buckets_count":2}})");
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() == zeros);
    for (int64_t i = 0; i < zeros; ++i) {
        REQUIRE(result.value()->GetIds()[i] < zeros);
        REQUIRE(result.value()->GetDistances()[i] < 1.0F);
    }
}
