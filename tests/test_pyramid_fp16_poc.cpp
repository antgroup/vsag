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

// Half input buffers and paths below are borrowed by Owner(false) datasets.
// Keep the fixture alive until all Build/Add/Search calls using it have finished.

#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "json_wrapper.h"
#include "simd/bf16_simd.h"
#include "simd/fp16_simd.h"
#include "vsag/vsag.h"

namespace vsag {
namespace {

constexpr int64_t HALF_DIM = 32;
constexpr int64_t HALF_COUNT = 128;
constexpr int64_t BUILD_COUNT = 96;
constexpr int64_t TOPK = 5;
const std::string SEARCH_PARAM = R"({"pyramid":{"ef_search":256}})";

struct HalfInputFixture {
    int64_t dim;
    std::vector<float> floats;
    std::vector<uint16_t> halves;
    std::vector<int64_t> ids;
    std::vector<std::string> paths;
    std::vector<std::string> source_ids;

    explicit HalfInputFixture(bool bf16, int64_t dimension = HALF_DIM)
        : dim(dimension),
          floats(dim * HALF_COUNT),
          halves(dim * HALF_COUNT),
          ids(HALF_COUNT),
          paths(HALF_COUNT, "all/half/leaf"),
          source_ids(HALF_COUNT) {
        std::mt19937 rng(42);
        for (int64_t i = 0; i < dim * HALF_COUNT; ++i) {
            // Non-binary fractions exercise rounding. Stay in the normal range:
            // the legacy generic FP16 converter does not handle zero/subnormals.
            const auto sample = static_cast<int>(rng() % 128) - 64;
            const float input = (sample < 0 ? sample : sample + 1) / 65.0F;
            halves[i] = bf16 ? generic::FloatToBF16(input) : generic::FloatToFP16(input);
            // Compare identical represented values, not the pre-rounding input.
            floats[i] = bf16 ? generic::BF16ToFloat(halves[i]) : generic::FP16ToFloat(halves[i]);
        }
        for (int64_t i = 0; i < HALF_COUNT; ++i) {
            ids[i] = 1000 + 3 * i;  // Never confuse external IDs with row offsets.
            source_ids[i] = "half_source_" + std::to_string(i);
        }
    }

    DatasetPtr
    Dataset(int64_t start, int64_t count, bool half = true) const {
        auto data = vsag::Dataset::Make()
                        ->NumElements(count)
                        ->Dim(dim)
                        ->Ids(ids.data() + start)
                        ->Paths(paths.data() + start)
                        ->SourceID(source_ids.data() + start)
                        ->Owner(false);
        if (half) {
            data->Float16Vectors(halves.data() + start * dim);
        } else {
            data->Float32Vectors(floats.data() + start * dim);
        }
        return data;
    }

    std::vector<std::pair<float, int64_t>>
    Neighbors(int64_t row, int64_t count) const {
        std::vector<std::pair<float, int64_t>> distances;
        for (int64_t i = 0; i < count; ++i) {
            float distance = 0;
            for (int64_t j = 0; j < dim; ++j) {
                const auto delta = floats[row * dim + j] - floats[i * dim + j];
                distance += delta * delta;
            }
            distances.emplace_back(distance, ids[i]);
        }
        std::sort(distances.begin(), distances.end());
        return distances;
    }
};

std::string
HalfPyramidParam(const std::string& dtype,
                 const std::string& quantization,
                 const std::string& graph = "nsw",
                 bool reorder = false,
                 int64_t dim = HALF_DIM) {
    const auto root = graph == "odescent" ? "single_layer" : "multi_layer";
    return R"({"dtype":")" + dtype + R"(","metric_type":"l2","dim":)" + std::to_string(dim) + R"(,
        "index_param":{"max_degree":16,"ef_construction":16,
        "build_thread_count":1,"alpha":1.2,"graph_type":")" +
           graph + R"(","root_graph_type":")" + root + R"(","no_build_levels":[1,2],
        "base_quantization_type":")" +
           quantization + R"(","precise_quantization_type":"fp32","use_reorder":)" +
           (reorder ? "true" : "false") +
           R"(,"index_min_size":8,"persist_source_id":true,"store_paths":true}})";
}

IndexPtr
MakeHalfIndex(const std::string& param) {
    auto made = Factory::CreateIndex("pyramid", param);
    INFO("Pyramid factory parameters: " << param);
    if (not made.has_value()) {
        INFO("Pyramid factory error: " << made.error().message);
        REQUIRE(made.has_value());
    }
    REQUIRE(made.has_value());
    return made.value();
}

void
CheckHalfSearch(const IndexPtr& index,
                const HalfInputFixture& data,
                int64_t count,
                bool half = true) {
    REQUIRE(index->GetNumElements() == count);
    // First, middle, and last rows detect byte-stride errors, including Add's
    // final row. The expected neighbors are computed independently of VSAG.
    for (const auto row : {int64_t{0}, count / 2, count - 1}) {
        CAPTURE(row, count, half);
        auto query = data.Dataset(row, 1, half);
        const auto expected = data.Neighbors(row, count);
        auto knn = index->KnnSearch(query, TOPK, SEARCH_PARAM);
        REQUIRE(knn.has_value());
        REQUIRE(knn.value()->GetNumElements() == 1);
        REQUIRE(knn.value()->GetDim() == TOPK);
        for (int64_t i = 0; i < TOPK; ++i) {
            CAPTURE(i, expected[i].first, knn.value()->GetDistances()[i]);
            // SIMD reduction/FMA ordering can differ from scalar accumulation.
            // Allow 2e-5 relative distance error only; IDs below must still have
            // the exact scalar distance at this rank (apart from genuine ties).
            REQUIRE(knn.value()->GetDistances()[i] ==
                    Catch::Approx(expected[i].first).epsilon(2e-5F).margin(1e-4F));
            // Permit legitimate equal-distance ties, but not arbitrary IDs.
            const auto id = knn.value()->GetIds()[i];
            const auto match = std::find_if(
                expected.begin(), expected.end(), [id](const auto& e) { return e.second == id; });
            REQUIRE(match != expected.end());
            REQUIRE(match->first == expected[i].first);
        }
        REQUIRE(knn.value()->GetIds()[0] == data.ids[row]);
        REQUIRE(expected[1].first > 0.01F);
        auto range = index->RangeSearch(query, 0.01F, SEARCH_PARAM);
        REQUIRE(range.has_value());
        REQUIRE(range.value()->GetNumElements() == 1);
        REQUIRE(range.value()->GetDim() == 1);
        REQUIRE(range.value()->GetIds()[0] == data.ids[row]);
        REQUIRE(range.value()->GetDistances()[0] == Catch::Approx(0.0F).margin(1e-4F));
    }
}

void
CheckHalfRoundTrip(const IndexPtr& index,
                   const std::string& param,
                   const HalfInputFixture& data,
                   int64_t count) {
    auto binary = index->Serialize();
    REQUIRE(binary.has_value());
    auto restored = MakeHalfIndex(param);
    REQUIRE(restored->Deserialize(binary.value()).has_value());
    CheckHalfSearch(restored, data, count);
}

}  // namespace

TEST_CASE("Pyramid half input Build Add and search roundtrip",
          "[ft][pyramid][fp16][bf16][half_input]") {
    const bool bf16 = GENERATE(false, true);
    const std::string graph = GENERATE("nsw", "odescent");
    const int64_t dim = GENERATE(int64_t{32}, int64_t{33});
    CAPTURE(bf16, graph, dim);
    HalfInputFixture data(bf16, dim);
    const auto quantization = bf16 ? "bf16" : "fp16";
    const auto param =
        HalfPyramidParam(bf16 ? "bfloat16" : "float16", quantization, graph, false, dim);
    auto index = MakeHalfIndex(param);
    auto built = index->Build(data.Dataset(0, BUILD_COUNT));
    REQUIRE(built.has_value());
    REQUIRE(built.value().empty());
    CheckHalfSearch(index, data, BUILD_COUNT);
    auto added = index->Add(data.Dataset(BUILD_COUNT, HALF_COUNT - BUILD_COUNT));
    REQUIRE(added.has_value());
    REQUIRE(added.value().empty());
    CheckHalfSearch(index, data, HALF_COUNT);
    CheckHalfRoundTrip(index, param, data, HALF_COUNT);

    auto baseline = MakeHalfIndex(HalfPyramidParam("float32", "fp32", graph, false, dim));
    auto baseline_build = baseline->Build(data.Dataset(0, HALF_COUNT, false));
    REQUIRE(baseline_build.has_value());
    REQUIRE(baseline_build.value().empty());
    CheckHalfSearch(baseline, data, HALF_COUNT, false);
}

TEST_CASE("Pyramid half input supports sq8 and fp32 quantization",
          "[ft][pyramid][fp16][bf16][half_input][half_quantization]") {
    const bool bf16 = GENERATE(false, true);
    const std::string quantization = GENERATE("sq8", "fp32");
    CAPTURE(bf16, quantization);
    HalfInputFixture data(bf16);
    // Precise fp32 reordering must also accept half inputs. No permissive
    // factory/build failure branch: these are supported production combinations.
    const auto param = HalfPyramidParam(bf16 ? "bfloat16" : "float16", quantization, "nsw", true);
    auto index = MakeHalfIndex(param);
    auto built = index->Build(data.Dataset(0, BUILD_COUNT));
    REQUIRE(built.has_value());
    REQUIRE(built.value().empty());
    auto added = index->Add(data.Dataset(BUILD_COUNT, HALF_COUNT - BUILD_COUNT));
    REQUIRE(added.has_value());
    REQUIRE(added.value().empty());
    CheckHalfSearch(index, data, HALF_COUNT);
    CheckHalfRoundTrip(index, param, data, HALF_COUNT);
}

TEST_CASE("Pyramid half input cache warm build preserves neighbors",
          "[ft][pyramid][fp16][bf16][half_input][cache]") {
    const bool bf16 = GENERATE(false, true);
    CAPTURE(bf16);
    HalfInputFixture data(bf16);
    const auto param = HalfPyramidParam(bf16 ? "bfloat16" : "float16", bf16 ? "bf16" : "fp16");
    auto cold = MakeHalfIndex(param);
    auto built = cold->Build(data.Dataset(0, HALF_COUNT));
    REQUIRE(built.has_value());
    REQUIRE(built.value().empty());
    std::stringstream cache;
    REQUIRE(cold->ExportCache(cache).has_value());
    REQUIRE(cache.tellp() > 0);
    cache.seekg(0);
    auto warm = MakeHalfIndex(param);
    REQUIRE(warm->ImportCache(cache).has_value());
    auto warm_build = warm->Build(data.Dataset(0, HALF_COUNT));
    REQUIRE(warm_build.has_value());
    REQUIRE(warm_build.value().empty());
    const auto stats = JsonWrapper::Parse(warm->GetStats());
    REQUIRE(stats["build_cache_hit_nodes"].GetInt() > 0);
    REQUIRE(stats["build_cache_restored_edges"].GetInt() > 0);
    CheckHalfSearch(warm, data, HALF_COUNT);
    CheckHalfRoundTrip(warm, param, data, HALF_COUNT);
}

TEST_CASE("Pyramid half storage serializes smaller than fp32 storage",
          "[ft][pyramid][fp16][bf16][half_input][half_storage]") {
    const bool bf16 = GENERATE(false, true);
    CAPTURE(bf16);
    HalfInputFixture data(bf16);
    std::vector<uint64_t> sizes;
    for (const std::string quantization :
         {std::string(bf16 ? "bf16" : "fp16"), std::string("fp32")}) {
        const auto param = HalfPyramidParam(bf16 ? "bfloat16" : "float16", quantization);
        auto index = MakeHalfIndex(param);
        auto built = index->Build(data.Dataset(0, HALF_COUNT));
        REQUIRE(built.has_value());
        REQUIRE(built.value().empty());
        auto binary = index->Serialize();
        REQUIRE(binary.has_value());
        uint64_t size = 0;
        for (const auto& key : binary.value().GetKeys()) {
            size += binary.value().Get(key).size;
        }
        sizes.push_back(size);
        CheckHalfRoundTrip(index, param, data, HALF_COUNT);
    }
    // Total serialized bytes include graphs/metadata, so do not claim that the
    // entire index shrinks by 50%; exact code-size checks belong to datacell UT.
    REQUIRE(sizes[0] < sizes[1]);
}

TEST_CASE("Pyramid half input rejects PiPNN and missing vectors",
          "[ft][pyramid][fp16][bf16][half_input][negative]") {
    const bool bf16 = GENERATE(false, true);
    const std::string dtype = bf16 ? "bfloat16" : "float16";
    const std::string quantization = bf16 ? "bf16" : "fp16";
    CAPTURE(dtype);
    SECTION("PiPNN requires float32") {
        const auto param = HalfPyramidParam(dtype, quantization, "pipnn");
        auto made = Factory::CreateIndex("pyramid", param);
        REQUIRE_FALSE(made.has_value());
        INFO("Pyramid factory error: " << made.error().message);
        REQUIRE(made.error().type == ErrorType::INVALID_ARGUMENT);
        REQUIRE(made.error().message.find("PiPNN") != std::string::npos);
    }
    SECTION("Missing half buffer is rejected") {
        HalfInputFixture data(bf16);
        auto index = MakeHalfIndex(HalfPyramidParam(dtype, quantization));
        // A float32 buffer is not a substitute for the declared half input.
        REQUIRE(index->Build(data.Dataset(0, BUILD_COUNT)).has_value());
        auto missing = data.Dataset(0, 1, false);
        auto knn = index->KnnSearch(missing, TOPK, SEARCH_PARAM);
        REQUIRE_FALSE(knn.has_value());
        INFO("Pyramid search error: " << knn.error().message);
        REQUIRE(knn.error().type == ErrorType::INVALID_ARGUMENT);
        auto range = index->RangeSearch(missing, 0.01F, SEARCH_PARAM);
        REQUIRE_FALSE(range.has_value());
        REQUIRE(range.error().type == ErrorType::INVALID_ARGUMENT);
    }
}

TEST_CASE("Pyramid rejects unsupported int8 input", "[ft][pyramid][half_input][negative]") {
    const std::string graph = GENERATE("nsw", "odescent");
    const auto param = HalfPyramidParam("int8", "fp32", graph);
    auto made = Factory::CreateIndex("pyramid", param);
    REQUIRE_FALSE(made.has_value());
    INFO("Pyramid factory error: " << made.error().message);
    REQUIRE(made.error().type == ErrorType::INVALID_ARGUMENT);
    REQUIRE(made.error().message.find("Pyramid does not support this data type") !=
            std::string::npos);
}

TEST_CASE("Pyramid half input native distance by ID", "[ft][pyramid][half_input][half_distance]") {
    const bool bf16 = GENERATE(false, true);
    const bool reorder = GENERATE(false, true);
    const bool precise = GENERATE(false, true);
    CAPTURE(bf16, reorder, precise);
    HalfInputFixture data(bf16, 33);
    const auto param = HalfPyramidParam(
        bf16 ? "bfloat16" : "float16", bf16 ? "bf16" : "fp16", "nsw", reorder, data.dim);
    auto index = MakeHalfIndex(param);
    REQUIRE(index->Build(data.Dataset(0, BUILD_COUNT)).has_value());
    auto baseline = MakeHalfIndex(HalfPyramidParam("float32", "fp32", "nsw", reorder, data.dim));
    REQUIRE(baseline->Build(data.Dataset(0, BUILD_COUNT, false)).has_value());
    constexpr int64_t first_row = 7;
    auto query = data.Dataset(first_row, 1);
    auto float_query = data.Dataset(first_row, 1, false);
    for (const auto row : {int64_t{0}, first_row, BUILD_COUNT - 1}) {
        const auto distance = index->CalcDistanceById(query, data.ids[row], precise);
        const auto expected = baseline->CalcDistanceById(float_query, data.ids[row], precise);
        REQUIRE(distance.has_value());
        REQUIRE(expected.has_value());
        REQUIRE(distance.value() == Catch::Approx(expected.value()).epsilon(2e-5F).margin(1e-4F));
    }
    // IDs are row-major and deliberately differ for the two query rows.
    const int64_t ids[] = {data.ids[first_row],
                           data.ids[0],
                           -123,
                           data.ids[BUILD_COUNT - 1],
                           -456,
                           data.ids[first_row + 1]};
    for (const int64_t topk : {int64_t{-1}, int64_t{2}}) {
        const auto distances =
            index->CalcDistancesById(data.Dataset(first_row, 2), ids, 3, precise, topk);
        const auto expected =
            baseline->CalcDistancesById(data.Dataset(first_row, 2, false), ids, 3, precise, topk);
        REQUIRE(distances.has_value());
        REQUIRE(expected.has_value());
        REQUIRE(distances.value()->GetNumElements() == 2);
        REQUIRE(distances.value()->GetDim() == (topk == -1 ? 3 : 2));
        for (int64_t i = 0; i < 2 * distances.value()->GetDim(); ++i) {
            REQUIRE(
                distances.value()->GetDistances()[i] ==
                Catch::Approx(expected.value()->GetDistances()[i]).epsilon(2e-5F).margin(1e-4F));
            if (topk != -1) {
                REQUIRE(distances.value()->GetIds()[i] == expected.value()->GetIds()[i]);
            }
        }
    }
    const auto raw_single = index->CalcDistanceById(data.floats.data(), data.ids[0], precise);
    REQUIRE_FALSE(raw_single.has_value());
    REQUIRE(raw_single.error().type == ErrorType::INVALID_ARGUMENT);
    const auto raw_batch = index->CalcDistancesById(data.floats.data(), ids, 3, precise);
    REQUIRE_FALSE(raw_batch.has_value());
    REQUIRE(raw_batch.error().type == ErrorType::INVALID_ARGUMENT);
    const auto float_single = index->CalcDistanceById(float_query, data.ids[0], precise);
    REQUIRE_FALSE(float_single.has_value());
    REQUIRE(float_single.error().type == ErrorType::INVALID_ARGUMENT);
    const auto float_batch = index->CalcDistancesById(float_query, ids, 3, precise);
    REQUIRE_FALSE(float_batch.has_value());
    REQUIRE(float_batch.error().type == ErrorType::INVALID_ARGUMENT);
    auto malformed = data.Dataset(first_row, 1);
    malformed->Dim(data.dim - 1);
    const auto bad_dim_single = index->CalcDistanceById(malformed, data.ids[0], precise);
    REQUIRE_FALSE(bad_dim_single.has_value());
    REQUIRE(bad_dim_single.error().type == ErrorType::INVALID_ARGUMENT);
    const auto bad_dim_batch = index->CalcDistancesById(malformed, ids, 3, precise);
    REQUIRE_FALSE(bad_dim_batch.has_value());
    REQUIRE(bad_dim_batch.error().type == ErrorType::INVALID_ARGUMENT);
    const auto multiple = index->CalcDistanceById(data.Dataset(first_row, 2), data.ids[0], precise);
    REQUIRE_FALSE(multiple.has_value());
    REQUIRE(multiple.error().type == ErrorType::INVALID_ARGUMENT);
    const auto empty_ids = index->CalcDistancesById(query, nullptr, 0, precise);
    REQUIRE(empty_ids.has_value());
    REQUIRE(empty_ids.value()->GetNumElements() == 1);
    REQUIRE(empty_ids.value()->GetDim() == 0);
    // Existing float32 raw-pointer overloads remain supported.
    REQUIRE(baseline->CalcDistanceById(data.floats.data(), data.ids[0], precise).has_value());
    REQUIRE(baseline->CalcDistancesById(data.floats.data(), ids, 3, precise).has_value());
}

TEST_CASE("Pyramid half input validates Build and Add before mutation",
          "[ft][pyramid][half_input][half_ingest_validation]") {
    const bool bf16 = GENERATE(false, true);
    const std::string graph = GENERATE("nsw", "odescent");
    const bool wrong_type = GENERATE(false, true);
    CAPTURE(bf16, graph, wrong_type);
    HalfInputFixture data(bf16);
    // SQ8 training must never receive the missing half buffer; precise FP32 also
    // exercises the second storage path when valid data is subsequently inserted.
    const auto param = HalfPyramidParam(bf16 ? "bfloat16" : "float16", "sq8", graph, true);
    const auto invalid = [&](int64_t start, int64_t count) {
        auto base = data.Dataset(start, count, false);
        if (not wrong_type) {
            base->Float32Vectors(nullptr);
        }
        return base;
    };
    const auto check_rejected = [](const auto& result) {
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == ErrorType::INVALID_ARGUMENT);
        REQUIRE(result.error().message.find("representation must match index dtype") !=
                std::string::npos);
    };
    SECTION("Build rejection leaves an empty index usable") {
        auto index = MakeHalfIndex(param);
        check_rejected(index->Build(invalid(0, BUILD_COUNT)));
        REQUIRE(index->GetNumElements() == 0);
        auto built = index->Build(data.Dataset(0, BUILD_COUNT));
        REQUIRE(built.has_value());
        REQUIRE(built.value().empty());
        CheckHalfSearch(index, data, BUILD_COUNT);
    }
    SECTION("First Add rejection does not reserve IDs or train") {
        auto index = MakeHalfIndex(param);
        check_rejected(index->Add(invalid(0, BUILD_COUNT)));
        REQUIRE(index->GetNumElements() == 0);
        auto added = index->Add(data.Dataset(0, BUILD_COUNT));
        REQUIRE(added.has_value());
        REQUIRE(added.value().empty());
        CheckHalfSearch(index, data, BUILD_COUNT);
    }
    SECTION("Populated Add rejection preserves prior points and reusable IDs") {
        auto index = MakeHalfIndex(param);
        REQUIRE(index->Build(data.Dataset(0, BUILD_COUNT)).has_value());
        check_rejected(index->Add(invalid(BUILD_COUNT, HALF_COUNT - BUILD_COUNT)));
        REQUIRE(index->GetNumElements() == BUILD_COUNT);
        CheckHalfSearch(index, data, BUILD_COUNT);
        auto added = index->Add(data.Dataset(BUILD_COUNT, HALF_COUNT - BUILD_COUNT));
        REQUIRE(added.has_value());
        REQUIRE(added.value().empty());
        CheckHalfSearch(index, data, HALF_COUNT);
    }
    SECTION("Cached Build rejects missing buffers before applying the cache") {
        if (graph == "nsw") {
            auto cold = MakeHalfIndex(param);
            REQUIRE(cold->Build(data.Dataset(0, BUILD_COUNT)).has_value());
            std::stringstream cache;
            REQUIRE(cold->ExportCache(cache).has_value());
            cache.seekg(0);
            auto warm = MakeHalfIndex(param);
            REQUIRE(warm->ImportCache(cache).has_value());
            check_rejected(warm->Build(invalid(0, BUILD_COUNT)));
            REQUIRE(warm->GetNumElements() == 0);
            auto built = warm->Build(data.Dataset(0, BUILD_COUNT));
            REQUIRE(built.has_value());
            REQUIRE(built.value().empty());
            CheckHalfSearch(warm, data, BUILD_COUNT);
        }
    }
    SECTION("Zero-element inputs remain successful no-ops without buffers") {
        auto index = MakeHalfIndex(param);
        auto empty = Dataset::Make()->NumElements(0)->Owner(false);
        REQUIRE(index->Build(empty).has_value());
        REQUIRE(index->Add(empty).has_value());
        REQUIRE(index->GetNumElements() == 0);
        REQUIRE(index->Build(data.Dataset(0, BUILD_COUNT)).has_value());
        REQUIRE(index->Add(empty).has_value());
        REQUIRE(index->GetNumElements() == BUILD_COUNT);
    }
}

}  // namespace vsag
