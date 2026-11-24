// Copyright 2026-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include <cstring>
#include <numeric>
#include <sstream>

#include "algorithm/bruteforce/half_dense_test_fixture.h"
#include "vsag/vsag.h"

namespace {
constexpr uint64_t STATS_COUNT = 10000;

bool
StatsFinite(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7f800000U) != 0x7f800000U;
}
}  // namespace

TEST_CASE("Pyramid native typed Stats executes large GRAPH recall after reload",
          "[PyramidNativeStats][pyramid]") {
    const uint64_t dim = GENERATE(7, 33);
    const std::string dtype =
        GENERATE(std::string("float32"), std::string("float16"), std::string("bfloat16"));
    CAPTURE(dim, dtype);
    auto config = vsag::JsonType::Parse(R"({
        "metric_type":"l2","index_param":{
            "base_quantization_type":"sq8","max_degree":32,"ef_construction":200,
            "alpha":1.2,"graph_iter_turn":15,"neighbor_sample_rate":0.2,
            "no_build_levels":[],"use_reorder":false,"graph_type":"odescent",
            "build_thread_count":4}})");
    config["dtype"].SetString(dtype);
    config["dim"].SetInt(dim);
    const auto create = [&]() {
        auto result = vsag::Factory::CreateIndex("pyramid", config.Dump());
        REQUIRE(result.has_value());
        return result.value();
    };
    std::vector<uint16_t> half(STATS_COUNT * dim);
    std::vector<float> floats(STATS_COUNT * dim);
    std::vector<int64_t> ids(STATS_COUNT);
    std::vector<std::string> paths(STATS_COUNT, "root/gist");
    std::iota(ids.begin(), ids.end(), 0);
    // Independent bit decoding gives the FLOAT control exactly the half-rounded values.
    const bool bf16 = dtype == "bfloat16";
    for (uint64_t i = 0; i < STATS_COUNT; ++i) {
        for (uint64_t j = 0; j < dim; ++j) {
            const uint16_t sign = ((i + j) % 3 == 0) ? 0x8000U : 0U;
            const uint16_t bits =
                sign | (bf16 ? 0x3e00U : 0x3000U) | ((i * 37 + j * 19) % (bf16 ? 512 : 2048));
            half[i * dim + j] = bits;
            floats[i * dim + j] = half_dense_test::Decode(bits, bf16);
        }
    }
    const auto dataset = [&](uint64_t offset, uint64_t count) {
        auto data = vsag::Dataset::Make()
                        ->Dim(dim)
                        ->NumElements(count)
                        ->Owner(false)
                        ->Ids(ids.data() + offset)
                        ->Paths(paths.data() + offset);
        if (dtype == "float32") {
            data->Float32Vectors(floats.data() + offset * dim);
        } else {
            data->Float16Vectors(half.data() + offset * dim);
        }
        return data;
    };
    const auto search = [&](const vsag::IndexPtr& index) {
        auto result = index->KnnSearch(dataset(0, 1), 10, R"({"pyramid":{"ef_search":500}})");
        REQUIRE(result.has_value());
        REQUIRE(result.value()->GetDim() == 10);
        REQUIRE(result.value()->GetIds() != nullptr);
        REQUIRE(result.value()->GetDistances() != nullptr);
        std::vector<std::pair<int64_t, float>> values;
        for (uint64_t i = 0; i < 10; ++i) {
            REQUIRE(StatsFinite(result.value()->GetDistances()[i]));
            values.emplace_back(result.value()->GetIds()[i], result.value()->GetDistances()[i]);
        }
        return values;
    };
    const auto check_stats = [&](const vsag::IndexPtr& index) {
        auto stats = vsag::JsonType::Parse(index->GetStats());
        // This must execute the >=10000 GRAPH leaf branch, not merely avoid a crash.
        REQUIRE(stats["graph_node_count"].GetInt() >= 1);
        REQUIRE(stats["total_graph_size"].GetInt() >= STATS_COUNT);
        const auto recall = stats["recall_base"].GetFloat();
        REQUIRE(StatsFinite(recall));
        // An exception swallowed into an empty search heap must not pass this fixture.
        REQUIRE(recall > 0.0F);
        REQUIRE(recall <= 1.0F);
    };
    auto index = create();
    auto built = index->Build(dataset(0, STATS_COUNT));
    REQUIRE(built.has_value());
    REQUIRE(built.value().empty());
    const auto before = search(index);
    check_stats(index);
    REQUIRE(search(index) == before);
    std::stringstream stream;
    REQUIRE(index->Serialize(stream).has_value());
    auto restored = create();
    REQUIRE(restored->Deserialize(stream).has_value());
    REQUIRE(search(restored) == before);
    check_stats(restored);
    REQUIRE(search(restored) == before);
}
