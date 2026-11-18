#include <fmt/format.h>
#include <vsag/vsag.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <sstream>

#include "unittest.h"
TEST_CASE("IVF public Factory native half lifecycle",
          "[IVFPublicHalfFactory][IVFHalfFullDistanceDiagnostic]") {
    const bool bf = GENERATE(false, true);
    const auto trainer =
        GENERATE(std::string("random"), std::string("kmeans"), std::string("e_means"));
    const auto quant = GENERATE(std::string("fp32"),
                                std::string("sq8"),
                                std::string("fp16"),
                                std::string("bf16"),
                                std::string("sq4"),
                                std::string("sq8_uniform"),
                                std::string("sq4_uniform"));
    CAPTURE(bf, trainer, quant);
    constexpr int n = 64, dim = 7;
    std::vector<uint16_t> half(n * dim);
    std::vector<float> rounded(n * dim);
    std::vector<int64_t> ids(n);
    std::iota(ids.begin(), ids.end(), 0);
    for (uint64_t i = 0; i < half.size(); ++i) {
        half[i] = uint16_t((bf ? 0x3e00 : 0x3400) + (i * 17) % 127);
        if (bf) {
            uint32_t bits = uint32_t(half[i]) << 16;
            std::memcpy(&rounded[i], &bits, 4);
        } else {
            const auto bits = half[i];
            rounded[i] = std::ldexp(1.0F + float(bits & 1023) / 1024, int((bits >> 10) & 31) - 15);
        }
    }
    auto config = [&](const std::string& dtype) {
        return fmt::format(
            R"({{"dtype":"{}","metric_type":"l2","dim":7,"index_param":{{"base_quantization_type":"{}","buckets_count":4,"ivf_train_type":"{}","use_route_graph":false,"use_residual":false,"use_reorder":false}}}})",
            dtype,
            quant,
            trainer);
    };
    const auto typed_config = config(bf ? "bfloat16" : "float16");
    auto a = vsag::Factory::CreateIndex("ivf", typed_config);
    auto b = vsag::Factory::CreateIndex("ivf", config("float32"));
    REQUIRE(a.has_value());
    REQUIRE(b.has_value());
    auto dataset = [&](bool typed, int start, int count) {
        auto d = vsag::Dataset::Make()
                     ->Dim(dim)
                     ->NumElements(count)
                     ->Ids(ids.data() + start)
                     ->Owner(false);
        if (typed)
            d->Float16Vectors(half.data() + start * dim);
        else
            d->Float32Vectors(rounded.data() + start * dim);
        return d;
    };
    REQUIRE(a.value()->Build(dataset(true, 0, 48)).has_value());
    REQUIRE(b.value()->Build(dataset(false, 0, 48)).has_value());
    REQUIRE(a.value()->Add(dataset(true, 48, 16)).has_value());
    REQUIRE(b.value()->Add(dataset(false, 48, 16)).has_value());
    const std::string search = R"({"ivf":{"scan_buckets_count":4}})";
    auto x = a.value()->KnnSearch(dataset(true, 3, 1), 64, search);
    auto y = b.value()->KnnSearch(dataset(false, 3, 1), 64, search);
    REQUIRE(x.has_value());
    REQUIRE(y.has_value());
    REQUIRE(x.value()->GetDim() == 64);
    REQUIRE(y.value()->GetDim() == 64);
    std::vector<uint32_t> typed_bits(64), float_bits(64);
    std::vector<bool> seen_typed(64), seen_float(64);
    for (int i = 0; i < 64; ++i) {
        auto xi = x.value()->GetIds()[i];
        auto yi = y.value()->GetIds()[i];
        REQUIRE(xi >= 0);
        REQUIRE(xi < 64);
        REQUIRE(yi >= 0);
        REQUIRE(yi < 64);
        REQUIRE_FALSE(seen_typed[xi]);
        REQUIRE_FALSE(seen_float[yi]);
        seen_typed[xi] = true;
        seen_float[yi] = true;
        std::memcpy(&typed_bits[xi], x.value()->GetDistances() + i, 4);
        std::memcpy(&float_bits[yi], y.value()->GetDistances() + i, 4);
    }
    for (int i = 0; i < 64; ++i) {
        CAPTURE(i, typed_bits[i], float_bits[i]);
        REQUIRE(typed_bits[i] == float_bits[i]);
    }
    auto tx = a.value()->KnnSearch(dataset(true, 3, 1), 10, search);
    auto ty = b.value()->KnnSearch(dataset(false, 3, 1), 10, search);
    REQUIRE(tx.has_value());
    REQUIRE(ty.has_value());
    REQUIRE(tx.value()->GetDim() == 10);
    REQUIRE(ty.value()->GetDim() == 10);
    std::vector<float> all_distances(64);
    for (int id = 0; id < 64; ++id) std::memcpy(&all_distances[id], &typed_bits[id], 4);
    std::sort(all_distances.begin(), all_distances.end());
    for (auto result : {tx.value(), ty.value()}) {
        std::vector<bool> seen(64);
        for (int rank = 0; rank < 10; ++rank) {
            const auto id = result->GetIds()[rank];
            REQUIRE(id >= 0);
            REQUIRE(id < 64);
            REQUIRE_FALSE(seen[id]);
            seen[id] = true;
            uint32_t actual, expected;
            std::memcpy(&actual, result->GetDistances() + rank, 4);
            std::memcpy(&expected, &all_distances[rank], 4);
            CAPTURE(rank, id, actual, expected);
            REQUIRE(actual == typed_bits[id]);
            REQUIRE(actual == expected);
        }
        const float boundary = result->GetDistances()[9];
        for (int id = 0; id < 64; ++id) {
            float distance;
            std::memcpy(&distance, &typed_bits[id], 4);
            REQUIRE((seen[id] || distance >= boundary));
        }
    }
    // The distance-only heap does not promise an ID order within an exact tie.
    for (int rank = 0; rank < 10; ++rank) {
        REQUIRE(typed_bits[tx.value()->GetIds()[rank]] == typed_bits[ty.value()->GetIds()[rank]]);
    }
    std::stringstream stream;
    REQUIRE(a.value()->Serialize(stream).has_value());
    auto restored = vsag::Factory::CreateIndex("ivf", typed_config);
    REQUIRE(restored.has_value());
    REQUIRE(restored.value()->Deserialize(stream).has_value());
    auto z = restored.value()->KnnSearch(dataset(true, 3, 1), 64, search);
    REQUIRE(z.has_value());
    auto restored_top = restored.value()->KnnSearch(dataset(true, 3, 1), 10, search);
    REQUIRE(restored_top.has_value());
    REQUIRE(restored_top.value()->GetDim() == 10);
    for (int rank = 0; rank < 10; ++rank) {
        REQUIRE(restored_top.value()->GetIds()[rank] == tx.value()->GetIds()[rank]);
        uint32_t bits;
        std::memcpy(&bits, restored_top.value()->GetDistances() + rank, 4);
        REQUIRE(bits == typed_bits[restored_top.value()->GetIds()[rank]]);
    }
    REQUIRE(z.value()->GetDim() == 64);
    for (int i = 0; i < 64; ++i) {
        REQUIRE(z.value()->GetIds()[i] == x.value()->GetIds()[i]);
        uint32_t bits;
        std::memcpy(&bits, z.value()->GetDistances() + i, 4);
        REQUIRE(bits == typed_bits[z.value()->GetIds()[i]]);
    }
}
