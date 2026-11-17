
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
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "datacell/rabitq_split_datacell.h"
#include "io/memory_io/memory_io_parameter.h"
#include "quantization/quantizer_adapter.h"
#include "quantization/rabitq_quantization/rabitq_quantizer.h"
#include "simd/bf16_simd.h"
#include "simd/fp16_simd.h"
#include "unittest.h"
#include "utils/half_input.h"
using namespace vsag;
namespace {
struct CountingAllocator : Allocator {
    uint64_t max_request = 0, calls = 0, live = 0, peak = 0, fail_at = 0;
    std::unordered_map<void*, uint64_t> sizes;
    std::string
    Name() override {
        return "bounded-training-test";
    }
    void*
    Allocate(uint64_t n) override {
        ++calls;
        if (fail_at && calls == fail_at)
            throw std::bad_alloc();
        auto* p = std::malloc(n);
        if (!p)
            throw std::bad_alloc();
        sizes[p] = n;
        live += n;
        peak = std::max(peak, live);
        max_request = std::max(max_request, n);
        return p;
    }
    void
    Deallocate(void* p) override {
        if (p) {
            live -= sizes.at(p);
            sizes.erase(p);
            std::free(p);
        }
    }
    void*
    Reallocate(void* p, uint64_t n) override {
        auto* q = Allocate(n);
        if (p) {
            std::memcpy(q, p, std::min(n, sizes.at(p)));
            Deallocate(p);
        }
        return q;
    }
    void
    Reset() {
        max_request = 0;
        calls = 0;
        peak = live;
        fail_at = 0;
    }
};
void
check(bool ok, const char* msg) {
    INFO(msg);
    REQUIRE(ok);
}
using Q = RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR>;
Q
make(Allocator* a, uint64_t dim, uint64_t pca = 0) {
    return Q(dim,
             pca,
             32,
             8,
             true,
             false,
             a,
             RaBitQuantizerParameter::RABITQ_VERSION_SPLIT,
             RaBitQuantizerParameter::DEFAULT_RABITQ_ERROR_RATE,
             3);
}
std::vector<uint8_t>
codes(Q& q, const float* p, uint64_t count, uint64_t dim) {
    std::vector<uint8_t> c(count * q.GetCodeSize());
    for (uint64_t i = 0; i < count; ++i) q.EncodeOne(p + i * dim, c.data() + i * q.GetCodeSize());
    return c;
}
}  // namespace
TEST_CASE("RaBitQ bounded half training", "[ut][RaBitQuantizer][half_train]") {
    constexpr uint64_t D = 32;
    for (auto type : {DataTypes::DATA_TYPE_FP16, DataTypes::DATA_TYPE_BF16}) {
        for (uint64_t n : {4095, 4096, 4097, 8193, 20000}) {
            std::vector<uint16_t> h(n * D);
            std::vector<float> f(n * D);
            for (uint64_t i = 0; i < f.size(); ++i) {
                float v = std::sin(float(i) * .133F) * 1.7F;
                h[i] = type == DataTypes::DATA_TYPE_FP16 ? generic::FloatToFP16(v)
                                                         : generic::FloatToBF16(v);
                f[i] = type == DataTypes::DATA_TYPE_FP16 ? half_input::FP16ToFloat(h[i])
                                                         : generic::BF16ToFloat(h[i]);
            }
            CountingAllocator a, b;
            auto typed = make(&a, D), reference = make(&b, D);
            a.Reset();
            check(typed.TrainHalf(h.data(), n, type), "typed train");
            check(a.max_request <= 4096 * D * sizeof(float), "unbounded allocation");
            check(a.peak - a.live <= 4096 * D * sizeof(float) + D * sizeof(float),
                  "unbounded live scratch");
            auto max = a.max_request;
            check(reference.Train(f.data(), n), "reference train");
            check(codes(typed, f.data(), std::min(n, 64UL), D) ==
                      codes(reference, f.data(), std::min(n, 64UL), D),
                  "encoded mismatch");
            a.Reset();
            check(typed.TrainHalf(
                      nullptr, std::numeric_limits<uint64_t>::max(), DataTypes::DATA_TYPE_FLOAT),
                  "trained early return");
            check(a.calls == 0, "trained allocates");
            for (uint64_t i = 0; i < f.size(); ++i) {
                h[i] = type == DataTypes::DATA_TYPE_FP16 ? generic::FloatToFP16(f[i] + 2.F)
                                                         : generic::FloatToBF16(f[i] + 2.F);
                f[i] = type == DataTypes::DATA_TYPE_FP16 ? half_input::FP16ToFloat(h[i])
                                                         : generic::BF16ToFloat(h[i]);
            }
            check(typed.ReTrainHalf(h.data(), n, type), "retrain");
            check(reference.ReTrain(f.data(), n), "reference retrain");
            check(codes(typed, f.data(), 64, D) == codes(reference, f.data(), 64, D),
                  "retrain encoded mismatch");
            std::cout << "PASS type=" << int(type) << " count=" << n << " max_request=" << max
                      << "\n";
        }
    }
    CountingAllocator a;
    auto q = make(&a, D);
    uint16_t dummy = 0;
    check(!q.TrainHalf(nullptr, 1, DataTypes::DATA_TYPE_FP16), "null");
    check(!q.TrainHalf(&dummy, 0, DataTypes::DATA_TYPE_FP16), "zero");
    for (auto bad : {DataTypes::DATA_TYPE_FLOAT, DataTypes::DATA_TYPE_INT8}) {
        bool threw = false;
        try {
            q.TrainHalf(&dummy, 1, bad);
        } catch (const std::exception&) {
            threw = true;
        }
        check(threw, "type reject");
    }
    bool threw = false;
    try {
        q.TrainHalf(&dummy, std::numeric_limits<uint64_t>::max(), DataTypes::DATA_TYPE_FP16);
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "overflow");
    std::vector<uint16_t> h(4097 * D, 0);
    a.Reset();
    a.fail_at = 1;
    threw = false;
    try {
        q.TrainHalf(h.data(), 4097, DataTypes::DATA_TYPE_FP16);
    } catch (const std::bad_alloc&) {
        threw = true;
    }
    check(threw, "scratch exception");
    a.Reset();
    check(q.TrainHalf(h.data(), 4097, DataTypes::DATA_TYPE_FP16), "retry");
    a.Reset();
    a.fail_at = 1;
    threw = false;
    try {
        q.ReTrainHalf(h.data(), 4097, DataTypes::DATA_TYPE_FP16);
    } catch (const std::bad_alloc&) {
        threw = true;
    }
    check(threw, "retrain scratch exception");
    a.Reset();
    check(q.TrainHalf(h.data(), 4097, DataTypes::DATA_TYPE_FP16) && a.calls > 0,
          "retrain exception resets state");
    check(!q.ReTrainHalf(nullptr, 1, DataTypes::DATA_TYPE_FP16), "retrain failure state");
    a.Reset();
    check(q.TrainHalf(h.data(), 4097, DataTypes::DATA_TYPE_FP16) && a.calls > 0,
          "failed retrain resets state");
    // PCA preserves whole-input conversion and the original train implementation.
    CountingAllocator pca_allocator;
    auto pca = make(&pca_allocator, D, 16);
    std::vector<uint16_t> ph(200 * D);
    std::vector<float> pf(ph.size());
    for (uint64_t i = 0; i < ph.size(); ++i) {
        ph[i] = generic::FloatToFP16(std::sin(float(i)));
        pf[i] = half_input::FP16ToFloat(ph[i]);
    }
    pca_allocator.Reset();
    check(pca.TrainHalf(ph.data(), 200, DataTypes::DATA_TYPE_FP16), "PCA fallback");
    check(pca_allocator.max_request >= 200 * D * sizeof(float), "PCA not fallback");
    CountingAllocator pca_ref_allocator;
    auto pca_ref = make(&pca_ref_allocator, D, 16);
    check(pca_ref.Train(pf.data(), 200), "PCA reference");
    check(codes(pca, pf.data(), 64, D) == codes(pca_ref, pf.data(), 64, D), "PCA reference codes");
    // ROM: compare against a reference centroid under the exact same trained rotation.
    CountingAllocator ra, rb;
    Q rom(D,
          0,
          32,
          8,
          false,
          false,
          &ra,
          RaBitQuantizerParameter::RABITQ_VERSION_SPLIT,
          RaBitQuantizerParameter::DEFAULT_RABITQ_ERROR_RATE,
          3);
    Q ref(D,
          0,
          32,
          8,
          false,
          false,
          &rb,
          RaBitQuantizerParameter::RABITQ_VERSION_SPLIT,
          RaBitQuantizerParameter::DEFAULT_RABITQ_ERROR_RATE,
          3);
    check(rom.TrainHalf(ph.data(), 200, DataTypes::DATA_TYPE_FP16), "ROM train");
    ref.ShareFusedModelFrom(rom);
    std::vector<float> centroid(D, 0);
    for (uint64_t i = 0; i < 200; ++i)
        for (uint64_t d = 0; d < D; ++d) centroid[d] += pf[i * D + d];
    for (auto& v : centroid) v /= float(200);
    ref.SetCentroid(centroid.data());
    check(codes(rom, pf.data(), 64, D) == codes(ref, pf.data(), 64, D),
          "ROM shared rotation centroid equality");
    for (auto type : {DataTypes::DATA_TYPE_FP16, DataTypes::DATA_TYPE_BF16}) {
        std::vector<uint16_t> special(D);
        std::vector<float> decoded(D);
        for (uint64_t i = 0; i < D; ++i) {
            special[i] = std::vector<uint16_t>{0, 0x8000, 1, 0x8001, 0x3c00}[i % 5];
            decoded[i] = type == DataTypes::DATA_TYPE_FP16 ? half_input::FP16ToFloat(special[i])
                                                           : generic::BF16ToFloat(special[i]);
        }
        CountingAllocator sa, sb;
        auto sq = make(&sa, D), sr = make(&sb, D);
        check(sq.TrainHalf(special.data(), 1, type) && sr.Train(decoded.data(), 1),
              "count1 bit patterns");
        check(codes(sq, decoded.data(), 1, D) == codes(sr, decoded.data(), 1, D), "special codes");
        auto recorded = std::make_shared<CountingAllocator>();
        IndexCommonParam common;
        common.allocator_ = recorded;
        common.dim_ = D;
        common.metric_ = MetricType::METRIC_TYPE_L2SQR;
        common.data_type_ = type;
        auto param = QuantizerParameter::GetQuantizerParameterByJson(JsonType::Parse(
            R"({"type":"rabitq","rabitq_version":"split","rabitq_bits_per_dim_query":32,"rabitq_bits_per_dim_base":8,"rabitq_bits_per_dim_filter":3,"use_fht":true})"));
        std::vector<uint16_t> input(8193 * D, 0x3c00);
        RaBitQSplitDataCell<MetricType::METRIC_TYPE_L2SQR, MemoryIO> split(
            param, std::make_shared<MemoryIOParameter>(), common);
        recorded->Reset();
        split.Train(input.data(), 8193);
        check(recorded->max_request <= 4096 * D * sizeof(float), "Split dispatch bounded");
        recorded->Reset();
        split.Train(nullptr, std::numeric_limits<uint64_t>::max());
        check(recorded->calls == 0, "Split trained no allocation");
        QuantizerAdapter<Q, uint16_t> adapter(param, common);
        recorded->Reset();
        check(adapter.Train(reinterpret_cast<const float*>(input.data()), 8193), "adapter Train");
        check(recorded->max_request <= 4096 * D * sizeof(float), "adapter dispatch bounded");
        recorded->Reset();
        check(adapter.Train(nullptr, 8193) && recorded->calls == 0,
              "adapter trained no allocation");
        recorded->Reset();
        recorded->fail_at = 1;
        bool failed = false;
        try {
            adapter.ReTrain(reinterpret_cast<const float*>(input.data()), 8193);
        } catch (const std::bad_alloc&) {
            failed = true;
        }
        check(failed, "adapter retrain throws");
        recorded->Reset();
        check(adapter.Train(reinterpret_cast<const float*>(input.data()), 8193) &&
                  recorded->calls > 0,
              "adapter retry after exception");
    }
    {
        CountingAllocator big;
        auto bq = make(&big, 1024);
        std::vector<uint16_t> input(4097 * 1024, 0x3c00);
        big.Reset();
        check(bq.TrainHalf(input.data(), 4097, DataTypes::DATA_TYPE_FP16), "D1024");
        check(big.max_request == 4096 * 1024 * sizeof(float), "D1024 16MiB scratch");
        check(big.peak - big.live <= 4096 * 1024 * sizeof(float) + 1024 * sizeof(float),
              "D1024 live bound");
        std::cout << "D1024_MAX_REQUEST " << big.max_request << "\n";
    }
    {
        auto alloc = std::make_shared<CountingAllocator>();
        IndexCommonParam common;
        common.allocator_ = alloc;
        common.dim_ = D;
        common.metric_ = MetricType::METRIC_TYPE_L2SQR;
        common.data_type_ = DataTypes::DATA_TYPE_FP16;
        auto param = QuantizerParameter::GetQuantizerParameterByJson(JsonType::Parse(
            R"({"type":"tq","tq_chain":"mrle, rabitq","mrle_dim":32,"rabitq_version":"split","rabitq_bits_per_dim_query":32,"rabitq_bits_per_dim_base":8,"rabitq_bits_per_dim_filter":3,"use_fht":true})"));
        using TQ = TransformQuantizer<Q, MetricType::METRIC_TYPE_L2SQR>;
        RaBitQSplitDataCell<MetricType::METRIC_TYPE_L2SQR, MemoryIO, MemoryIO, TQ> split(
            param, std::make_shared<MemoryIOParameter>(), common);
        std::vector<uint16_t> input(8193 * D, 0x3c00);
        alloc->Reset();
        split.Train(input.data(), 8193);
        check(alloc->max_request >= 8193 * D * sizeof(float), "TQ preserves dense fallback");
    }
    std::cout << "ALL_TYPED_TESTS_PASS\n";
}
