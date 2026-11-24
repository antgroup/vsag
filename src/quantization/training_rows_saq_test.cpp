#include <sstream>

#include "impl/allocator/safe_allocator.h"
#include "quantization/quantizer_adapter.h"
#include "quantization/saq_quantization/saq_quantizer.h"
#include "simd/bf16_simd.h"
#include "simd/fp16_simd.h"
#include "storage/stream_writer.h"
#include "unittest.h"
using namespace vsag;
namespace {
template <class Q>
std::string
FullSAQ(Q& q) {
    std::ostringstream o;
    IOStreamWriter w(o);
    q.Serialize(w);
    return o.str();
}
template <class Q>
std::string
InnerSAQ(Q& q) {
    std::ostringstream o;
    IOStreamWriter w(o);
    q.SerializeImpl(w);
    return o.str();
}
float
ExpandSAQ(uint16_t v, bool bf) {
    if (bf) {
        uint32_t b = uint32_t(v) << 16;
        float f;
        std::memcpy(&f, &b, 4);
        return f;
    }
    int e = (v >> 10) & 31;
    float f =
        e ? std::ldexp(1.f + float(v & 1023) / 1024, e - 15) : std::ldexp(float(v & 1023), -24);
    return v & 0x8000 ? -f : f;
}
}  // namespace
TEST_CASE("SAQ TrainingRows complete model codes query retry", "[TrainingRowsSAQ]") {
    for (bool bf : {false, true})
        for (int d : {7, 33}) {
            auto a = SafeAllocator::FactoryDefaultAllocator();
            uint64_t n = 41;
            std::vector<uint16_t> h(n * d);
            std::vector<float> x(n * d), y(n * d);
            for (size_t i = 0; i < x.size(); ++i) {
                float v = std::sin(float(i) * .137f) + std::cos(float(i) * .271f) * .3f;
                h[i] = bf ? generic::FloatToBF16(v) : generic::FloatToFP16(v);
                x[i] = ExpandSAQ(h[i], bf);
                y[i] = std::cos(float(i) * .183f);
            }
            TrainingRows raw(h.data(),
                             n,
                             d,
                             bf ? DataTypes::DATA_TYPE_BF16 : DataTypes::DATA_TYPE_FP16,
                             d * 2,
                             h.size() * 2);
            auto run = [&]<MetricType M>() {
                SAQQuantizer<M> p(d, 4, 1, false, a.get()), f(d, 4, 1, false, a.get()),
                    fresh(d, 4, 1, false, a.get());
                REQUIRE(p.TrainRows(raw));
                REQUIRE(f.Train(x.data(), n));
                REQUIRE(FullSAQ(p) == FullSAQ(f));
                std::vector<uint8_t> pc(n * p.GetCodeSize()), fc(pc.size());
                p.EncodeBatch(x.data(), pc.data(), n);
                f.EncodeBatch(x.data(), fc.data(), n);
                REQUIRE(pc == fc);
                auto pq = p.FactoryComputer();
                auto fq = f.FactoryComputer();
                pq->SetQuery(x.data());
                fq->SetQuery(x.data());
                for (uint64_t i = 0; i < n; ++i)
                    REQUIRE(p.ComputeDist(*pq, pc.data() + i * p.GetCodeSize()) ==
                            f.ComputeDist(*fq, fc.data() + i * f.GetCodeSize()));
                auto before = FullSAQ(p);
                TrainingRows invalid(
                    1,
                    d,
                    [](const void*, uint64_t, float*) {
                        throw std::runtime_error("must not read");
                    },
                    nullptr);
                REQUIRE_THROWS(p.TrainRows(invalid));
                REQUIRE_THROWS(p.ReTrainRows(invalid));
                REQUIRE(FullSAQ(p) == before);
                TrainingRows other(y.data(), n, d, DataTypes::DATA_TYPE_FLOAT, d * 4, y.size() * 4);
                REQUIRE(p.ReTrainRows(other));
                REQUIRE(fresh.Train(y.data(), n));
                REQUIRE(FullSAQ(p) == FullSAQ(fresh));
                TrainingRows bad(
                    n,
                    d,
                    [](const void*, uint64_t, float*) {
                        throw std::runtime_error("SAQ provider fail");
                    },
                    nullptr);
                REQUIRE_THROWS(p.ReTrainRows(bad));
                REQUIRE(p.ReTrainRows(other));
                REQUIRE(FullSAQ(p) == FullSAQ(fresh));
            };
            run.template operator()<MetricType::METRIC_TYPE_L2SQR>();
            run.template operator()<MetricType::METRIC_TYPE_IP>();
            run.template operator()<MetricType::METRIC_TYPE_COSINE>();
        }
}
TEST_CASE("SAQ deterministic capped provider selection preserves exact order",
          "[TrainingRowsSAQ]") {
    for (uint64_t n : {65535UL, 65536UL, 65537UL, 100001UL})
        for (bool cosine : {false, true}) {
            auto a = SafeAllocator::FactoryDefaultAllocator();
            std::vector<float> x(n * 7);
            for (size_t i = 0; i < x.size(); ++i)
                x[i] = std::sin(float(i) * .137f) + std::cos(float(i) * .271f) * .3f;
            struct C {
                const float* x;
                std::vector<uint64_t>* ids;
            };
            std::vector<uint64_t> ids;
            C c{x.data(), &ids};
            TrainingRows rows(
                n,
                7,
                [](const void* p, uint64_t i, float* out) {
                    auto& c = *static_cast<const C*>(p);
                    c.ids->push_back(i);
                    std::memcpy(out, c.x + i * 7, 7 * 4);
                },
                &c);
            auto run = [&]<MetricType M>() {
                SAQQuantizer<M> p(7, 4, 1, false, a.get()), f(7, 4, 1, false, a.get());
                REQUIRE(p.TrainRows(rows));
                REQUIRE(f.Train(x.data(), n));
                REQUIRE(FullSAQ(p) == FullSAQ(f));
            };
            if (cosine)
                run.template operator()<MetricType::METRIC_TYPE_COSINE>();
            else
                run.template operator()<MetricType::METRIC_TYPE_L2SQR>();
            const uint64_t s = std::min<uint64_t>(n, 65536);
            REQUIRE(ids.size() == 3 * s);
            for (uint64_t pass = 0; pass < 3; ++pass)
                for (uint64_t i = 0; i < s; ++i)
                    REQUIRE(ids[pass * s + i] == i * (n / s) + i * (n % s) / s);
        }
}
TEST_CASE("SAQ native half adapter matches rounded FP32 model codes query", "[TrainingRowsSAQ]") {
    for (bool bf : {false, true})
        for (int d : {7, 33}) {
            auto a = SafeAllocator::FactoryDefaultAllocator();
            IndexCommonParam c;
            c.allocator_ = a;
            c.dim_ = d;
            c.metric_ = MetricType::METRIC_TYPE_COSINE;
            c.data_type_ = bf ? DataTypes::DATA_TYPE_BF16 : DataTypes::DATA_TYPE_FP16;
            auto param = std::make_shared<SAQQuantizerParameter>();
            param->avg_bits_ = 4;
            param->segment_count_ = 1;
            param->random_rotation_ = false;
            QuantizerAdapter<SAQQuantizer<MetricType::METRIC_TYPE_COSINE>, uint16_t> p(param, c);
            SAQQuantizer<MetricType::METRIC_TYPE_COSINE> f(d, 4, 1, false, a.get());
            uint64_t n = 41;
            std::vector<uint16_t> h(n * d);
            std::vector<float> x(n * d);
            for (size_t i = 0; i < x.size(); ++i) {
                float v = std::sin(float(i) * .137f);
                h[i] = bf ? generic::FloatToBF16(v) : generic::FloatToFP16(v);
                x[i] = ExpandSAQ(h[i], bf);
            }
            REQUIRE(p.Train(reinterpret_cast<const float*>(h.data()), n));
            REQUIRE(f.Train(x.data(), n));
            REQUIRE(InnerSAQ(p) == InnerSAQ(f));
            std::vector<uint8_t> pc(n * p.GetCodeSize()), fc(pc.size());
            p.EncodeBatch(reinterpret_cast<const float*>(h.data()), pc.data(), n);
            f.EncodeBatch(x.data(), fc.data(), n);
            REQUIRE(pc == fc);
            auto pq = p.FactoryComputer();
            auto fq = f.FactoryComputer();
            pq->SetQuery(reinterpret_cast<const float*>(h.data()));
            fq->SetQuery(x.data());
            for (uint64_t i = 0; i < n; ++i)
                REQUIRE(p.ComputeDist(*pq, pc.data() + i * p.GetCodeSize()) ==
                        f.ComputeDist(*fq, fc.data() + i * f.GetCodeSize()));
        }
}

TEST_CASE("SAQ cosine provider in-place normalization matches original out-of-place extremes",
          "[TrainingRowsSAQ]") {
    for (bool bf : {false, true})
        for (int d : {7, 33}) {
            auto a = SafeAllocator::FactoryDefaultAllocator();
            const uint64_t n = 41;
            std::vector<uint16_t> h(n * d);
            std::vector<float> x(n * d);
            for (size_t i = 0; i < h.size(); ++i) {
                float v = std::sin(float(i) * .137f);
                h[i] = bf ? generic::FloatToBF16(v) : generic::FloatToFP16(v);
            }
            const uint16_t minsub = 1, maxfinite = bf ? 0x7f7f : 0x7bff, inf = bf ? 0x7f80 : 0x7c00,
                           nan = bf ? 0x7fc0 : 0x7e00;
            std::fill(h.begin(), h.begin() + d, uint16_t(0));
            h[d] = 0x8000;
            h[2 * d] = minsub;
            h[3 * d] = maxfinite;
            h[3 * d + 1] = uint16_t(maxfinite | 0x8000);
            h[4 * d] = inf;
            h[5 * d] = nan;
            for (size_t i = 0; i < h.size(); ++i) {
                uint16_t v = h[i];
                if (!bf && ((v >> 10) & 31) == 31) {
                    uint32_t bits =
                        (v & 0x8000 ? 0x80000000U : 0) | 0x7f800000U | (v & 1023 ? 0x400000U : 0);
                    std::memcpy(&x[i], &bits, 4);
                } else
                    x[i] = ExpandSAQ(v, bf);
            }
            TrainingRows raw(h.data(),
                             n,
                             d,
                             bf ? DataTypes::DATA_TYPE_BF16 : DataTypes::DATA_TYPE_FP16,
                             d * 2,
                             h.size() * 2);
            SAQQuantizer<MetricType::METRIC_TYPE_COSINE> p(d, 4, 1, false, a.get()),
                f(d, 4, 1, false, a.get());
            REQUIRE(p.TrainRows(raw));
            REQUIRE(f.Train(x.data(), n));
            REQUIRE(FullSAQ(p) == FullSAQ(f));
            std::vector<uint8_t> pc(n * p.GetCodeSize()), fc(pc.size());
            p.EncodeBatch(x.data(), pc.data(), n);
            f.EncodeBatch(x.data(), fc.data(), n);
            REQUIRE(pc == fc);
            auto pq = p.FactoryComputer();
            auto fq = f.FactoryComputer();
            for (uint64_t i = 0; i < n; ++i) {
                pq->SetQuery(x.data() + i * d);
                fq->SetQuery(x.data() + i * d);
                float pd = p.ComputeDist(*pq, pc.data() + 6 * p.GetCodeSize()),
                      fd = f.ComputeDist(*fq, fc.data() + 6 * f.GetCodeSize());
                REQUIRE(std::memcmp(&pd, &fd, 4) == 0);
            }
        }
}
