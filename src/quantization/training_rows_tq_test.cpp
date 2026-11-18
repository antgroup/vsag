#include <sstream>

#include "impl/allocator/safe_allocator.h"
#include "quantization/quantizer_adapter.h"
#include "quantization/scalar_quantization/scalar_quantizer.h"
#include "quantization/transform_quantization/transform_quantizer.h"
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

TEST_CASE("TQ typed providers preserve deterministic full transforms and original pointer models",
          "[TrainingRowsTQ]") {
    for (bool bf : {false, true})
        for (int d : {7, 33})
            for (const char* chain : {"fht, sq8", "mrle, fht, sq8"}) {
                auto a = SafeAllocator::FactoryDefaultAllocator();
                IndexCommonParam c;
                c.allocator_ = a;
                c.dim_ = d;
                c.metric_ = MetricType::METRIC_TYPE_L2SQR;
                c.data_type_ = bf ? DataTypes::DATA_TYPE_BF16 : DataTypes::DATA_TYPE_FP16;
                auto param = std::make_shared<TransformQuantizerParameter>();
                param->FromJson(JsonType::Parse(
                    fmt::format(R"({{"tq_chain":"{}","mrle_dim":{}}})", chain, d - 1)));
                using Inner = ScalarQuantizer<MetricType::METRIC_TYPE_L2SQR, 8>;
                using Q = TransformQuantizer<Inner>;
                Q p(param, c), f(param, c), fresh(param, c);
                REQUIRE(p.CanTrainRows());
                const uint64_t n = 41;
                std::vector<uint16_t> h(n * d);
                std::vector<float> x(n * d), y(n * d);
                for (size_t i = 0; i < x.size(); ++i) {
                    float v = std::sin(float(i) * .137f);
                    h[i] = bf ? generic::FloatToBF16(v) : generic::FloatToFP16(v);
                    x[i] = ExpandSAQ(h[i], bf);
                    y[i] = std::cos(float(i) * .173f);
                }
                TrainingRows raw(h.data(), n, d, c.data_type_, d * 2, h.size() * 2);
                REQUIRE(p.TrainRows(raw));
                REQUIRE(f.Train(x.data(), n));
                REQUIRE(FullSAQ(p) == FullSAQ(f));
                REQUIRE(p.GetTransformedDim() ==
                        (std::string(chain).find("mrle") != std::string::npos ? d - 1 : d));
                auto compare = [&](Q& left, Q& right, const float* input) {
                    std::vector<uint8_t> pc(n * left.GetCodeSize()), fc(pc.size());
                    left.EncodeBatch(input, pc.data(), n);
                    right.EncodeBatch(input, fc.data(), n);
                    REQUIRE(pc == fc);
                    auto pq = left.FactoryComputer();
                    auto fq = right.FactoryComputer();
                    pq->SetQuery(input);
                    fq->SetQuery(input);
                    for (uint64_t i = 0; i < n; ++i) {
                        float pd = left.ComputeDist(*pq, pc.data() + i * left.GetCodeSize()),
                              fd = right.ComputeDist(*fq, fc.data() + i * right.GetCodeSize());
                        REQUIRE(std::memcmp(&pd, &fd, 4) == 0);
                    }
                    std::vector<float> decoded(d);
                    REQUIRE_FALSE(left.DecodeBatch(pc.data(), decoded.data(), 1));
                };
                compare(p, f, x.data());
                struct C {
                    const float* x;
                    uint64_t d;
                };
                C ctx{x.data(), uint64_t(d)};
                TrainingRows rowonly(
                    n,
                    d,
                    [](const void* v, uint64_t i, float* out) {
                        const auto& c = *static_cast<const C*>(v);
                        std::memcpy(out, c.x + i * c.d, c.d * 4);
                    },
                    &ctx);
                Q rowq(param, c);
                REQUIRE(rowq.TrainRows(rowonly));
                REQUIRE(FullSAQ(rowq) == FullSAQ(f));
                TrainingRows other(y.data(), n, d, DataTypes::DATA_TYPE_FLOAT, d * 4, y.size() * 4);
                REQUIRE(p.ReTrainRows(other));
                REQUIRE(fresh.Train(y.data(), n));
                REQUIRE(FullSAQ(p) == FullSAQ(fresh));
                compare(p, fresh, y.data());
                TrainingRows bad(
                    n,
                    d,
                    [](const void*, uint64_t, float*) { throw std::runtime_error("TQ callback"); },
                    nullptr);
                REQUIRE_THROWS(p.ReTrainRows(bad));
                REQUIRE(p.ReTrainRows(other));
                REQUIRE(FullSAQ(p) == FullSAQ(fresh));
                QuantizerAdapter<Q, uint16_t> adapter(param, c);
                REQUIRE(adapter.Train(reinterpret_cast<const float*>(h.data()), n));
                REQUIRE(InnerSAQ(adapter) == InnerSAQ(f));
                std::vector<uint8_t> ac(n * adapter.GetCodeSize()), fc(ac.size());
                adapter.EncodeBatch(reinterpret_cast<const float*>(h.data()), ac.data(), n);
                f.EncodeBatch(x.data(), fc.data(), n);
                REQUIRE(ac == fc);
                auto aq = adapter.FactoryComputer();
                auto fq = f.FactoryComputer();
                aq->SetQuery(reinterpret_cast<const float*>(h.data()));
                fq->SetQuery(x.data());
                for (uint64_t i = 0; i < n; ++i) {
                    float ad = adapter.ComputeDist(*aq, ac.data() + i * adapter.GetCodeSize()),
                          fd = f.ComputeDist(*fq, fc.data() + i * f.GetCodeSize());
                    REQUIRE(std::memcmp(&ad, &fd, 4) == 0);
                }
            }
}
TEST_CASE("TQ PCA keeps native half fallback and ROM retains random model scope",
          "[TrainingRowsTQ]") {
    auto a = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam c;
    c.allocator_ = a;
    c.dim_ = 7;
    c.data_type_ = DataTypes::DATA_TYPE_FP16;
    using I = ScalarQuantizer<MetricType::METRIC_TYPE_L2SQR, 8>;
    using Q = TransformQuantizer<I>;
    auto p = std::make_shared<TransformQuantizerParameter>();
    p->FromJson(JsonType::Parse(R"({"tq_chain":"pca, sq8","pca_dim":3})"));
    Q q(p, c);
    REQUIRE_FALSE(q.CanTrainRows());
    std::vector<float> x(41 * 7);
    std::vector<uint16_t> h(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        h[i] = generic::FloatToFP16(std::sin(float(i) * .137f));
        x[i] = ExpandSAQ(h[i], false);
    }
    REQUIRE(q.Train(x.data(), 41));
    auto before = FullSAQ(q);
    TrainingRows raw(h.data(), 41, 7, c.data_type_, 14, h.size() * 2);
    REQUIRE_THROWS(q.ReTrainRows(raw));
    REQUIRE(FullSAQ(q) == before);
    QuantizerAdapter<Q, uint16_t> adapter(p, c);
    REQUIRE(adapter.Train(reinterpret_cast<const float*>(h.data()), 41));
    REQUIRE(InnerSAQ(adapter) == InnerSAQ(q));
    p = std::make_shared<TransformQuantizerParameter>();
    p->FromJson(JsonType::Parse(R"({"tq_chain":"rom, sq8"})"));
    Q rom(p, c);
    REQUIRE(rom.CanTrainRows());
    REQUIRE(rom.TrainRows(raw));
    std::vector<uint8_t> codes(rom.GetCodeSize());
    REQUIRE(rom.EncodeOne(x.data(), codes.data()));
    auto computer = rom.FactoryComputer();
    computer->SetQuery(x.data());
    float distance = rom.ComputeDist(*computer, codes.data());
    uint32_t bits;
    std::memcpy(&bits, &distance, 4);
    REQUIRE((bits & 0x7f800000) != 0x7f800000);  // No random cross-model equality claim.
}
