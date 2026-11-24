#include <cstdlib>
#include <sstream>

#include "impl/allocator/safe_allocator.h"
#include "impl/transform/pca_transformer.h"
#include "quantization/quantizer_adapter.h"
#include "quantization/rabitq_quantization/rabitq_quantizer.h"
#include "simd/bf16_simd.h"
#include "simd/fp16_simd.h"
#include "storage/stream_writer.h"
#include "unittest.h"
using namespace vsag;
namespace {
template <class Q>
std::string
BytesPCA(Q& q) {
    std::ostringstream o;
    IOStreamWriter w(o);
    q.Serialize(w);
    return o.str();
}
float
ExpandPCA(uint16_t v, bool bf) {
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
TEST_CASE("PCA TrainingRows preserves ordered mean covariance and eigen model",
          "[TrainingRowsPCA]") {
    for (bool bf : {false, true})
        for (int d : {7, 33}) {
            auto a = SafeAllocator::FactoryDefaultAllocator();
            uint64_t n = 41;
            std::vector<uint16_t> h(n * d);
            std::vector<float> x(n * d), y(n * d);
            for (size_t i = 0; i < x.size(); ++i) {
                float f = std::sin(float(i) * .137f) + std::cos(float(i) * .271f) * .3f;
                h[i] = bf ? generic::FloatToBF16(f) : generic::FloatToFP16(f);
                x[i] = ExpandPCA(h[i], bf);
                y[i] = std::cos(float(i) * .183f);
            }
            TrainingRows raw(h.data(),
                             n,
                             d,
                             bf ? DataTypes::DATA_TYPE_BF16 : DataTypes::DATA_TYPE_FP16,
                             d * 2,
                             h.size() * 2);
            PCATransformer p(a.get(), d, d - 1), f(a.get(), d, d - 1);
            p.TrainRows(raw);
            f.Train(x.data(), n);
            REQUIRE(BytesPCA(p) == BytesPCA(f));
            std::vector<float> mp(d), mf(d), cp(n * d), cf(n * d), vp(d * d), vf(d * d), op(d),
                of(d);
            p.CopyMean(mp.data());
            f.CopyMean(mf.data());
            REQUIRE(std::memcmp(mp.data(), mf.data(), d * 4) == 0);
            for (uint64_t i = 0; i < n; ++i) {
                p.CentralizeData(x.data() + i * d, cp.data() + i * d);
                f.CentralizeData(x.data() + i * d, cf.data() + i * d);
                p.Transform(x.data() + i * d, op.data());
                f.Transform(x.data() + i * d, of.data());
                REQUIRE(std::memcmp(op.data(), of.data(), (d - 1) * 4) == 0);
            }
            REQUIRE(cp == cf);
            p.ComputeCovarianceMatrix(cp.data(), n, vp.data());
            f.ComputeCovarianceMatrix(cf.data(), n, vf.data());
            REQUIRE(vp == vf);
            struct C {
                const float* x;
                int d;
            };
            C ctx{y.data(), d};
            TrainingRows other(
                n,
                d,
                [](const void* c, uint64_t i, float* out) {
                    auto& v = *static_cast<const C*>(c);
                    std::memcpy(out, v.x + i * v.d, v.d * 4);
                },
                &ctx);
            TrainingRows bad(
                n,
                d,
                [](const void* c, uint64_t i, float* out) {
                    if (i == 3)
                        throw std::runtime_error("PCA provider fail");
                    auto& v = *static_cast<const C*>(c);
                    std::memcpy(out, v.x + i * v.d, v.d * 4);
                },
                &ctx);
            REQUIRE_THROWS(p.TrainRows(bad));
            p.TrainRows(other);
            PCATransformer fresh(a.get(), d, d - 1);
            fresh.Train(y.data(), n);
            REQUIRE(BytesPCA(p) == BytesPCA(fresh));
            auto before = BytesPCA(p);
            TrainingRows huge(
                uint64_t(PTRDIFF_MAX) / 4 / d + 1,
                d,
                [](const void*, uint64_t, float*) {
                    throw std::runtime_error("must not read overflow");
                },
                nullptr);
            REQUIRE_THROWS(p.TrainRows(huge));
            REQUIRE(BytesPCA(p) == before);
        }
}
TEST_CASE("RaBitQ PCA TrainingRows full payload codes query and retry", "[TrainingRowsPCA]") {
    for (bool bf : {false, true})
        for (int d : {7, 33}) {
            auto a = SafeAllocator::FactoryDefaultAllocator();
            uint64_t n = 41;
            std::vector<uint16_t> h(n * d);
            std::vector<float> x(n * d), y(n * d);
            for (size_t i = 0; i < x.size(); ++i) {
                float f = std::sin(float(i) * .137f) + std::cos(float(i) * .271f) * .3f;
                h[i] = bf ? generic::FloatToBF16(f) : generic::FloatToFP16(f);
                x[i] = ExpandPCA(h[i], bf);
                y[i] = std::cos(float(i) * .183f);
            }
            TrainingRows raw(h.data(),
                             n,
                             d,
                             bf ? DataTypes::DATA_TYPE_BF16 : DataTypes::DATA_TYPE_FP16,
                             d * 2,
                             h.size() * 2);
            auto run = [&]<MetricType M>() {
                RaBitQuantizer<M> p(d, d - 1, 32, 1, true, false, a.get()),
                    f(d, d - 1, 32, 1, true, false, a.get()),
                    fresh(d, d - 1, 32, 1, true, false, a.get());
                REQUIRE(p.CanTrainRows());
                REQUIRE(p.TrainRows(raw));
                REQUIRE(f.Train(x.data(), n));
                REQUIRE(BytesPCA(p) == BytesPCA(f));
                auto before = BytesPCA(p);
                TrainingRows invalid(
                    1,
                    d,
                    [](const void*, uint64_t, float*) {
                        throw std::runtime_error("must not read invalid PCA count");
                    },
                    nullptr);
                REQUIRE_THROWS(p.TrainRows(invalid));
                REQUIRE_THROWS(p.ReTrainRows(invalid));
                REQUIRE(BytesPCA(p) == before);
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
                TrainingRows other(y.data(), n, d, DataTypes::DATA_TYPE_FLOAT, d * 4, y.size() * 4);
                REQUIRE(p.ReTrainRows(other));
                REQUIRE(fresh.Train(y.data(), n));
                REQUIRE(BytesPCA(p) == BytesPCA(fresh));
                TrainingRows bad(
                    n,
                    d,
                    [](const void*, uint64_t, float*) {
                        throw std::runtime_error("PCA provider fail");
                    },
                    nullptr);
                REQUIRE_THROWS(p.ReTrainRows(bad));
                REQUIRE(p.ReTrainRows(other));
                REQUIRE(BytesPCA(p) == BytesPCA(fresh));
            };
            run.template operator()<MetricType::METRIC_TYPE_L2SQR>();
            run.template operator()<MetricType::METRIC_TYPE_IP>();
            run.template operator()<MetricType::METRIC_TYPE_COSINE>();
        }
}

namespace {
class PCAAllocationCounter final : public Allocator {
public:
    std::string
    Name() override {
        return "pca-allocation-counter";
    }
    void*
    Allocate(uint64_t n) override {
        if (n == watched)
            ++watched_calls;
        auto p = std::malloc(n);
        if (!p && n)
            throw std::bad_alloc();
        return p;
    }
    void
    Deallocate(void* p) override {
        std::free(p);
    }
    void*
    Reallocate(void* p, uint64_t n) override {
        if (n == watched)
            ++watched_calls;
        auto q = std::realloc(p, n);
        if (!q && n)
            throw std::bad_alloc();
        return q;
    }
    uint64_t watched = 0, watched_calls = 0;
};
template <class Q>
std::string
InnerPCA(Q& q) {
    std::ostringstream o;
    IOStreamWriter w(o);
    q.SerializeImpl(w);
    return o.str();
}
}  // namespace
TEST_CASE("Native half adapter PCA has only PCA centralized ND allocation", "[TrainingRowsPCA]") {
    for (bool bf : {false, true})
        for (int d : {7, 33}) {
            auto a = std::make_shared<PCAAllocationCounter>();
            IndexCommonParam c;
            c.allocator_ = a;
            c.dim_ = d;
            c.metric_ = MetricType::METRIC_TYPE_L2SQR;
            c.data_type_ = bf ? DataTypes::DATA_TYPE_BF16 : DataTypes::DATA_TYPE_FP16;
            auto param = std::make_shared<RaBitQuantizerParameter>();
            param->pca_dim_ = d - 1;
            param->use_fht_ = true;
            QuantizerAdapter<RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR>, uint16_t> p(param, c);
            RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR> f(d, d - 1, 32, 1, true, false, a.get());
            uint64_t n = 41;
            std::vector<uint16_t> h(n * d);
            std::vector<float> x(n * d);
            for (size_t i = 0; i < x.size(); ++i) {
                float v = std::sin(float(i) * .137f) + std::cos(float(i) * .271f) * .3f;
                h[i] = bf ? generic::FloatToBF16(v) : generic::FloatToFP16(v);
                x[i] = ExpandPCA(h[i], bf);
            }
            a->watched = n * d * sizeof(float);
            a->watched_calls = 0;
            REQUIRE(p.Train(reinterpret_cast<const float*>(h.data()), n));
            // The one N*D allocation is PCA centralized_data; no second adapter mirror.
            REQUIRE(a->watched_calls == 1);
            a->watched_calls = 0;
            REQUIRE(f.Train(x.data(), n));
            REQUIRE(a->watched_calls == 1);
            REQUIRE(InnerPCA(p) == InnerPCA(f));
            std::vector<uint8_t> pc(n * p.GetCodeSize()), fc(n * f.GetCodeSize());
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
            auto before = BytesPCA(p);
            TrainingRows invalid(
                1,
                d,
                [](const void*, uint64_t, float*) { throw std::runtime_error("must not read"); },
                nullptr);
            REQUIRE_THROWS(p.ReTrainRows(invalid));
            REQUIRE(BytesPCA(p) == before);
            TrainingRows wrong(
                41,
                d - 1,
                [](const void*, uint64_t, float*) { throw std::runtime_error("must not read"); },
                nullptr);
            REQUIRE_THROWS(p.ReTrainRows(wrong));
            REQUIRE(BytesPCA(p) == before);
        }
}
