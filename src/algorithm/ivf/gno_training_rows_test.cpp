#include <cmath>
#include <cstring>
#include <numeric>

#include "gno_imi_partition.h"
#include "gno_residual_rows.h"
#include "impl/allocator/safe_allocator.h"
#include "ivf_partition_strategy_parameter.h"
#include "storage/stream_writer.h"
#include "unittest.h"
#include "utils/half_input.h"
using namespace vsag;
namespace {
float
DecodeGNO(uint16_t bits, bool bf) {
    if (bf) {
        uint32_t expanded = uint32_t(bits) << 16;
        float v;
        std::memcpy(&v, &expanded, 4);
        return v;
    }
    const int sign = (bits & 0x8000) ? -1 : 1;
    const int e = (bits >> 10) & 31;
    const int f = bits & 1023;
    return sign * (e ? std::ldexp(float(1024 + f), e - 25) : std::ldexp(float(f), -24));
}
IVFPartitionStrategyParametersPtr
GNOParams(uint32_t k) {
    auto p = std::make_shared<IVFPartitionStrategyParameters>();
    auto j = JsonType::Parse(
        R"({"partition_strategy_type":"gno_imi","ivf_train_type":"kmeans","gno_imi":{"first_order_buckets_count":2,"second_order_buckets_count":2}})");
    p->FromJson(j);
    p->gnoimi_param->first_order_buckets_count = k;
    p->gnoimi_param->second_order_buckets_count = k;
    return p;
}
std::string
GNOBytes(GNOIMIPartition& p) {
    std::ostringstream s;
    IOStreamWriter w(s);
    p.Serialize(w);
    return s.str();
}
}  // namespace
TEST_CASE("GNO provider four immutable generations equal original X minus BF centroid",
          "[GNOTrainingRows]") {
    auto alloc = SafeAllocator::FactoryDefaultAllocator();
    for (bool bf : {false, true})
        for (int d : {7, 33})
            for (bool norm : {false, true}) {
                INFO(bf);
                INFO(d);
                INFO(norm);
                const uint64_t n = 17, k = 3;
                std::vector<uint16_t> h(n * d);
                std::vector<float> x(n * d);
                for (uint64_t i = 0; i < h.size(); ++i) {
                    h[i] =
                        bf ? uint16_t(0x3e00 + (i * 29) % 512) : uint16_t(0x3000 + (i * 37) % 1024);
                    x[i] = DecodeGNO(h[i], bf);
                }
                h[0] = 0;
                h[1] = 0x8000;
                h[2] = 1;
                h[3] = 0x8001;
                h[4] = bf ? 0x007f : 0x03ff;
                for (uint64_t j = 0; j < h.size(); ++j) x[j] = DecodeGNO(h[j], bf);
                TrainingRows source(h.data(),
                                    n,
                                    d,
                                    bf ? DataTypes::DATA_TYPE_BF16 : DataTypes::DATA_TYPE_FP16,
                                    d * 2,
                                    h.size() * 2);
                GNOResidualRows ctx(source, norm, alloc.get());
                std::vector<float> current = x, row(d), expected(d);
                if (norm) {
                    std::vector<float> normalized(x.size());
                    for (uint64_t i = 0; i < n; ++i)
                        Normalize(x.data() + i * d, normalized.data() + i * d, d);
                    x.swap(normalized);
                }
                current = x;
                for (int stage = 0; stage < 4; ++stage) {
                    for (uint64_t i = 0; i < n; ++i) {
                        GNOResidualRows::Read(&ctx, i, row.data());
                        REQUIRE(std::memcmp(row.data(), current.data() + i * d, d * 4) == 0);
                    }
                    std::vector<float> c(k * d);
                    for (uint64_t j = 0; j < c.size(); ++j)
                        c[j] = float((int(j) + stage * 13) % 19 - 9) * .03125F;
                    // Synthetic valid centroid IDs vary across rows/stages; not a BF mapping oracle.
                    Vector<BucketIdType> labels(n, alloc.get());
                    for (uint64_t i = 0; i < n; ++i) labels[i] = (i + stage) % k;
                    std::vector<float> next(n * d);
                    for (uint64_t i = 0; i < n; ++i) {
                        std::memcpy(next.data() + i * d, x.data() + i * d, d * 4);
                        BlasFunction::Saxpy(
                            d, -1.0, c.data() + labels[i] * d, 1, next.data() + i * d, 1);
                    }
                    ctx.Replace(c.data(), k, std::move(labels));
                    std::fill(c.begin(), c.end(), 999.0F);
                    current.swap(next);
                }
                for (uint64_t i = 0; i < n; ++i) {
                    GNOResidualRows::Read(&ctx, i, row.data());
                    REQUIRE(std::memcmp(row.data(), current.data() + i * d, d * 4) == 0);
                }
                Vector<BucketIdType> invalid(n, alloc.get());
                std::fill(invalid.begin(), invalid.end(), k);
                std::vector<float> c(k * d);
                REQUIRE_THROWS(ctx.Replace(c.data(), k, std::move(invalid)));
                GNOResidualRows::Read(&ctx, 0, row.data());
                REQUIRE(std::memcmp(row.data(), current.data(), d * 4) == 0);
            }
}
TEST_CASE("GNO actual typed partition nondegenerate training and collapsed reference",
          "[GNOTrainingRows]") {
    auto alloc = SafeAllocator::FactoryDefaultAllocator();
    for (bool bf : {false, true})
        for (int d : {7, 33})
            for (auto metric : {MetricType::METRIC_TYPE_L2SQR,
                                MetricType::METRIC_TYPE_IP,
                                MetricType::METRIC_TYPE_COSINE}) {
                INFO(bf);
                INFO(d);
                INFO(static_cast<int>(metric));
                IndexCommonParam cp;
                cp.dim_ = d;
                cp.metric_ = metric;
                cp.allocator_ = alloc;
                cp.data_type_ = bf ? DataTypes::DATA_TYPE_BF16 : DataTypes::DATA_TYPE_FP16;
                const uint64_t n = 48;
                std::vector<uint16_t> h(n * d);
                std::vector<float> x(n * d);
                for (uint64_t i = 0; i < h.size(); ++i) {
                    h[i] =
                        bf ? uint16_t(0x3e00 + (i * 29) % 512) : uint16_t(0x3000 + (i * 37) % 1024);
                    x[i] = DecodeGNO(h[i], bf);
                }
                auto dataset =
                    Dataset::Make()->Dim(d)->NumElements(n)->Float16Vectors(h.data())->Owner(false);
                GNOIMIPartition typed(cp, GNOParams(2));
                typed.Train(dataset);
                auto assignments = typed.ClassifyDatas(x.data(), n, 1, nullptr);
                REQUIRE(assignments.size() == n);
                for (auto a : assignments) REQUIRE(a < 4);
                auto bytes = GNOBytes(typed);
                REQUIRE(!bytes.empty());
                Vector<float> centroid(d, alloc.get());
                for (uint32_t a = 0; a < 4; ++a) {
                    typed.GetCentroid(a, centroid);
                    for (float v : centroid) {
                        uint32_t b;
                        std::memcpy(&b, &v, 4);
                        REQUIRE((b & 0x7f800000) != 0x7f800000);
                    }
                }
                // k1 is only a weak deterministic complete-model parity check; k2 above is
                // nondegenerate functional coverage, NOT random-default cross-object equality.
                GNOIMIPartition one(cp, GNOParams(1));
                one.Train(dataset);
                auto fp = cp;
                fp.data_type_ = DataTypes::DATA_TYPE_FLOAT;
                GNOIMIPartition reference(fp, GNOParams(1));
                reference.Train(
                    Dataset::Make()->Dim(d)->NumElements(n)->Float32Vectors(x.data())->Owner(
                        false));
                REQUIRE(GNOBytes(one) == GNOBytes(reference));
                auto gpu = GNOParams(2);
                gpu->enable_gpu_build = true;
                GNOIMIPartition reject(cp, gpu);
                REQUIRE_THROWS(reject.Train(dataset));
            }
}
