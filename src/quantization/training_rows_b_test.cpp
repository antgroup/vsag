#include <sstream>

#include "datacell/bucket_interface.h"
#include "datacell/flatten_interface.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/transform/fht_kac_rotate_transformer.h"
#include "impl/transform/random_orthogonal_transformer.h"
#include "quantization/fp32_quantizer.h"
#include "quantization/rabitq_quantization/rabitq_quantizer.h"
#include "quantization/scalar_quantization/half_precision_quantizer.h"
#include "simd/bf16_simd.h"
#include "simd/fp16_simd.h"
#include "storage/stream_writer.h"
#include "unittest.h"
using namespace vsag;
namespace {
template <class Q>
std::string
Payload(Q& q) {
    std::ostringstream o;
    IOStreamWriter w(o);
    q.Serialize(w);
    return o.str();
}
}  // namespace
TEST_CASE("Shape-only provider never reads values", "[TrainingRowsB]") {
    auto a = SafeAllocator::FactoryDefaultAllocator();
    TrainingRows rows(
        17,
        7,
        [](const void*, uint64_t, float*) { throw std::runtime_error("must not read"); },
        nullptr);
    FP32Quantizer<MetricType::METRIC_TYPE_L2SQR> f(7, a.get());
    REQUIRE(f.CanTrainRows());
    REQUIRE(f.TrainRows(rows));
    REQUIRE(f.ReTrainRows(rows));
    IndexCommonParam c;
    c.dim_ = 7;
    c.allocator_ = a;
    c.metric_ = MetricType::METRIC_TYPE_L2SQR;
    FP16Quantizer<MetricType::METRIC_TYPE_L2SQR> h(7, a.get());
    BF16Quantizer<MetricType::METRIC_TYPE_L2SQR> b(7, a.get());
    REQUIRE(h.TrainRows(rows));
    REQUIRE(h.ReTrainRows(rows));
    REQUIRE(b.TrainRows(rows));
    REQUIRE(b.ReTrainRows(rows));
}
TEST_CASE("No-PCA RaBitQ provider ordered global training", "[TrainingRowsB]") {
    const auto metric = GENERATE(
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_IP, MetricType::METRIC_TYPE_COSINE);
    const bool bf16 = GENERATE(false, true);
    const int dim = GENERATE(7, 33);
    auto a = SafeAllocator::FactoryDefaultAllocator();
    auto run = [&]<MetricType M>() {
        RaBitQuantizer<M> p(dim, dim, 32, 1, true, false, a.get()),
            f(dim, dim, 32, 1, true, false, a.get()), fresh(dim, dim, 32, 1, true, false, a.get());
        std::vector<float> x(17 * dim), y(17 * dim);
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] = std::sin(float(i) * .13f);
            y[i] = std::cos(float(i) * .27f);
        }
        std::vector<uint16_t> half(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            half[i] = bf16 ? generic::FloatToBF16(x[i]) : generic::FloatToFP16(x[i]);
            const uint16_t v = half[i];
            if (bf16) {
                uint32_t bits = uint32_t(v) << 16;
                std::memcpy(&x[i], &bits, 4);
            } else {
                int exp = (v >> 10) & 31;
                float value = exp == 0 ? std::ldexp(float(v & 1023), -24)
                                       : std::ldexp(1.f + float(v & 1023) / 1024, exp - 15);
                x[i] = (v & 0x8000) ? -value : value;
            }
        }
        TrainingRows raw(half.data(),
                         17,
                         dim,
                         bf16 ? DataTypes::DATA_TYPE_BF16 : DataTypes::DATA_TYPE_FP16,
                         dim * 2,
                         half.size() * 2);
        TrainingRows rows(
            17,
            dim,
            [](const void* c, uint64_t i, float* out) {
                static_cast<const TrainingRows*>(c)->Read(i, out);
            },
            &raw);
        REQUIRE(p.TrainRows(rows));
        REQUIRE(f.Train(x.data(), 17));
        REQUIRE(Payload(p) == Payload(f));
        std::vector<uint8_t> cp(17 * p.GetCodeSize()), cf(cp.size());
        p.EncodeBatch(x.data(), cp.data(), 17);
        f.EncodeBatch(x.data(), cf.data(), 17);
        REQUIRE(cp == cf);
        auto pc = p.FactoryComputer(), fc = f.FactoryComputer();
        pc->SetQuery(x.data());
        fc->SetQuery(x.data());
        for (int i = 0; i < 17; ++i)
            REQUIRE(p.ComputeDist(*pc, cp.data() + i * p.GetCodeSize()) ==
                    f.ComputeDist(*fc, cf.data() + i * f.GetCodeSize()));
        TrainingRows other(y.data(), 17, dim, DataTypes::DATA_TYPE_FLOAT, dim * 4, y.size() * 4);
        REQUIRE(p.ReTrainRows(other));
        REQUIRE(fresh.Train(y.data(), 17));
        REQUIRE(Payload(p) == Payload(fresh));
        TrainingRows bad(
            17,
            dim,
            [](const void*, uint64_t, float*) { throw std::runtime_error("provider fail"); },
            nullptr);
        REQUIRE_THROWS(p.ReTrainRows(bad));
        REQUIRE(p.ReTrainRows(other));
        REQUIRE(Payload(p) == Payload(fresh));
        // PCA capability and actual model parity are covered in TrainingRowsPCA.
    };
    if (metric == MetricType::METRIC_TYPE_L2SQR)
        run.template operator()<MetricType::METRIC_TYPE_L2SQR>();
    else if (metric == MetricType::METRIC_TYPE_IP)
        run.template operator()<MetricType::METRIC_TYPE_IP>();
    else
        run.template operator()<MetricType::METRIC_TYPE_COSINE>();
}

TEST_CASE("Provider capability is configuration-specific in consumers", "[TrainingRowsB]") {
    auto a = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam c;
    c.dim_ = 33;
    c.allocator_ = a;
    c.metric_ = MetricType::METRIC_TYPE_L2SQR;
    c.data_type_ = DataTypes::DATA_TYPE_FP16;
    for (bool pca : {false, true}) {
        auto p = std::make_shared<BucketDataCellParameter>();
        p->FromJson(JsonType::Parse(
            R"({"io_params":{"type":"memory_io"},"quantization_params":{"type":"rabitq","rabitq_use_fht":true},"buckets_count":1,"use_residual":false})"));
        auto q = std::dynamic_pointer_cast<RaBitQuantizerParameter>(p->quantizer_parameter);
        q->pca_dim_ = pca ? 16 : 33;
        auto bucket = BucketInterface::MakeInstance(p, c);
        REQUIRE(bucket->SupportsTrainingRows());
        auto flatp = std::make_shared<FlattenDataCellParameter>();
        flatp->io_parameter = p->io_parameter;
        flatp->quantizer_parameter = p->quantizer_parameter;
        auto flat = FlattenInterface::MakeInstance(flatp, c);
        REQUIRE(flat->SupportsTrainingRows());
    }
}

TEST_CASE("RaBitQ rotation training is explicitly data independent", "[TrainingRowsB]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    RandomOrthogonalMatrix first(allocator.get(), 7, 3, 2026);
    RandomOrthogonalMatrix second(allocator.get(), 7, 3, 2026);
    REQUIRE(first.IsTrainingDataIndependent());
    float ignored = 123.0F;
    first.Train(nullptr, 0);
    second.Train(&ignored, 999);
    REQUIRE(Payload(first) == Payload(second));
    FhtKacRotator fht_first(allocator.get(), 7);
    FhtKacRotator fht_second(allocator.get(), 7);
    REQUIRE(fht_first.IsTrainingDataIndependent());
    fht_first.Train(nullptr, 0);
    fht_second.Train(&ignored, 999);
    REQUIRE(Payload(fht_first) == Payload(fht_second));
}
