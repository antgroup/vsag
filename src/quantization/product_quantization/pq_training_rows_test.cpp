#include "quantization/product_quantization/pq_training_rows.h"

#include <sstream>

#include "impl/allocator/safe_allocator.h"
#include "quantization/product_quantization/pq_fastscan_quantizer.h"
#include "quantization/product_quantization/product_quantizer.h"
#include "simd/bf16_simd.h"
#include "simd/fp16_simd.h"
#include "storage/stream_writer.h"
#include "unittest.h"
using namespace vsag;
namespace {
struct Prefix {
    uint64_t count, dim, reads = 0, columns = 0;
    static float
    Value(uint64_t i, uint64_t d) {
        if (i == 0)
            return 0.f;
        if (i == 1) {
            uint32_t bits = (d % 2) ? 0x80000000 : 0;
            float v;
            std::memcpy(&v, &bits, 4);
            return v;
        }
        return std::sin(float(i % 997) * .117f + float(d) * .313f);
    }
    static void
    Row(const void* c, uint64_t i, float* out) {
        auto& p = *const_cast<Prefix*>(static_cast<const Prefix*>(c));
        if (i >= std::min<uint64_t>(p.count, 65536))
            throw std::runtime_error("prefix tail read");
        ++p.reads;
        for (uint64_t d = 0; d < p.dim; ++d) out[d] = Value(i, d);
    }
    static void
    Col(const void* c, uint64_t i, uint64_t b, uint64_t n, float* out) {
        auto& p = *const_cast<Prefix*>(static_cast<const Prefix*>(c));
        if (i >= std::min<uint64_t>(p.count, 65536))
            throw std::runtime_error("prefix tail column");
        ++p.columns;
        for (uint64_t d = 0; d < n; ++d) out[d] = Value(i, b + d);
    }
};
template <class Q>
std::string
Payload(Q& q) {
    std::ostringstream o;
    IOStreamWriter w(o);
    q.Serialize(w);
    return o.str();
}
}  // namespace
TEST_CASE("PQ actual packed subspace bitwise prefix oracle", "[ut][training_rows][pq]") {
    auto a = SafeAllocator::FactoryDefaultAllocator();
    const uint64_t n = GENERATE(65535, 65536, 65537, 100001);
    const uint64_t dim = GENERATE(7, 33);
    const bool norm = GENERATE(false, true);
    Prefix p{n, dim};
    TrainingRows rows(n, dim, Prefix::Row, &p, Prefix::Col);
    detail::PQTrainingRows actual(rows, norm, a.get());
    const auto count = std::min<uint64_t>(n, 65536);
    REQUIRE(actual.Count() == count);
    std::vector<float> expected(count * dim), packed(count), row(dim);
    for (uint64_t i = 0; i < count; ++i) {
        for (uint64_t d = 0; d < dim; ++d) row[d] = Prefix::Value(i, d);
        if (norm)
            Normalize(row.data(), row.data(), dim);
        std::memcpy(expected.data() + i * dim, row.data(), dim * 4);
    }
    for (uint64_t d = 0; d < dim; ++d) {
        actual.Pack(d, 1, packed.data());
        bool same = true;
        for (uint64_t i = 0; i < count; ++i)
            if (std::memcmp(packed.data() + i, expected.data() + i * dim + d, 4) != 0) {
                same = false;
                break;
            }
        REQUIRE(same);
    }
    std::vector<float> full(count * dim);
    actual.Pack(0, dim, full.data());
    REQUIRE(std::memcmp(full.data(), expected.data(), full.size() * 4) == 0);
    REQUIRE(p.reads == (norm ? count : 0));
    REQUIRE(p.columns == count * (dim + 1));
}
TEST_CASE("PQ provider deterministic collapsed smoke and retry", "[ut][training_rows][pq]") {
    auto a = SafeAllocator::FactoryDefaultAllocator();
    const bool fast = GENERATE(false, true);
    const bool norm = GENERATE(false, true);
    const bool bf16 = GENERATE(false, true);
    const int dim = 8;
    const int n = 256;
    auto run = [&]<class Q>() {
        Q p(dim, 2, a.get()), f(dim, 2, a.get()), fresh(dim, 2, a.get());
        std::vector<float> x(n * dim), y(n * dim);
        std::vector<uint16_t> half(n * dim);
        for (int i = 0; i < n; ++i)
            for (int d = 0; d < dim; ++d) {
                float v = float(d + 1) * .0625f;
                half[i * dim + d] = bf16 ? generic::FloatToBF16(v) : generic::FloatToFP16(v);
                x[i * dim + d] = v;
                y[i * dim + d] = v * .5f;
            }
        TrainingRows rows(half.data(),
                          n,
                          dim,
                          bf16 ? DataTypes::DATA_TYPE_BF16 : DataTypes::DATA_TYPE_FP16,
                          dim * 2,
                          half.size() * 2);
        REQUIRE(p.TrainRows(rows));
        REQUIRE(f.Train(x.data(), n));
        REQUIRE(Payload(p) == Payload(f));
        std::vector<uint8_t> cp(n * p.GetCodeSize()), cf(cp.size());
        p.EncodeBatch(x.data(), cp.data(), n);
        f.EncodeBatch(x.data(), cf.data(), n);
        REQUIRE(cp == cf);
        auto pc = p.FactoryComputer(), fc = f.FactoryComputer();
        pc->SetQuery(x.data());
        fc->SetQuery(x.data());
        REQUIRE(p.ComputeDist(*pc, cp.data()) == f.ComputeDist(*fc, cf.data()));
        TrainingRows b(y.data(), n, dim, DataTypes::DATA_TYPE_FLOAT, dim * 4, y.size() * 4);
        REQUIRE(p.ReTrainRows(b));
        REQUIRE(fresh.Train(y.data(), n));
        REQUIRE(Payload(p) == Payload(fresh));
        TrainingRows bad(
            n,
            dim,
            [](const void*, uint64_t, float*) { throw std::runtime_error("provider fail"); },
            nullptr);
        REQUIRE_THROWS(p.ReTrainRows(bad));
        REQUIRE(p.ReTrainRows(b));
        REQUIRE(Payload(p) == Payload(fresh));
    };
    if (fast) {
        if (norm)
            run.template operator()<PQFastScanQuantizer<MetricType::METRIC_TYPE_COSINE>>();
        else
            run.template operator()<PQFastScanQuantizer<MetricType::METRIC_TYPE_L2SQR>>();
    } else {
        if (norm)
            run.template operator()<ProductQuantizer<MetricType::METRIC_TYPE_COSINE>>();
        else
            run.template operator()<ProductQuantizer<MetricType::METRIC_TYPE_L2SQR>>();
    }
}

TEST_CASE("PQ packing finite half edges retains native norm semantics", "[ut][training_rows][pq]") {
    auto a = SafeAllocator::FactoryDefaultAllocator();
    for (bool bf16 : {false, true}) {
        std::vector<uint16_t> h =
            bf16 ? std::vector<uint16_t>{0, 0x8000, 1, 0x7f, 0x80, 0x7f7f, 0xff7f}
                 : std::vector<uint16_t>{0, 0x8000, 1, 0x3ff, 0x400, 0x7bff, 0xfbff};
        std::vector<float> ref(7), out(7);
        for (int d = 0; d < 7; ++d) {
            auto v = h[d];
            if (bf16) {
                uint32_t bits = uint32_t(v) << 16;
                std::memcpy(&ref[d], &bits, 4);
            } else {
                int e = (v >> 10) & 31;
                float x = e == 0 ? std::ldexp(float(v & 1023), -24)
                                 : std::ldexp(1.f + float(v & 1023) / 1024, e - 15);
                ref[d] = (v & 0x8000) ? -x : x;
            }
        }
        Normalize(ref.data(), ref.data(), 7);
        TrainingRows rows(
            h.data(), 1, 7, bf16 ? DataTypes::DATA_TYPE_BF16 : DataTypes::DATA_TYPE_FP16, 14, 14);
        detail::PQTrainingRows actual(rows, true, a.get());
        actual.Pack(0, 7, out.data());
        REQUIRE(std::memcmp(ref.data(), out.data(), 28) == 0);
    }
}

TEST_CASE("PQ packing SIMD widths and offsets", "[ut][training_rows][pq]") {
    auto a = SafeAllocator::FactoryDefaultAllocator();
    const int dim = 65;
    const int n = 17;
    for (bool norm : {false, true}) {
        Prefix p{n, dim};
        TrainingRows rows(n, dim, Prefix::Row, &p, Prefix::Col);
        detail::PQTrainingRows actual(rows, norm, a.get());
        std::vector<float> ref(n * dim), row(dim);
        for (int i = 0; i < n; ++i) {
            for (int d = 0; d < dim; ++d) row[d] = Prefix::Value(i, d);
            if (norm)
                Normalize(row.data(), row.data(), dim);
            std::memcpy(ref.data() + i * dim, row.data(), dim * 4);
        }
        for (uint64_t width : {1, 7, 8, 9, 31, 32, 33})
            for (uint64_t begin : {uint64_t(0), uint64_t(1), uint64_t(dim) - width}) {
                std::vector<float> packed(n * width);
                actual.Pack(begin, width, packed.data());
                bool same = true;
                for (int i = 0; i < n; ++i)
                    if (std::memcmp(packed.data() + i * width,
                                    ref.data() + i * dim + begin,
                                    width * 4) != 0) {
                        same = false;
                        break;
                    }
                REQUIRE(same);
            }
    }
}

TEST_CASE("PQ row-only residual provider packed parity", "[ut][training_rows][pq]") {
    auto a = SafeAllocator::FactoryDefaultAllocator();
    const int dim = 65, n = 17;
    struct Residual {
        static void
        Read(const void*, uint64_t i, float* out) {
            for (int d = 0; d < 65; ++d) out[d] = Prefix::Value(i, d) - float(d) * .03125f;
        }
        static void
        Col(const void*, uint64_t i, uint64_t b, uint64_t n, float* out) {
            for (uint64_t d = 0; d < n; ++d)
                out[d] = Prefix::Value(i, b + d) - float(b + d) * .03125f;
        }
    };
    for (bool norm : {false, true}) {
        TrainingRows rows(n, dim, Residual::Read, nullptr),
            columns(n, dim, Residual::Read, nullptr, Residual::Col);
        detail::PQTrainingRows actual(rows, norm, a.get()), other(columns, norm, a.get());
        std::vector<float> ref(n * dim), row(dim);
        for (int i = 0; i < n; ++i) {
            Residual::Read(nullptr, i, row.data());
            if (norm)
                Normalize(row.data(), row.data(), dim);
            std::memcpy(ref.data() + i * dim, row.data(), dim * 4);
        }
        for (uint64_t width : {1, 7, 8, 9, 31, 32, 33})
            for (uint64_t begin : {uint64_t(0), uint64_t(1), uint64_t(dim) - width}) {
                std::vector<float> x(n * width), y(n * width);
                actual.Pack(begin, width, x.data());
                other.Pack(begin, width, y.data());
                REQUIRE(std::memcmp(x.data(), y.data(), x.size() * 4) == 0);
                bool same = true;
                for (int i = 0; i < n; ++i)
                    if (std::memcmp(
                            x.data() + i * width, ref.data() + i * dim + begin, width * 4) != 0) {
                        same = false;
                        break;
                    }
                REQUIRE(same);
            }
        REQUIRE_THROWS(actual.Pack(66, 1, row.data()));
        REQUIRE_THROWS(actual.Pack(64, 2, row.data()));
        REQUIRE_THROWS(actual.Pack(0, 1, nullptr));
    }
}
