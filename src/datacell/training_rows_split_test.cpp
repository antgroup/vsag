
#include <numeric>
#include <sstream>

#include "datacell/flatten_datacell_parameter.h"
#include "datacell/rabitq_split_datacell.h"
#include "impl/allocator/safe_allocator.h"
#include "simd/bf16_simd.h"
#include "simd/fp16_simd.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "unittest.h"
using namespace vsag;
namespace {
float
ExpandSplitRows(uint16_t v, bool bf) {
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
std::string
SaveSplitRows(FlattenInterface& q) {
    std::ostringstream o;
    IOStreamWriter w(o);
    q.Serialize(w);
    return o.str();
}
}  // namespace
TEST_CASE("Ordinary split native half training matches original float pointer and reload",
          "[TrainingRowsSplit]") {
    for (bool bf : {false, true})
        for (int d : {7, 33})
            for (auto metric : {MetricType::METRIC_TYPE_L2SQR,
                                MetricType::METRIC_TYPE_IP,
                                MetricType::METRIC_TYPE_COSINE})
                for (int mode : {0, 1, 2}) {
                    INFO(bf << " " << d << " " << static_cast<int>(metric) << " " << mode);
                    auto a = SafeAllocator::FactoryDefaultAllocator();
                    IndexCommonParam c;
                    c.allocator_ = a;
                    c.dim_ = d;
                    c.metric_ = metric;
                    c.data_type_ = bf ? DataTypes::DATA_TYPE_BF16 : DataTypes::DATA_TYPE_FP16;
                    auto fcommon = c;
                    fcommon.data_type_ = DataTypes::DATA_TYPE_FLOAT;
                    const auto type = mode == 2 ? "mrle, rabitq" : "rabitq";
                    auto json = JsonType::Parse(fmt::format(
                        R"({{"codes_type":"rabitq_split","io_params":{{"type":"memory_io"}},"quantization_params":{{"type":"rabitq","tq_chain":"{}","mrle_dim":{},"pca_dim":{},"rabitq_version":"split","rabitq_bits_per_dim_query":32,"rabitq_bits_per_dim_base":8,"rabitq_bits_per_dim_filter":3,"use_fht":true}}}})",
                        type,
                        d - 1,
                        mode == 1 ? d - 1 : 0));
                    json["quantization_params"]["type"].SetString(mode == 2 ? "tq" : "rabitq");
                    auto param = std::make_shared<FlattenDataCellParameter>();
                    param->FromJson(json);
                    auto p = FlattenInterface::MakeInstance(param, c);
                    auto f = FlattenInterface::MakeInstance(param, fcommon);
                    REQUIRE(p != nullptr);
                    REQUIRE(f != nullptr);
                    constexpr uint64_t n = 41;
                    std::vector<uint16_t> h(n * d);
                    std::vector<float> x(n * d);
                    for (size_t i = 0; i < x.size(); ++i) {
                        h[i] = bf ? generic::FloatToBF16(std::sin(float(i) * .137f))
                                  : generic::FloatToFP16(std::sin(float(i) * .137f));
                        x[i] = ExpandSplitRows(h[i], bf);
                    }
                    p->Train(h.data(), n);
                    f->Train(x.data(), n);
                    p->Resize(n);
                    f->Resize(n);
                    p->BatchInsertVector(h.data(), n);
                    f->BatchInsertVector(x.data(), n);
                    auto ps = std::dynamic_pointer_cast<RaBitQSplitDataCellInterface>(p);
                    REQUIRE(ps != nullptr);
                    std::vector<uint8_t> pf(ps->OneBitCodeSize()), pt(ps->SupplementCodeSize()),
                        ff(pf.size()), ft(pt.size());
                    auto fs = std::dynamic_pointer_cast<RaBitQSplitDataCellInterface>(f);
                    REQUIRE(fs != nullptr);
                    std::vector<InnerIdType> ids(n);
                    std::iota(ids.begin(), ids.end(), 0);
                    for (auto id : ids) {
                        REQUIRE(ps->CopySplitCodes(id, pf.data(), pt.data()));
                        REQUIRE(fs->CopySplitCodes(id, ff.data(), ft.data()));
                        REQUIRE(pf == ff);
                        REQUIRE(pt == ft);
                    }
                    auto pc = p->FactoryComputer(h.data());
                    auto fc = f->FactoryComputer(x.data());
                    std::vector<float> pd(n), fd(n);
                    p->Query(pd.data(), pc, ids.data(), n);
                    f->Query(fd.data(), fc, ids.data(), n);
                    REQUIRE(std::memcmp(pd.data(), fd.data(), n * 4) == 0);
                    auto payload = SaveSplitRows(*p);
                    auto loaded = FlattenInterface::MakeInstance(param, c);
                    std::istringstream in(payload);
                    IOStreamReader reader(in);
                    loaded->Deserialize(reader);
                    auto lc = loaded->FactoryComputer(h.data());
                    std::vector<float> ld(n);
                    loaded->Query(ld.data(), lc, ids.data(), n);
                    REQUIRE(std::memcmp(pd.data(), ld.data(), n * 4) == 0);
                    REQUIRE(SaveSplitRows(*loaded) == payload);
                }
}
