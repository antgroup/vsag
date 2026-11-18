#include <atomic>
#include <sstream>

#include "bucket_datacell.h"
#include "impl/allocator/safe_allocator.h"
#include "io/memory_io/memory_io.h"
#include "quantization/scalar_quantization/scalar_quantizer.h"
#include "storage/stream_writer.h"
#include "unittest.h"
using namespace vsag;
namespace {
class FixedCentroidPartitionStrategy : public IVFPartitionStrategy {
public:
    FixedCentroidPartitionStrategy(const IndexCommonParam& common_param,
                                   BucketIdType bucket_count,
                                   float centroid_scale = 1.0F)
        : IVFPartitionStrategy(common_param, bucket_count), centroid_scale_(centroid_scale) {
    }

    void
    Train(const DatasetPtr) override {
        is_trained_ = true;
    }

    Vector<BucketIdType>
    ClassifyDatas(const void*, int64_t count, BucketIdType, QueryContext*) const override {
        return Vector<BucketIdType>(count, 0, allocator_);
    }

    void
    GetCentroid(BucketIdType bucket_id, Vector<float>& centroid) override {
        centroid_requests_.fetch_add(1, std::memory_order_relaxed);
        for (uint64_t i = 0; i < centroid.size(); ++i) {
            centroid[i] = static_cast<float>((bucket_id + 1) * (i + 1)) * 0.01F * centroid_scale_;
        }
    }

    [[nodiscard]] uint64_t
    GetCentroidRequestCount() const {
        return centroid_requests_.load(std::memory_order_relaxed);
    }

private:
    std::atomic<uint64_t> centroid_requests_{0};
    float centroid_scale_{1.0F};
};

}  // namespace
TEST_CASE("SQ8 residual TrainingRows fixed centroid and callback bounds", "[IVFRowsBucket]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common;
    common.allocator_ = allocator;
    common.dim_ = 7;
    common.metric_ = GENERATE(
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_IP, MetricType::METRIC_TYPE_COSINE);
    common.data_type_ = DataTypes::DATA_TYPE_FLOAT;
    auto p = std::make_shared<BucketDataCellParameter>();
    p->FromJson(JsonType::Parse(
        R"({"io_params":{"type":"memory_io"},"quantization_params":{"type":"sq8"},"buckets_count":1,"use_residual":true})"));
    auto make = [&] {
        auto b = BucketInterface::MakeInstance(p, common);
        b->SetStrategy(std::make_shared<FixedCentroidPartitionStrategy>(common, 1));
        return b;
    };
    auto a = make(), b = make();
    std::vector<float> data(17 * 7);
    for (size_t i = 0; i < data.size(); ++i) data[i] = std::sin(float(i) * .21f);
    TrainingRows rows(data.data(), 17, 7, DataTypes::DATA_TYPE_FLOAT, 28, data.size() * 4);
    // Callback route forces provider residual composition rather than contiguous fastpath.
    TrainingRows provider(
        17,
        7,
        [](const void* c, uint64_t i, float* out) {
            static_cast<const TrainingRows*>(c)->Read(i, out);
        },
        &rows);
    a->Train(data.data(), 17);
    b->TrainRows(provider);
    std::ostringstream oa, ob;
    IOStreamWriter wa(oa), wb(ob);
    a->Serialize(wa);
    b->Serialize(wb);
    REQUIRE(oa.str() == ob.str());
    bool read = false;
    TrainingRows huge(
        uint64_t(PTRDIFF_MAX),
        7,
        [](const void* c, uint64_t, float*) { *static_cast<bool*>(const_cast<void*>(c)) = true; },
        &read);
    auto fresh = make();
    REQUIRE_THROWS(fresh->TrainRows(huge));
    REQUIRE_FALSE(read);
    REQUIRE(fresh->SupportsTrainingRows());
    fresh->TrainRows(provider);
    std::ostringstream oc;
    IOStreamWriter wc(oc);
    fresh->Serialize(wc);
    REQUIRE(oc.str() == ob.str());
}
