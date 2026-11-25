#include <stdexcept>

#include "impl/allocator/safe_allocator.h"
#include "quantization/fp32_quantizer.h"
#include "quantization/scalar_quantization/half_precision_quantizer.h"
#include "unittest.h"

using namespace vsag;

TEST_CASE("Shape-only provider never reads values", "[ut][training_rows][shape_only]") {
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
    FP16Quantizer<MetricType::METRIC_TYPE_L2SQR> h(7, a.get());
    BF16Quantizer<MetricType::METRIC_TYPE_L2SQR> b(7, a.get());
    REQUIRE(h.TrainRows(rows));
    REQUIRE(h.ReTrainRows(rows));
    REQUIRE(b.TrainRows(rows));
    REQUIRE(b.ReTrainRows(rows));
}
