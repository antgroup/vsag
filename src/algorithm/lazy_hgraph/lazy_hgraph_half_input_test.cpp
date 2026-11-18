// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include "algorithm/bruteforce/half_dense_test_fixture.h"

TEST_CASE("LazyHGraph native half flat and graph lifecycle", "[ut][lazy_hgraph][half_input]") {
    for (bool bf16 : {false, true}) {
        for (auto metric : {vsag::MetricType::METRIC_TYPE_L2SQR,
                            vsag::MetricType::METRIC_TYPE_IP,
                            vsag::MetricType::METRIC_TYPE_COSINE}) {
            for (uint64_t threshold : {4, 100}) {
                CAPTURE(bf16, metric, threshold);
                half_dense_test::Exercise(true, bf16, metric, threshold);
            }
        }
    }
}
