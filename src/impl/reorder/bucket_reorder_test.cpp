// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "bucket_reorder.h"

#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "datacell/bucket_datacell_parameter.h"
#include "impl/filter/iterator_filter.h"
#include "impl/heap/standard_heap.h"
#include "index_common_param.h"
#include "io/memory_io/memory_io_parameter.h"
#include "quantization/fp32_quantizer_parameter.h"
#include "query_context.h"
#include "unittest.h"
#include "vsag/engine.h"
#include "vsag_exception.h"

namespace vsag {

TEST_CASE("BucketReorder reranks bucket locations", "[ut][reorder][bucket]") {
    auto allocator = Engine::CreateDefaultAllocator();
    auto bucket_param = std::make_shared<BucketDataCellParameter>();
    bucket_param->quantizer_parameter = std::make_shared<FP32QuantizerParameter>();
    bucket_param->io_parameter = std::make_shared<MemoryIOParameter>();
    bucket_param->buckets_count = 2;

    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common_param.dim_ = 1;
    auto bucket = BucketInterface::MakeInstance(bucket_param, common_param);

    float vectors[] = {std::numeric_limits<float>::quiet_NaN(), 1.0F, 2.0F, 3.0F};
    bucket->Train(vectors, 4);
    bucket->InsertVector(vectors, 0, 0);
    bucket->InsertVector(vectors + 1, 1, 1);
    bucket->InsertVector(vectors + 2, 0, 2);
    bucket->InsertVector(vectors + 3, 1, 3);

    std::vector<std::pair<BucketIdType, InnerIdType>> locations{{0, 0}, {1, 0}, {0, 1}, {1, 1}};
    std::vector<InnerIdType> resolved_ids;
    BucketReorder reorder(
        bucket,
        [&locations, &resolved_ids](InnerIdType inner_id) {
            resolved_ids.emplace_back(inner_id);
            return locations[inner_id];
        },
        allocator.get());

    auto candidates = std::make_shared<StandardHeap<true, false>>(allocator.get(), -1);
    candidates->Push(20.0F, 2);
    candidates->Push(40.0F, 0);
    candidates->Push(10.0F, 3);
    candidates->Push(30.0F, 1);
    std::vector<InnerIdType> candidate_order;
    for (uint64_t i = 0; i < candidates->Size(); ++i) {
        candidate_order.emplace_back(candidates->GetData()[i].second);
    }

    float query = 0.0F;
    SearchStatistics stats;
    QueryContext ctx{.alloc = allocator.get(), .stats = &stats};
    auto result =
        reorder.Reorder(candidates, &query, 2, ctx, nullptr, nullptr, std::optional<float>{4.0F});

    REQUIRE(resolved_ids == candidate_order);
    REQUIRE(stats.reorder_distance_count.load(std::memory_order_relaxed) == candidates->Size());
    REQUIRE(result->Size() == 2);
    REQUIRE(result->Top().second == 2);
    REQUIRE(result->Top().first == 4.0F);
    result->Pop();
    REQUIRE(result->Top().second == 1);
    REQUIRE(result->Top().first == 1.0F);

    SECTION("resolver observes updated locations") {
        locations[3] = locations[2];
        auto remapped_candidate = std::make_shared<StandardHeap<true, false>>(allocator.get(), -1);
        remapped_candidate->Push(1.0F, 3);
        QueryContext remapped_ctx{.alloc = allocator.get()};
        auto remapped = reorder.Reorder(remapped_candidate, &query, 1, remapped_ctx);
        REQUIRE(remapped->Size() == 1);
        REQUIRE(remapped->Top().second == 3);
        REQUIRE(remapped->Top().first == 4.0F);
    }

    SECTION("unsupported optional inputs are rejected") {
        auto require_unsupported = [](const auto& operation) {
            try {
                operation();
                FAIL("unsupported BucketReorder input should be rejected");
            } catch (const VsagException& error) {
                REQUIRE(error.error_.type == ErrorType::UNSUPPORTED_INDEX_OPERATION);
            }
        };

        IteratorFilterContext iter_ctx;
        QueryContext iter_query_ctx{.alloc = allocator.get()};
        require_unsupported(
            [&]() { (void)reorder.Reorder(candidates, &query, 1, iter_query_ctx, &iter_ctx); });

        DistanceRecordVector lower_bound_candidates(allocator.get());
        lower_bound_candidates.emplace_back(0.0F, 0);
        QueryContext lower_bound_query_ctx{.alloc = allocator.get()};
        require_unsupported([&]() {
            (void)reorder.Reorder(
                candidates, &query, 1, lower_bound_query_ctx, nullptr, &lower_bound_candidates);
        });
    }
}

}  // namespace vsag
