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

#include "flatten_reorder.h"

#include <limits>
#include <memory>
#include <optional>

#include "datacell/flatten_datacell_parameter.h"
#include "impl/filter/iterator_filter.h"
#include "impl/heap/standard_heap.h"
#include "index_common_param.h"
#include "io/memory_io/memory_io_parameter.h"
#include "quantization/fp32_quantizer_parameter.h"
#include "query_context.h"
#include "unittest.h"
#include "vsag/engine.h"

namespace vsag {

TEST_CASE("FlattenReorder filters non-finite exact distances before top-k",
          "[ut][reorder][threshold][nonfinite]") {
    auto allocator = Engine::CreateDefaultAllocator();
    auto flatten_param = std::make_shared<FlattenDataCellParameter>();
    flatten_param->quantizer_parameter = std::make_shared<FP32QuantizerParameter>();
    flatten_param->io_parameter = std::make_shared<MemoryIOParameter>();

    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common_param.dim_ = 1;
    auto flatten = FlattenInterface::MakeInstance(flatten_param, common_param);

    float vectors[] = {std::numeric_limits<float>::quiet_NaN(), 1.0F, 0.0F};
    flatten->Train(vectors, 3);
    flatten->BatchInsertVector(vectors, 3);

    auto candidates = std::make_shared<StandardHeap<true, false>>(allocator.get(), -1);
    candidates->Push(100.0F, 0);
    candidates->Push(1.0F, 1);
    candidates->Push(2.0F, 2);

    float query = 1.0F;
    QueryContext ctx{.alloc = allocator.get()};
    IteratorFilterContext iter_ctx;
    REQUIRE(iter_ctx.init(3, 3, allocator.get()).has_value());
    FlattenReorder reorder(flatten, allocator.get());
    auto result =
        reorder.Reorder(candidates, &query, 1, ctx, &iter_ctx, nullptr, std::optional<float>{0.0F});

    REQUIRE(result->Size() == 1);
    REQUIRE(result->Top().second == 1);
    REQUIRE(result->Top().first == 0.0F);
    REQUIRE_FALSE(iter_ctx.CheckPoint(0));
}

}  // namespace vsag
