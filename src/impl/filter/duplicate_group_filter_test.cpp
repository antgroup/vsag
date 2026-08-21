// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "duplicate_group_filter.h"

#include "datacell/graph_datacell_parameter.h"
#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
#include "io/memory_io/memory_io_parameter.h"
#include "unittest.h"

namespace vsag {
namespace {

class SingleIdFilter final : public Filter {
public:
    explicit SingleIdFilter(int64_t valid_id) : valid_id_(valid_id) {
    }

    [[nodiscard]] bool
    CheckValid(int64_t id) const override {
        return id == valid_id_;
    }

    [[nodiscard]] float
    ValidRatio() const override {
        return 0.0F;
    }

private:
    int64_t valid_id_;
};

}  // namespace

TEST_CASE("DuplicateGroupFilter keeps representatives traversable for alias-only filters",
          "[ut][DuplicateGroupFilter]") {
    IndexCommonParam common_param;
    common_param.dim_ = 32;
    common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();

    auto graph_param = std::make_shared<GraphDataCellParameter>();
    graph_param->io_parameter_ = std::make_shared<MemoryIOParameter>();
    graph_param->support_duplicate_ = true;
    auto graph = GraphInterface::MakeInstance(graph_param, common_param);
    graph->Resize(4);
    graph->SetDuplicateId(0, 1);
    graph->SetDuplicateId(0, 2);

    FilterPtr alias_only = std::make_shared<SingleIdFilter>(2);
    const auto group_filter = MakeDuplicateGroupFilter(alias_only, graph, true);
    REQUIRE(group_filter != alias_only);
    REQUIRE(group_filter->ValidRatio() == 0.0F);
    REQUIRE(group_filter->CheckValid(int64_t{0}));
    REQUIRE(group_filter->CheckValid(int64_t{1}));
    REQUIRE(group_filter->CheckValid(int64_t{2}));
    REQUIRE_FALSE(group_filter->CheckValid(int64_t{3}));

    REQUIRE(MakeDuplicateGroupFilter(alias_only, graph, false) == alias_only);
    REQUIRE(MakeDuplicateGroupFilter(nullptr, graph, true) == nullptr);
}

}  // namespace vsag
