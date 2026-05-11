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

#pragma once

#include <string>
#include <vector>

#include "basic_types.h"
#include "common.h"
#include "vsag/index.h"
#include "datacell/flatten_interface.h"
#include "datacell/graph_interface.h"
#include "vsag/attribute.h"

namespace vsag {

class HGraph;
class Dataset;
using DatasetPtr = std::shared_ptr<Dataset>;


class HGraphModifier {
public:
    static uint32_t
    Remove(HGraph& graph, const std::vector<int64_t>& ids, RemoveMode mode);

    static void
    UpdateAttribute(HGraph& graph, int64_t id, const AttributeSet& new_attrs);

    static void
    UpdateAttribute(HGraph& graph, int64_t id, const AttributeSet& new_attrs, const AttributeSet& origin_attrs);

    static bool
    UpdateVector(HGraph& graph, int64_t id, const DatasetPtr& new_base, bool force_update);

    static void
    RecoverRemove(HGraph& graph, int64_t id);

    static bool
    TryRecoverTombstone(HGraph& graph, const DatasetPtr& data, std::vector<int64_t>& failed_ids);

private:
    static uint32_t
    ForceRemoveOne(HGraph& graph, int64_t label);

    static void
    FindNewEntryPoint(HGraph& graph);

    static void
    GraphForceRemoveOne(HGraph& graph,
                         const InnerIdType& inner_id,
                         const FlattenInterfacePtr& flatten,
                         const GraphInterfacePtr& graph_ptr);

    static void
    MoveId(HGraph& graph, InnerIdType from, InnerIdType to);

    static void
    ShrinkToFit(HGraph& graph);
};

}  // namespace vsag
