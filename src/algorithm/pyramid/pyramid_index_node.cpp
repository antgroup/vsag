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

#include "pyramid_index_node.h"

#include <fmt/format.h>

#include "datacell/graph_datacell_parameter.h"
#include "datacell/sparse_graph_datacell.h"
#include "impl/reasoning/reasoning_context.h"
#include "storage/serialization.h"
#include "vsag/constants.h"

namespace vsag {

static constexpr uint64_t MAX_ROOT_ROUTE_GRAPH_COUNT = 1024;

static inline uint64_t
get_suitable_max_degree(int64_t data_num) {
    if (data_num < 100'000) {
        return 24;
    }
    if (data_num < 1000'000) {
        return 32;
    }
    return 64;
}

IndexNode::IndexNode(Allocator* allocator,
                     GraphInterfaceParamPtr graph_param,
                     uint32_t index_min_size,
                     const IndexCommonParam& common_param,
                     GraphInterfaceParamPtr child_graph_param)
    : ids_(allocator),
      index_min_size_(index_min_size),
      children_(allocator),
      allocator_(allocator),
      common_param_(common_param),
      graph_param_(std::move(graph_param)),
      child_graph_param_(std::move(child_graph_param)) {
}

void
IndexNode::enable_routing(GraphInterfaceParamPtr graph_param) {
    routing_ = std::make_unique<RoutingOverlay>(allocator_, std::move(graph_param));
}

GraphInterfacePtr
IndexNode::make_route_graph() const {
    return std::make_shared<SparseGraphDataCell>(
        std::dynamic_pointer_cast<SparseGraphDatacellParameter>(routing_->graph_param), allocator_);
}

void
IndexNode::Build(ODescent& odescent) {
    std::unique_lock lock(mutex_);
    // Build an index when the level corresponding to the current node requires indexing
    if (not ids_.empty()) {
        Init();
    }
    if (status_ == Status::GRAPH) {
        entry_point_ = ids_[0];
        odescent.SetMaxDegree(static_cast<int32_t>(graph_param_->max_degree_));
        odescent.Build(ids_);
        odescent.SaveGraph(graph_);
        Vector<InnerIdType>(allocator_).swap(ids_);
    }
    for (const auto& item : children_) {
        item.second->Build(odescent);
    }
}

void
IndexNode::AddChild(const std::string& key) {
    // AddChild is not thread-safe; ensure thread safety in calls to it.
    children_[key] = std::make_unique<IndexNode>(
        allocator_, child_graph_param_, index_min_size_, common_param_, child_graph_param_);
    children_[key]->level_ = level_ + 1;
}

IndexNode*
IndexNode::GetChild(const std::string& key, bool need_init) {
    std::unique_lock lock(mutex_);
    auto result = children_.find(key);
    if (result != children_.end()) {
        return result->second.get();
    }
    if (not need_init) {
        return nullptr;
    }
    AddChild(key);
    return children_[key].get();
}

void
IndexNode::Deserialize(StreamReader& reader) {
    // deserialize `entry_point_`
    StreamReader::ReadObj(reader, entry_point_);
    // deserialize `level_`
    StreamReader::ReadObj(reader, level_);
    // deserialize `status_`
    StreamReader::ReadObj(reader, status_);
    if (status_ == Status::GRAPH) {
        graph_ = GraphInterface::MakeInstance(graph_param_, common_param_);
        graph_->Deserialize(reader);
        if (graph_param_->graph_storage_type_ == GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_FLAT ||
            graph_param_->graph_storage_type_ ==
                GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_COMPRESSED) {
            graph_capacity_ = graph_->MaxCapacity();
        }
    } else if (status_ == Status::FLAT) {
        StreamReader::ReadVector(reader, ids_);
    }
    // deserialize `children`
    uint64_t children_size = 0;
    StreamReader::ReadObj(reader, children_size);
    for (uint64_t i = 0; i < children_size; ++i) {
        std::string key = StreamReader::ReadString(reader);
        AddChild(key);
        children_[key]->Deserialize(reader);
    }
    deserialize_routing_unlocked(reader);
}

void
IndexNode::Serialize(StreamWriter& writer) const {
    std::unique_lock lock(mutex_);
    // serialize `entry_point_`
    StreamWriter::WriteObj(writer, entry_point_);
    // serialize `level_`
    StreamWriter::WriteObj(writer, level_);
    // serialize `status_`
    StreamWriter::WriteObj(writer, status_);
    if (status_ == Status::GRAPH) {
        graph_->Serialize(writer);
    } else if (status_ == Status::FLAT) {
        StreamWriter::WriteVector(writer, ids_);
    }
    // serialize `children`
    uint64_t children_size = children_.size();
    StreamWriter::WriteObj(writer, children_size);
    for (const auto& item : children_) {
        // calculate size of `key`
        StreamWriter::WriteString(writer, item.first);
        // calculate size of `content`
        item.second->Serialize(writer);
    }
    serialize_routing_unlocked(writer);
}

void
IndexNode::serialize_routing_unlocked(StreamWriter& writer) const {
    if (not has_routing()) {
        return;
    }
    StreamWriter::WriteObj(writer, static_cast<uint64_t>(routing_->graphs.size()));
    for (const auto& graph : routing_->graphs) {
        graph->Serialize(writer);
    }
}

void
IndexNode::deserialize_routing_unlocked(StreamReader& reader) {
    if (not has_routing()) {
        return;
    }
    uint64_t route_count = 0;
    StreamReader::ReadObj(reader, route_count);
    CHECK_ARGUMENT(route_count <= MAX_ROOT_ROUTE_GRAPH_COUNT,
                   fmt::format("invalid root route graph count: {}", route_count));
    routing_->graphs.clear();
    routing_->graphs.reserve(route_count);
    for (uint64_t i = 0; i < route_count; ++i) {
        auto graph = make_route_graph();
        graph->Deserialize(reader);
        routing_->graphs.push_back(std::move(graph));
    }
}

std::pair<uint64_t, uint64_t>
IndexNode::get_memory_usage_detail() const {
    uint64_t memory = sizeof(IndexNode);
    uint64_t routing_memory = 0;
    std::shared_lock lock(mutex_);
    if (has_routing()) {
        routing_memory +=
            sizeof(RoutingOverlay) + routing_->graphs.capacity() * sizeof(GraphInterfacePtr);
        for (const auto& graph : routing_->graphs) {
            routing_memory += graph->GetMemoryUsage();
        }
        memory += routing_memory;
    }
    memory += ids_.capacity() * sizeof(InnerIdType);
    memory +=
        children_.bucket_count() * (sizeof(decltype(children_)::value_type) + sizeof(uint32_t));
    for (const auto& [key, child] : children_) {
        memory += key.capacity() + 1;
        const auto [child_memory, child_routing_memory] = child->get_memory_usage_detail();
        memory += child_memory;
        routing_memory += child_routing_memory;
    }
    if (graph_ != nullptr) {
        memory += graph_->GetMemoryUsage();
    }
    return {memory, routing_memory};
}

JsonType
IndexNode::get_graph_stats() const {
    std::shared_lock lock(mutex_);
    JsonType stats;
    stats[PYRAMID_ROOT_GRAPH_TYPE].SetString(has_routing() ? PYRAMID_ROOT_GRAPH_TYPE_MULTI_LAYER
                                                           : PYRAMID_ROOT_GRAPH_TYPE_SINGLE_LAYER);
    const char* storage_type = "sparse";
    if (graph_param_->graph_storage_type_ == GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_FLAT) {
        storage_type = GRAPH_STORAGE_TYPE_VALUE_FLAT;
    } else if (graph_param_->graph_storage_type_ ==
               GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_COMPRESSED) {
        storage_type = GRAPH_STORAGE_TYPE_VALUE_COMPRESSED;
    }
    stats["bottom_graph_storage_type"].SetString(storage_type);
    stats["bottom_graph_node_count"].SetUint64(graph_ == nullptr ? 0 : graph_->TotalCount());
    stats["bottom_graph_size"].SetUint64(graph_ == nullptr ? 0 : graph_->GetMemoryUsage());

    const auto route_graph_count = has_routing() ? routing_->graphs.size() : 0;
    stats["route_graph_count"].SetUint64(route_graph_count);
    std::vector<int32_t> route_node_counts;
    route_node_counts.reserve(route_graph_count);
    uint64_t route_graph_size = 0;
    if (has_routing()) {
        for (const auto& graph : routing_->graphs) {
            route_node_counts.push_back(static_cast<int32_t>(graph->TotalCount()));
            route_graph_size += graph->GetMemoryUsage();
        }
    }
    stats["route_node_counts"].SetVector(route_node_counts);
    stats["route_graph_size"].SetUint64(route_graph_size);
    return stats;
}

Vector<InnerIdType>
IndexNode::get_ids_unlocked() const {
    if (status_ == Status::FLAT) {
        return ids_;
    }
    Vector<InnerIdType> ids(allocator_);
    if (graph_ == nullptr) {
        return ids;
    }
    if (graph_param_->graph_storage_type_ != GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_FLAT) {
        return graph_->GetIds();
    }
    const auto total_count = graph_->TotalCount();
    ids.reserve(total_count);
    for (InnerIdType id = 0; id < total_count; ++id) {
        if (graph_->CheckIdExists(id)) {
            ids.push_back(id);
        }
    }
    return ids;
}

void
IndexNode::resize_graph(InnerIdType new_capacity) {
    std::unique_lock lock(mutex_);
    if (graph_param_->graph_storage_type_ != GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_FLAT &&
        graph_param_->graph_storage_type_ !=
            GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_COMPRESSED) {
        return;
    }
    graph_capacity_ = std::max(graph_capacity_, new_capacity);
    auto flat_param = std::dynamic_pointer_cast<GraphDataCellParameter>(graph_param_);
    if (flat_param != nullptr) {
        flat_param->init_max_capacity_ = static_cast<uint64_t>(graph_capacity_);
    }
    if (graph_ != nullptr) {
        graph_->Resize(graph_capacity_);
    }
}

void
IndexNode::Init() {
    if (status_ == Status::NO_INDEX) {
        if (has_routing() || ids_.size() >= index_min_size_) {
            if (not ids_.empty() and level_ != 0) {
                auto new_max_degree = get_suitable_max_degree(static_cast<int64_t>(ids_.size()));
                if (new_max_degree < graph_param_->max_degree_) {
                    auto new_graph_param = std::make_shared<SparseGraphDatacellParameter>();
                    new_graph_param->FromJson(graph_param_->ToJson());
                    new_graph_param->max_degree_ =
                        get_suitable_max_degree(static_cast<int64_t>(ids_.size()));
                    graph_param_ = new_graph_param;
                }
            }
            graph_ = GraphInterface::MakeInstance(graph_param_, common_param_);
            if (graph_capacity_ == 0 && graph_param_->graph_storage_type_ ==
                                            GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_FLAT) {
                const auto flat_param =
                    std::static_pointer_cast<GraphDataCellParameter>(graph_param_);
                graph_capacity_ = static_cast<InnerIdType>(flat_param->init_max_capacity_);
            }
            if (graph_capacity_ > 0) {
                graph_->Resize(graph_capacity_);
            }
            status_ = Status::GRAPH;
        } else {
            status_ = Status::FLAT;
        }
    }
}

void
IndexNode::Search(const SearchFunc& search_func,
                  const VisitedListPtr& vl,
                  const DistHeapPtr& search_result,
                  uint64_t ef_search,
                  ReasoningContext* reasoning_ctx) const {
    bool has_index = false;
    {
        std::shared_lock lock(mutex_);
        has_index = status_ != IndexNode::Status::NO_INDEX;
    }
    if (has_index) {
        auto self_search_result = search_func(this, vl);
        search_result->Merge(*self_search_result);
        while (search_result->Size() > ef_search) {
            if (reasoning_ctx != nullptr) {
                // [reasoning] RecordEviction
                reasoning_ctx->RecordEviction(search_result->Top().second, level_);
            }
            search_result->Pop();
        }
        return;
    }

    Vector<IndexNode*> children(allocator_);
    {
        std::shared_lock lock(mutex_);
        children.reserve(children_.size());
        for (const auto& [key, node] : children_) {
            children.push_back(node.get());
        }
    }
    for (const auto* node : children) {
        node->Search(search_func, vl, search_result, ef_search, reasoning_ctx);
    }
}

}  // namespace vsag
