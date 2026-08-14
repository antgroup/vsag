
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

#include "pyramid.h"

#include <fmt/format.h>

#include <chrono>
#include <exception>
#include <limits>

#include "algorithm/inner_index_interface.h"
#include "analyzer/analyzer.h"
#include "datacell/flatten_interface.h"
#include "impl/distance_provider_for_graph.h"
#include "impl/heap/standard_heap.h"
#include "impl/odescent/odescent_graph_builder.h"
#include "impl/pruning_strategy.h"
#include "query_context.h"
#include "storage/empty_index_binary_set.h"
#include "storage/serialization.h"
#include "storage/serialization_tags.h"
#include "storage/tlv_section.h"
#include "utils/search_threshold.h"
#include "utils/slow_task_timer.h"
#include "utils/util_functions.h"
namespace vsag {

const static float RADIUS_EPSILON = 1.1F;
static constexpr uint64_t SOURCE_ID_TABLE_MAGIC = 0x534F555243454944ULL;  // SOURCEID

std::vector<std::string>
split(const std::string& str, char delimiter) {
    auto vec = split_string(str, delimiter);
    vec.erase(
        std::remove_if(vec.begin(), vec.end(), [](const std::string& s) { return s.empty(); }),
        vec.end());
    return vec;
}

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

static inline uint64_t
get_suitable_ef_search(int64_t topk, int64_t data_num, uint64_t subindex_ef_search = 50) {
    auto topk_float = static_cast<float>(topk);
    if (data_num < 1'000) {
        return std::max(static_cast<uint64_t>(1.5F * topk_float), subindex_ef_search);
    }
    if (data_num < 100'000) {
        return std::max(static_cast<uint64_t>(2.0F * topk_float), subindex_ef_search * 2);
    }
    if (data_num < 1'000'000) {
        return std::max(static_cast<uint64_t>(3.0F * topk_float), subindex_ef_search * 4);
    }
    return std::max(static_cast<uint64_t>(4.0F * topk_float), subindex_ef_search * 8);
}

IndexNode::IndexNode(Allocator* allocator,
                     GraphInterfaceParamPtr graph_param,
                     uint32_t index_min_size)
    : ids_(allocator),
      children_(allocator),
      allocator_(allocator),
      graph_param_(std::move(graph_param)),
      index_min_size_(index_min_size) {
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
    children_[key] = std::make_unique<IndexNode>(allocator_, graph_param_, index_min_size_);
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
        graph_ = std::make_shared<SparseGraphDataCell>(
            std::dynamic_pointer_cast<SparseGraphDatacellParameter>(graph_param_), allocator_);
        graph_->Deserialize(reader);
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
}

void
IndexNode::Serialize(StreamWriter& writer) const {
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
}
void
IndexNode::Init() {
    if (status_ == Status::NO_INDEX) {
        if (ids_.size() >= index_min_size_) {
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
            graph_ = std::make_shared<SparseGraphDataCell>(
                std::dynamic_pointer_cast<SparseGraphDatacellParameter>(graph_param_), allocator_);
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
                  uint64_t ef_search) const {
    bool has_index = false;
    {
        std::shared_lock lock(mutex_);
        has_index = status_ != IndexNode::Status::NO_INDEX;
    }
    if (has_index) {
        auto self_search_result = search_func(this, vl);
        search_result->Merge(*self_search_result);
        while (search_result->Size() > ef_search) {
            search_result->Pop();
        }
        return;
    }

    for (const auto& [key, node] : children_) {
        node->Search(search_func, vl, search_result, ef_search);
    }
}

std::vector<int64_t>
Pyramid::build_by_odescent(const DatasetPtr& base) {
    int64_t data_num = base->GetNumElements();
    const auto* data_vectors = base->GetFloat32Vectors();
    const auto* data_ids = base->GetIds();
    const auto* source_ids = base->GetSourceID();

    resize(data_num);
    for (InnerIdType inner_id = 0; inner_id < data_num; ++inner_id) {
        label_table_->Insert(inner_id, data_ids[inner_id]);
        if (source_ids != nullptr) {
            label_table_->InsertSourceId(inner_id, source_ids[inner_id]);
        }
    }

    base_codes_->BatchInsertVector(data_vectors, data_num);
    if (has_precise_reorder()) {
        precise_codes_->BatchInsertVector(data_vectors, data_num);
    }
    if (raw_vector_ != nullptr) {
        raw_vector_->BatchInsertVector(data_vectors, data_num);
    }
    auto codes = has_precise_reorder() ? precise_codes_ : base_codes_;

    if (thread_pool_ != nullptr && hierarchies_.size() > 1) {
        auto build_flatten = ODescent::CreateBuildFlatten(codes, data_vectors, data_num);
        Vector<std::future<void>> futures(allocator_);
        for (const auto& [hname, h_ptr] : hierarchies_) {
            auto* root_ptr = h_ptr->root.get();
            futures.push_back(thread_pool_->GeneralEnqueue([&, codes, build_flatten, root_ptr]() {
                ODescent builder(odescent_param_,
                                 codes,
                                 allocator_,
                                 nullptr,
                                 true,
                                 data_vectors,
                                 data_num,
                                 build_flatten);
                root_ptr->Build(builder);
            }));
        }
        for (auto& f : futures) {
            f.get();
        }
    } else {
        ODescent graph_builder(odescent_param_,
                               codes,
                               allocator_,
                               this->thread_pool_.get(),
                               true,
                               data_vectors,
                               data_num);
        for (const auto& [hname, h_ptr] : hierarchies_) {
            h_ptr->root->Build(graph_builder);
        }
    }
    cur_element_count_ = data_num;
    return {};
}

DatasetPtr
Pyramid::KnnSearch(const DatasetPtr& query,
                   int64_t k,
                   const std::string& parameters,
                   const FilterPtr& filter) const {
    SearchStatistics stats;
    QueryContext ctx{.stats = &stats};

    const auto threshold = ParseSearchThreshold(parameters);
    auto parsed_param = PyramidSearchParameters::FromJson(parameters);
    ctx.rabitq_error_rate = parsed_param.rabitq_error_rate;
    CHECK_ARGUMENT(k > 0, fmt::format("k({}) must be greater than 0", k));
    CHECK_ARGUMENT(parsed_param.hierarchy_op == PyramidSearchParameters::HierarchyOp::SINGLE,
                   "multi-hierarchy search (union/intersection) is not yet implemented");
    auto ef_search_threshold = std::max<uint64_t>(AMPLIFICATION_FACTOR * k, 1000L);
    CHECK_ARGUMENT(  // NOLINT
        (1 <= parsed_param.ef_search) and (parsed_param.ef_search <= ef_search_threshold),
        fmt::format(
            "ef_search({}) must in range[1, {}]", parsed_param.ef_search, ef_search_threshold));

    InnerSearchParam search_param;
    search_param.ef = std::max<uint64_t>(parsed_param.ef_search, static_cast<uint64_t>(k));
    search_param.radius = std::numeric_limits<float>::max();
    search_param.topk = threshold.has_value() ? static_cast<int64_t>(search_param.ef) : k;
    search_param.distance_threshold = threshold;
    search_param.enable_reorder = use_reorder_;
    search_param.search_mode = KNN_SEARCH;
    search_param.parallel_search_thread_count = parsed_param.parallel_search_thread_count;
    search_param.enable_rabitq_one_bit_search = parsed_param.has_rabitq_one_bit_search
                                                    ? parsed_param.rabitq_one_bit_search
                                                    : default_rabitq_one_bit_search_;

    // Keep the same contract as HGraph: a useful hop cap must exceed ef_search.
    if (static_cast<uint64_t>(parsed_param.hops_limit) <= parsed_param.ef_search) {
        search_param.hops_limit = std::numeric_limits<uint32_t>::max();
        if (parsed_param.hops_limit != std::numeric_limits<uint32_t>::max()) {
            logger::warn(
                fmt::format("hops_limit({}) is not greater than ef_search({}), ignoring hops_limit",
                            parsed_param.hops_limit,
                            parsed_param.ef_search));
        }
    } else {
        search_param.hops_limit = parsed_param.hops_limit;
    }
    if (this->support_duplicate_) {
        search_param.consider_duplicate = true;
    }

    if (parsed_param.enable_time_record) {
        search_param.time_cost = std::make_shared<Timer>();
        search_param.time_cost->SetThreshold(parsed_param.timeout_ms);
    }

    search_param.is_inner_id_allowed = this->create_search_filter(filter);
    const bool collect_rabitq_lower_bounds = search_param.enable_rabitq_one_bit_search and
                                             use_reorder_ and
                                             base_codes_->SupportSplitCodeStorage();
    DistanceRecordVector rabitq_lower_bound_candidates(allocator_);
    std::mutex rabitq_lower_bound_mutex;
    SearchFunc search_func = [&](const IndexNode* node, const VisitedListPtr& vl) {
        DistanceRecordVector local_candidates(allocator_);
        auto* candidates = collect_rabitq_lower_bounds ? &local_candidates : nullptr;
        auto result = this->search_node(node,
                                        vl,
                                        search_param,
                                        query,
                                        base_codes_,
                                        ctx,
                                        parsed_param.subindex_ef_search,
                                        candidates);
        if (candidates != nullptr and not candidates->empty()) {
            std::lock_guard lock(rabitq_lower_bound_mutex);
            rabitq_lower_bound_candidates.insert(
                rabitq_lower_bound_candidates.end(), candidates->begin(), candidates->end());
        }
        return result;
    };

    std::string hierarchy_name =
        parsed_param.hierarchies.empty() ? "" : parsed_param.hierarchies[0];
    auto result =
        this->search_impl(query,
                          search_func,
                          search_param,
                          ctx,
                          hierarchy_name,
                          collect_rabitq_lower_bounds ? &rabitq_lower_bound_candidates : nullptr);
    result->Statistics(stats.Dump());
    return FilterDatasetByThreshold(result, threshold, allocator_, k);
}

DatasetPtr
Pyramid::RangeSearch(const DatasetPtr& query,
                     float radius,
                     const std::string& parameters,
                     const FilterPtr& filter,
                     int64_t limited_size) const {
    CHECK_ARGUMENT(radius >= 0.0F, "radius must be non-negative");

    SearchStatistics stats;
    QueryContext ctx{.stats = &stats};

    auto parsed_param = PyramidSearchParameters::FromJson(parameters);
    ctx.rabitq_error_rate = parsed_param.rabitq_error_rate;
    CHECK_ARGUMENT(parsed_param.hierarchy_op == PyramidSearchParameters::HierarchyOp::SINGLE,
                   "multi-hierarchy search (union/intersection) is not yet implemented");
    InnerSearchParam search_param;
    search_param.ef = parsed_param.ef_search;
    search_param.radius = radius * RADIUS_EPSILON;
    search_param.search_mode = RANGE_SEARCH;
    search_param.parallel_search_thread_count = parsed_param.parallel_search_thread_count;
    search_param.enable_rabitq_one_bit_search = parsed_param.has_rabitq_one_bit_search
                                                    ? parsed_param.rabitq_one_bit_search
                                                    : default_rabitq_one_bit_search_;
    search_param.topk = limited_size == -1 ? std::numeric_limits<int64_t>::max() : limited_size;

    if (parsed_param.enable_time_record) {
        search_param.time_cost = std::make_shared<Timer>();
        search_param.time_cost->SetThreshold(parsed_param.timeout_ms);
    }

    if (this->support_duplicate_) {
        search_param.consider_duplicate = true;
    }

    search_param.is_inner_id_allowed = this->create_search_filter(filter);
    const bool collect_rabitq_lower_bounds = search_param.enable_rabitq_one_bit_search and
                                             use_reorder_ and
                                             base_codes_->SupportSplitCodeStorage();
    DistanceRecordVector rabitq_lower_bound_candidates(allocator_);
    std::mutex rabitq_lower_bound_mutex;
    SearchFunc search_func = [&](const IndexNode* node, const VisitedListPtr& vl) {
        DistanceRecordVector local_candidates(allocator_);
        auto* candidates = collect_rabitq_lower_bounds ? &local_candidates : nullptr;
        auto result = this->search_node(node,
                                        vl,
                                        search_param,
                                        query,
                                        base_codes_,
                                        ctx,
                                        parsed_param.subindex_ef_search,
                                        candidates);
        if (candidates != nullptr and not candidates->empty()) {
            std::lock_guard lock(rabitq_lower_bound_mutex);
            rabitq_lower_bound_candidates.insert(
                rabitq_lower_bound_candidates.end(), candidates->begin(), candidates->end());
        }
        return result;
    };

    std::string hierarchy_name =
        parsed_param.hierarchies.empty() ? "" : parsed_param.hierarchies[0];
    auto result =
        this->search_impl(query,
                          search_func,
                          search_param,
                          ctx,
                          hierarchy_name,
                          collect_rabitq_lower_bounds ? &rabitq_lower_bound_candidates : nullptr);
    result->Statistics(stats.Dump());
    return result;
}

DatasetPtr
Pyramid::search_impl(const DatasetPtr& query,
                     const SearchFunc& search_func,
                     InnerSearchParam& search_param,
                     QueryContext& ctx,
                     const std::string& hierarchy_name,
                     const DistanceRecordVector* rabitq_lower_bound_candidates) const {
    auto h_iter = hierarchies_.find(hierarchy_name);
    CHECK_ARGUMENT(h_iter != hierarchies_.end(),
                   fmt::format("unknown hierarchy name: '{}'", hierarchy_name));
    const auto& h = *h_iter->second;

    const auto* query_path = query->GetPaths(hierarchy_name);
    if (query_path == nullptr) {
        query_path = query->GetPaths();
    }
    // NOLINTNEXTLINE(readability-simplify-boolean-expr)
    CHECK_ARGUMENT(query_path != nullptr || h.root->status_ != IndexNode::Status::NO_INDEX,
                   "query_path is required when level0 is not built");
    CHECK_ARGUMENT(query->GetFloat32Vectors() != nullptr, "query vectors is required");

    DistHeapPtr search_result = std::make_shared<StandardHeap<true, false>>(allocator_, -1);

    std::shared_lock<std::shared_mutex> lock(resize_mutex_);
    VisitedListGuard vl_guard(pool_.get());
    const VisitedListPtr& vl = vl_guard.get();
    if (query_path != nullptr) {
        const std::string& current_path = query_path[0];
        search_hierarchy(h, search_func, vl, search_result, current_path, search_param);
    } else {
        h.root->Search(search_func, vl, search_result, search_param.ef);
    }

    if (use_reorder_) {
        search_result = this->reorder_->Reorder(search_result,
                                                query->GetFloat32Vectors(),
                                                search_param.topk,
                                                ctx,
                                                nullptr,
                                                rabitq_lower_bound_candidates);
    }

    if (search_result->Empty()) {
        return DatasetImpl::MakeEmptyDataset();
    }

    while (not search_result->Empty() && (search_result->Size() > search_param.topk ||
                                          search_result->Top().first > search_param.radius)) {
        search_result->Pop();
    }

    // return result
    auto result = Dataset::Make();
    auto target_size = static_cast<int64_t>(search_result->Size());
    if (target_size == 0) {
        result->Dim(0)->NumElements(1);
        return result;
    }
    result->Dim(target_size)->NumElements(1)->Owner(true, allocator_);
    auto* ids = static_cast<int64_t*>(allocator_->Allocate(sizeof(int64_t) * target_size));
    result->Ids(ids);
    auto* dists = static_cast<float*>(allocator_->Allocate(sizeof(float) * target_size));
    result->Distances(dists);
    for (int64_t j = target_size - 1; j >= 0; --j) {
        dists[j] = search_result->Top().first;
        ids[j] = label_table_->GetLabelById(search_result->Top().second);

        search_result->Pop();
    }
    return result;
}

int64_t
Pyramid::GetNumElements() const {
    auto total = static_cast<int64_t>(base_codes_->TotalCount());
    auto deleted = delete_count_.load();
    return total > deleted ? total - deleted : 0;
}

int64_t
Pyramid::GetNumberRemoved() const {
    return delete_count_.load();
}

uint32_t
Pyramid::Remove(const std::vector<int64_t>& ids, RemoveMode mode) {
    if (mode != RemoveMode::MARK_REMOVE) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "Pyramid only supports MARK_REMOVE");
    }
    std::scoped_lock lock(this->label_lookup_mutex_, this->cur_element_count_mutex_);
    uint32_t delete_count = this->label_table_->MarkRemove(ids);
    delete_count_.fetch_add(delete_count, std::memory_order_relaxed);
    return delete_count;
}

void
Pyramid::Serialize(StreamWriter& writer) const {
    label_table_->Serialize(writer);
    serialize_source_id_table(writer);
    base_codes_->Serialize(writer);
    if (has_precise_reorder()) {
        precise_codes_->Serialize(writer);
    }
    if (raw_vector_ != nullptr) {
        raw_vector_->Serialize(writer);
    }

    auto pyramid_param = std::dynamic_pointer_cast<PyramidParameters>(create_param_ptr_);
    if (pyramid_param && pyramid_param->has_hierarchies) {
        uint64_t hierarchy_count = hierarchies_.size();
        StreamWriter::WriteObj(writer, hierarchy_count);
        for (const auto& [hname, h_ptr] : hierarchies_) {
            StreamWriter::WriteString(writer, hname);
            h_ptr->root->Serialize(writer);
        }
    } else {
        hierarchies_.at("")->root->Serialize(writer);
    }

    // serialize footer (introduced since v0.15)
    JsonType basic_info;
    basic_info["max_capacity"].SetInt(max_capacity_);
    basic_info[INDEX_PARAM].SetString(this->create_param_ptr_->ToString());
    write_index_footer(writer, basic_info);
}

void
Pyramid::serialize_source_id_table(StreamWriter& writer) const {
    if (not persist_source_id_) {
        return;
    }
    const auto& source_ids = label_table_->GetSourceIdTableRef();
    StreamWriter::WriteObj(writer, SOURCE_ID_TABLE_MAGIC);
    StreamWriter::WriteObj(writer, static_cast<uint64_t>(source_ids.size()));
    for (const auto& source_id : source_ids) {
        StreamWriter::WriteString(writer, source_id);
    }
}

void
Pyramid::deserialize_source_id_table(StreamReader& reader) {
    if (not persist_source_id_) {
        return;
    }
    uint64_t magic = 0;
    StreamReader::ReadObj(reader, magic);
    if (magic != SOURCE_ID_TABLE_MAGIC) {
        throw VsagException(ErrorType::READ_ERROR, "missing Pyramid source_id_table marker");
    }
    uint64_t count = 0;
    StreamReader::ReadObj(reader, count);
    if (count > label_table_->label_table_.size()) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "corrupted Pyramid source_id_table count");
    }
    Vector<std::string> source_ids(count, std::string{}, allocator_);
    for (uint64_t i = 0; i < count; ++i) {
        source_ids[i] = StreamReader::ReadString(reader);
    }
    label_table_->ReplaceSourceIdTable(std::move(source_ids));
}

MetadataPtr
Pyramid::collect_streaming_header() const {
    auto metadata = std::make_shared<Metadata>();
    metadata->Set("format", "vsag_stream_v1");
    metadata->Set("index_name", this->GetName());

    JsonType basic_info;
    basic_info["max_capacity"].SetInt(max_capacity_);
    basic_info["dim"].SetInt(dim_);
    basic_info["metric"].SetInt(static_cast<int64_t>(metric_));
    basic_info["data_type"].SetInt(static_cast<int64_t>(data_type_));
    basic_info["extra_info_size"].SetInt(static_cast<int64_t>(extra_info_size_));
    basic_info[INDEX_PARAM].SetString(this->create_param_ptr_->ToString());
    metadata->Set(BASIC_INFO, basic_info);

    JsonType manifest;
    auto label_tag = static_cast<uint32_t>(StreamSerializationTag::LABEL_TABLE);
    auto base_tag = static_cast<uint32_t>(StreamSerializationTag::BASE_CODES);
    auto hierarchy_tag = static_cast<uint32_t>(StreamSerializationTag::PYRAMID_HIERARCHIES);
    AppendStreamingManifestBlock(manifest,
                                 label_tag,
                                 StreamSerializationBlockCurrentVersion(label_tag),
                                 StreamSerializationTagCritical(label_tag));
    AppendStreamingManifestBlock(manifest,
                                 base_tag,
                                 StreamSerializationBlockCurrentVersion(base_tag),
                                 StreamSerializationTagCritical(base_tag));
    if (this->has_precise_reorder()) {
        auto tag = static_cast<uint32_t>(StreamSerializationTag::HIGH_PRECISION_CODES);
        AppendStreamingManifestBlock(manifest,
                                     tag,
                                     StreamSerializationBlockCurrentVersion(tag),
                                     StreamSerializationTagCritical(tag));
    }
    if (this->raw_vector_ != nullptr) {
        auto tag = static_cast<uint32_t>(StreamSerializationTag::RAW_VECTOR);
        AppendStreamingManifestBlock(manifest,
                                     tag,
                                     StreamSerializationBlockCurrentVersion(tag),
                                     StreamSerializationTagCritical(tag));
    }
    AppendStreamingManifestBlock(manifest,
                                 hierarchy_tag,
                                 StreamSerializationBlockCurrentVersion(hierarchy_tag),
                                 StreamSerializationTagCritical(hierarchy_tag));
    metadata->Set("block_manifest", manifest);
    metadata->SetEmptyIndex(this->GetNumElements() == 0);
    return metadata;
}

void
Pyramid::serialize_hierarchies(StreamWriter& writer) const {
    auto pyramid_param = std::dynamic_pointer_cast<PyramidParameters>(create_param_ptr_);
    if (pyramid_param && pyramid_param->has_hierarchies) {
        uint64_t hierarchy_count = hierarchies_.size();
        StreamWriter::WriteObj(writer, hierarchy_count);
        for (const auto& [hname, h_ptr] : hierarchies_) {
            StreamWriter::WriteString(writer, hname);
            h_ptr->root->Serialize(writer);
        }
    } else {
        hierarchies_.at("")->root->Serialize(writer);
    }
}

void
Pyramid::serialize_streaming_body(StreamWriter& writer) const {
    auto label_tag = static_cast<uint32_t>(StreamSerializationTag::LABEL_TABLE);
    auto base_tag = static_cast<uint32_t>(StreamSerializationTag::BASE_CODES);
    auto hierarchy_tag = static_cast<uint32_t>(StreamSerializationTag::PYRAMID_HIERARCHIES);

    WriteStreamingBlock(
        writer, label_tag, StreamSerializationTagCritical(label_tag), [this](StreamWriter& w) {
            this->label_table_->Serialize(w);
        });
    WriteStreamingBlock(
        writer, base_tag, StreamSerializationTagCritical(base_tag), [this](StreamWriter& w) {
            this->base_codes_->Serialize(w);
        });
    if (this->has_precise_reorder()) {
        auto tag = static_cast<uint32_t>(StreamSerializationTag::HIGH_PRECISION_CODES);
        WriteStreamingBlock(
            writer, tag, StreamSerializationTagCritical(tag), [this](StreamWriter& w) {
                this->precise_codes_->Serialize(w);
            });
    }
    if (this->raw_vector_ != nullptr) {
        auto tag = static_cast<uint32_t>(StreamSerializationTag::RAW_VECTOR);
        WriteStreamingBlock(
            writer, tag, StreamSerializationTagCritical(tag), [this](StreamWriter& w) {
                this->raw_vector_->Serialize(w);
            });
    }
    WriteStreamingBlock(writer,
                        hierarchy_tag,
                        StreamSerializationTagCritical(hierarchy_tag),
                        [this](StreamWriter& w) { this->serialize_hierarchies(w); });
}

void
Pyramid::deserialize_hierarchies(StreamReader& reader, const JsonType& basic_info) {
    auto param_json = JsonType::Parse(basic_info[INDEX_PARAM].GetString());
    if (param_json.Contains(PYRAMID_HIERARCHIES)) {
        uint64_t hierarchy_count = 0;
        StreamReader::ReadObj(reader, hierarchy_count);
        CHECK_ARGUMENT(hierarchy_count == hierarchies_.size(),
                       fmt::format("serialized hierarchy count ({}) != config ({})",
                                   hierarchy_count,
                                   hierarchies_.size()));
        for (uint64_t i = 0; i < hierarchy_count; ++i) {
            std::string hname = StreamReader::ReadString(reader);
            auto h_iter = hierarchies_.find(hname);
            CHECK_ARGUMENT(h_iter != hierarchies_.end(),
                           fmt::format("deserialized hierarchy '{}' not in config", hname));
            h_iter->second->root->Deserialize(reader);
        }
    } else {
        auto h_iter = hierarchies_.find("");
        CHECK_ARGUMENT(
            h_iter != hierarchies_.end(),
            "deserialized single-hierarchy index but current config has named hierarchies");
        h_iter->second->root->Deserialize(reader);
    }
}

void
Pyramid::deserialize_streaming_body(StreamReader& reader, const MetadataPtr& metadata) {
    this->read_streaming_body(reader, metadata);
}

void
Pyramid::load_streaming_body(StreamReader& reader,
                             const MetadataPtr& metadata,
                             const LoadParameters& parameters) {
    (void)parameters;
    this->read_streaming_body(reader, metadata);
}

void
Pyramid::read_streaming_body(StreamReader& reader, const MetadataPtr& metadata) {
    auto basic_info = metadata->Get(BASIC_INFO);
    auto max_capacity = basic_info["max_capacity"].GetInt();
    if (basic_info.Contains(INDEX_PARAM)) {
        auto index_param = std::make_shared<PyramidParameters>();
        index_param->FromString(basic_info[INDEX_PARAM].GetString());
        if (not this->create_param_ptr_->CheckCompatibility(index_param)) {
            auto message = fmt::format("Pyramid index parameter not match, current: {}, new: {}",
                                       this->create_param_ptr_->ToString(),
                                       index_param->ToString());
            logger::error(message);
            throw VsagException(ErrorType::INVALID_ARGUMENT, message);
        }
    }

    bool loaded_label_table = false;
    bool loaded_base_codes = false;
    bool loaded_precise_codes = false;
    bool loaded_raw_vector = false;
    bool loaded_hierarchies = false;

    while (true) {
        auto block_header = StreamBlockHeader::Read(reader);
        if (block_header.IsSectionEnd()) {
            break;
        }
        BoundedForwardReader block_reader(&reader, block_header.value_len);
        if (!StreamSerializationBlockVersionSupported(block_header.tag,
                                                      block_header.block_version)) {
            if (block_header.IsCritical()) {
                throw VsagException(
                    ErrorType::UNSUPPORTED_INDEX_OPERATION,
                    fmt::format("unsupported Pyramid streaming block version: tag={}, "
                                "name={}, version={}, flags={}, value_len={}",
                                block_header.tag,
                                StreamSerializationTagName(block_header.tag),
                                block_header.block_version,
                                block_header.flags,
                                block_header.value_len));
            }
            block_reader.SkipRemaining();
            continue;
        }

        switch (static_cast<StreamSerializationTag>(block_header.tag)) {
            case StreamSerializationTag::LABEL_TABLE:
                ReadSeekableBlockPayload(block_reader, block_header, [this](StreamReader& block) {
                    this->label_table_->Deserialize(block);
                });
                this->delete_count_.store(
                    static_cast<int64_t>(this->label_table_->GetAllDeletedIds().size()),
                    std::memory_order_relaxed);
                loaded_label_table = true;
                break;
            case StreamSerializationTag::BASE_CODES:
                ReadSeekableBlockPayload(block_reader, block_header, [this](StreamReader& block) {
                    this->base_codes_->Deserialize(block);
                });
                this->cur_element_count_ = this->base_codes_->TotalCount();
                loaded_base_codes = true;
                break;
            case StreamSerializationTag::HIGH_PRECISION_CODES:
                if (this->has_precise_reorder()) {
                    ReadSeekableBlockPayload(
                        block_reader, block_header, [this](StreamReader& block) {
                            this->precise_codes_->Deserialize(block);
                        });
                    loaded_precise_codes = true;
                }
                break;
            case StreamSerializationTag::RAW_VECTOR:
                if (this->raw_vector_ != nullptr) {
                    ReadSeekableBlockPayload(
                        block_reader, block_header, [this](StreamReader& block) {
                            this->raw_vector_->Deserialize(block);
                        });
                    loaded_raw_vector = true;
                }
                break;
            case StreamSerializationTag::PYRAMID_HIERARCHIES:
                ReadSeekableBlockPayload(
                    block_reader, block_header, [this, &basic_info](StreamReader& block) {
                        this->deserialize_hierarchies(block, basic_info);
                    });
                loaded_hierarchies = true;
                break;
            default:
                if (block_header.IsCritical()) {
                    throw VsagException(
                        ErrorType::UNSUPPORTED_INDEX_OPERATION,
                        fmt::format("unknown Pyramid streaming serialization block: "
                                    "tag={}, name={}, version={}, flags={}, "
                                    "value_len={}",
                                    block_header.tag,
                                    StreamSerializationTagName(block_header.tag),
                                    block_header.block_version,
                                    block_header.flags,
                                    block_header.value_len));
                }
                break;
        }
        block_reader.SkipRemaining();
    }

    if (!loaded_label_table || !loaded_base_codes || !loaded_hierarchies) {
        throw VsagException(ErrorType::READ_ERROR,
                            "Pyramid streaming serialization required block is missing");
    }
    if (this->has_precise_reorder() && !loaded_precise_codes) {
        throw VsagException(ErrorType::READ_ERROR,
                            "Pyramid streaming serialization precise codes block is missing");
    }
    if (this->raw_vector_ != nullptr && !loaded_raw_vector) {
        throw VsagException(ErrorType::READ_ERROR,
                            "Pyramid streaming serialization raw vector block is missing");
    }

    resize(max_capacity);
    this->current_memory_usage_ = static_cast<int64_t>(this->CalSerializeSize());
}

void
Pyramid::Deserialize(StreamReader& reader) {
    // try to deserialize footer (only in new version)
    JsonType basic_info;
    if (not read_index_footer(reader, basic_info)) {
        throw VsagException(ErrorType::READ_ERROR, "failed to read index footer");
    }
    auto max_capacity = basic_info["max_capacity"].GetInt();

    BufferStreamReader buffer_reader(
        &reader, std::numeric_limits<uint64_t>::max(), this->allocator_);

    label_table_->Deserialize(buffer_reader);
    deserialize_source_id_table(buffer_reader);
    delete_count_.store(static_cast<int64_t>(label_table_->GetAllDeletedIds().size()),
                        std::memory_order_relaxed);
    base_codes_->Deserialize(buffer_reader);
    if (has_precise_reorder()) {
        precise_codes_->Deserialize(buffer_reader);
    }
    if (raw_vector_ != nullptr) {
        raw_vector_->Deserialize(buffer_reader);
    }
    cur_element_count_ = base_codes_->TotalCount();

    auto param_json = JsonType::Parse(basic_info[INDEX_PARAM].GetString());
    if (param_json.Contains(PYRAMID_HIERARCHIES)) {
        uint64_t hierarchy_count = 0;
        StreamReader::ReadObj(buffer_reader, hierarchy_count);
        CHECK_ARGUMENT(hierarchy_count == hierarchies_.size(),
                       fmt::format("serialized hierarchy count ({}) != config ({})",
                                   hierarchy_count,
                                   hierarchies_.size()));
        for (uint64_t i = 0; i < hierarchy_count; ++i) {
            std::string hname = StreamReader::ReadString(buffer_reader);
            auto h_iter = hierarchies_.find(hname);
            CHECK_ARGUMENT(h_iter != hierarchies_.end(),
                           fmt::format("deserialized hierarchy '{}' not in config", hname));
            h_iter->second->root->Deserialize(buffer_reader);
        }
    } else {
        auto h_iter = hierarchies_.find("");
        CHECK_ARGUMENT(
            h_iter != hierarchies_.end(),
            "deserialized single-hierarchy index but current config has named hierarchies");
        h_iter->second->root->Deserialize(buffer_reader);
    }

    resize(max_capacity);
    this->current_memory_usage_ = this->CalSerializeSize();
}

InnerIndexPtr
Pyramid::ExportModel(const IndexCommonParam& param) const {
    auto index = std::make_shared<Pyramid>(this->create_param_ptr_, param);
    if (index->use_reorder_ != this->use_reorder_) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "Export model's pyramid reorder config mismatched");
    }
    this->base_codes_->ExportModel(index->base_codes_);
    if (has_precise_reorder()) {
        if (index->precise_codes_ == nullptr) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                "Export model's pyramid precise codes is empty");
        }
        this->precise_codes_->ExportModel(index->precise_codes_);
    }
    if (raw_vector_ != nullptr) {
        if (index->raw_vector_ == nullptr) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                "Export model's pyramid raw vector is empty");
        }
        this->raw_vector_->ExportModel(index->raw_vector_);
    }
    index->current_memory_usage_ = index->CalSerializeSize();
    return index;
}

std::vector<int64_t>
Pyramid::Add(const DatasetPtr& base) {
    const int64_t data_num = base->GetNumElements();
    const auto* data_vectors = base->GetFloat32Vectors();
    const auto* data_ids = base->GetIds();
    const auto* source_ids = base->GetSourceID();
    std::vector<int64_t> failed_ids;
    Vector<int64_t> data_biases(allocator_);
    int64_t local_cur_element_count = 0;
    {
        std::lock_guard lock(cur_element_count_mutex_);
        local_cur_element_count = cur_element_count_;
        auto new_capacity = max_capacity_;
        if (max_capacity_ == 0) {
            new_capacity = std::max(INIT_CAPACITY, data_num);
        } else if (max_capacity_ < data_num + cur_element_count_) {
            new_capacity = std::min(MAX_CAPACITY_EXTEND, max_capacity_);
            new_capacity = std::max(data_num + cur_element_count_ - max_capacity_, new_capacity) +
                           max_capacity_;
        }
        bool base_storage_resized = false;
        bool precise_storage_resized = false;
        bool raw_storage_resized = false;
        if (new_capacity > max_capacity_) {
            base_storage_resized = new_capacity > static_cast<int64_t>(base_codes_->max_capacity_);
            precise_storage_resized =
                not has_precise_reorder() ||
                new_capacity > static_cast<int64_t>(precise_codes_->max_capacity_);
            raw_storage_resized = raw_vector_ == nullptr ||
                                  new_capacity > static_cast<int64_t>(raw_vector_->max_capacity_);
            resize(new_capacity);
        }

        data_biases.reserve(data_num);
        for (int64_t i = 0; i < data_num; ++i) {
            if (not label_table_->CheckLabel(data_ids[i])) {
                const auto inner_id =
                    static_cast<InnerIdType>(local_cur_element_count + data_biases.size());
                label_table_->Insert(inner_id, data_ids[i]);
                if (source_ids != nullptr) {
                    label_table_->InsertSourceId(inner_id, source_ids[i]);
                }
                data_biases.push_back(i);
            } else {
                logger::warn("Label {} already exists, skip adding.", data_ids[i]);
                failed_ids.push_back(data_ids[i]);
            }
        }

        if (local_cur_element_count == 0 and not data_biases.empty()) {
            this->Train(base);
        }

        const auto encode_range = [this, data_vectors, local_cur_element_count, &data_biases](
                                      uint64_t begin, uint64_t end) {
            for (uint64_t offset = begin; offset < end; ++offset) {
                const auto* vector = data_vectors + dim_ * data_biases[offset];
                const auto inner_id = static_cast<InnerIdType>(local_cur_element_count + offset);
                base_codes_->InsertVector(vector, inner_id);
                if (has_precise_reorder()) {
                    precise_codes_->InsertVector(vector, inner_id);
                }
                if (raw_vector_ != nullptr) {
                    raw_vector_->InsertVector(vector, inner_id);
                }
            }
        };
        const auto supports_parallel_encode = [](const FlattenInterfacePtr& codes) {
            const auto name = codes->GetQuantizerName();
            return codes->InMemory() &&
                   (name == QUANTIZATION_TYPE_VALUE_FP32 ||
                    name == QUANTIZATION_TYPE_VALUE_RABITQ || name == QUANTIZATION_TYPE_VALUE_SQ8);
        };
        const bool use_parallel_encode =
            local_cur_element_count == 0 && thread_pool_ != nullptr && build_thread_count_ > 1 &&
            data_biases.size() > 1 && supports_parallel_encode(base_codes_) &&
            (not has_precise_reorder() || supports_parallel_encode(precise_codes_)) &&
            (raw_vector_ == nullptr || supports_parallel_encode(raw_vector_)) &&
            base_storage_resized && precise_storage_resized && raw_storage_resized;
        if (use_parallel_encode) {
            const uint64_t worker_count =
                std::min<uint64_t>(build_thread_count_, data_biases.size());
            const uint64_t block_size = (data_biases.size() + worker_count - 1) / worker_count;
            Vector<std::future<void>> futures(allocator_);
            futures.reserve(worker_count);
            const auto wait_futures = [&futures]() {
                std::exception_ptr first_exception = nullptr;
                for (auto& future : futures) {
                    try {
                        future.get();
                    } catch (...) {
                        if (not first_exception) {
                            first_exception = std::current_exception();
                        }
                    }
                }
                if (first_exception) {
                    std::rethrow_exception(first_exception);
                }
            };
            try {
                for (uint64_t begin = 0; begin < data_biases.size(); begin += block_size) {
                    const uint64_t end = std::min<uint64_t>(begin + block_size, data_biases.size());
                    futures.push_back(thread_pool_->GeneralEnqueue(
                        [encode_range, begin, end]() { encode_range(begin, end); }));
                }
            } catch (...) {
                const auto enqueue_exception = std::current_exception();
                try {
                    wait_futures();
                } catch (...) {
                }
                std::rethrow_exception(enqueue_exception);
            }
            wait_futures();
        } else {
            encode_range(0, data_biases.size());
        }
        cur_element_count_ += static_cast<int64_t>(data_biases.size());
    }
    std::shared_lock<std::shared_mutex> lock(resize_mutex_);

    for (const auto& [hname, h_ptr] : hierarchies_) {
        const auto* hpath = base->GetPaths(hname);
        if (hpath != nullptr) {
            add_to_hierarchy(*h_ptr, data_vectors, hpath, data_biases, local_cur_element_count);
        }
    }
    return failed_ids;
}

void
Pyramid::resize(int64_t new_max_capacity) {
    std::unique_lock<std::shared_mutex> lock(resize_mutex_);
    if (new_max_capacity <= max_capacity_) {
        return;
    }
    pool_ = std::make_unique<VisitedListPool>(1, allocator_, new_max_capacity, allocator_);
    label_table_->Resize(new_max_capacity);
    base_codes_->Resize(new_max_capacity);
    if (has_precise_reorder()) {
        precise_codes_->Resize(new_max_capacity);
    }
    if (raw_vector_ != nullptr) {
        raw_vector_->Resize(new_max_capacity);
    }
    points_mutex_->Resize(new_max_capacity);
    max_capacity_ = new_max_capacity;
}

void
Pyramid::InitFeatures() {
    // add & build
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_BUILD,
        IndexFeature::SUPPORT_ADD_AFTER_BUILD,
        IndexFeature::SUPPORT_ADD_FROM_EMPTY,
    });

    // search
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_KNN_SEARCH,
        IndexFeature::SUPPORT_KNN_SEARCH_WITH_ID_FILTER,
        IndexFeature::SUPPORT_RANGE_SEARCH,
        IndexFeature::SUPPORT_RANGE_SEARCH_WITH_ID_FILTER,
    });

    // calculate distance by id

    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_CAL_DISTANCE_BY_ID,
        IndexFeature::SUPPORT_BATCH_CALC_DISTANCE_BY_ID,
    });

    // concurrency
    this->index_feature_list_->SetFeatures({IndexFeature::SUPPORT_SEARCH_CONCURRENT,
                                            IndexFeature::SUPPORT_ADD_CONCURRENT,
                                            IndexFeature::SUPPORT_ADD_SEARCH_CONCURRENT});

    // serialize
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_SERIALIZE_FILE,
        IndexFeature::SUPPORT_DESERIALIZE_FILE,
        IndexFeature::SUPPORT_SERIALIZE_BINARY_SET,
        IndexFeature::SUPPORT_DESERIALIZE_BINARY_SET,
        IndexFeature::SUPPORT_DESERIALIZE_BINARY_SET,
    });

    // other
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_CLONE,
        IndexFeature::SUPPORT_EXPORT_MODEL,
        IndexFeature::SUPPORT_GET_MEMORY_USAGE,
    });
    if (has_raw_vector_) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_GET_RAW_VECTOR_BY_IDS);
    }

    this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_DELETE_BY_ID);
}

static const std::string HGRAPH_PARAMS_TEMPLATE =
    R"(
    {
        "{TYPE_KEY}": "{INDEX_TYPE_PYRAMID}",
        "{USE_REORDER_KEY}": false,
        "{GRAPH_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH_KEY}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "{GRAPH_TYPE_KEY}": "{GRAPH_TYPE_VALUE_NSW}",
            "{GRAPH_STORAGE_TYPE_KEY}": "{GRAPH_STORAGE_TYPE_VALUE_FLAT}",
            "{ODESCENT_PARAMETER_BUILD_BLOCK_SIZE}": 10000,
            "{ODESCENT_PARAMETER_MIN_IN_DEGREE}": 1,
            "{ODESCENT_PARAMETER_ALPHA}": 1.2,
            "{ODESCENT_PARAMETER_GRAPH_ITER_TURN}": 30,
            "{ODESCENT_PARAMETER_NEIGHBOR_SAMPLE_RATE}": 0.2,
            "{GRAPH_PARAM_MAX_DEGREE_KEY}": 64,
            "{GRAPH_PARAM_INIT_MAX_CAPACITY_KEY}": 100,
            "{GRAPH_SUPPORT_REMOVE}": false,
            "{REMOVE_FLAG_BIT}": 8,
            "{SUPPORT_DUPLICATE}": false
        },
        "{BASE_CODES_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH_KEY}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "{CODES_TYPE_KEY}": "flatten",
            "{QUANTIZATION_PARAMS_KEY}": {
                "{TYPE_KEY}": "{QUANTIZATION_TYPE_VALUE_FP32}",
                "{SQ4_UNIFORM_QUANTIZATION_TRUNC_RATE_KEY}": 0.05,
                "{PCA_DIM_KEY}": 0,
                "{MRLE_DIM_KEY}": 0,
                "{RABITQ_QUANTIZATION_VERSION_KEY}": "standard",
                "{RABITQ_QUANTIZATION_BITS_PER_DIM_QUERY_KEY}": 32,
                "{RABITQ_QUANTIZATION_BITS_PER_DIM_BASE_KEY}": 1,
                "{RABITQ_QUANTIZATION_BITS_PER_DIM_FILTER_KEY}": 1,
                "{FAST_ENCODE_RABITQ_KEY}": true,
                "{FAST_ENCODE_RABITQ_ROUNDS_KEY}": 6,
                "{TQ_CHAIN_KEY}": "",
                "nbits": 8,
                "{PRODUCT_QUANTIZATION_DIM_KEY}": 1,
                "{HOLD_MOLDS}": false
            }
        },
        "{PRECISE_CODES_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH_KEY}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "{CODES_TYPE_KEY}": "flatten",
            "{QUANTIZATION_PARAMS_KEY}": {
                "{TYPE_KEY}": "{QUANTIZATION_TYPE_VALUE_FP32}",
                "{SQ4_UNIFORM_QUANTIZATION_TRUNC_RATE_KEY}": 0.05,
                "{PCA_DIM_KEY}": 0,
                "{FAST_ENCODE_RABITQ_KEY}": true,
                "{FAST_ENCODE_RABITQ_ROUNDS_KEY}": 6,
                "{PRODUCT_QUANTIZATION_DIM_KEY}": 1,
                "{HOLD_MOLDS}": false
            }
        },
        "{STORE_RAW_VECTOR_KEY}": false,
        "{RAW_VECTOR_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH_KEY}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "{CODES_TYPE_KEY}": "flatten",
            "{QUANTIZATION_PARAMS_KEY}": {
                "{TYPE_KEY}": "{QUANTIZATION_TYPE_VALUE_FP32}",
                "{HOLD_MOLDS}": true
            }
        },
        "{BUILD_THREAD_COUNT_KEY}": 1,
        "{EF_CONSTRUCTION_KEY}": 400,
        "{NO_BUILD_LEVELS}":[],
        "{INDEX_MIN_SIZE}": 0,
        "{SUPPORT_DUPLICATE}": false
    })";

ParamPtr
Pyramid::CheckAndMappingExternalParam(const JsonType& external_param,
                                      const IndexCommonParam& common_param) {
    const ConstParamMap external_mapping = {
        {PYRAMID_EF_CONSTRUCTION, {EF_CONSTRUCTION_KEY}},
        {PYRAMID_USE_REORDER, {USE_REORDER_KEY}},
        {PYRAMID_BASE_QUANTIZATION_TYPE, {BASE_CODES_KEY, QUANTIZATION_PARAMS_KEY, TYPE_KEY}},
        {INDEX_TQ_CHAIN, {BASE_CODES_KEY, QUANTIZATION_PARAMS_KEY, TQ_CHAIN_KEY}},
        {INDEX_MRLE_DIM, {BASE_CODES_KEY, QUANTIZATION_PARAMS_KEY, MRLE_DIM_KEY}},
        {PYRAMID_RABITQ_BITS_PER_DIM_BASE,
         {BASE_CODES_KEY, QUANTIZATION_PARAMS_KEY, RABITQ_QUANTIZATION_BITS_PER_DIM_BASE_KEY}},
        {PYRAMID_RABITQ_BITS_PER_DIM_QUERY,
         {BASE_CODES_KEY, QUANTIZATION_PARAMS_KEY, RABITQ_QUANTIZATION_BITS_PER_DIM_QUERY_KEY}},
        {PYRAMID_RABITQ_BITS_PER_DIM_PRECISE,
         {PRECISE_CODES_KEY, QUANTIZATION_PARAMS_KEY, RABITQ_QUANTIZATION_BITS_PER_DIM_BASE_KEY}},
        {PYRAMID_RABITQ_PCA_DIM, {BASE_CODES_KEY, QUANTIZATION_PARAMS_KEY, PCA_DIM_KEY}},
        {PYRAMID_RABITQ_USE_FHT, {BASE_CODES_KEY, QUANTIZATION_PARAMS_KEY, USE_FHT_KEY}},
        {RABITQ_ERROR_RATE,
         {BASE_CODES_KEY, QUANTIZATION_PARAMS_KEY, RABITQ_QUANTIZATION_ERROR_RATE_KEY}},
        {RABITQ_ERROR_RATE,
         {PRECISE_CODES_KEY, QUANTIZATION_PARAMS_KEY, RABITQ_QUANTIZATION_ERROR_RATE_KEY}},
        {PYRAMID_FAST_ENCODE_RABITQ,
         {BASE_CODES_KEY, QUANTIZATION_PARAMS_KEY, FAST_ENCODE_RABITQ_KEY}},
        {PYRAMID_FAST_ENCODE_RABITQ,
         {PRECISE_CODES_KEY, QUANTIZATION_PARAMS_KEY, FAST_ENCODE_RABITQ_KEY}},
        {PYRAMID_FAST_ENCODE_RABITQ_ROUNDS,
         {BASE_CODES_KEY, QUANTIZATION_PARAMS_KEY, FAST_ENCODE_RABITQ_ROUNDS_KEY}},
        {PYRAMID_FAST_ENCODE_RABITQ_ROUNDS,
         {PRECISE_CODES_KEY, QUANTIZATION_PARAMS_KEY, FAST_ENCODE_RABITQ_ROUNDS_KEY}},
        {PYRAMID_PRECISE_QUANTIZATION_TYPE, {PRECISE_CODES_KEY, QUANTIZATION_PARAMS_KEY, TYPE_KEY}},
        {PYRAMID_GRAPH_MAX_DEGREE, {GRAPH_KEY, GRAPH_PARAM_MAX_DEGREE_KEY}},
        {PYRAMID_BASE_IO_TYPE, {BASE_CODES_KEY, IO_PARAMS_KEY, TYPE_KEY}},
        {PYRAMID_BASE_SUPPLEMENT_IO_TYPE, {BASE_CODES_KEY, SUPPLEMENT_IO_PARAMS_KEY, TYPE_KEY}},
        {PYRAMID_BASE_SUPPLEMENT_FILE_PATH,
         {BASE_CODES_KEY, SUPPLEMENT_IO_PARAMS_KEY, IO_FILE_PATH_KEY}},
        {PYRAMID_BUILD_ALPHA, {GRAPH_KEY, ODESCENT_PARAMETER_ALPHA}},
        {PYRAMID_GRAPH_TYPE, {GRAPH_KEY, GRAPH_TYPE_KEY}},
        {PYRAMID_GRAPH_STORAGE_TYPE, {GRAPH_KEY, GRAPH_STORAGE_TYPE_KEY}},
        {PYRAMID_PRECISE_IO_TYPE, {PRECISE_CODES_KEY, IO_PARAMS_KEY, TYPE_KEY}},
        {PYRAMID_BUILD_THREAD_COUNT, {BUILD_THREAD_COUNT_KEY}},
        {STORE_RAW_VECTOR, {STORE_RAW_VECTOR_KEY}},
        {PYRAMID_NO_BUILD_LEVELS, {NO_BUILD_LEVELS}},
        {PYRAMID_HIERARCHIES, {PYRAMID_HIERARCHIES}},
        {PYRAMID_PERSIST_SOURCE_ID, {PYRAMID_PERSIST_SOURCE_ID_KEY}},
        {PYRAMID_BASE_PQ_DIM,
         {BASE_CODES_KEY, QUANTIZATION_PARAMS_KEY, PRODUCT_QUANTIZATION_DIM_KEY}},
        {PYRAMID_BASE_FILE_PATH, {BASE_CODES_KEY, IO_PARAMS_KEY, IO_FILE_PATH_KEY}},
        {PYRAMID_PRECISE_FILE_PATH, {PRECISE_CODES_KEY, IO_PARAMS_KEY, IO_FILE_PATH_KEY}},
        {STORE_RAW_VECTOR, {STORE_RAW_VECTOR_KEY}},
        {ODESCENT_PARAMETER_BUILD_BLOCK_SIZE, {GRAPH_KEY, ODESCENT_PARAMETER_BUILD_BLOCK_SIZE}},
        {ODESCENT_PARAMETER_MIN_IN_DEGREE, {GRAPH_KEY, ODESCENT_PARAMETER_MIN_IN_DEGREE}},
        {ODESCENT_PARAMETER_GRAPH_ITER_TURN, {GRAPH_KEY, ODESCENT_PARAMETER_GRAPH_ITER_TURN}},
        {ODESCENT_PARAMETER_NEIGHBOR_SAMPLE_RATE,
         {GRAPH_KEY, ODESCENT_PARAMETER_NEIGHBOR_SAMPLE_RATE}},
        {PYRAMID_INDEX_MIN_SIZE, {INDEX_MIN_SIZE}},
        {PYRAMID_SUPPORT_DUPLICATE, {SUPPORT_DUPLICATE}},
        {PYRAMID_SUPPORT_DUPLICATE, {GRAPH_KEY, SUPPORT_DUPLICATE}}};

    std::string str = format_map(HGRAPH_PARAMS_TEMPLATE, DEFAULT_MAP);
    auto inner_json = JsonType::Parse(str);
    mapping_external_param_to_inner(external_param, external_mapping, inner_json);
    MapRaBitQSplitParam(external_param, inner_json);
    ValidateMRLEDim(external_param, common_param.dim_);
    if (RequiresRawVectorForTransformQuantizer(inner_json) and
        not RequiresRawVectorForMRLERaBitQSplit(inner_json)) {
        inner_json[STORE_RAW_VECTOR_KEY].SetBool(true);
    }
    auto pyramid_params = std::make_shared<PyramidParameters>();
    pyramid_params->FromJson(inner_json);
    return pyramid_params;
}

void
Pyramid::Train(const DatasetPtr& base) {
    this->base_codes_->Train(base->GetFloat32Vectors(), base->GetNumElements());
    if (has_precise_reorder()) {
        this->precise_codes_->Train(base->GetFloat32Vectors(), base->GetNumElements());
    }
    if (raw_vector_ != nullptr) {
        this->raw_vector_->Train(base->GetFloat32Vectors(), base->GetNumElements());
    }
}
std::vector<int64_t>
Pyramid::Build(const DatasetPtr& base) {
    CHECK_ARGUMENT(GetNumElements() == 0, "index is not empty");
    int64_t data_num = base->GetNumElements();

    if (graph_type_ == GRAPH_TYPE_VALUE_NSW && not support_duplicate_ && has_loaded_cache() &&
        base->GetSourceID() != nullptr) {
        UnorderedSet<std::string> source_ids(allocator_);
        UnorderedSet<LabelType> labels(allocator_);
        source_ids.reserve(data_num);
        labels.reserve(data_num);
        bool unique = true;
        const auto* source_id_data = base->GetSourceID();
        const auto* data_ids = base->GetIds();
        for (int64_t i = 0; i < data_num; ++i) {
            unique =
                source_ids.emplace(source_id_data[i]).second && labels.emplace(data_ids[i]).second;
            if (not unique) {
                break;
            }
        }
        if (unique) {
            return build_with_cache(base);
        }
        logger::warn(
            "[pyramid_build_cache] duplicate source_id or label; falling back to cold build");
    }

    this->Train(base);
    std::vector<int64_t> ret;

    if (thread_pool_ != nullptr && hierarchies_.size() > 1) {
        Vector<std::future<void>> futures(allocator_);
        for (const auto& [hname, h_ptr] : hierarchies_) {
            const auto* hpath = base->GetPaths(hname);
            if (hpath != nullptr) {
                futures.push_back(
                    thread_pool_->GeneralEnqueue([&h = *h_ptr, hpath, data_num, this]() {
                        populate_path_tree(h, hpath, data_num);
                    }));
            }
        }
        for (auto& f : futures) {
            f.get();
        }
    } else {
        for (const auto& [hname, h_ptr] : hierarchies_) {
            const auto* hpath = base->GetPaths(hname);
            if (hpath != nullptr) {
                populate_path_tree(*h_ptr, hpath, data_num);
            }
        }
    }

    if (graph_type_ == GRAPH_TYPE_VALUE_NSW) {
        ret = this->Add(base);
    } else {
        ret = this->build_by_odescent(base);
    }
    return ret;
}

void
Pyramid::add_one_point(const Hierarchy& h,
                       IndexNode* node,
                       InnerIdType inner_id,
                       const float* vector,
                       uint64_t ef_construction,
                       bool use_self_as_entry) {
    std::unique_lock graph_lock(node->mutex_);

    if (node->status_ == IndexNode::Status::NO_INDEX) {
        node->Init();
        Vector<InnerIdType>(allocator_).swap(node->ids_);
    }

    if (node->status_ == IndexNode::Status::FLAT) {
        node->ids_.push_back(inner_id);
        if (node->ids_.size() < node->index_min_size_) {
            return;
        }

        // Keep the FLAT node intact until the replacement graph is complete.
        IndexNode graph_node(allocator_, node->graph_param_, node->index_min_size_);
        graph_node.level_ = node->level_;
        graph_node.ids_ = node->ids_;
        graph_node.Init();

        if (base_codes_->SupportSplitCodeStorage() and raw_vector_ == nullptr) {
            for (const auto id : node->ids_) {
                add_one_point(h, &graph_node, id, nullptr);
            }
        } else {
            auto codes = decodable_codes();
            Vector<float> decoded_vector(dim_, allocator_);
            for (const auto id : node->ids_) {
                bool need_release = false;
                const auto* buffer = codes->GetCodesById(id, need_release);
                const bool decoded = codes->Decode(buffer, decoded_vector.data());
                if (need_release) {
                    codes->Release(buffer);
                }
                if (not decoded) {
                    throw VsagException(ErrorType::INTERNAL_ERROR,
                                        "Pyramid graph promotion requires decodable vectors");
                }
                add_one_point(h, &graph_node, id, decoded_vector.data());
            }
        }

        node->graph_ = std::move(graph_node.graph_);
        node->graph_param_ = std::move(graph_node.graph_param_);
        node->entry_point_ = graph_node.entry_point_;
        node->status_ = IndexNode::Status::GRAPH;
        Vector<InnerIdType>(allocator_).swap(node->ids_);
        return;
    }

    if (node->graph_->TotalCount() == 0) {
        node->graph_->InsertNeighborsById(inner_id, Vector<InnerIdType>(allocator_));
        node->entry_point_ = inner_id;
    } else {
        const uint64_t effective_ef = ef_construction == 0 ? h.ef_construction : ef_construction;
        InnerSearchParam search_param;
        search_param.ef = effective_ef;
        search_param.topk = static_cast<int64_t>(effective_ef);
        search_param.search_mode = KNN_SEARCH;
        search_param.hops_limit = 10000;
        if (support_duplicate_) {
            search_param.find_duplicate = true;
            search_param.duplicate_query_id = inner_id;
        }
        auto codes = has_precise_reorder() ? precise_codes_ : base_codes_;
        bool update_entry_point;
        {
            std::scoped_lock<std::mutex> entry_point_lock(entry_point_mutex_);
            update_entry_point = is_update_entry_point(node->graph_->TotalCount());
        }
        Vector<InnerIdType> cached_neighbors(allocator_);
        node->graph_->GetNeighbors(inner_id, cached_neighbors);
        search_param.ep =
            (use_self_as_entry && not cached_neighbors.empty()) ? inner_id : node->entry_point_;
        if (not update_entry_point) {
            graph_lock.unlock();
        }

        VisitedListGuard vl_guard(pool_.get());
        const VisitedListPtr& vl = vl_guard.get();
        DistHeapPtr results;
        if (vector != nullptr) {
            results = searcher_->Search(
                node->graph_, codes, vl, vector, search_param, (LabelTablePtr) nullptr, nullptr);
        } else {
            FlattenIdDistanceProvider distance_provider(base_codes_, inner_id);
            results = searcher_->Search(
                node->graph_, distance_provider, vl, search_param, nullptr, nullptr);
            if (support_duplicate_ and not results->Empty()) {
                // StandardHeap exposes heap storage rather than sorted output, so inspect every
                // candidate to find the actual nearest neighbor for duplicate detection.
                const auto* data = results->GetData();
                auto min_distance = data[0].first;
                auto min_index = data[0].second;
                for (uint32_t i = 1; i < results->Size(); ++i) {
                    if (data[i].first < min_distance) {
                        min_distance = data[i].first;
                        min_index = data[i].second;
                    }
                }
                if (search_param.duplicate_distance_threshold > 0.0F) {
                    if (min_distance <= search_param.duplicate_distance_threshold) {
                        search_param.duplicate_id = min_index;
                    }
                } else if (codes->CompareVectors(inner_id, min_index)) {
                    search_param.duplicate_id = min_index;
                }
            }
        }
        // HGraph-style cache-hit refinement: retain the restored row as local seeds,
        // start from self, then merge current-vector search candidates. Cold/miss construction
        // keeps its search heap unchanged and therefore does not allocate a merge heap.
        if (use_self_as_entry && not cached_neighbors.empty()) {
            auto merged_results = std::make_shared<StandardHeap<true, false>>(allocator_, -1);
            UnorderedSet<InnerIdType> seen(allocator_);
            seen.reserve(cached_neighbors.size() + results->Size());
            for (const auto neighbor : cached_neighbors) {
                if (neighbor != inner_id && seen.emplace(neighbor).second) {
                    merged_results->Push(codes->ComputePairVectors(inner_id, neighbor), neighbor);
                }
            }
            while (not results->Empty()) {
                const auto candidate = results->Top();
                results->Pop();
                if (candidate.second != inner_id && seen.emplace(candidate.second).second) {
                    merged_results->Push(candidate.first, candidate.second);
                }
            }
            results = std::move(merged_results);
        }
        if (this->support_duplicate_ && search_param.duplicate_id >= 0) {
            std::unique_lock lock(this->label_lookup_mutex_);
            node->graph_->SetDuplicateId(static_cast<InnerIdType>(search_param.duplicate_id),
                                         inner_id);
            return;
        }
        mutually_connect_new_element(
            inner_id, results, node->graph_, codes, points_mutex_, allocator_, h.alpha);
        if (update_entry_point) {
            node->entry_point_ = inner_id;
        }
    }
}

void
Pyramid::populate_path_tree(Hierarchy& h, const std::string* paths, int64_t count) {
    for (int64_t i = 0; i < count; ++i) {
        std::string current_path = paths[i];
        auto path_slices = split(current_path, PART_SLASH);
        IndexNode* node = h.root.get();
        if (std::find(h.no_build_levels.begin(), h.no_build_levels.end(), node->level_) ==
            h.no_build_levels.end()) {
            node->ids_.push_back(i);
        }
        for (auto& path_slice : path_slices) {
            node = node->GetChild(path_slice, true);
            if (std::find(h.no_build_levels.begin(), h.no_build_levels.end(), node->level_) ==
                h.no_build_levels.end()) {
                node->ids_.push_back(i);
            }
        }
    }
}

void
Pyramid::add_to_hierarchy(Hierarchy& h,
                          const float* data_vectors,
                          const std::string* paths,
                          const Vector<int64_t>& data_biases,
                          int64_t local_cur_element_count) {
    auto add_func = [&](int64_t i, int64_t data_bias) {
        std::string current_path = paths[data_bias];
        auto path_slices = split(current_path, PART_SLASH);
        IndexNode* node = h.root.get();
        auto inner_id = static_cast<InnerIdType>(i + local_cur_element_count);
        const auto* vector = base_codes_->SupportSplitCodeStorage() and raw_vector_ == nullptr
                                 ? nullptr
                                 : data_vectors + dim_ * data_bias;
        int no_build_level_index = 0;
        for (int j = 0; j <= static_cast<int>(path_slices.size()); ++j) {
            IndexNode* new_node = nullptr;
            if (j != static_cast<int>(path_slices.size())) {
                new_node = node->GetChild(path_slices[j], true);
            }
            if (no_build_level_index < static_cast<int>(h.no_build_levels.size()) &&
                j == h.no_build_levels[no_build_level_index]) {
                node = new_node;
                no_build_level_index++;
                continue;
            }
            add_one_point(h, node, inner_id, vector);
            node = new_node;
        }
    };

    Vector<std::future<void>> futures(allocator_);
    for (int64_t i = 0; i < static_cast<int64_t>(data_biases.size()); ++i) {
        auto data_bias = data_biases[i];
        if (this->thread_pool_ != nullptr) {
            futures.push_back(this->thread_pool_->GeneralEnqueue(add_func, i, data_bias));
        } else {
            add_func(i, data_bias);
        }
    }
    if (this->thread_pool_ != nullptr) {
        for (auto& future : futures) {
            future.get();
        }
    }
}

void
Pyramid::search_hierarchy(const Hierarchy& h,
                          const SearchFunc& search_func,
                          const VisitedListPtr& vl,
                          DistHeapPtr& search_result,
                          const std::string& path,
                          const InnerSearchParam& search_param) const {
    std::vector<std::future<void>> futures;
    auto parsed_path = parse_path(path);
    Vector<DistHeapPtr> search_result_lists(parsed_path.size(), allocator_);
    for (uint32_t i = 0; i < parsed_path.size(); ++i) {
        const auto& one_path = parsed_path[i];
        search_result_lists[i] = std::make_shared<StandardHeap<true, false>>(allocator_, -1);
        IndexNode* node = h.root.get();
        bool valid = true;
        for (const auto& item : one_path) {
            node = node->GetChild(item, false);
            if (node == nullptr) {
                valid = false;
                break;
            }
        }
        if (valid) {
            if (thread_pool_ != nullptr && search_param.parallel_search_thread_count > 1) {
                futures.push_back(thread_pool_->GeneralEnqueue([&, node, i]() -> void {
                    VisitedListGuard vl_guard(pool_.get());
                    const VisitedListPtr& local_vl = vl_guard.get();
                    node->Search(search_func, local_vl, search_result_lists[i], search_param.ef);
                }));
            } else {
                node->Search(search_func, vl, search_result_lists[i], search_param.ef);
            }
        }
    }

    for (auto& future : futures) {
        future.get();
    }

    for (uint32_t i = 0; i < search_result_lists.size(); ++i) {
        if (i != 0) {
            search_result->Merge(*search_result_lists[i]);
        } else {
            search_result = search_result_lists[i];
        }
    }
}

std::vector<std::vector<std::string>>
Pyramid::parse_path(const std::string& path) {
    auto multi_paths = split(path, PART_BAR);
    std::vector<std::vector<std::string>> parsed_paths;
    parsed_paths.reserve(multi_paths.size());
    for (const auto& single_path : multi_paths) {
        parsed_paths.push_back(split(single_path, PART_SLASH));
    }
    return parsed_paths;
}

DistHeapPtr
Pyramid::search_node(const IndexNode* node,
                     const VisitedListPtr& vl,
                     const InnerSearchParam& search_param,
                     const DatasetPtr& query,
                     const FlattenInterfacePtr& codes,
                     QueryContext& ctx,
                     uint64_t subindex_ef_search,
                     DistanceRecordVector* rabitq_lower_bound_candidates) const {
    std::shared_lock lock(node->mutex_);
    DistHeapPtr results = nullptr;

    if (node->status_ == IndexNode::Status::FLAT) {
        results = std::make_shared<StandardHeap<true, false>>(allocator_, -1);
        if (search_param.time_cost != nullptr and search_param.time_cost->CheckOvertime() and
            ctx.stats != nullptr) {
            ctx.stats->is_timeout.store(true, std::memory_order_relaxed);
            return results;
        }
        const auto* ids_ptr = node->ids_.data();
        auto id_count = node->ids_.size();
        Vector<InnerIdType> valid_ids(allocator_);

        if (search_param.is_inner_id_allowed != nullptr) {
            const auto& inner_filter = search_param.is_inner_id_allowed;
            valid_ids.reserve(node->ids_.size());
            for (uint64_t i = 0; i < id_count; ++i) {
                if (inner_filter->CheckValid(ids_ptr[i])) {
                    valid_ids.push_back(ids_ptr[i]);
                }
            }
            ids_ptr = valid_ids.data();
            id_count = valid_ids.size();
        }

        Vector<float> dists(id_count, allocator_);
        auto computer = codes->FactoryComputer(query->GetFloat32Vectors());
        codes->Query(dists.data(), computer, ids_ptr, id_count, &ctx);

        for (int i = 0; i < id_count; ++i) {
            if (search_param.distance_threshold.has_value() and
                (not std::isfinite(dists[i]) ||
                 (not search_param.enable_reorder and
                  dists[i] > search_param.distance_threshold.value()))) {
                continue;
            }
            results->Push(dists[i], ids_ptr[i]);
            if (results->Size() > search_param.ef) {
                results->Pop();
            }
        }
    } else if (node->status_ == IndexNode::Status::GRAPH) {
        InnerSearchParam modified_param = search_param;
        modified_param.ep = node->entry_point_;
        if (node->level_ != 0) {
            modified_param.hops_limit = std::numeric_limits<uint32_t>::max();
            if (search_param.search_mode == KNN_SEARCH) {
                modified_param.ef = std::min(
                    modified_param.ef,
                    get_suitable_ef_search(
                        search_param.topk, node->graph_->TotalCount(), subindex_ef_search));
            }
        }
        modified_param.topk = static_cast<int64_t>(modified_param.ef);
        results = searcher_->Search(node->graph_,
                                    codes,
                                    vl,
                                    query->GetFloat32Vectors(),
                                    modified_param,
                                    label_table_,
                                    &ctx,
                                    rabitq_lower_bound_candidates);
    }

    return results;
}
void
Pyramid::SetImmutable() {
    if (this->immutable_.load(std::memory_order_acquire)) {
        return;
    }
    label_table_->SetImmutable();
    this->points_mutex_.reset();
    this->points_mutex_ = std::make_shared<EmptyMutex>();
    this->searcher_->SetMutexArray(this->points_mutex_);
    this->immutable_.store(true, std::memory_order_release);
}

float
Pyramid::CalcDistanceById(const float* query, int64_t id, bool calculate_precise_distance) const {
    std::shared_lock<std::shared_mutex> lock(resize_mutex_);
    auto flat = this->base_codes_;
    if (has_precise_reorder() && calculate_precise_distance) {
        flat = this->precise_codes_;
    }
    if (raw_vector_ != nullptr && calculate_precise_distance) {
        flat = this->raw_vector_;
    }
    return InnerIndexInterface::calc_distance_by_id(query, id, flat);
}

DatasetPtr
Pyramid::CalcDistancesById(const float* query,
                           const int64_t* ids,
                           int64_t count,
                           bool calculate_precise_distance) const {
    return this->CalDistanceById(query, ids, count, calculate_precise_distance);
}

DatasetPtr
Pyramid::CalDistanceById(const float* query,
                         const int64_t* ids,
                         int64_t count,
                         bool calculate_precise_distance,
                         int64_t topk) const {
    std::shared_lock<std::shared_mutex> lock(resize_mutex_);
    auto flat = this->base_codes_;
    if (has_precise_reorder() && calculate_precise_distance) {
        flat = this->precise_codes_;
    }
    if (raw_vector_ != nullptr && calculate_precise_distance) {
        flat = this->raw_vector_;
    }
    std::vector<bool> validity;
    auto result = InnerIndexInterface::cal_distance_by_id(query, ids, count, flat, &validity);
    if (topk == -1) {
        return result;
    }
    return ApplyTopkWithValidity(result->GetDistances(), ids, count, 1, topk, validity, allocator_);
}

void
Pyramid::GetVectorByInnerId(InnerIdType inner_id, float* data) const {
    std::shared_lock<std::shared_mutex> lock(resize_mutex_);
    auto codes = decodable_codes();
    bool release = false;
    const auto* buffer = codes->GetCodesById(inner_id, release);
    const bool decoded = codes->Decode(buffer, data);
    if (release) {
        codes->Release(buffer);
    }
    if (not decoded) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "Pyramid vector source does not support decode");
    }
}

std::string
Pyramid::GetStats() const {
    AnalyzerParam analyzer_param(allocator_);
    analyzer_param.topk = 10;
    analyzer_param.base_sample_size = std::min<uint64_t>(10, this->GetNumElements());
    analyzer_param.search_params = R"({"pyramid": {"ef_search": 500}})";
    auto analyzer = CreateAnalyzer(this, analyzer_param);
    JsonType stats = analyzer->GetStats();
    if (build_cache_hit_rate_ >= 0.0F) {
        stats["build_cache_hit_rate"].SetFloat(build_cache_hit_rate_);
        stats["build_cache_hit_nodes"].SetUint64(build_cache_hit_nodes_);
        stats["build_cache_missed_nodes"].SetUint64(build_cache_missed_nodes_);
    } else {
        stats["build_cache_hit_rate"]["skipped_reason"].SetString(
            "index was not built from an imported cache");
    }
    return stats.Dump(4);
}
void
Pyramid::collect_graph_nodes(IndexNode* node,
                             const std::string& node_path,

                             std::vector<std::pair<std::string, IndexNode*>>& out) {
    if (node == nullptr) {
        return;
    }
    if (node->status_ == IndexNode::Status::GRAPH) {
        out.emplace_back(node_path, node);
    }
    for (const auto& [key, child] : node->children_) {
        std::string child_path = node_path;
        if (child_path.empty()) {
            child_path = key;
        } else {
            child_path.push_back(PART_SLASH);
            child_path.append(key);
        }
        collect_graph_nodes(child.get(), child_path, out);
    }
}

void
Pyramid::init_index_nodes_with_ids(IndexNode* node) const {
    if (node == nullptr) {
        return;
    }
    if (not node->ids_.empty()) {
        node->Init();
    }
    for (const auto& [key, child] : node->children_) {
        init_index_nodes_with_ids(child.get());
    }
}

void
Pyramid::fulfill_cache(PyramidBuildCache& cache_snapshot) const {
    const auto& source_id_table = label_table_->GetSourceIdTableRef();
    if (source_id_table.empty()) {
        return;
    }

    UnorderedSet<std::string> seen_source_ids(allocator_);
    seen_source_ids.reserve(source_id_table.size());
    for (const auto& source_id : source_id_table) {
        if (source_id.empty()) {
            continue;
        }
        auto inserted = seen_source_ids.emplace(source_id);
        if (not inserted.second) {
            logger::warn(
                "[pyramid_build_cache] skip export because source_id "
                "is duplicated");
            return;
        }
    }

    for (const auto& [hname, h_ptr] : hierarchies_) {
        std::vector<std::pair<std::string, IndexNode*>> graph_nodes;
        collect_graph_nodes(h_ptr->root.get(), std::string{}, graph_nodes);
        for (const auto& [node_path, gnode] : graph_nodes) {
            std::shared_lock lock(gnode->mutex_);
            auto graph_ids = gnode->graph_->GetIds();
            if (graph_ids.empty()) {
                continue;
            }
            BuildCache graph_cache(allocator_);
            UnorderedMap<InnerIdType, InnerIdType> global_to_local(allocator_);
            global_to_local.reserve(graph_ids.size());
            for (auto inner_id : graph_ids) {
                if (static_cast<uint64_t>(inner_id) >= source_id_table.size()) {
                    continue;
                }
                const auto& source_id = source_id_table[inner_id];
                if (source_id.empty()) {
                    continue;
                }
                auto local_id = static_cast<InnerIdType>(graph_cache.source_ids_.size());
                global_to_local.emplace(inner_id, local_id);
                graph_cache.source_ids_.push_back(source_id);
            }
            for (auto inner_id : graph_ids) {
                auto source_iter = global_to_local.find(inner_id);
                if (source_iter == global_to_local.end()) {
                    continue;
                }
                Vector<InnerIdType> neighbors(allocator_);
                gnode->graph_->GetNeighbors(inner_id, neighbors);
                if (neighbors.empty()) {
                    continue;
                }
                Vector<InnerIdType> entry(allocator_);
                entry.push_back(source_iter->second);
                for (auto n : neighbors) {
                    auto neighbor_iter = global_to_local.find(n);
                    if (neighbor_iter != global_to_local.end()) {
                        entry.push_back(neighbor_iter->second);
                    }
                }
                const auto& source_id = graph_cache.source_ids_[source_iter->second];
                graph_cache.neighbors_.insert_or_assign(source_id, std::move(entry));
            }
            if (not graph_cache.neighbors_.empty()) {
                auto& target_cache = cache_snapshot.CreateGraphCache(hname, node_path);
                target_cache.source_ids_ = std::move(graph_cache.source_ids_);
                target_cache.neighbors_ = std::move(graph_cache.neighbors_);
            }
        }
    }
}

void
Pyramid::ExportCache(std::ostream& out_stream) const {
    IOStreamWriter writer(out_stream);
    PyramidBuildCache cache_snapshot(allocator_);
    if (not support_duplicate_) {
        this->fulfill_cache(cache_snapshot);
    } else {
        logger::warn(
            "[pyramid_build_cache] skip export because duplicate "
            "labels are enabled");
    }
    cache_snapshot.Serialize(writer);
}

void
Pyramid::ImportCache(std::istream& in_stream) {
    IOStreamReader reader(in_stream);
    this->cache_->Deserialize(reader);
}

std::vector<int64_t>
Pyramid::build_with_cache(const DatasetPtr& base) {
    build_cache_hit_rate_ = -1.0F;
    build_cache_hit_nodes_ = 0;
    build_cache_missed_nodes_ = 0;

    auto start = std::chrono::steady_clock::now();
    int64_t data_num = base->GetNumElements();
    const auto* data_vectors = base->GetFloat32Vectors();
    const auto* data_ids = base->GetIds();
    const auto* source_ids = base->GetSourceID();

    CHECK_ARGUMENT(source_ids != nullptr, "build_with_cache requires dataset with source_ids");
    CHECK_ARGUMENT(not support_duplicate_, "build_with_cache does not support duplicate labels");

    this->Train(base);
    resize(data_num);
    for (int64_t i = 0; i < data_num; ++i) {
        auto inner_id = static_cast<InnerIdType>(i);
        // Insert updates both the label remap and label-table total_count_.
        label_table_->Insert(inner_id, data_ids[i]);
        label_table_->InsertSourceId(inner_id, source_ids[i]);
    }
    base_codes_->BatchInsertVector(data_vectors, data_num);
    if (has_precise_reorder()) {
        precise_codes_->BatchInsertVector(data_vectors, data_num);
    }
    cur_element_count_ = data_num;

    for (const auto& [hname, h_ptr] : hierarchies_) {
        const auto* hpath = base->GetPaths(hname);
        if (hpath != nullptr) {
            populate_path_tree(*h_ptr, hpath, data_num);
        }
    }

    for (const auto& [hname, h_ptr] : hierarchies_) {
        init_index_nodes_with_ids(h_ptr->root.get());
    }

    auto codes = has_precise_reorder() ? precise_codes_ : base_codes_;
    std::vector<bool> global_hits(static_cast<size_t>(data_num), false);

    for (const auto& [hname, h_ptr] : hierarchies_) {
        std::vector<std::pair<std::string, IndexNode*>> graph_nodes;
        collect_graph_nodes(h_ptr->root.get(), std::string{}, graph_nodes);
        std::vector<bool> hierarchy_hits(static_cast<size_t>(data_num), false);

        UnorderedMap<std::string, InnerIdType> source_id_to_inner(allocator_);
        source_id_to_inner.reserve(data_num);
        for (InnerIdType id = 0; id < static_cast<InnerIdType>(data_num); ++id) {
            source_id_to_inner[source_ids[id]] = id;
        }

        for (const auto& [node_path, gnode] : graph_nodes) {
            Vector<InnerIdType> node_member_ids(allocator_);
            {
                std::shared_lock lock(gnode->mutex_);
                node_member_ids = gnode->ids_;
            }

            Vector<InnerIdType> node_missed_ids(allocator_);
            Vector<InnerIdType> node_hit_ids(allocator_);
            auto* graph_cache = cache_->GetGraphCache(hname, node_path);
            if (graph_cache != nullptr) {
                std::unique_lock lock(gnode->mutex_);
                UnorderedSet<InnerIdType> node_ids(allocator_);
                node_ids.reserve(node_member_ids.size());
                for (auto inner_id : node_member_ids) {
                    node_ids.insert(inner_id);
                }

                for (auto inner_id : node_member_ids) {
                    if (inner_id >= static_cast<InnerIdType>(data_num)) {
                        continue;
                    }
                    auto source_id = source_ids[inner_id];
                    auto cached = graph_cache->GetNeighbors(source_id);
                    if (cached.empty()) {
                        node_missed_ids.push_back(inner_id);
                        continue;
                    }

                    Vector<InnerIdType> new_neighbors(allocator_);
                    for (const auto& nb_src : cached) {
                        auto it = source_id_to_inner.find(nb_src);
                        if (it != source_id_to_inner.end() && it->second != inner_id &&
                            node_ids.find(it->second) != node_ids.end()) {
                            new_neighbors.push_back(it->second);
                        }
                    }
                    std::sort(new_neighbors.begin(), new_neighbors.end());
                    new_neighbors.erase(std::unique(new_neighbors.begin(), new_neighbors.end()),
                                        new_neighbors.end());

                    if (new_neighbors.empty()) {
                        node_missed_ids.push_back(inner_id);
                        continue;
                    }

                    if (gnode->graph_->TotalCount() == 0) {
                        gnode->entry_point_ = inner_id;
                    }

                    const auto max_deg = gnode->graph_->MaximumDegree();
                    if (new_neighbors.size() > max_deg) {
                        DistHeapPtr candidates =
                            std::make_shared<StandardHeap<true, false>>(allocator_, -1);
                        for (auto nb : new_neighbors) {
                            float dist = codes->ComputePairVectors(inner_id, nb);
                            candidates->Push(dist, nb);
                        }
                        while (candidates->Size() > max_deg) {
                            candidates->Pop();
                        }
                        new_neighbors.clear();
                        new_neighbors.reserve(max_deg);
                        while (!candidates->Empty()) {
                            new_neighbors.push_back(candidates->Top().second);
                            candidates->Pop();
                        }
                    }
                    // Cache entries seed only outgoing edges. add_one_point() below refines them
                    // against current vectors and installs deduplicated reverse edges.
                    gnode->graph_->InsertNeighborsById(inner_id, new_neighbors);
                    node_hit_ids.push_back(inner_id);
                    hierarchy_hits[static_cast<size_t>(inner_id)] = true;
                }
            } else {
                node_missed_ids = node_member_ids;
            }

            IndexNode* const graph_node = gnode;
            auto refine_nodes = [this, &h = *h_ptr, graph_node, data_vectors](
                                    const Vector<InnerIdType>& ids,
                                    uint64_t ef_construction,
                                    bool use_self_as_entry) {
                auto refine_one =
                    [this, &h, graph_node, data_vectors, ef_construction, use_self_as_entry](
                        InnerIdType inner_id) {
                        add_one_point(h,
                                      graph_node,
                                      inner_id,
                                      data_vectors + dim_ * inner_id,
                                      ef_construction,
                                      use_self_as_entry);
                    };

                Vector<std::future<void>> futures(allocator_);
                // All futures are joined below before graph_node or data_vectors can go out of scope.
                std::exception_ptr first_error;
                for (const auto inner_id : ids) {
                    try {
                        if (thread_pool_ != nullptr) {
                            futures.push_back(thread_pool_->GeneralEnqueue(refine_one, inner_id));
                        } else {
                            refine_one(inner_id);
                        }
                    } catch (...) {
                        first_error = std::current_exception();
                        break;
                    }
                }
                for (auto& future : futures) {
                    try {
                        future.get();
                    } catch (...) {
                        if (first_error == nullptr) {
                            first_error = std::current_exception();
                        }
                    }
                }
                if (first_error != nullptr) {
                    std::rethrow_exception(first_error);
                }
            };

            refine_nodes(node_missed_ids, h_ptr->ef_construction, false);
            // Match HGraph's warm hit phase: self-entry keeps refinement local and the
            // reduced budget limits work while merged cache rows preserve graph connectivity.
            refine_nodes(node_hit_ids, std::max<uint64_t>(h_ptr->ef_construction / 3, 1), true);
            Vector<InnerIdType>(allocator_).swap(gnode->ids_);
        }

        for (InnerIdType id = 0; id < static_cast<InnerIdType>(data_num); ++id) {
            if (hierarchy_hits[static_cast<size_t>(id)]) {
                global_hits[static_cast<size_t>(id)] = true;
            }
        }
    }

    for (bool is_hit : global_hits) {
        if (is_hit) {
            ++build_cache_hit_nodes_;
        }
    }
    build_cache_missed_nodes_ = static_cast<uint64_t>(data_num) - build_cache_hit_nodes_;

    uint64_t total = build_cache_hit_nodes_ + build_cache_missed_nodes_;
    if (total > 0) {
        build_cache_hit_rate_ =
            static_cast<float>(build_cache_hit_nodes_) / static_cast<float>(total);
    } else {
        build_cache_hit_rate_ = 0.0F;
    }

    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    logger::info(
        "[pyramid_build_cache] completed in {} ms, hit_rate={:.4f}, "
        "hit={}, missed={}",
        elapsed_ms,
        build_cache_hit_rate_,
        build_cache_hit_nodes_,
        build_cache_missed_nodes_);

    // Imported-cache eligibility rejects duplicate labels, so this matches the normal Build result
    // for the same valid dataset: no failed ids.
    return {};
}

std::string
Pyramid::AnalyzeIndexBySearch(const SearchRequest& request) {
    CHECK_ARGUMENT(request.mode_ == SearchMode::KNN_SEARCH,
                   "Pyramid AnalyzeIndexBySearch only supports KNN search");
    const bool is_supported_search =
        not request.enable_filter_ && not request.enable_bitset_filter_ &&
        not request.enable_attribute_filter_ && not request.enable_iterator_search_;
    CHECK_ARGUMENT(is_supported_search,
                   "Pyramid AnalyzeIndexBySearch does not support "
                   "filtered or iterator search");
    CHECK_ARGUMENT(request.topk_ > 0,
                   fmt::format("topk({}) must be greater than 0", request.topk_));
    CHECK_ARGUMENT(request.topk_ <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
                   fmt::format("topk({}) exceeds the supported maximum", request.topk_));
    CHECK_ARGUMENT(base_codes_->TotalCount() > 0,
                   "Pyramid AnalyzeIndexBySearch requires a built index");
    AnalyzerParam analyzer_param(allocator_);
    analyzer_param.topk = request.topk_;
    auto analyzer = CreateAnalyzer(this, analyzer_param);
    return analyzer->AnalyzeIndexBySearch(request).Dump(4);
}

}  // namespace vsag
