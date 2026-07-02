
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

#include <chrono>

#include "algorithm/build_cache.h"
#include "algorithm/inner_index_interface.h"
#include "analyzer/analyzer.h"
#include "datacell/flatten_interface.h"
#include "impl/heap/standard_heap.h"
#include "impl/odescent/odescent_graph_builder.h"
#include "impl/pruning_strategy.h"
#include "io/memory_io_parameter.h"
#include "query_context.h"
#include "storage/empty_index_binary_set.h"
#include "storage/serialization.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "utils/slow_task_timer.h"
#include "utils/util_functions.h"
namespace vsag {

const static float RADIUS_EPSILON = 1.1F;

static constexpr uint64_t SOURCE_ID_TABLE_MAGIC = 0x534F555243454944ULL;

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
    if (status_ != IndexNode::Status::NO_INDEX) {
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

    resize(data_num);
    std::memcpy(label_table_->label_table_.data(), data_ids, sizeof(LabelType) * data_num);
    const auto* source_ids = base->GetSourceID();
    if (source_ids != nullptr) {
        for (int64_t i = 0; i < data_num; ++i) {
            label_table_->InsertSourceId(static_cast<InnerIdType>(i), source_ids[i]);
        }
    }

    label_table_->total_count_.store(data_num);
    base_codes_->BatchInsertVector(data_vectors, data_num);
    if (use_reorder_) {
        precise_codes_->BatchInsertVector(data_vectors, data_num);
    }
    auto codes = use_reorder_ ? precise_codes_ : base_codes_;

    if (thread_pool_ != nullptr && hierarchies_.size() > 1) {
        Vector<std::future<void>> futures(allocator_);
        for (const auto& [hname, h_ptr] : hierarchies_) {
            auto* root_ptr = h_ptr->root.get();
            futures.push_back(thread_pool_->GeneralEnqueue([&, codes, root_ptr]() {
                ODescent builder(odescent_param_, codes, allocator_, nullptr);
                root_ptr->Build(builder);
            }));
        }
        for (auto& f : futures) {
            f.get();
        }
    } else {
        ODescent graph_builder(odescent_param_, codes, allocator_, this->thread_pool_.get());
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

    auto parsed_param = PyramidSearchParameters::FromJson(parameters);
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
    search_param.topk = k;
    search_param.search_mode = KNN_SEARCH;
    search_param.parallel_search_thread_count = parsed_param.parallel_search_thread_count;
    if (this->support_duplicate_) {
        search_param.consider_duplicate = true;
    }

    if (parsed_param.enable_time_record) {
        search_param.time_cost = std::make_shared<Timer>();
        search_param.time_cost->SetThreshold(parsed_param.timeout_ms);
    }

    search_param.is_inner_id_allowed = this->create_search_filter(filter);
    SearchFunc search_func = [&](const IndexNode* node, const VisitedListPtr& vl) {
        return this->search_node(
            node, vl, search_param, query, base_codes_, ctx, parsed_param.subindex_ef_search);
    };

    std::string hierarchy_name =
        parsed_param.hierarchies.empty() ? "" : parsed_param.hierarchies[0];
    auto result = this->search_impl(query, search_func, search_param, hierarchy_name);
    result->Statistics(stats.Dump());
    return result;
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
    CHECK_ARGUMENT(parsed_param.hierarchy_op == PyramidSearchParameters::HierarchyOp::SINGLE,
                   "multi-hierarchy search (union/intersection) is not yet implemented");
    InnerSearchParam search_param;
    search_param.ef = parsed_param.ef_search;
    search_param.radius = radius * RADIUS_EPSILON;
    search_param.search_mode = RANGE_SEARCH;
    search_param.parallel_search_thread_count = parsed_param.parallel_search_thread_count;
    search_param.topk = limited_size == -1 ? std::numeric_limits<int64_t>::max() : limited_size;

    if (parsed_param.enable_time_record) {
        search_param.time_cost = std::make_shared<Timer>();
        search_param.time_cost->SetThreshold(parsed_param.timeout_ms);
    }

    if (this->support_duplicate_) {
        search_param.consider_duplicate = true;
    }

    search_param.is_inner_id_allowed = this->create_search_filter(filter);
    SearchFunc search_func = [&](const IndexNode* node, const VisitedListPtr& vl) {
        return this->search_node(
            node, vl, search_param, query, base_codes_, ctx, parsed_param.subindex_ef_search);
    };

    std::string hierarchy_name =
        parsed_param.hierarchies.empty() ? "" : parsed_param.hierarchies[0];
    auto result = this->search_impl(query, search_func, search_param, hierarchy_name);
    result->Statistics(stats.Dump());
    return result;
}

DatasetPtr
Pyramid::search_impl(const DatasetPtr& query,
                     const SearchFunc& search_func,
                     InnerSearchParam& search_param,
                     const std::string& hierarchy_name) const {
    SearchStatistics stats;
    QueryContext ctx{.stats = &stats};

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
    auto vl = pool_->TakeOne();
    if (query_path != nullptr) {
        const std::string& current_path = query_path[0];
        search_hierarchy(h, search_func, vl, search_result, current_path, search_param);
    } else {
        h.root->Search(search_func, vl, search_result, search_param.ef);
    }
    pool_->ReturnOne(vl);

    if (use_reorder_) {
        search_result = this->reorder_->Reorder(
            search_result, query->GetFloat32Vectors(), search_param.topk, ctx);
    }

    if (search_result->Empty()) {
        return DatasetImpl::MakeEmptyDataset();
    }

    while (search_result->Size() > search_param.topk ||
           search_result->Top().first > search_param.radius) {
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

    if (this->persist_source_id_) {
        const auto& sid_table = this->label_table_->GetSourceIdTableRef();
        StreamWriter::WriteObj(writer, SOURCE_ID_TABLE_MAGIC);
        uint64_t sid_count = sid_table.size();
        StreamWriter::WriteObj(writer, sid_count);
        for (uint64_t i = 0; i < sid_count; ++i) {
            StreamWriter::WriteString(writer, sid_table[i]);
        }
    }

    base_codes_->Serialize(writer);
    if (use_reorder_) {
        precise_codes_->Serialize(writer);
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

    {
        const uint64_t cursor_before = buffer_reader.GetCursor();
        if (buffer_reader.Length() >= cursor_before + sizeof(uint64_t)) {
            uint64_t magic = 0;
            StreamReader::ReadObj(buffer_reader, magic);
            if (magic == SOURCE_ID_TABLE_MAGIC) {
                uint64_t sid_count = 0;
                StreamReader::ReadObj(buffer_reader, sid_count);
                const uint64_t label_table_size = this->label_table_->label_table_.size();
                if (sid_count > label_table_size) {
                    throw VsagException(
                        ErrorType::INVALID_ARGUMENT,
                        fmt::format("corrupted index: source_id_table sid_count ({}) "
                                    "exceeds label_table size ({})",
                                    sid_count,
                                    label_table_size));
                }
                Vector<std::string> sid_table(sid_count, std::string{}, allocator_);
                for (uint64_t i = 0; i < sid_count; ++i) {
                    sid_table[i] = StreamReader::ReadString(buffer_reader);
                }
                this->label_table_->ReplaceSourceIdTable(std::move(sid_table));
            } else {
                buffer_reader.Seek(cursor_before);
            }
        }
    }

    delete_count_.store(static_cast<int64_t>(label_table_->GetAllDeletedIds().size()),
                        std::memory_order_relaxed);
    base_codes_->Deserialize(buffer_reader);
    if (use_reorder_) {
        precise_codes_->Deserialize(buffer_reader);
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
    this->current_memory_usage_ = static_cast<int64_t>(this->CalSerializeSize());
}

InnerIndexPtr
Pyramid::ExportModel(const IndexCommonParam& param) const {
    auto index = std::make_shared<Pyramid>(this->create_param_ptr_, param);
    if (index->use_reorder_ != this->use_reorder_) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "Export model's pyramid reorder config mismatched");
    }
    this->base_codes_->ExportModel(index->base_codes_);
    if (use_reorder_) {
        if (index->precise_codes_ == nullptr) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                "Export model's pyramid precise codes is empty");
        }
        this->precise_codes_->ExportModel(index->precise_codes_);
    }
    index->current_memory_usage_ = static_cast<int64_t>(index->CalSerializeSize());
    return index;
}

std::vector<int64_t>
Pyramid::Add(const DatasetPtr& base) {
    int64_t data_num = base->GetNumElements();
    const auto* data_vectors = base->GetFloat32Vectors();
    const auto* data_ids = base->GetIds();
    std::vector<int64_t> failed_ids;
    Vector<int64_t> data_biases(allocator_);
    int64_t local_cur_element_count = 0;
    {
        std::lock_guard lock(cur_element_count_mutex_);
        local_cur_element_count = cur_element_count_;
        if (max_capacity_ == 0) {
            auto new_capacity = std::max(INIT_CAPACITY, data_num);
            resize(new_capacity);
        } else if (max_capacity_ < data_num + cur_element_count_) {
            auto new_capacity = std::min(MAX_CAPACITY_EXTEND, max_capacity_);
            new_capacity = std::max(data_num + cur_element_count_ - max_capacity_, new_capacity) +
                           max_capacity_;
            resize(new_capacity);
        }
        int64_t valid_id_count = 0;
        for (int64_t i = 0; i < data_num; ++i) {
            if (not label_table_->CheckLabel(data_ids[i])) {
                label_table_->Insert(valid_id_count + local_cur_element_count, data_ids[i]);
                {
                    const auto* src_ids = base->GetSourceID();
                    if (src_ids != nullptr) {
                        label_table_->InsertSourceId(valid_id_count + local_cur_element_count,
                                                     src_ids[i]);
                    }
                }
                base_codes_->InsertVector(data_vectors + dim_ * i,
                                          valid_id_count + local_cur_element_count);
                if (use_reorder_) {
                    precise_codes_->InsertVector(data_vectors + dim_ * i,
                                                 valid_id_count + local_cur_element_count);
                }
                valid_id_count++;
                data_biases.push_back(i);
            } else {
                logger::warn("Label {} already exists, skip adding.", data_ids[i]);
                failed_ids.push_back(data_ids[i]);
            }
        }
        cur_element_count_ += valid_id_count;
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
    if (use_reorder_) {
        precise_codes_->Resize(new_max_capacity);
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
                "{RABITQ_QUANTIZATION_BITS_PER_DIM_QUERY_KEY}": 32,
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
                "{PRODUCT_QUANTIZATION_DIM_KEY}": 1,
                "{HOLD_MOLDS}": false
            }
        },
        "{BUILD_THREAD_COUNT_KEY}": 1,
        "{EF_CONSTRUCTION_KEY}": 400,
        "{NO_BUILD_LEVELS}":[],
        "{INDEX_MIN_SIZE}": 0,
        "{SUPPORT_DUPLICATE}": false,
        "{HGRAPH_PERSIST_SOURCE_ID_KEY}": false
    })";

ParamPtr
Pyramid::CheckAndMappingExternalParam(const JsonType& external_param,
                                      const IndexCommonParam& common_param) {
    const ConstParamMap external_mapping = {
        {PYRAMID_EF_CONSTRUCTION, {EF_CONSTRUCTION_KEY}},
        {PYRAMID_USE_REORDER, {USE_REORDER_KEY}},
        {PYRAMID_BASE_QUANTIZATION_TYPE, {BASE_CODES_KEY, QUANTIZATION_PARAMS_KEY, TYPE_KEY}},
        {PYRAMID_RABITQ_BITS_PER_DIM_BASE,
         {BASE_CODES_KEY, QUANTIZATION_PARAMS_KEY, RABITQ_QUANTIZATION_BITS_PER_DIM_BASE_KEY}},
        {PYRAMID_RABITQ_BITS_PER_DIM_QUERY,
         {BASE_CODES_KEY, QUANTIZATION_PARAMS_KEY, RABITQ_QUANTIZATION_BITS_PER_DIM_QUERY_KEY}},
        {PYRAMID_RABITQ_PCA_DIM, {BASE_CODES_KEY, QUANTIZATION_PARAMS_KEY, PCA_DIM_KEY}},
        {PYRAMID_RABITQ_USE_FHT, {BASE_CODES_KEY, QUANTIZATION_PARAMS_KEY, USE_FHT_KEY}},
        {PYRAMID_PRECISE_QUANTIZATION_TYPE, {PRECISE_CODES_KEY, QUANTIZATION_PARAMS_KEY, TYPE_KEY}},
        {PYRAMID_GRAPH_MAX_DEGREE, {GRAPH_KEY, GRAPH_PARAM_MAX_DEGREE_KEY}},
        {PYRAMID_BASE_IO_TYPE, {BASE_CODES_KEY, IO_PARAMS_KEY, TYPE_KEY}},
        {PYRAMID_BUILD_ALPHA, {GRAPH_KEY, ODESCENT_PARAMETER_ALPHA}},
        {PYRAMID_GRAPH_TYPE, {GRAPH_KEY, GRAPH_TYPE_KEY}},
        {PYRAMID_GRAPH_STORAGE_TYPE, {GRAPH_KEY, GRAPH_STORAGE_TYPE_KEY}},
        {PYRAMID_PRECISE_IO_TYPE, {PRECISE_CODES_KEY, IO_PARAMS_KEY, TYPE_KEY}},
        {PYRAMID_BUILD_THREAD_COUNT, {BUILD_THREAD_COUNT_KEY}},
        {PYRAMID_NO_BUILD_LEVELS, {NO_BUILD_LEVELS}},
        {PYRAMID_PERSIST_SOURCE_ID, {HGRAPH_PERSIST_SOURCE_ID_KEY}},
        {PYRAMID_HIERARCHIES, {PYRAMID_HIERARCHIES}},
        {PYRAMID_BASE_PQ_DIM,
         {BASE_CODES_KEY, QUANTIZATION_PARAMS_KEY, PRODUCT_QUANTIZATION_DIM_KEY}},
        {PYRAMID_BASE_FILE_PATH, {BASE_CODES_KEY, IO_PARAMS_KEY, IO_FILE_PATH_KEY}},
        {PYRAMID_PRECISE_FILE_PATH, {PRECISE_CODES_KEY, IO_PARAMS_KEY, IO_FILE_PATH_KEY}},
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
    auto pyramid_params = std::make_shared<PyramidParameters>();
    pyramid_params->FromJson(inner_json);
    return pyramid_params;
}

void
Pyramid::Train(const DatasetPtr& base) {
    this->base_codes_->Train(base->GetFloat32Vectors(), base->GetNumElements());
    if (use_reorder_) {
        this->precise_codes_->Train(base->GetFloat32Vectors(), base->GetNumElements());
    }
}
std::vector<int64_t>
Pyramid::Build(const DatasetPtr& base) {
    CHECK_ARGUMENT(GetNumElements() == 0, "index is not empty");
    int64_t data_num = base->GetNumElements();

    this->build_cache_hit_rate_ = -1.0F;
    this->build_cache_hit_nodes_ = 0;
    this->build_cache_missed_nodes_ = 0;

    if (this->has_loaded_cache()) {
        auto ret = this->build_with_cache(base);
        return ret;
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
                       const float* vector) {
    std::unique_lock graph_lock(node->mutex_);

    if (node->status_ == IndexNode::Status::NO_INDEX) {
        node->Init();
        Vector<InnerIdType>(allocator_).swap(node->ids_);
    }

    if (node->status_ == IndexNode::Status::FLAT) {
        node->ids_.push_back(inner_id);
        return;
    }

    if (node->graph_->TotalCount() == 0) {
        node->graph_->InsertNeighborsById(inner_id, Vector<InnerIdType>(allocator_));
        node->entry_point_ = inner_id;
    } else {
        InnerSearchParam search_param;
        search_param.ef = h.ef_construction;
        search_param.topk = static_cast<int64_t>(h.ef_construction);
        search_param.search_mode = KNN_SEARCH;
        search_param.hops_limit = 10000;
        if (support_duplicate_) {
            search_param.find_duplicate = true;
            search_param.duplicate_query_id = inner_id;
        }
        auto codes = use_reorder_ ? precise_codes_ : base_codes_;
        bool update_entry_point;
        {
            std::scoped_lock<std::mutex> entry_point_lock(entry_point_mutex_);
            update_entry_point = is_update_entry_point(node->graph_->TotalCount());
        }
        search_param.ep = node->entry_point_;
        if (not update_entry_point) {
            graph_lock.unlock();
        }

        auto vl = pool_->TakeOne();
        auto results = searcher_->Search(
            node->graph_, codes, vl, vector, search_param, (LabelTablePtr) nullptr, nullptr);
        pool_->ReturnOne(vl);
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
        const auto* vector = data_vectors + dim_ * data_bias;
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
                    auto local_vl = pool_->TakeOne();
                    node->Search(search_func, local_vl, search_result_lists[i], search_param.ef);
                    pool_->ReturnOne(local_vl);
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
                     uint64_t subindex_ef_search) const {
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
        codes->Query(dists.data(), computer, ids_ptr, id_count);

        for (int i = 0; i < id_count; ++i) {
            results->Push(dists[i], ids_ptr[i]);
            if (results->Size() > search_param.ef) {
                results->Pop();
            }
        }
    } else if (node->status_ == IndexNode::Status::GRAPH) {
        InnerSearchParam modified_param = search_param;
        modified_param.ep = node->entry_point_;
        if (node->level_ != 0 && search_param.search_mode == KNN_SEARCH) {
            modified_param.ef =
                std::min(modified_param.ef,
                         get_suitable_ef_search(
                             search_param.topk, node->graph_->TotalCount(), subindex_ef_search));
        }
        modified_param.topk = static_cast<int64_t>(modified_param.ef);
        results = searcher_->Search(node->graph_,
                                    codes,
                                    vl,
                                    query->GetFloat32Vectors(),
                                    modified_param,
                                    label_table_,
                                    &ctx);
    }

    return results;
}
void
Pyramid::SetImmutable() {
    if (this->immutable_) {
        return;
    }
    label_table_->SetImmutable();
    this->points_mutex_.reset();
    this->points_mutex_ = std::make_shared<EmptyMutex>();
    this->searcher_->SetMutexArray(this->points_mutex_);
    immutable_ = true;
}

float
Pyramid::CalcDistanceById(const float* query, int64_t id, bool calculate_precise_distance) const {
    std::shared_lock<std::shared_mutex> lock(resize_mutex_);
    auto flat = this->base_codes_;
    if (use_reorder_ && calculate_precise_distance) {
        flat = this->precise_codes_;
    }
    return InnerIndexInterface::calc_distance_by_id(query, id, flat);
}

DatasetPtr
Pyramid::CalDistanceById(const float* query,
                         const int64_t* ids,
                         int64_t count,
                         bool calculate_precise_distance) const {
    std::shared_lock<std::shared_mutex> lock(resize_mutex_);
    auto flat = this->base_codes_;
    if (use_reorder_ && calculate_precise_distance) {
        flat = this->precise_codes_;
    }
    return InnerIndexInterface::cal_distance_by_id(query, ids, count, flat);
}

void
Pyramid::GetVectorByInnerId(InnerIdType inner_id, float* data) const {
    std::shared_lock<std::shared_mutex> lock(resize_mutex_);
    auto codes = (use_reorder_) ? precise_codes_ : base_codes_;
    bool release = false;
    const auto* buffer = codes->GetCodesById(inner_id, release);
    codes->Decode(buffer, data);
    if (release) {
        codes->Release(buffer);
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

    if (this->build_cache_hit_rate_ >= 0.0F) {
        stats["build_cache_hit_rate"].SetFloat(this->build_cache_hit_rate_);
        stats["build_cache_hit_nodes"].SetInt(static_cast<int64_t>(this->build_cache_hit_nodes_));
        stats["build_cache_missed_nodes"].SetInt(
            static_cast<int64_t>(this->build_cache_missed_nodes_));
    } else {
        stats["build_cache_hit_rate"]["skipped_reason"].SetString(
            "index was not built from an imported cache");
    }

    return stats.Dump(4);
}

void
Pyramid::collect_graph_nodes(IndexNode* node, std::vector<IndexNode*>& out) {
    if (node == nullptr) {
        return;
    }
    if (node->status_ == IndexNode::Status::GRAPH) {
        out.push_back(node);
    }
    for (auto& [key, child] : node->children_) {
        collect_graph_nodes(child.get(), out);
    }
}

void
Pyramid::fullfill_cache() {
    auto total_count = static_cast<InnerIdType>(label_table_->total_count_.load());
    // Track which inner_ids we've already captured so that the same vector
    // appearing in multiple IndexNode graphs (e.g. ancestor levels) doesn't
    // produce duplicate entries in cache_->neighbors_, which only stores the
    // most recent snapshot via emplace().
    UnorderedSet<InnerIdType> seen(allocator_);
    for (auto& [hname, h_ptr] : hierarchies_) {
        std::vector<IndexNode*> graph_nodes;
        collect_graph_nodes(h_ptr->root.get(), graph_nodes);
        for (auto* gnode : graph_nodes) {
            std::shared_lock lock(gnode->mutex_);
            // IndexNode::Build drains ids_ after persisting the graph, so we
            // can't use gnode->ids_ here. Walk the graph's id space directly:
            // every id in [0, TotalCount()) was inserted via
            // InsertNeighborsById (even the seed entry, which carries an
            // empty neighbor list). GetNeighbors() on ids that were routed
            // into a sibling IndexNode simply returns empty -- we can't tell
            // those apart from a freshly-seeded entry by neighbor count
            // alone, so we rely on the `seen` set to keep a single canonical
            // entry per global inner_id. Downstream, build_with_cache()
            // translates each cached neighbor inner_id back to its source_id
            // and re-discovers the right IndexNode via populate_path_tree;
            // neighbors that physically live in a different IndexNode will
            // miss the path filter and be dropped before insertion.
            InnerIdType graph_total = static_cast<InnerIdType>(gnode->graph_->TotalCount());
            InnerIdType scan_count = std::min<InnerIdType>(graph_total, total_count);
            for (InnerIdType inner_id = 0; inner_id < scan_count; ++inner_id) {
                if (seen.find(inner_id) != seen.end()) {
                    continue;
                }
                auto source_id = label_table_->GetSourceId(inner_id);
                if (source_id.empty()) {
                    continue;
                }
                seen.insert(inner_id);
                Vector<InnerIdType> neighbors(allocator_);
                gnode->graph_->GetNeighbors(inner_id, neighbors);
                if (static_cast<uint64_t>(inner_id) >= cache_->source_ids_.size()) {
                    cache_->source_ids_.resize(static_cast<uint64_t>(inner_id) + 1);
                }
                cache_->source_ids_[inner_id] = source_id;
                Vector<InnerIdType> entry(allocator_);
                entry.push_back(inner_id);
                for (auto n : neighbors) {
                    entry.push_back(n);
                }
                cache_->neighbors_.emplace(source_id, std::move(entry));
            }
        }
    }
}

void
Pyramid::ExportCache(std::ostream& out_stream) const {
    IOStreamWriter writer(out_stream);
    const_cast<Pyramid*>(this)->fullfill_cache();
    this->cache_->Serialize(writer);
}

void
Pyramid::ImportCache(std::istream& in_stream) {
    IOStreamReader reader(in_stream);
    this->cache_->Deserialize(reader);
}

std::vector<int64_t>
Pyramid::build_with_cache(const DatasetPtr& base) {
    auto start = std::chrono::steady_clock::now();
    int64_t data_num = base->GetNumElements();
    const auto* data_vectors = base->GetFloat32Vectors();
    const auto* data_ids = base->GetIds();
    const auto* source_ids = base->GetSourceID();

    CHECK_ARGUMENT(source_ids != nullptr, "build_with_cache requires dataset with source_ids");

    this->Train(base);
    resize(data_num);
    std::memcpy(label_table_->label_table_.data(), data_ids, sizeof(LabelType) * data_num);
    for (int64_t i = 0; i < data_num; ++i) {
        label_table_->InsertSourceId(static_cast<InnerIdType>(i), source_ids[i]);
    }
    label_table_->total_count_.store(data_num);
    base_codes_->BatchInsertVector(data_vectors, data_num);
    if (use_reorder_) {
        precise_codes_->BatchInsertVector(data_vectors, data_num);
    }
    cur_element_count_ = data_num;

    for (const auto& [hname, h_ptr] : hierarchies_) {
        const auto* hpath = base->GetPaths(hname);
        if (hpath != nullptr) {
            populate_path_tree(*h_ptr, hpath, data_num);
        }
    }

    for (auto& [hname, h_ptr] : hierarchies_) {
        std::function<void(IndexNode*)> init_all = [&](IndexNode* node) {
            if (node == nullptr)
                return;
            node->Init();
            for (auto& [k, child] : node->children_) {
                init_all(child.get());
            }
        };
        init_all(h_ptr->root.get());
    }

    Vector<InnerIdType> hit_ids(allocator_);
    Vector<InnerIdType> missed_ids(allocator_);
    hit_ids.reserve(data_num);
    missed_ids.reserve(data_num);

    auto codes = use_reorder_ ? precise_codes_ : base_codes_;

    for (auto& [hname, h_ptr] : hierarchies_) {
        std::vector<IndexNode*> graph_nodes;
        collect_graph_nodes(h_ptr->root.get(), graph_nodes);
        if (graph_nodes.empty()) {
            for (InnerIdType id = 0; id < static_cast<InnerIdType>(data_num); ++id) {
                missed_ids.push_back(id);
            }
            continue;
        }

        UnorderedMap<std::string, InnerIdType> source_id_to_inner(allocator_);
        source_id_to_inner.reserve(data_num);
        for (InnerIdType id = 0; id < static_cast<InnerIdType>(data_num); ++id) {
            source_id_to_inner[source_ids[id]] = id;
        }

        for (auto* gnode : graph_nodes) {
            std::unique_lock lock(gnode->mutex_);
            for (auto inner_id : gnode->ids_) {
                if (inner_id >= static_cast<InnerIdType>(data_num)) {
                    continue;
                }
                auto source_id = source_ids[inner_id];
                auto cached = cache_->GetNeighbors(source_id);
                if (cached.empty()) {
                    continue;
                }

                Vector<InnerIdType> new_neighbors(allocator_);
                for (const auto& nb_src : cached) {
                    auto it = source_id_to_inner.find(nb_src);
                    if (it != source_id_to_inner.end() && it->second != inner_id) {
                        new_neighbors.push_back(it->second);
                    }
                }
                std::sort(new_neighbors.begin(), new_neighbors.end());
                new_neighbors.erase(std::unique(new_neighbors.begin(), new_neighbors.end()),
                                    new_neighbors.end());

                if (new_neighbors.empty()) {
                    continue;
                }

                if (gnode->graph_->TotalCount() == 0) {
                    gnode->entry_point_ = inner_id;
                }

                // SparseGraphDataCell enforces a hard max_degree on every
                // InsertNeighborsById call. When the warm seed already fits,
                // keep the translated list as-is so the refined graph keeps
                // the topology captured by the cache. Only when the cached
                // neighbour set is wider than the cap do we need distance
                // information to drop the farthest entries.
                const auto max_deg = gnode->graph_->MaximumDegree();
                if (new_neighbors.size() > max_deg) {
                    DistHeapPtr candidates =
                        std::make_shared<StandardHeap<true, false>>(allocator_, -1);
                    for (auto nb : new_neighbors) {
                        float dist = codes->ComputePairVectors(inner_id, nb);
                        candidates->Push(dist, nb);
                    }
                    new_neighbors.clear();
                    new_neighbors.reserve(max_deg);
                    while (!candidates->Empty() && new_neighbors.size() < max_deg) {
                        new_neighbors.push_back(candidates->Top().second);
                        candidates->Pop();
                    }
                }
                gnode->graph_->InsertNeighborsById(inner_id, new_neighbors);
                for (auto nb : new_neighbors) {
                    Vector<InnerIdType> existing(allocator_);
                    gnode->graph_->GetNeighbors(nb, existing);
                    bool has_reverse = false;
                    for (auto e : existing) {
                        if (e == inner_id) {
                            has_reverse = true;
                            break;
                        }
                    }
                    if (has_reverse) {
                        continue;
                    }
                    if (existing.size() < max_deg) {
                        existing.push_back(inner_id);
                        gnode->graph_->InsertNeighborsById(nb, existing);
                    }
                    // else: neighbor is already at the degree cap; let the
                    // downstream Add/odescent refinement step decide whether
                    // to expand this adjacency later.
                }
                hit_ids.push_back(inner_id);
            }
        }

        std::vector<bool> is_hit(data_num, false);
        for (auto hid : hit_ids) {
            if (hid < static_cast<InnerIdType>(data_num)) {
                is_hit[hid] = true;
            }
        }
        for (InnerIdType id = 0; id < static_cast<InnerIdType>(data_num); ++id) {
            if (!is_hit[id]) {
                missed_ids.push_back(id);
            }
        }
        build_cache_hit_nodes_ += hit_ids.size();
        build_cache_missed_nodes_ += missed_ids.size();

        hit_ids.clear();
        const auto* hpath = base->GetPaths(hname);
        for (auto mid : missed_ids) {
            int64_t bias = static_cast<int64_t>(mid);
            if (hpath == nullptr)
                continue;
            std::string current_path = hpath[bias];
            auto path_slices = split(current_path, PART_SLASH);
            IndexNode* node = h_ptr->root.get();
            for (auto& slice : path_slices) {
                auto* child = node->GetChild(slice, false);
                if (child == nullptr)
                    break;
                node = child;
            }
            add_one_point(*h_ptr, node, mid, data_vectors + dim_ * bias);
        }
        missed_ids.clear();

        if (graph_type_ != GRAPH_TYPE_VALUE_NSW) {
            ODescent builder(odescent_param_, codes, allocator_, this->thread_pool_.get());
            h_ptr->root->Build(builder);
        }
    }

    uint64_t total = build_cache_hit_nodes_ + build_cache_missed_nodes_;
    if (total > 0) {
        build_cache_hit_rate_ =
            static_cast<float>(build_cache_hit_nodes_) / static_cast<float>(total);
    } else {
        build_cache_hit_rate_ = 0.0F;
    }

    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    logger::info("[pyramid_build_cache] completed in {} ms, hit_rate={:.4f}, hit={}, missed={}",
                 elapsed_ms,
                 build_cache_hit_rate_,
                 build_cache_hit_nodes_,
                 build_cache_missed_nodes_);

    return {};
}

}  // namespace vsag
