
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

#include "algorithm/inner_index_interface.h"
#include "datacell/flatten_interface.h"
#include "impl/heap/standard_heap.h"
#include "impl/odescent/odescent_graph_builder.h"
#include "impl/pruning_strategy.h"
#include "io/memory_io_parameter.h"
#include "storage/empty_index_binary_set.h"
#include "storage/serialization.h"
#include "utils/slow_task_timer.h"
#include "utils/util_functions.h"
namespace vsag {

std::vector<std::string>
split(const std::string& str, char delimiter) {
    auto vec = split_string(str, delimiter);
    vec.erase(
        std::remove_if(vec.begin(), vec.end(), [](const std::string& s) { return s.empty(); }),
        vec.end());
    return vec;
}

IndexNode::IndexNode(IndexCommonParam* common_param, GraphInterfaceParamPtr graph_param)
    : ids_(common_param->allocator_.get()),
      children_(common_param->allocator_.get()),
      common_param_(common_param),
      graph_param_(std::move(graph_param)) {
}

void
IndexNode::BuildGraph(ODescent& odescent) {
    std::unique_lock lock(mutex_);
    // Build an index when the level corresponding to the current node requires indexing
    if (has_index_ && not ids_.empty()) {
        InitGraph();
        entry_point_ = ids_[0];
        odescent.Build(ids_);
        odescent.SaveGraph(graph_);
        Vector<InnerIdType>(common_param_->allocator_.get()).swap(ids_);
    }
    for (const auto& item : children_) {
        item.second->BuildGraph(odescent);
    }
}

void
IndexNode::AddChild(const std::string& key) {
    // AddChild is not thread-safe; ensure thread safety in calls to it.
    children_[key] = std::make_shared<IndexNode>(common_param_, graph_param_);
    children_[key]->level_ = level_ + 1;
}

std::shared_ptr<IndexNode>
IndexNode::GetChild(const std::string& key, bool need_init) {
    std::unique_lock lock(mutex_);
    auto result = children_.find(key);
    if (result != children_.end()) {
        return result->second;
    }
    if (not need_init) {
        return nullptr;
    }
    AddChild(key);
    return children_[key];
}

void
IndexNode::Deserialize(StreamReader& reader) {
    // deserialize `entry_point_`
    StreamReader::ReadObj(reader, entry_point_);
    // deserialize `level_`
    StreamReader::ReadObj(reader, level_);
    // serialize `has_index_`
    bool has_index;
    StreamReader::ReadObj(reader, has_index);
    // deserialize `graph`
    if (has_index) {
        InitGraph();
        graph_->Deserialize(reader);
    }
    // deserialize `children`
    size_t children_size = 0;
    StreamReader::ReadObj(reader, children_size);
    for (int i = 0; i < children_size; ++i) {
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
    // serialize `has_index_`
    bool has_index = this->graph_ != nullptr;
    StreamWriter::WriteObj(writer, has_index);
    // serialize `graph_`
    if (has_index) {
        graph_->Serialize(writer);
    }
    // serialize `children`
    size_t children_size = children_.size();
    StreamWriter::WriteObj(writer, children_size);
    for (const auto& item : children_) {
        // calculate size of `key`
        StreamWriter::WriteString(writer, item.first);
        // calculate size of `content`
        item.second->Serialize(writer);
    }
}
void
IndexNode::InitGraph() {
    graph_ = GraphInterface::MakeInstance(graph_param_, *common_param_);
}

void
IndexNode::SearchGraph(const SearchFunc& search_func,
                       const VisitedListPtr& vl,
                       const DistHeapPtr& search_result,
                       int64_t ef_search) const {
    if (graph_ != nullptr && graph_->TotalCount() > 0) {
        auto self_search_result = search_func(this, vl);
        while (not self_search_result->Empty()) {
            auto result = self_search_result->Top();
            self_search_result->Pop();
            search_result->Push(result.first, result.second);
            if (search_result->Size() > ef_search) {
                search_result->Pop();
            }
        }
        return;
    }

    for (const auto& [key, node] : children_) {
        node->SearchGraph(search_func, vl, search_result, ef_search);
    }
}

std::vector<int64_t>
Pyramid::build_by_odescent(const DatasetPtr& base) {
    const auto* path = base->GetPaths();
    CHECK_ARGUMENT(path != nullptr, "path is required");
    int64_t data_num = base->GetNumElements();
    const auto* data_vectors = base->GetFloat32Vectors();
    const auto* data_ids = base->GetIds();
    const auto& no_build_levels = pyramid_param_->no_build_levels;

    resize(data_num);
    std::memcpy(label_table_->label_table_.data(), data_ids, sizeof(LabelType) * data_num);

    base_codes_->BatchInsertVector(data_vectors, data_num);
    if (use_reorder_) {
        precise_codes_->BatchInsertVector(data_vectors, data_num);
    }
    auto codes = use_reorder_ ? precise_codes_ : base_codes_;

    ODescent graph_builder(
        pyramid_param_->odescent_param, codes, allocator_, common_param_.thread_pool_.get());
    for (int i = 0; i < data_num; ++i) {
        std::string current_path = path[i];
        auto path_slices = split(current_path, PART_SLASH);
        std::shared_ptr<IndexNode> node = root_;
        for (auto& path_slice : path_slices) {
            node = node->GetChild(path_slice, true);
            node->ids_.push_back(i);
            node->has_index_ =
                std::find(no_build_levels.begin(), no_build_levels.end(), node->level_) ==
                no_build_levels.end();
        }
    }
    root_->BuildGraph(graph_builder);
    cur_element_count_ = data_num;
    max_capacity_ = data_num;
    return {};
}

DatasetPtr
Pyramid::KnnSearch(const DatasetPtr& query,
                   int64_t k,
                   const std::string& parameters,
                   const FilterPtr& filter) const {
    auto parsed_param = PyramidSearchParameters::FromJson(parameters);
    InnerSearchParam search_param;
    search_param.ef = parsed_param.ef_search;
    search_param.topk = k;
    search_param.search_mode = KNN_SEARCH;

    if (parsed_param.enable_time_record) {
        search_param.time_cost = std::make_shared<Timer>();
        search_param.time_cost->SetThreshold(parsed_param.timeout_ms);
    }

    if (filter != nullptr) {
        search_param.is_inner_id_allowed =
            std::make_shared<InnerIdWrapperFilter>(filter, *label_table_);
    }
    Statistics stats;
    auto codes = use_reorder_ ? precise_codes_ : base_codes_;
    SearchFunc search_func = [&](const IndexNode* node, const VisitedListPtr& vl) {
        std::shared_lock lock(node->mutex_);
        search_param.ep = node->entry_point_;
        auto results = searcher_->Search(node->graph_,
                                         codes,
                                         vl,
                                         query->GetFloat32Vectors(),
                                         search_param,
                                         (LabelTablePtr) nullptr,
                                         stats);
        return results;
    };

    auto result = this->search_impl(query, k, search_func, parsed_param.ef_search);
    result->Statistics(stats.Dump());
    return result;
}

DatasetPtr
Pyramid::RangeSearch(const DatasetPtr& query,
                     float radius,
                     const std::string& parameters,
                     const FilterPtr& filter,
                     int64_t limited_size) const {
    auto parsed_param = PyramidSearchParameters::FromJson(parameters);
    InnerSearchParam search_param;
    search_param.ef = parsed_param.ef_search;
    search_param.radius = radius;
    search_param.search_mode = RANGE_SEARCH;

    if (parsed_param.enable_time_record) {
        search_param.time_cost = std::make_shared<Timer>();
        search_param.time_cost->SetThreshold(parsed_param.timeout_ms);
    }

    if (filter != nullptr) {
        search_param.is_inner_id_allowed =
            std::make_shared<InnerIdWrapperFilter>(filter, *label_table_);
    }
    Statistics stats;
    auto codes = use_reorder_ ? precise_codes_ : base_codes_;
    SearchFunc search_func = [&](const IndexNode* node, const VisitedListPtr& vl) {
        std::shared_lock lock(node->mutex_);
        search_param.ep = node->entry_point_;
        auto results = searcher_->Search(node->graph_,
                                         codes,
                                         vl,
                                         query->GetFloat32Vectors(),
                                         search_param,
                                         (LabelTablePtr) nullptr,
                                         stats);
        return results;
    };
    int64_t final_limit = limited_size == -1 ? std::numeric_limits<int64_t>::max() : limited_size;

    auto result = this->search_impl(query, final_limit, search_func, parsed_param.ef_search);
    result->Statistics(stats.Dump());
    return result;
}

DatasetPtr
Pyramid::search_impl(const DatasetPtr& query,
                     int64_t limit,
                     const SearchFunc& search_func,
                     int64_t ef_search) const {
    const auto* query_path = query->GetPaths();
    CHECK_ARGUMENT(query_path != nullptr || root_->graph_ != nullptr,  // NOLINT
                   "query_path is required when level0 is not built");
    CHECK_ARGUMENT(query->GetFloat32Vectors() != nullptr, "query vectors is required");

    auto search_result = std::make_shared<StandardHeap<true, false>>(allocator_, -1);

    auto vl = pool_->TakeOne();
    if (query_path != nullptr) {
        const std::string& current_path = query_path[0];
        auto parsed_path = parse_path(current_path);
        for (const auto& one_path : parsed_path) {
            std::shared_ptr<IndexNode> node = root_;
            bool valid = true;
            for (const auto& item : one_path) {
                node = node->GetChild(item, false);
                if (node == nullptr) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                node->SearchGraph(search_func, vl, search_result, ef_search);
            }
        }
    } else {
        root_->SearchGraph(search_func, vl, search_result, ef_search);
    }
    pool_->ReturnOne(vl);

    if (search_result->Empty()) {
        return DatasetImpl::MakeEmptyDataset();
    }

    while (search_result->Size() > limit) {
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
    auto* ids = (int64_t*)allocator_->Allocate(sizeof(int64_t) * target_size);
    result->Ids(ids);
    auto* dists = (float*)allocator_->Allocate(sizeof(float) * target_size);
    result->Distances(dists);
    for (auto j = target_size - 1; j >= 0; --j) {
        if (j < target_size) {
            dists[j] = search_result->Top().first;
            ids[j] = label_table_->GetLabelById(search_result->Top().second);
        }
        search_result->Pop();
    }
    return result;
}

int64_t
Pyramid::GetNumElements() const {
    return base_codes_->TotalCount();
}

void
Pyramid::Serialize(StreamWriter& writer) const {
    StreamWriter::WriteVector(writer, label_table_->label_table_);
    base_codes_->Serialize(writer);
    if (use_reorder_) {
        precise_codes_->Serialize(writer);
    }
    root_->Serialize(writer);

    // serialize footer (introduced since v0.15)
    JsonType basic_info;
    basic_info["max_capacity"].SetInt(max_capacity_);
    auto metadata = std::make_shared<Metadata>();
    metadata->Set(BASIC_INFO, basic_info);
    auto footer = std::make_shared<Footer>(metadata);
    footer->Write(writer);
}

void
Pyramid::Deserialize(StreamReader& reader) {
    // try to deserialize footer (only in new version)
    auto footer = Footer::Parse(reader);
    auto metadata = footer->GetMetadata();
    auto basic_info = metadata->Get(BASIC_INFO);
    auto max_capacity = basic_info["max_capacity"].GetInt();

    BufferStreamReader buffer_reader(
        &reader, std::numeric_limits<uint64_t>::max(), this->allocator_);

    StreamReader::ReadVector(buffer_reader, label_table_->label_table_);
    base_codes_->Deserialize(buffer_reader);
    if (use_reorder_) {
        precise_codes_->Deserialize(buffer_reader);
    }
    cur_element_count_ = base_codes_->TotalCount();
    root_->Deserialize(buffer_reader);
    resize(max_capacity);
}

std::vector<int64_t>
Pyramid::Add(const DatasetPtr& base) {
    const auto* path = base->GetPaths();
    CHECK_ARGUMENT(path != nullptr, "path is required");
    int64_t data_num = base->GetNumElements();
    const auto* data_vectors = base->GetFloat32Vectors();
    const auto* data_ids = base->GetIds();
    const auto& no_build_levels = pyramid_param_->no_build_levels;
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
        cur_element_count_ += data_num;
        base_codes_->BatchInsertVector(data_vectors, data_num);
        if (use_reorder_) {
            precise_codes_->BatchInsertVector(data_vectors, data_num);
        }
    }
    std::shared_lock<std::shared_mutex> lock(resize_mutex_);

    std::memcpy(label_table_->label_table_.data() + local_cur_element_count,
                data_ids,
                sizeof(LabelType) * data_num);

    auto add_func = [&](int i) {
        std::string current_path = path[i];
        auto path_slices = split(current_path, PART_SLASH);
        std::shared_ptr<IndexNode> node = root_;
        auto inner_id = static_cast<InnerIdType>(i + local_cur_element_count);
        const auto* vector = data_vectors + dim_ * i;
        int no_build_level_index = 0;
        for (int j = 0; j <= path_slices.size(); ++j) {
            std::shared_ptr<IndexNode> new_node = nullptr;
            if (j != path_slices.size()) {
                new_node = node->GetChild(path_slices[j], true);
            }
            if (no_build_level_index < no_build_levels.size() &&
                j == no_build_levels[no_build_level_index]) {
                node = new_node;
                no_build_level_index++;
                continue;
            }
            add_one_point(node, inner_id, vector);
            node = new_node;
        }
    };

    Vector<std::future<void>> futures(allocator_);
    for (auto i = 0; i < data_num; ++i) {
        if (this->build_pool_ != nullptr) {
            futures.push_back(this->build_pool_->GeneralEnqueue(add_func, i));
        } else {
            add_func(i);
        }
    }
    if (this->build_pool_ != nullptr) {
        for (auto& future : futures) {
            future.get();
        }
    }
    return {};
}

void
Pyramid::resize(int64_t new_max_capacity) {
    std::unique_lock<std::shared_mutex> lock(resize_mutex_);
    if (new_max_capacity <= max_capacity_) {
        return;
    }
    pool_ = std::make_unique<VisitedListPool>(1, allocator_, new_max_capacity, allocator_);
    label_table_->label_table_.resize(new_max_capacity);
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
    });
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
            "{REMOVE_FLAG_BIT}": 8
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
        "{BUILD_THREAD_COUNT_KEY}": 16,
        "{EF_CONSTRUCTION_KEY}": 400,
        "{NO_BUILD_LEVELS}":[]
    })";

ParamPtr
Pyramid::CheckAndMappingExternalParam(const JsonType& external_param,
                                      const IndexCommonParam& common_param) {
    const ConstParamMap external_mapping = {
        {PYRAMID_EF_CONSTRUCTION, {EF_CONSTRUCTION_KEY}},
        {PYRAMID_USE_REORDER, {USE_REORDER_KEY}},
        {PYRAMID_BASE_QUANTIZATION_TYPE, {BASE_CODES_KEY, QUANTIZATION_PARAMS_KEY, TYPE_KEY}},
        {PYRAMID_PRECISE_QUANTIZATION_TYPE, {PRECISE_CODES_KEY, QUANTIZATION_PARAMS_KEY, TYPE_KEY}},
        {PYRAMID_GRAPH_MAX_DEGREE, {GRAPH_KEY, GRAPH_PARAM_MAX_DEGREE_KEY}},
        {PYRAMID_BASE_IO_TYPE, {BASE_CODES_KEY, IO_PARAMS_KEY, TYPE_KEY}},
        {PYRAMID_BUILD_ALPHA, {GRAPH_KEY, ODESCENT_PARAMETER_ALPHA}},
        {PYRAMID_GRAPH_TYPE, {GRAPH_KEY, GRAPH_TYPE_KEY}},
        {PYRAMID_GRAPH_STORAGE_TYPE, {GRAPH_KEY, GRAPH_STORAGE_TYPE_KEY}},
        {PYRAMID_PRECISE_IO_TYPE, {PRECISE_CODES_KEY, IO_PARAMS_KEY, TYPE_KEY}},
        {PYRAMID_BUILD_THREAD_COUNT, {BUILD_THREAD_COUNT_KEY}},
        {PYRAMID_NO_BUILD_LEVELS, {NO_BUILD_LEVELS}},
        {PYRAMID_BASE_PQ_DIM,
         {BASE_CODES_KEY, QUANTIZATION_PARAMS_KEY, PRODUCT_QUANTIZATION_DIM_KEY}},
        {PYRAMID_BASE_FILE_PATH, {BASE_CODES_KEY, IO_PARAMS_KEY, IO_FILE_PATH_KEY}},
        {PYRAMID_PRECISE_FILE_PATH, {PRECISE_CODES_KEY, IO_PARAMS_KEY, IO_FILE_PATH_KEY}},
        {ODESCENT_PARAMETER_BUILD_BLOCK_SIZE, {GRAPH_KEY, ODESCENT_PARAMETER_BUILD_BLOCK_SIZE}},
        {ODESCENT_PARAMETER_MIN_IN_DEGREE, {GRAPH_KEY, ODESCENT_PARAMETER_MIN_IN_DEGREE}},
        {ODESCENT_PARAMETER_GRAPH_ITER_TURN, {GRAPH_KEY, ODESCENT_PARAMETER_GRAPH_ITER_TURN}},
        {ODESCENT_PARAMETER_NEIGHBOR_SAMPLE_RATE,
         {GRAPH_KEY, ODESCENT_PARAMETER_NEIGHBOR_SAMPLE_RATE}}};

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
    this->Train(base);
    std::vector<int64_t> ret;
    if (graph_type_ == GRAPH_TYPE_VALUE_NSW) {
        ret = this->Add(base);
    } else {
        ret = this->build_by_odescent(base);
    }
    return ret;
}

void
Pyramid::add_one_point(const std::shared_ptr<IndexNode>& node,
                       InnerIdType inner_id,
                       const float* vector) {
    std::unique_lock graph_lock(node->mutex_);
    InnerSearchParam search_param;
    search_param.ef = pyramid_param_->ef_construction;
    search_param.topk = pyramid_param_->max_degree;
    search_param.search_mode = KNN_SEARCH;

    auto codes = use_reorder_ ? precise_codes_ : base_codes_;
    // add one point
    if (node->graph_ == nullptr) {
        node->InitGraph();
    }
    if (node->graph_->TotalCount() == 0) {
        node->graph_->InsertNeighborsById(inner_id, Vector<InnerIdType>(allocator_));
        node->entry_point_ = inner_id;
    } else {
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
        Statistics discard_stats;
        auto results = searcher_->Search(
            node->graph_, codes, vl, vector, search_param, (LabelTablePtr) nullptr, discard_stats);
        pool_->ReturnOne(vl);
        mutually_connect_new_element(
            inner_id, results, node->graph_, codes, points_mutex_, allocator_, alpha_);
        if (update_entry_point) {
            node->entry_point_ = inner_id;
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

}  // namespace vsag
