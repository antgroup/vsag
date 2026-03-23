
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
#include "query_context.h"
#include "storage/empty_index_binary_set.h"
#include "storage/serialization.h"
#include "utils/slow_task_timer.h"
#include "utils/util_functions.h"
namespace vsag {

const static float RADIUS_EPSILON = 1.1F;

std::vector<std::string>
split(const std::string& str, char delimiter) {
    auto vec = split_string(str, delimiter);
    vec.erase(
        std::remove_if(vec.begin(), vec.end(), [](const std::string& s) { return s.empty(); }),
        vec.end());
    return vec;
}

static uint64_t
get_suitable_max_degree(int64_t data_num) {
    if (data_num < 100'000) {
        return 16;
    }
    if (data_num < 1000'000) {
        return 32;
    }
    return 64;
}

static uint64_t
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
    const auto* path = base->GetPaths();
    CHECK_ARGUMENT(path != nullptr, "path is required");
    int64_t data_num = base->GetNumElements();
    const auto* data_vectors = base->GetFloat32Vectors();
    const auto* data_ids = base->GetIds();

    resize(data_num);
    std::memcpy(label_table_->label_table_.data(), data_ids, sizeof(LabelType) * data_num);

    base_codes_->BatchInsertVector(data_vectors, data_num);
    if (use_reorder_) {
        precise_codes_->BatchInsertVector(data_vectors, data_num);
    }
    auto codes = use_reorder_ ? precise_codes_ : base_codes_;

    ODescent graph_builder(odescent_param_, codes, allocator_, this->thread_pool_.get());
    root_->Build(graph_builder);
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
    auto ef_search_threshold = std::max<uint64_t>(AMPLIFICATION_FACTOR * k, 1000L);
    CHECK_ARGUMENT(  // NOLINT
        (1 <= parsed_param.ef_search) and (parsed_param.ef_search <= ef_search_threshold),
        fmt::format(
            "ef_search({}) must in range[1, {}]", parsed_param.ef_search, ef_search_threshold));

    InnerSearchParam search_param;
    search_param.ef = parsed_param.ef_search;
    search_param.radius = std::numeric_limits<float>::max();
    search_param.topk = k;
    search_param.search_mode = KNN_SEARCH;
    search_param.parallel_search_thread_count = parsed_param.parallel_search_thread_count;
    if (this->label_table_->CompressDuplicateData()) {
        search_param.consider_duplicate = true;
    }

    if (parsed_param.enable_time_record) {
        search_param.time_cost = std::make_shared<Timer>();
        search_param.time_cost->SetThreshold(parsed_param.timeout_ms);
    }

    if (filter != nullptr) {
        search_param.is_inner_id_allowed =
            std::make_shared<InnerIdWrapperFilter>(filter, *label_table_);
    }
    SearchFunc search_func = [&](const IndexNode* node, const VisitedListPtr& vl) {
        return this->search_node(
            node, vl, search_param, query, base_codes_, ctx, parsed_param.subindex_ef_search);
    };

    auto result = this->search_impl(query, search_func, search_param);
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

    if (this->label_table_->CompressDuplicateData()) {
        search_param.consider_duplicate = true;
    }

    if (filter != nullptr) {
        search_param.is_inner_id_allowed =
            std::make_shared<InnerIdWrapperFilter>(filter, *label_table_);
    }
    SearchFunc search_func = [&](const IndexNode* node, const VisitedListPtr& vl) {
        return this->search_node(
            node, vl, search_param, query, base_codes_, ctx, parsed_param.subindex_ef_search);
    };

    auto result = this->search_impl(query, search_func, search_param);
    result->Statistics(stats.Dump());
    return result;
}

DatasetPtr
Pyramid::search_impl(const DatasetPtr& query,
                     const SearchFunc& search_func,
                     InnerSearchParam& search_param) const {
    SearchStatistics stats;
    QueryContext ctx{.stats = &stats};

    const auto* query_path = query->GetPaths();
    CHECK_ARGUMENT(  // NOLINT
        query_path != nullptr || root_->status_ != IndexNode::Status::NO_INDEX,
        "query_path is required when level0 is not built");
    CHECK_ARGUMENT(query->GetFloat32Vectors() != nullptr, "query vectors is required");

    DistHeapPtr search_result = std::make_shared<StandardHeap<true, false>>(allocator_, -1);

    std::shared_lock<std::shared_mutex> lock(resize_mutex_);
    auto vl = pool_->TakeOne();
    if (query_path != nullptr) {
        std::vector<std::future<void>> futures;
        const std::string& current_path = query_path[0];
        auto parsed_path = parse_path(current_path);
        Vector<DistHeapPtr> search_result_lists(parsed_path.size(), allocator_);
        for (uint32_t i = 0; i < parsed_path.size(); ++i) {
            const auto& one_path = parsed_path[i];
            search_result_lists[i] = std::make_shared<StandardHeap<true, false>>(allocator_, -1);
            IndexNode* node = root_.get();
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
                        node->Search(search_func, vl, search_result_lists[i], search_param.ef);
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

    } else {
        root_->Search(search_func, vl, search_result, search_param.ef);
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
    return base_codes_->TotalCount();
}

void
Pyramid::Serialize(StreamWriter& writer) const {
    label_table_->Serialize(writer);
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

    label_table_->Deserialize(buffer_reader);
    base_codes_->Deserialize(buffer_reader);
    if (use_reorder_) {
        precise_codes_->Deserialize(buffer_reader);
    }
    cur_element_count_ = base_codes_->TotalCount();
    root_->Deserialize(buffer_reader);
    resize(max_capacity);
    this->current_memory_usage_ = static_cast<int64_t>(this->CalSerializeSize());
}

std::vector<int64_t>
Pyramid::Add(const DatasetPtr& base, AddMode mode) {
    const auto* path = base->GetPaths();
    CHECK_ARGUMENT(path != nullptr, "path is required");
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

    auto add_func = [&](int64_t i, int64_t data_bias) {
        std::string current_path = path[data_bias];
        auto path_slices = split(current_path, PART_SLASH);
        IndexNode* node = root_.get();
        auto inner_id = static_cast<InnerIdType>(i + local_cur_element_count);
        const auto* vector = data_vectors + dim_ * data_bias;
        int no_build_level_index = 0;
        for (int j = 0; j <= path_slices.size(); ++j) {
            IndexNode* new_node = nullptr;
            if (j != path_slices.size()) {
                new_node = node->GetChild(path_slices[j], true);
            }
            if (no_build_level_index < no_build_levels_.size() &&
                j == no_build_levels_[no_build_level_index]) {
                node = new_node;
                no_build_level_index++;
                continue;
            }
            add_one_point(node, inner_id, vector);
            node = new_node;
        }
    };

    Vector<std::future<void>> futures(allocator_);
    for (int64_t i = 0; i < data_biases.size(); ++i) {
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
        IndexFeature::SUPPORT_GET_MEMORY_USAGE,
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
         {GRAPH_KEY, ODESCENT_PARAMETER_NEIGHBOR_SAMPLE_RATE}},
        {PYRAMID_INDEX_MIN_SIZE, {INDEX_MIN_SIZE}},
        {PYRAMID_SUPPORT_DUPLICATE, {SUPPORT_DUPLICATE}}};

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
    const auto* path = base->GetPaths();
    CHECK_ARGUMENT(path != nullptr, "path is required");
    int64_t data_num = base->GetNumElements();
    for (int i = 0; i < data_num; ++i) {
        std::string current_path = path[i];
        auto path_slices = split(current_path, PART_SLASH);
        IndexNode* node = root_.get();
        if (std::find(no_build_levels_.begin(), no_build_levels_.end(), node->level_) ==
            no_build_levels_.end()) {
            node->ids_.push_back(i);
        }
        for (auto& path_slice : path_slices) {
            node = node->GetChild(path_slice, true);
            if (std::find(no_build_levels_.begin(), no_build_levels_.end(), node->level_) ==
                no_build_levels_.end()) {
                node->ids_.push_back(i);
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
Pyramid::add_one_point(IndexNode* node, InnerIdType inner_id, const float* vector) {
    std::unique_lock graph_lock(node->mutex_);
    // add one point
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
        search_param.ef = ef_construction_;
        search_param.topk = static_cast<int64_t>(ef_construction_);
        search_param.search_mode = KNN_SEARCH;
        if (label_table_->CompressDuplicateData()) {
            search_param.find_duplicate = true;
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
        if (this->label_table_->CompressDuplicateData() && search_param.duplicate_id >= 0) {
            std::unique_lock lock(this->label_lookup_mutex_);
            label_table_->SetDuplicateId(static_cast<InnerIdType>(search_param.duplicate_id),
                                         inner_id);
            return;
        }
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

        for (const auto& id : node->ids_) {
            vl->Set(id);
        }

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
}  // namespace vsag
