
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

#include "data_cell/flatten_interface.h"
#include "empty_index_binary_set.h"
#include "impl/heap/standard_heap.h"
#include "impl/odescent_graph_builder.h"
#include "impl/pruning_strategy.h"
#include "io/memory_io_parameter.h"
#include "storage/serialization.h"
#include "utils/slow_task_timer.h"

namespace vsag {

std::vector<std::string>
split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    size_t start = 0;
    size_t end = str.find(delimiter);
    if (str.empty()) {
        throw std::runtime_error("fail to parse empty path");
    }

    while (end != std::string::npos) {
        std::string token = str.substr(start, end - start);
        if (token.empty()) {
            throw std::runtime_error("fail to parse path:" + str);
        }
        tokens.push_back(str.substr(start, end - start));
        start = end + 1;
        end = str.find(delimiter, start);
    }
    std::string last_token = str.substr(start);
    if (last_token.empty()) {
        throw std::runtime_error("fail to parse path:" + str);
    }
    tokens.push_back(str.substr(start, end - start));
    return tokens;
}

IndexNode::IndexNode(IndexCommonParam* common_param, GraphInterfaceParamPtr graph_param)
    : ids_(common_param->allocator_.get()),
      children_(common_param->allocator_.get()),
      common_param_(common_param),
      graph_param_(std::move(graph_param)) {
}

void
IndexNode::BuildGraph(ODescent& odescent) {
    std::lock_guard<std::mutex> lock(mutex_);
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
    std::lock_guard<std::mutex> lock(mutex_);
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
    StreamReader::ReadObj(reader, has_index_);
    // deserialize `graph`
    if (has_index_) {
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
    StreamWriter::WriteObj(writer, has_index_);
    // serialize `graph_`
    if (has_index_) {
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

DistHeapPtr
IndexNode::SearchGraph(const SearchFunc& search_func) const {
    if (graph_ != nullptr && graph_->TotalCount() > 0) {
        return search_func(this);
    }
    auto search_result =
        std::make_shared<StandardHeap<true, false>>(common_param_->allocator_.get(), -1);
    for (const auto& [key, node] : children_) {
        DistHeapPtr child_search_result = node->SearchGraph(search_func);
        while (not child_search_result->Empty()) {
            auto result = child_search_result->Top();
            child_search_result->Pop();
            search_result->Push(result.first, result.second);
        }
    }
    return search_result;
}

std::vector<int64_t>
Pyramid::Build(const DatasetPtr& base) {
    const auto* path = base->GetPaths();
    CHECK_ARGUMENT(path != nullptr, "path is required");
    int64_t data_num = base->GetNumElements();
    const auto* data_vectors = base->GetFloat32Vectors();
    const auto* data_ids = base->GetIds();
    const auto& no_build_levels = pyramid_param_->no_build_levels;

    resize(data_num);
    std::memcpy(label_table_->label_table_.data(), data_ids, sizeof(LabelType) * data_num);

    flatten_interface_ptr_->Train(data_vectors, data_num);
    flatten_interface_ptr_->BatchInsertVector(data_vectors, data_num);

    ODescent graph_builder(pyramid_param_->odescent_param,
                           flatten_interface_ptr_,
                           allocator_,
                           common_param_.thread_pool_.get());
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
    if (filter != nullptr) {
        search_param.is_inner_id_allowed =
            std::make_shared<InnerIdWrapperFilter>(filter, *label_table_);
    }
    SearchFunc search_func = [&](const IndexNode* node) {
        search_param.ep = node->entry_point_;
        std::lock_guard<std::mutex> lock(node->mutex_);
        auto vl = pool_->TakeOne();
        auto results = searcher_->Search(
            node->graph_, flatten_interface_ptr_, vl, query->GetFloat32Vectors(), search_param);
        pool_->ReturnOne(vl);
        return results;
    };
    return this->search_impl(query, k, search_func);
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
    if (filter != nullptr) {
        search_param.is_inner_id_allowed =
            std::make_shared<InnerIdWrapperFilter>(filter, *label_table_);
    }
    SearchFunc search_func = [&](const IndexNode* node) {
        search_param.ep = node->entry_point_;
        std::lock_guard<std::mutex> lock(node->mutex_);
        auto vl = pool_->TakeOne();
        auto results = searcher_->Search(
            node->graph_, flatten_interface_ptr_, vl, query->GetFloat32Vectors(), search_param);
        pool_->ReturnOne(vl);
        return results;
    };
    int64_t final_limit = limited_size == -1 ? std::numeric_limits<int64_t>::max() : limited_size;
    return this->search_impl(query, final_limit, search_func);
}

DatasetPtr
Pyramid::search_impl(const DatasetPtr& query, int64_t limit, const SearchFunc& search_func) const {
    const auto* path = query->GetPaths();
    CHECK_ARGUMENT(path != nullptr, "path is required");
    CHECK_ARGUMENT(query->GetFloat32Vectors() != nullptr, "query vectors is required");
    std::string current_path = path[0];
    auto path_slices = split(current_path, PART_SLASH);
    std::shared_ptr<IndexNode> node = root_;
    for (auto& path_slice : path_slices) {
        node = node->GetChild(path_slice, false);
        if (node == nullptr) {
            return DatasetImpl::MakeEmptyDataset();
        }
    }
    auto search_result = node->SearchGraph(search_func);
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
    return flatten_interface_ptr_->TotalCount();
}

int64_t
Pyramid::GetMemoryUsage() const {
    return 0;
}

void
Pyramid::Serialize(StreamWriter& writer) const {
    // FIXME(wxyu): only for testing, remove before merge into the main branch
    // if (not Options::Instance().new_version()) {
    //     StreamWriter::WriteVector(writer, label_table_->label_table_);
    //     flatten_interface_ptr_->Serialize(writer);
    //     root_->Serialize(writer);
    //     return;
    // }

    StreamWriter::WriteVector(writer, label_table_->label_table_);
    flatten_interface_ptr_->Serialize(writer);
    root_->Serialize(writer);

    // serialize footer (introduced since v0.15)
    auto metadata = std::make_shared<Metadata>();
    auto footer = std::make_shared<Footer>(metadata);
    footer->Write(writer);
}

void
Pyramid::Deserialize(StreamReader& reader) {
    // try to deserialize footer (only in new version)
    auto footer = Footer::Parse(reader);

    if (footer == nullptr) {  // old format, DON'T EDIT, remove in the future
        StreamReader::ReadVector(reader, label_table_->label_table_);
        flatten_interface_ptr_->Deserialize(reader);
        root_->Deserialize(reader);
        pool_ = std::make_unique<VisitedListPool>(
            1, allocator_, flatten_interface_ptr_->TotalCount(), allocator_);
    } else {  // create like `else if ( ver in [v0.15, v0.17] )` here if need in the future
        logger::debug("parse with new version format");
        auto metadata = footer->GetMetadata();

        StreamReader::ReadVector(reader, label_table_->label_table_);
        flatten_interface_ptr_->Deserialize(reader);
        root_->Deserialize(reader);
        pool_ = std::make_unique<VisitedListPool>(
            1, allocator_, flatten_interface_ptr_->TotalCount(), allocator_);
    }
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
        flatten_interface_ptr_->BatchInsertVector(data_vectors, data_num);
    }
    std::shared_lock<std::shared_mutex> lock(resize_mutex_);

    std::memcpy(label_table_->label_table_.data() + local_cur_element_count,
                data_ids,
                sizeof(LabelType) * data_num);

    InnerSearchParam search_param;
    search_param.ef = pyramid_param_->ef_construction;
    search_param.topk = pyramid_param_->odescent_param->max_degree;
    search_param.search_mode = KNN_SEARCH;
    auto empty_mutex = std::make_shared<EmptyMutex>();
    for (auto i = 0; i < data_num; ++i) {
        std::string current_path = path[i];
        auto path_slices = split(current_path, PART_SLASH);
        std::shared_ptr<IndexNode> node = root_;
        auto inner_id = static_cast<InnerIdType>(i + local_cur_element_count);
        for (auto& path_slice : path_slices) {
            node = node->GetChild(path_slice, true);
            std::lock_guard<std::mutex> graph_lock(node->mutex_);
            // add one point
            if (node->graph_ == nullptr) {
                node->has_index_ =
                    std::find(no_build_levels.begin(), no_build_levels.end(), node->level_) ==
                    no_build_levels.end();
                node->InitGraph();
            }
            if (node->graph_->TotalCount() == 0) {
                if (node->has_index_) {
                    node->graph_->InsertNeighborsById(inner_id, Vector<InnerIdType>(allocator_));
                    node->entry_point_ = inner_id;
                }
            } else {
                search_param.ep = node->entry_point_;
                auto vl = pool_->TakeOne();
                auto results = searcher_->Search(node->graph_,
                                                 flatten_interface_ptr_,
                                                 vl,
                                                 data_vectors + dim_ * i,
                                                 search_param);
                pool_->ReturnOne(vl);
                mutually_connect_new_element(inner_id,
                                             results,
                                             node->graph_,
                                             flatten_interface_ptr_,
                                             empty_mutex,
                                             allocator_);
            }
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
    flatten_interface_ptr_->Resize(new_max_capacity);
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
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_SEARCH_CONCURRENT,
        IndexFeature::SUPPORT_ADD_CONCURRENT,
    });

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

ParamPtr
Pyramid::CheckAndMappingExternalParam(const JsonType& external_param,
                                      const IndexCommonParam& common_param) {
    auto pyramid_params = std::make_shared<PyramidParameters>();
    pyramid_params->FromJson(external_param);
    return pyramid_params;
}

}  // namespace vsag
