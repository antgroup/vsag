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

#include "lazy_hgraph.h"

#include <fmt/format.h>

#include <algorithm>
#include <limits>
#include <mutex>
#include <vector>

#include "dataset_impl.h"
#include "index_common_param.h"
#include "index_feature_list.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "vsag/constants.h"

namespace vsag {

namespace {

const char* const LAZY_HGRAPH_TRANSITION_THRESHOLD = "transition_threshold";
const char* const LAZY_HGRAPH_HGRAPH = "hgraph";

constexpr uint64_t LAZY_HGRAPH_MAGIC = 0x4C415A5948475246ULL;  // "LAZYHGRF"
constexpr uint64_t LAZY_HGRAPH_VERSION = 1;

void
reject_flat_param(const JsonType& external_param, const char* key) {
    CHECK_ARGUMENT(
        not external_param.Contains(key),
        fmt::format("lazy_hgraph flat phase is fixed to fp32 and does not accept {}", key));
}

uint64_t
parse_transition_threshold(const JsonType& threshold_json) {
    CHECK_ARGUMENT(threshold_json.IsNumberInteger(),
                   "lazy_hgraph transition_threshold must be an integer");
    if (threshold_json.IsNumberUnsigned()) {
        const auto threshold_value = threshold_json.GetUint64();
        CHECK_ARGUMENT(threshold_value > 0, "lazy_hgraph transition_threshold must be positive");
        return threshold_value;
    }
    const auto threshold_value = threshold_json.GetInt();
    CHECK_ARGUMENT(threshold_value > 0, "lazy_hgraph transition_threshold must be positive");
    return static_cast<uint64_t>(threshold_value);
}

JsonType
make_lazy_inner_json(uint64_t transition_threshold,
                     const ParamPtr& flat_param,
                     const ParamPtr& graph_param) {
    JsonType inner_json;
    inner_json["type"].SetString(INDEX_LAZY_HGRAPH);
    inner_json[LAZY_HGRAPH_TRANSITION_THRESHOLD].SetUint64(transition_threshold);
    inner_json["flat"].SetJson(flat_param->ToJson());
    inner_json[LAZY_HGRAPH_HGRAPH].SetJson(graph_param->ToJson());
    return inner_json;
}

LazyHGraphParameterPtr
checked_lazy_hgraph_param(const ParamPtr& param) {
    auto lazy_param = std::dynamic_pointer_cast<LazyHGraphParameter>(param);
    CHECK_ARGUMENT(lazy_param != nullptr, "lazy_hgraph parameter must be LazyHGraphParameter");
    return lazy_param;
}

const LazyHGraphParameterPtr&
checked_lazy_hgraph_param(const LazyHGraphParameterPtr& param) {
    CHECK_ARGUMENT(param != nullptr, "lazy_hgraph parameter must be LazyHGraphParameter");
    return param;
}

IndexCommonParam
make_wrapper_common_param(IndexCommonParam common_param) {
    common_param.extra_info_size_ = 0;
    return common_param;
}

}  // namespace

ParamPtr
LazyHGraph::CheckAndMappingExternalParam(const JsonType& external_param,
                                         const IndexCommonParam& common_param) {
    CHECK_ARGUMENT(common_param.data_type_ == DataTypes::DATA_TYPE_FLOAT,
                   "lazy_hgraph only supports float32 vectors");
    reject_flat_param(external_param, INDEX_BRUTE_FORCE);
    reject_flat_param(external_param, "bruteforce");
    reject_flat_param(external_param, "flat");
    reject_flat_param(external_param, "flat_quantization_type");
    reject_flat_param(external_param, "flat_base_quantization_type");

    uint64_t transition_threshold = 1000;
    if (external_param.Contains(LAZY_HGRAPH_TRANSITION_THRESHOLD)) {
        transition_threshold =
            parse_transition_threshold(external_param[LAZY_HGRAPH_TRANSITION_THRESHOLD]);
    }
    CHECK_ARGUMENT(transition_threshold > 0, "lazy_hgraph transition_threshold must be positive");

    JsonType hgraph_json;
    if (external_param.Contains(LAZY_HGRAPH_HGRAPH)) {
        hgraph_json = external_param[LAZY_HGRAPH_HGRAPH];
    }

    JsonType flat_json;
    auto flat_param = BruteForce::CheckAndMappingExternalParam(flat_json, common_param);
    auto graph_param = HGraph::CheckAndMappingExternalParam(hgraph_json, common_param);

    auto lazy_param = std::make_shared<LazyHGraphParameter>();
    lazy_param->FromJson(make_lazy_inner_json(transition_threshold, flat_param, graph_param));
    return lazy_param;
}

LazyHGraph::LazyHGraph(const ParamPtr& param, const IndexCommonParam& common_param)
    : LazyHGraph(checked_lazy_hgraph_param(param), common_param) {
}

LazyHGraph::LazyHGraph(const LazyHGraphParameterPtr& param, const IndexCommonParam& common_param)
    : InnerIndexInterface(checked_lazy_hgraph_param(param),
                          make_wrapper_common_param(common_param)),
      transition_threshold_(param->transition_threshold),
      flat_param_(param->flat_param),
      graph_param_(param->graph_param),
      common_param_(common_param) {
    this->flat_index_ = std::make_shared<BruteForce>(flat_param_, common_param);
    this->flat_index_->InitFeatures();
    this->has_raw_vector_ = true;
}

InnerIndexInterface*
LazyHGraph::ActiveIndex() const {
    if (phase_.load(std::memory_order_acquire) == Phase::FLAT) {
        return flat_index_.get();
    }
    CHECK_ARGUMENT(graph_index_ != nullptr, "lazy_hgraph graph index is not initialized");
    return graph_index_.get();
}

std::vector<int64_t>
LazyHGraph::Build(const DatasetPtr& data) {
    std::unique_lock lock(this->phase_mutex_);
    CHECK_ARGUMENT(ActiveIndex()->GetNumElements() == 0, "index is not empty");
    if (static_cast<uint64_t>(data->GetNumElements()) < transition_threshold_) {
        phase_.store(Phase::FLAT, std::memory_order_release);
        return flat_index_->Build(data);
    }
    graph_index_ = std::make_shared<HGraph>(graph_param_, this->common_param_);
    graph_index_->InitFeatures();
    auto failed_ids = graph_index_->Build(data);
    phase_.store(Phase::GRAPH, std::memory_order_release);
    flat_index_.reset();
    return failed_ids;
}

std::vector<int64_t>
LazyHGraph::Add(const DatasetPtr& data) {
    std::vector<int64_t> failed_ids;
    bool need_transition = false;
    {
        std::shared_lock lock(this->phase_mutex_);
        if (phase_.load(std::memory_order_acquire) == Phase::FLAT) {
            failed_ids = flat_index_->Add(data);
            need_transition =
                static_cast<uint64_t>(flat_index_->GetNumElements()) >= transition_threshold_;
        } else {
            failed_ids = graph_index_->Add(data);
        }
    }
    if (need_transition) {
        TransitionToGraph();
    }
    return failed_ids;
}

void
LazyHGraph::TransitionToGraph() {
    std::unique_lock lock(this->phase_mutex_);
    if (phase_.load(std::memory_order_acquire) == Phase::GRAPH) {
        return;
    }
    auto id_dataset = flat_index_->ExportIDs();
    if (id_dataset == nullptr) {
        return;
    }
    const auto total = id_dataset->GetNumElements();
    const auto* ids = id_dataset->GetIds();
    if (ids == nullptr) {
        return;
    }
    std::vector<int64_t> valid_ids;
    valid_ids.reserve(static_cast<uint64_t>(total));
    for (int64_t i = 0; i < total; ++i) {
        if (flat_index_->CheckIdExist(ids[i])) {
            valid_ids.emplace_back(ids[i]);
        }
    }
    if (valid_ids.empty()) {
        return;
    }

    CHECK_ARGUMENT(valid_ids.size() <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                   "lazy_hgraph transition id count is too large");
    const auto valid_count = static_cast<int64_t>(valid_ids.size());

    auto build_data = flat_index_->GetDataByIds(valid_ids.data(), valid_count);
    if (build_data == nullptr) {
        throw VsagException(ErrorType::NO_ENOUGH_MEMORY, "failed to get lazy_hgraph vectors");
    }

    auto new_graph = std::make_shared<HGraph>(graph_param_, this->common_param_);
    new_graph->InitFeatures();
    auto failed_ids = new_graph->Build(build_data);
    CHECK_ARGUMENT(failed_ids.empty(), "lazy_hgraph transition failed to build all ids");
    graph_index_ = new_graph;
    flat_index_.reset();
    phase_.store(Phase::GRAPH, std::memory_order_release);
}

DatasetPtr
LazyHGraph::KnnSearch(const DatasetPtr& query,
                      int64_t k,
                      const std::string& parameters,
                      const FilterPtr& filter) const {
    std::shared_lock lock(this->phase_mutex_);
    return ActiveIndex()->KnnSearch(query, k, parameters, filter);
}

DatasetPtr
LazyHGraph::RangeSearch(const DatasetPtr& query,
                        float radius,
                        const std::string& parameters,
                        const FilterPtr& filter,
                        int64_t limited_size) const {
    std::shared_lock lock(this->phase_mutex_);
    return ActiveIndex()->RangeSearch(query, radius, parameters, filter, limited_size);
}

float
LazyHGraph::CalcDistanceById(const float* query,
                             int64_t id,
                             bool calculate_precise_distance) const {
    std::shared_lock lock(this->phase_mutex_);
    return ActiveIndex()->CalcDistanceById(query, id, calculate_precise_distance);
}

bool
LazyHGraph::CheckIdExist(int64_t id) const {
    std::shared_lock lock(this->phase_mutex_);
    return ActiveIndex()->CheckIdExist(id);
}

uint32_t
LazyHGraph::Remove(const std::vector<int64_t>& ids, RemoveMode mode) {
    std::shared_lock lock(this->phase_mutex_);
    if (phase_.load(std::memory_order_acquire) == Phase::FLAT) {
        return flat_index_->Remove(ids, RemoveMode::FORCE_REMOVE);
    }
    return graph_index_->Remove(ids, mode);
}

int64_t
LazyHGraph::GetNumElements() const {
    std::shared_lock lock(this->phase_mutex_);
    return ActiveIndex()->GetNumElements();
}

int64_t
LazyHGraph::GetNumberRemoved() const {
    std::shared_lock lock(this->phase_mutex_);
    return ActiveIndex()->GetNumberRemoved();
}

void
LazyHGraph::GetVectorByInnerId(InnerIdType inner_id, float* data) const {
    std::shared_lock lock(this->phase_mutex_);
    ActiveIndex()->GetVectorByInnerId(inner_id, data);
}

uint64_t
LazyHGraph::EstimateMemory(uint64_t num_elements) const {
    if (num_elements < transition_threshold_) {
        auto flat_index = std::make_shared<BruteForce>(flat_param_, this->common_param_);
        return flat_index->EstimateMemory(num_elements);
    }
    auto graph_index = std::make_shared<HGraph>(graph_param_, this->common_param_);
    return graph_index->EstimateMemory(num_elements);
}

DatasetPtr
LazyHGraph::ExportIDs() const {
    std::shared_lock lock(this->phase_mutex_);
    return ActiveIndex()->ExportIDs();
}

DatasetPtr
LazyHGraph::GetDataByIds(const int64_t* ids, int64_t count) const {
    std::shared_lock lock(this->phase_mutex_);
    return ActiveIndex()->GetDataByIds(ids, count);
}

DatasetPtr
LazyHGraph::GetDataByIdsWithFlag(const int64_t* ids,
                                 int64_t count,
                                 uint64_t selected_data_flag) const {
    std::shared_lock lock(this->phase_mutex_);
    return ActiveIndex()->GetDataByIdsWithFlag(ids, count, selected_data_flag);
}

void
LazyHGraph::GetExtraInfoByIds(const int64_t* ids, int64_t count, char* extra_infos) const {
    std::shared_lock lock(this->phase_mutex_);
    ActiveIndex()->GetExtraInfoByIds(ids, count, extra_infos);
}

uint64_t
LazyHGraph::GetMemoryUsage() const {
    std::shared_lock lock(this->phase_mutex_);
    return ActiveIndex()->GetMemoryUsage();
}

DatasetPtr
LazyHGraph::GetVectorByIds(const int64_t* ids,
                           int64_t count,
                           Allocator* specified_allocator) const {
    std::shared_lock lock(this->phase_mutex_);
    return ActiveIndex()->GetVectorByIds(ids, count, specified_allocator);
}

void
LazyHGraph::Serialize(StreamWriter& writer) const {
    std::shared_lock lock(this->phase_mutex_);
    StreamWriter::WriteObj(writer, LAZY_HGRAPH_MAGIC);
    StreamWriter::WriteObj(writer, LAZY_HGRAPH_VERSION);
    auto phase = phase_.load(std::memory_order_acquire);
    auto phase_value = static_cast<uint8_t>(phase);
    StreamWriter::WriteObj(writer, phase_value);
    if (phase == Phase::GRAPH) {
        graph_index_->Serialize(writer);
    } else {
        flat_index_->Serialize(writer);
    }
}

bool
LazyHGraph::UpdateExtraInfo(const DatasetPtr& new_base) {
    std::shared_lock lock(this->phase_mutex_);
    return ActiveIndex()->UpdateExtraInfo(new_base);
}

bool
LazyHGraph::UpdateVector(int64_t id, const DatasetPtr& new_base, bool force_update) {
    std::shared_lock lock(this->phase_mutex_);
    return ActiveIndex()->UpdateVector(id, new_base, force_update);
}

void
LazyHGraph::Deserialize(StreamReader& reader) {
    uint64_t magic = 0;
    uint64_t version = 0;
    StreamReader::ReadObj(reader, magic);
    StreamReader::ReadObj(reader, version);
    CHECK_ARGUMENT(magic == LAZY_HGRAPH_MAGIC, "invalid lazy_hgraph magic");
    CHECK_ARGUMENT(version == LAZY_HGRAPH_VERSION, "unsupported lazy_hgraph version");

    uint8_t phase_value = 0;
    StreamReader::ReadObj(reader, phase_value);
    CHECK_ARGUMENT(phase_value <= static_cast<uint8_t>(Phase::GRAPH), "invalid lazy_hgraph phase");
    auto phase = static_cast<Phase>(phase_value);
    std::unique_lock lock(this->phase_mutex_);
    if (phase == Phase::FLAT) {
        flat_index_ = std::make_shared<BruteForce>(flat_param_, this->common_param_);
        flat_index_->InitFeatures();
        flat_index_->Deserialize(reader);
        graph_index_.reset();
        phase_.store(Phase::FLAT, std::memory_order_release);
        return;
    }
    graph_index_ = std::make_shared<HGraph>(graph_param_, this->common_param_);
    graph_index_->InitFeatures();
    graph_index_->Deserialize(reader);
    flat_index_.reset();
    phase_.store(Phase::GRAPH, std::memory_order_release);
}

void
LazyHGraph::InitFeatures() {
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_ADD_FROM_EMPTY,
        IndexFeature::SUPPORT_BUILD,
        IndexFeature::SUPPORT_ADD_AFTER_BUILD,
        IndexFeature::SUPPORT_DELETE_BY_ID,
        IndexFeature::SUPPORT_KNN_SEARCH,
        IndexFeature::SUPPORT_KNN_SEARCH_WITH_ID_FILTER,
        IndexFeature::SUPPORT_RANGE_SEARCH,
        IndexFeature::SUPPORT_RANGE_SEARCH_WITH_ID_FILTER,
        IndexFeature::SUPPORT_CAL_DISTANCE_BY_ID,
        IndexFeature::SUPPORT_GET_RAW_VECTOR_BY_IDS,
        IndexFeature::SUPPORT_SEARCH_CONCURRENT,
        IndexFeature::SUPPORT_ADD_CONCURRENT,
        IndexFeature::SUPPORT_DELETE_CONCURRENT,
        IndexFeature::SUPPORT_DESERIALIZE_BINARY_SET,
        IndexFeature::SUPPORT_DESERIALIZE_FILE,
        IndexFeature::SUPPORT_DESERIALIZE_READER_SET,
        IndexFeature::SUPPORT_SERIALIZE_BINARY_SET,
        IndexFeature::SUPPORT_SERIALIZE_FILE,
        IndexFeature::SUPPORT_SERIALIZE_WRITE_FUNC,
        IndexFeature::SUPPORT_ESTIMATE_MEMORY,
        IndexFeature::SUPPORT_GET_MEMORY_USAGE,
        IndexFeature::SUPPORT_CHECK_ID_EXIST,
        IndexFeature::SUPPORT_CLONE,
        IndexFeature::SUPPORT_UPDATE_VECTOR_CONCURRENT,
    });
    if (this->common_param_.extra_info_size_ > 0) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_GET_EXTRA_INFO_BY_ID);
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_KNN_SEARCH_WITH_EX_FILTER);
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_UPDATE_EXTRA_INFO_CONCURRENT);
    }
}

}  // namespace vsag
