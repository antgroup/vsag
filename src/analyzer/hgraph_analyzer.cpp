
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use hgraph_ file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "hgraph_analyzer.h"

#include "impl/heap/standard_heap.h"

namespace vsag {

Vector<int64_t>
HGraphAnalyzer::GetComponentCount() {
    // graph connection
    Vector<bool> visited(total_count_, false, allocator_);
    Vector<int64_t> component_sizes(allocator_);
    if (hgraph_->label_table_->CompressDuplicateData()) {
        for (InnerIdType i = 0; i < hgraph_->total_count_; ++i) {
            if (visited[i]) {
                continue;
            }
            const auto& dup_ids = hgraph_->label_table_->duplicate_ids_;
            auto current_id = i;
            while (dup_ids[current_id] != i) {
                visited[dup_ids[current_id]] = true;
                current_id = dup_ids[current_id];
            }
        }
    }
    for (int64_t i = 0; i < total_count_; ++i) {
        if (not visited[i] and not hgraph_->label_table_->IsRemoved(i)) {
            int64_t component_size = 0;
            std::queue<int64_t> q;
            q.push(i);
            visited[i] = true;
            while (not q.empty()) {
                auto node = q.front();
                q.pop();
                component_size++;
                Vector<InnerIdType> neighbors(allocator_);
                hgraph_->bottom_graph_->GetNeighbors(node, neighbors);
                for (const auto& nb : neighbors) {
                    if (not visited[nb] and not hgraph_->label_table_->IsRemoved(nb)) {
                        visited[nb] = true;
                        q.push(nb);
                    }
                }
            }
            component_sizes.push_back(component_size);
        }
    }
    return component_sizes;
}

void
HGraphAnalyzer::calculate_base_groundtruth() {
    if (not base_ground_truth_.empty()) {
        return;
    }
    is_duplicate_ids_.resize(this->total_count_, false);
    Vector<bool> visited(this->total_count_, false, allocator_);
    if (this->hgraph_->label_table_->CompressDuplicateData()) {
        for (InnerIdType i = 0; i < this->total_count_; ++i) {
            if (visited[i]) {
                continue;
            }
            visited[i] = true;
            const auto& dup_ids = hgraph_->label_table_->duplicate_ids_;
            auto current_id = i;
            while (dup_ids[current_id] != i) {
                visited[dup_ids[current_id]] = true;
                current_id = dup_ids[current_id];
            }
        }
    }

    {
        base_sample_ids_.resize(this->total_count_);
        std::iota(base_sample_ids_.begin(), base_sample_ids_.end(), 0);
        std::random_device rd;
        std::mt19937 rng(rd());
        std::shuffle(base_sample_ids_.begin(), base_sample_ids_.end(), rng);
        Vector<InnerIdType> unique_sample_ids(allocator_);
        for (const auto& id : base_sample_ids_) {
            if (not is_duplicate_ids_[id]) {
                unique_sample_ids.push_back(id);
            }
            if (unique_sample_ids.size() >= static_cast<size_t>(this->base_sample_size_)) {
                break;
            }
        }
        this->base_sample_size_ = unique_sample_ids.size();
        base_sample_ids_.swap(unique_sample_ids);
    }
    base_sample_datas_.resize(static_cast<std::vector<float>::size_type>(base_sample_size_) *
                              static_cast<std::vector<float>::size_type>(dim_));
    for (uint64_t i = 0; i < this->base_sample_size_; ++i) {
        InnerIdType sample_id = base_sample_ids_[i];
        hgraph_->GetVectorByInnerId(sample_id, base_sample_datas_.data() + i * dim_);
    }
    calculate_groundtruth(
        base_sample_datas_, base_sample_ids_, base_ground_truth_, this->base_sample_size_);
}

float
HGraphAnalyzer::GetBaseAvgDistance() {
    calculate_base_groundtruth();
    return get_avg_distance(base_sample_ids_, base_ground_truth_);
}

float
HGraphAnalyzer::GetNeighborRecall() {
    calculate_base_groundtruth();
    float neighbor_recall = 0.0F;
    for (const auto& id : base_sample_ids_) {
        // get neighbors from graph
        Vector<InnerIdType> neighbors(allocator_);
        hgraph_->bottom_graph_->GetNeighbors(id, neighbors);

        DistHeapPtr groundtruth = base_ground_truth_[id];
        std::unordered_set<InnerIdType> gt_set;
        const auto* gt_data = groundtruth->GetData();
        auto neighbor_count = std::min<uint64_t>(neighbors.size(), groundtruth->Size());
        for (uint32_t i = 0; i < neighbor_count; ++i) {
            gt_set.insert(gt_data[i].second);
        }

        uint32_t hit_count = 0;
        for (const auto& nb : neighbors) {
            if (gt_set.find(nb) != gt_set.end()) {
                hit_count++;
            }
        }
        neighbor_recall += static_cast<float>(hit_count) / static_cast<float>(neighbor_count);
    }
    return neighbor_recall / static_cast<float>(this->base_sample_size_);
}

float
HGraphAnalyzer::GetDuplicateRatio() {
    if (hgraph_->label_table_->CompressDuplicateData()) {
        auto duplicate_count = 0;
        for (int i = 0; i < hgraph_->label_table_->duplicate_ids_.size(); ++i) {
            if (hgraph_->label_table_->duplicate_ids_[i] != i) {
                is_duplicate_ids_[i] = true;
                duplicate_count++;
            }
        }
        return static_cast<float>(duplicate_count) / static_cast<float>(this->total_count_);
    }
    return 0.0F;
}

float
HGraphAnalyzer::GetBaseSearchRecall(const std::string& search_param) {
    calculate_base_groundtruth();
    calculate_base_search_result(search_param);
    return get_search_recall(
        this->base_sample_size_, base_sample_ids_, base_ground_truth_, base_search_result_);
}

void
HGraphAnalyzer::calculate_base_search_result(const std::string& search_param) {
    if (base_search_result_.empty()) {
        base_search_time_ms_ = calculate_search_result(base_sample_datas_,
                                                       base_sample_ids_,
                                                       base_search_result_,
                                                       search_param,
                                                       this->base_sample_size_);
    }
}

float
HGraphAnalyzer::GetQuantizationError(const std::string& search_param) {
    calculate_base_search_result(search_param);
    if (not hgraph_->use_reorder_) {
        return 0.0F;
    }
    return std::get<0>(calculate_quantization_result(
        base_sample_datas_, base_sample_ids_, base_search_result_, this->base_sample_size_));
}

std::tuple<float, float>
HGraphAnalyzer::calculate_quantization_result(
    const Vector<float>& sample_datas,
    const Vector<InnerIdType>& sample_ids,
    const UnorderedMap<InnerIdType, Vector<LabelType>>& search_result,
    uint32_t sample_size) {
    float total_quantization_error = 0.0F;
    float total_quantization_inversion_count_rate = 0.0F;
    for (int i = 0; i < sample_size; ++i) {
        auto id = sample_ids[i];
        const auto& result = search_result.at(id);
        float sample_error = 0.0F;
        hgraph_->use_reorder_ = false;
        auto base_result = hgraph_->CalDistanceById(sample_datas.data() + i, result.data(), topk_);
        const auto* base_distance = base_result->GetDistances();
        hgraph_->use_reorder_ = true;
        auto precise_result =
            hgraph_->CalDistanceById(sample_datas.data() + i, result.data(), topk_);
        const auto* precise_distance = precise_result->GetDistances();
        uint32_t inversion_count = 0;
        for (uint32_t j = 0; j < topk_; ++j) {
            sample_error += std::abs(base_distance[j] - precise_distance[j]);
            for (uint32_t k = j + 1; k < topk_; ++k) {
                if ((base_distance[j] - base_distance[k]) *
                        (precise_distance[j] - precise_distance[k]) <
                    0) {
                    inversion_count++;
                }
            }
        }
        total_quantization_error += sample_error / static_cast<float>(topk_);
        total_quantization_inversion_count_rate +=
            static_cast<float>(inversion_count) /
            (static_cast<float>(topk_) * static_cast<float>(topk_ - 1) / 2.0F);
    }
    return {total_quantization_error / static_cast<float>(sample_size),
            total_quantization_inversion_count_rate / static_cast<float>(sample_size)};
}

float
HGraphAnalyzer::GetQuantizationInversionRatio(const std::string& search_param) {
    calculate_base_search_result(search_param);
    if (not hgraph_->use_reorder_) {
        return 0.0F;
    }
    return std::get<1>(calculate_quantization_result(
        base_sample_datas_, base_sample_ids_, base_search_result_, this->base_sample_size_));
}

bool
HGraphAnalyzer::SetQuery(const DatasetPtr& query) {
    if (query_sample_size_ != 0) {
        query_sample_ids_.clear();
        query_sample_datas_.clear();
        query_ground_truth_.clear();
        query_search_result_.clear();
    }
    query_sample_size_ = query->GetNumElements();
    query_sample_ids_.resize(query_sample_size_);
    query_sample_datas_.resize(static_cast<std::vector<float>::size_type>(query_sample_size_) *
                               static_cast<std::vector<float>::size_type>(dim_));
    if (const float* query_vectors = query->GetFloat32Vectors()) {
        std::memcpy(query_sample_datas_.data(),
                    query_vectors,
                    static_cast<unsigned long>(query_sample_size_ * dim_) * sizeof(float));
    } else {
        logger::error("Query dataset is empty or not float32");
        return false;
    }
    std::iota(query_sample_ids_.begin(), query_sample_ids_.end(), 0);
    return true;
}

void
HGraphAnalyzer::calculate_groundtruth(const Vector<float>& sample_datas,
                                      const Vector<InnerIdType>& sample_ids,
                                      UnorderedMap<InnerIdType, DistHeapPtr>& ground_truth,
                                      uint32_t sample_size) {
    if (not ground_truth.empty()) {
        return;
    }
    // calculate duplicate ratio while calculating groundtruth
    uint32_t duplicate_count = 0;
    Vector<float> distances_array(this->total_count_, allocator_);
    Vector<InnerIdType> ids_array(this->total_count_, allocator_);
    std::iota(ids_array.begin(), ids_array.end(), 0);
    auto codes = hgraph_->reorder_ ? hgraph_->high_precise_codes_ : hgraph_->basic_flatten_codes_;
    for (uint64_t i = 0; i < sample_size; ++i) {
        if (i % 10 == 0) {
            logger::info("calculate groundtruth for sample {} of {}", i, i + 10);
        }
        auto comp = codes->FactoryComputer(sample_datas.data() + i * dim_);
        codes->Query(distances_array.data(), comp, ids_array.data(), this->total_count_);
        DistHeapPtr groundtruth = std::make_shared<StandardHeap<true, false>>(allocator_, -1);
        for (uint64_t j = 0; j < this->total_count_; ++j) {
            float dist = distances_array[j];
            if (groundtruth->Size() < topk_) {
                groundtruth->Push({dist, j});
            } else if (dist < groundtruth->Top().first) {
                groundtruth->Push({dist, j});
                groundtruth->Pop();
            }
        }
        ground_truth.insert({sample_ids[i], groundtruth});
    }
}

void
HGraphAnalyzer::calculate_query_groundtruth() {
    if (query_ground_truth_.empty()) {
        calculate_groundtruth(
            query_sample_datas_, query_sample_ids_, query_ground_truth_, this->query_sample_size_);
    }
}

void
HGraphAnalyzer::calculate_query_search_result(const std::string& search_param) {
    if (query_search_result_.empty()) {
        query_search_time_ms_ = calculate_search_result(query_sample_datas_,
                                                        query_sample_ids_,
                                                        query_search_result_,
                                                        search_param,
                                                        this->query_sample_size_);
    }
}

float
HGraphAnalyzer::calculate_search_result(const Vector<float>& sample_datas,
                                        const Vector<InnerIdType>& sample_ids,
                                        UnorderedMap<InnerIdType, Vector<LabelType>>& search_result,
                                        const std::string& search_param,
                                        uint32_t sample_size) {
    auto time_cost = 0.0F;
    for (int i = 0; i < sample_size; ++i) {
        auto query = Dataset::Make();
        query->Dim(dim_)->NumElements(1)->Owner(false)->Float32Vectors(
            sample_datas.data() + static_cast<uint64_t>(i * dim_));
        double single_query_time;
        DatasetPtr result = nullptr;
        {
            Timer t(single_query_time);
            result = hgraph_->KnnSearch(query, topk_, search_param, nullptr);
        }
        auto result_size = result->GetDim();
        const auto* ids = result->GetIds();
        Vector<LabelType> result_labels(allocator_);
        result_labels.resize(result_size);
        std::memcpy(result_labels.data(), ids, result_size * sizeof(LabelType));
        search_result.insert({sample_ids[i], result_labels});
        time_cost += static_cast<float>(single_query_time);
    }
    return time_cost / static_cast<float>(sample_size);
}

float
HGraphAnalyzer::GetQueryQuantizationError(const std::string& search_param) {
    calculate_query_search_result(search_param);
    if (not hgraph_->use_reorder_) {
        return 0.0F;
    }
    return std::get<0>(calculate_quantization_result(
        query_sample_datas_, query_sample_ids_, query_search_result_, this->query_sample_size_));
}

float
HGraphAnalyzer::GetQueryQuantizationInversionRatio(const std::string& search_param) {
    calculate_query_search_result(search_param);
    if (not hgraph_->use_reorder_) {
        return 0.0F;
    }
    return std::get<1>(calculate_quantization_result(
        query_sample_datas_, query_sample_ids_, query_search_result_, this->query_sample_size_));
}

float
HGraphAnalyzer::GetQueryAvgDistance() {
    calculate_query_groundtruth();
    return get_avg_distance(query_sample_ids_, query_ground_truth_);
}

float
HGraphAnalyzer::GetQuerySearchRecall(const std::string& search_param) {
    calculate_query_groundtruth();
    calculate_query_search_result(search_param);
    return get_search_recall(
        this->query_sample_size_, query_sample_ids_, query_ground_truth_, query_search_result_);
}

float
HGraphAnalyzer::get_search_recall(
    uint32_t sample_size,
    const Vector<InnerIdType>& sample_ids,
    const UnorderedMap<InnerIdType, DistHeapPtr>& ground_truth,
    const UnorderedMap<InnerIdType, Vector<LabelType>>& search_result) {
    float total_recall = 0.0F;
    for (int i = 0; i < sample_size; ++i) {
        const auto& real_result = ground_truth.at(sample_ids[i]);
        std::unordered_set<InnerIdType> gt_set;
        const auto* gt_data = real_result->GetData();
        for (uint32_t j = 0; j < real_result->Size(); ++j) {
            gt_set.insert(hgraph_->label_table_->GetLabelById(gt_data[j].second));
        }
        uint32_t hit_count = 0;
        const auto& result = search_result.at(sample_ids[i]);
        for (long r_id : result) {
            if (gt_set.find(r_id) != gt_set.end()) {
                hit_count++;
            }
        }
        total_recall += static_cast<float>(hit_count) / static_cast<float>(real_result->Size());
    }
    return total_recall / static_cast<float>(sample_size);
}

float
HGraphAnalyzer::GetQuerySearchTimeCost(const std::string& search_param) {
    calculate_query_search_result(search_param);
    return query_search_time_ms_;
}

float
HGraphAnalyzer::GetBaseSearchTimeCost(const std::string& search_param) {
    calculate_base_search_result(search_param);
    return base_search_time_ms_;
}

JsonType
HGraphAnalyzer::GetStats() {
    JsonType stats;
    stats["avg_distance_base"].SetFloat(GetBaseAvgDistance());
    auto components = GetComponentCount();
    stats["connect_components"].SetInt(components.size());
    stats["maximal_component_size"].SetInt(*std::max_element(components.begin(), components.end()));
    stats["deleted_count"].SetInt(hgraph_->delete_count_);
    if (hgraph_->label_table_->CompressDuplicateData()) {
        stats["duplicate_ratio"].SetFloat(GetDuplicateRatio());
    }
    const auto& [count_in_degree, count_out_degree, avg_degree] = GetDegreeDistribution();
    stats["in_degree_distribution"].SetVector<uint32_t>(count_in_degree);
    stats["out_degree_distribution"].SetVector<uint32_t>(count_out_degree);
    stats["average_degree"].SetFloat(avg_degree);
    stats["duplicate_ratio"].SetFloat(GetDuplicateRatio());
    stats["proximity_recall_neighbor"].SetFloat(GetNeighborRecall());
    stats["quantization_bias_ratio"].SetFloat(GetQuantizationError(search_params_));
    stats["quantization_inversion_count_rate"].SetFloat(
        GetQuantizationInversionRatio(search_params_));
    stats["recall_base"].SetFloat(GetBaseSearchRecall(search_params_));
    stats["time_cost_query"].SetFloat(GetBaseSearchTimeCost(search_params_));
    stats["total_count"].SetInt(total_count_);
    return stats;
}
JsonType
HGraphAnalyzer::AnalyzeIndexBySearch(const SearchRequest& request) {
    SetQuery(request.query_);
    JsonType stats;
    stats["avg_distance_query"].SetFloat(GetQueryAvgDistance());
    stats["recall_query"].SetFloat(GetQuerySearchRecall(request.params_str_));
    stats["time_cost_query"].SetFloat(GetQuerySearchTimeCost(request.params_str_));
    stats["quantization_bias_ratio_query"].SetFloat(GetQueryQuantizationError(request.params_str_));
    stats["quantization_inversion_count_rate_query"].SetFloat(
        GetQueryQuantizationInversionRatio(request.params_str_));
    return stats;
}

std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, float>
HGraphAnalyzer::GetDegreeDistribution() {
    Vector<uint32_t> in_degree(this->total_count_, allocator_);
    Vector<uint32_t> out_degree(this->total_count_, allocator_);
    for (InnerIdType i = 0; i < this->total_count_; ++i) {
        Vector<InnerIdType> neighbors(allocator_);
        hgraph_->bottom_graph_->GetNeighbors(i, neighbors);
        out_degree[i] = neighbors.size();
        for (const auto& nb : neighbors) {
            in_degree[nb]++;
        }
    }
    auto max_degree = hgraph_->bottom_graph_->maximum_degree_;
    std::vector<uint32_t> count_in_degree(max_degree + 1);
    std::vector<uint32_t> count_out_degree(max_degree + 1);
    uint64_t total_degree = 0;
    uint64_t valid_id_count = 0;
    for (InnerIdType i = 0; i < this->total_count_; ++i) {
        if (not is_duplicate_ids_[i]) {
            count_in_degree[std::min(in_degree[i], max_degree)]++;
            count_out_degree[std::min(out_degree[i], max_degree)]++;
            total_degree += in_degree[i];
            valid_id_count++;
        }
    }
    auto avg_degree = static_cast<float>(total_degree) / static_cast<float>(valid_id_count);
    return {count_in_degree, count_out_degree, avg_degree};
}

}  // namespace vsag
