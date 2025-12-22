
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
        for (int i = 0; i < hgraph_->total_count_; ++i) {
            if (hgraph_->label_table_->duplicate_records_[i] != nullptr) {
                for (const auto& dup_id :
                     hgraph_->label_table_->duplicate_records_[i]->duplicate_ids) {
                    visited[dup_id] = true;
                }
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
    base_sample_ids_.resize(this->total_count_);
    std::iota(base_sample_ids_.begin(), base_sample_ids_.end(), 0);

    std::random_device rd;
    std::mt19937 rng(rd());

    std::shuffle(base_sample_ids_.begin(), base_sample_ids_.end(), rng);

    base_sample_ids_.resize(base_sample_size_);

    for (uint64_t i = 0; i < this->base_sample_size_; ++i) {
        InnerIdType sample_id = base_sample_ids_[i];
        hgraph_->GetVectorByInnerId(sample_id, base_sample_datas_.data() + i * dim_);
    }
    calculate_groundtruth(base_sample_datas_, base_sample_ids_,
                                  base_ground_truth_, this->base_sample_size_);
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
        auto neighbor_count = std::min(neighbors.size(), groundtruth->Size());
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
        size_t duplicate_num = 0;
        for (int i = 0; i < this->total_count_; ++i) {
            if (hgraph_->label_table_->duplicate_records_[i] != nullptr) {
                duplicate_num += hgraph_->label_table_->duplicate_records_[i]->duplicate_ids.size();
            }
        }
        return static_cast<float>(duplicate_num) / static_cast<float>(this->total_count_);
    } else {
        calculate_base_groundtruth();
        return duplicate_ratio_;
    }
}

float
HGraphAnalyzer::GetBaseSearchRecall(const std::string& search_param) {
    calculate_base_groundtruth();
    calculate_base_search_result(search_param);
    return get_search_recall(this->base_sample_size_, base_sample_ids_,
                             base_ground_truth_, base_search_result_);
}

void
HGraphAnalyzer::calculate_base_search_result(const std::string& search_param) {
        if (base_search_result_.empty()) {
            base_search_time_ms_ = calculate_search_result(base_sample_datas_, base_sample_ids_,
                                    base_search_result_, search_param, this->base_sample_size_);
        }
}

float
HGraphAnalyzer::GetQuantizationError(const std::string& search_param) {
    calculate_base_search_result(search_param);
    if (not hgraph_->use_reorder_) {
        return 0.0F;
    }
    return get<0>(calculate_quantization_result(base_sample_datas_, base_sample_ids_,
                                         base_search_result_, this->base_sample_size_));
}

std::tuple<float, float>
HGraphAnalyzer::calculate_quantization_result(const Vector<float>& sample_datas,
                                             const Vector<InnerIdType>& sample_ids,
                                             const UnorderedMap<InnerIdType, Vector<LabelType>>& search_result,
                                             int sample_size) {
    float total_quantization_error = 0.0F;
    float total_quantization_inversion_count_rate = 0.0F;
    for (int i = 0; i < sample_size; ++i) {
        auto id = sample_ids[i];
        const auto& result = search_result.at(id);
        float sample_error = 0.0F;
        hgraph_->use_reorder_ = false;
        auto base_result = hgraph_->CalDistanceById(sample_datas.data() + i, result.data(), topk_);
        auto base_distance = base_result->GetDistances();
        hgraph_->use_reorder_ = true;
        auto precise_result = hgraph_->CalDistanceById(sample_datas.data() + i, result.data(), topk_);
        auto precise_distance = precise_result->GetDistances();
        uint32_t inversion_count = 0;
        for (uint32_t j = 0; j < topk_; ++j) {
            sample_error += std::abs(base_distance[j] - precise_distance[j]);
            for (uint32_t k = j + 1; k < topk_; ++k) {
                if ((base_distance[j] - base_distance[k]) * (precise_distance[j] - precise_distance[k]) < 0) {
                    inversion_count++;
                }
            }
        }
        total_quantization_error += sample_error / static_cast<float>(topk_);
        total_quantization_inversion_count_rate += static_cast<float>(inversion_count) /
                                                   static_cast<float>(topk_ * (topk_ - 1) / 2);
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
    return get<1>(calculate_quantization_result(base_sample_datas_, base_sample_ids_,
                                                base_search_result_, this->base_sample_size_));
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
    query_sample_datas_.resize(query_sample_size_ * dim_);
    std::iota(query_sample_ids_.begin(), query_sample_ids_.end(), 0);
    return true;
}

void
HGraphAnalyzer::calculate_groundtruth(const Vector<float>& sample_datas,
                                      const Vector<InnerIdType>& sample_ids,
                                      UnorderedMap<InnerIdType, DistHeapPtr>& ground_truth,
                                      int sample_size) {

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
        std::sort(distances_array.begin(), distances_array.end());
        for (uint64_t j = 0; j < this->total_count_; ++j) {
            if (j > 0 and
                std::abs(distances_array[j] - distances_array[j - 1]) <= THRESHOLD_ERROR) {
                duplicate_count++;
            }
        }
        duplicate_ratio_ +=
            static_cast<float>(duplicate_count) / static_cast<float>(this->total_count_);
    }
    duplicate_ratio_ /= static_cast<float>(this->base_sample_size_);
}

void
HGraphAnalyzer::calculate_query_groundtruth() {
    if (query_ground_truth_.empty()) {
        calculate_groundtruth(query_sample_datas_, query_sample_ids_,
                              query_ground_truth_, this->query_sample_size_);
    }
}

void
HGraphAnalyzer::calculate_query_search_result(const std::string& search_param) {
    if (query_search_result_.empty()) {
        query_search_time_ms_ = calculate_search_result(query_sample_datas_, query_sample_ids_,
                                query_search_result_, search_param, this->query_sample_size_);
    }
}

float
HGraphAnalyzer::calculate_search_result(const Vector<float>& sample_datas,
                                        const Vector<InnerIdType>& sample_ids,
                                        UnorderedMap<InnerIdType, Vector<LabelType>>& search_result,
                                        const std::string& search_param,
                                        int sample_size) {
    auto time_cost = 0.0F;
    for (int i = 0; i < sample_size; ++i) {
        auto query = Dataset::Make();
        query->Dim(dim_)->NumElements(1)->Owner(false)->Float32Vectors(sample_datas.data() +
                                                                       i * dim_);
        double single_query_time;
        DatasetPtr result = nullptr;
        {
            Timer t(single_query_time);
            result = hgraph_->KnnSearch(query, topk_, search_param, nullptr);
        }
        auto result_size = result->GetDim();
        auto ids = result->GetIds();
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
    return get<0>(calculate_quantization_result(query_sample_datas_, query_sample_ids_,
                                                query_search_result_, this->query_sample_size_));
}

float
HGraphAnalyzer::GetQueryQuantizationInversionRatio(const std::string& search_param) {
    calculate_query_search_result(search_param);
    if (not hgraph_->use_reorder_) {
        return 0.0F;
    }
    return get<1>(calculate_quantization_result(query_sample_datas_, query_sample_ids_,
                                                query_search_result_, this->query_sample_size_));
}

float
HGraphAnalyzer::GetQueryAvgDistance() {
    calculate_query_groundtruth();
    return get_avg_distance(query_sample_ids_, query_ground_truth_);
}

float
HGraphAnalyzer::get_avg_distance(const Vector<InnerIdType>& sample_ids,
                                 const UnorderedMap<InnerIdType, DistHeapPtr>& ground_truth) {
    float dist_sum = 0.0F;
    uint32_t dist_count = 0;
    for (const auto& id : sample_ids) {
        const auto& result = ground_truth.at(id);
        const auto* data = result->GetData();
        for (uint32_t i = 0; i < result->Size(); ++i) {
            dist_sum += data[i].first;
            dist_count++;
        }
    }
    return dist_sum / dist_count;
}
float
HGraphAnalyzer::GetQuerySearchRecall(const std::string& search_param) {
    calculate_query_groundtruth();
    calculate_query_search_result(search_param);
    return get_search_recall(this->query_sample_size_, query_sample_ids_,
                             query_ground_truth_, query_search_result_);
}

float
HGraphAnalyzer::get_search_recall(uint32_t sample_size,
                                  const Vector<InnerIdType>& sample_ids,
    const UnorderedMap<InnerIdType, DistHeapPtr>& ground_truth,
    const UnorderedMap<InnerIdType, Vector<LabelType>>& search_result) {
    float total_recall = 0.0F;
    for (int i = 0; i < sample_size; ++i) {
        const auto& real_result = ground_truth.at(sample_ids[i]);
        std::unordered_set<InnerIdType> gt_set;
        const auto* gt_data = real_result->GetData();
        for (uint32_t j = 0; j < real_result->Size(); ++j) {
            gt_set.insert(hgraph_->label_table_->GetIdByLabel(gt_data[j].second));
        }
        uint32_t hit_count = 0;
        const auto& result = search_result.at(sample_ids[i]);
        for (uint32_t j = 0; j < result.size(); ++j) {
            if (gt_set.find(result[j]) != gt_set.end()) {
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
    stats["avg_base_distance"].SetFloat(GetBaseAvgDistance());
    return stats;
}

}  // namespace vsag