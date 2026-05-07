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

#include "pyramid_analyzer.h"

#include <algorithm>
#include <cmath>
#include <random>

#include "impl/heap/standard_heap.h"
#include "impl/logger/logger.h"
#include "impl/searcher/basic_searcher.h"
#include "query_context.h"
#include "typing.h"
#include "vsag/dataset.h"

namespace vsag {

JsonType
PyramidAnalyzer::GetStats() {
    JsonType stats;

    auto start = std::chrono::steady_clock::now();
    stats["index_node_structure"].SetJson(get_index_node_structure());
    auto end = std::chrono::steady_clock::now();
    logger::info("[PyramidAnalyzer] get_index_node_structure: {}ms",
                 std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    start = std::chrono::steady_clock::now();
    stats["leaf_node_size_distribution"].SetJson(get_leaf_node_size_distribution());
    end = std::chrono::steady_clock::now();
    logger::info("[PyramidAnalyzer] get_leaf_node_size_distribution: {}ms",
                 std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    start = std::chrono::steady_clock::now();
    stats["subindex_quality"].SetJson(get_subindex_quality());
    end = std::chrono::steady_clock::now();
    logger::info("[PyramidAnalyzer] get_subindex_quality: {}ms",
                 std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    start = std::chrono::steady_clock::now();
    sample_global();
    end = std::chrono::steady_clock::now();
    logger::info("[PyramidAnalyzer] sample_global: {}ms",
                 std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    if (not sample_ids_.empty()) {
        if (not search_params_.empty()) {
            start = std::chrono::steady_clock::now();
            auto recall_stats = get_graph_node_recall_stats(search_params_);
            end = std::chrono::steady_clock::now();
            logger::info(
                "[PyramidAnalyzer] get_graph_node_recall_stats: {}ms",
                std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

            stats["recall_base"].SetFloat(recall_stats["weighted_recall"].GetFloat());
            stats["graph_node_count"].SetInt(recall_stats["node_count"].GetInt());
            stats["total_graph_size"].SetInt(recall_stats["total_size"].GetInt());
            stats["skipped_node_count"].SetInt(recall_stats["skipped_node_count"].GetInt());
            if (recall_stats.Contains("low_recall_nodes")) {
                stats["low_recall_nodes"].SetJson(recall_stats["low_recall_nodes"]);
            }
        }
    }

    start = std::chrono::steady_clock::now();
    stats["duplicate_ratio"].SetFloat(GetDuplicateRatio());
    end = std::chrono::steady_clock::now();
    logger::info("[PyramidAnalyzer] GetDuplicateRatio: {}ms",
                 std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    stats["total_count"].SetInt(this->total_count_);

    return stats;
}

JsonType
PyramidAnalyzer::AnalyzeIndexBySearch(const SearchRequest& request) {
    JsonType stats;
    if (request.query_ == nullptr) {
        return stats;
    }

    auto query = request.query_;
    auto query_num = query->GetNumElements();
    const float* query_vectors = query->GetFloat32Vectors();
    if (query_vectors == nullptr || query_num == 0) {
        return stats;
    }

    Vector<InnerIdType> query_ids(allocator_);
    query_ids.resize(query_num);
    std::iota(query_ids.begin(), query_ids.end(), 0);

    Vector<float> query_datas(allocator_);
    query_datas.resize(static_cast<size_t>(query_num) * dim_);
    std::memcpy(query_datas.data(), query_vectors, query_num * dim_ * sizeof(float));

    UnorderedMap<InnerIdType, DistHeapPtr> query_ground_truth(allocator_);
    calculate_groundtruth(query_datas, query_ids, query_ground_truth, query_num);

    UnorderedMap<InnerIdType, Vector<LabelType>> query_search_result(allocator_);
    float query_time_ms = calculate_search_result(
        query_datas, query_ids, query_search_result, request.params_str_, query_num);

    stats["avg_distance_query"].SetFloat(
        get_avg_distance_from_groundtruth(query_ids, query_ground_truth));
    stats["recall_query"].SetFloat(
        get_search_recall(query_num, query_ids, query_ground_truth, query_search_result));
    stats["time_cost_query"].SetFloat(query_time_ms);

    if (pyramid_->use_reorder_) {
        auto [q_error, q_inversion] =
            calculate_quantization_result(query_datas, query_ids, query_search_result, query_num);
        stats["quantization_error_query"].SetFloat(q_error);
        stats["quantization_inversion_ratio_query"].SetFloat(q_inversion);
    }

    return stats;
}

JsonType
PyramidAnalyzer::get_index_node_structure() {
    JsonType structure;

    uint32_t total_nodes = 0;
    uint32_t max_depth = 0;
    std::map<uint32_t, uint32_t> nodes_by_level;
    std::map<std::string, uint32_t> status_distribution;

    std::function<void(IndexNode*, int)> traverse = [&](IndexNode* node, int level) {
        if (node == nullptr) {
            return;
        }

        total_nodes++;
        max_depth = std::max(max_depth, static_cast<uint32_t>(level));
        nodes_by_level[level]++;

        std::string status_str;
        switch (node->status_) {
            case IndexNode::Status::GRAPH:
                status_str = "GRAPH";
                break;
            case IndexNode::Status::FLAT:
                status_str = "FLAT";
                break;
            case IndexNode::Status::NO_INDEX:
                status_str = "NO_INDEX";
                break;
        }
        status_distribution[status_str]++;

        for (const auto& child : node->children_) {
            traverse(child.second.get(), level + 1);
        }
    };

    traverse(pyramid_->root_.get(), 0);

    structure["total_nodes"].SetInt(total_nodes);
    structure["max_depth"].SetInt(max_depth);

    JsonType nodes_by_level_json;
    for (const auto& [level, count] : nodes_by_level) {
        nodes_by_level_json[std::to_string(level)].SetInt(count);
    }
    structure["nodes_by_level"].SetJson(nodes_by_level_json);

    JsonType status_json;
    for (const auto& [status, count] : status_distribution) {
        status_json[status].SetInt(count);
    }
    structure["status_distribution"].SetJson(status_json);

    return structure;
}

JsonType
PyramidAnalyzer::get_leaf_node_size_distribution() {
    JsonType distribution;

    Vector<uint32_t> leaf_sizes(allocator_);
    collect_leaf_sizes(pyramid_->root_.get(), leaf_sizes);

    if (leaf_sizes.empty()) {
        distribution["total_leaf_nodes"].SetInt(0);
        return distribution;
    }

    std::sort(leaf_sizes.begin(), leaf_sizes.end());

    uint32_t total = leaf_sizes.size();
    uint32_t min_size = leaf_sizes.front();
    uint32_t max_size = leaf_sizes.back();

    double sum = 0.0;
    for (auto s : leaf_sizes) {
        sum += s;
    }
    double avg_size = sum / total;

    double median_size = 0.0;
    if (total % 2 == 0) {
        median_size = (leaf_sizes[total / 2 - 1] + leaf_sizes[total / 2]) / 2.0;
    } else {
        median_size = leaf_sizes[total / 2];
    }

    double variance = 0.0;
    for (auto s : leaf_sizes) {
        variance += (s - avg_size) * (s - avg_size);
    }
    double std_dev = std::sqrt(variance / total);

    auto percentile = [&](double p) -> uint32_t {
        auto idx = static_cast<size_t>(p / 100.0 * static_cast<double>(total - 1));
        return leaf_sizes[idx];
    };

    auto get_bucket = [](uint32_t size) -> std::string {
        if (size < 10000) {
            return "0-10000";
        }
        if (size < 50000) {
            return "10000-50000";
        }
        if (size < 100000) {
            return "50000-100000";
        }
        if (size < 200000) {
            return "100000-200000";
        }
        if (size < 500000) {
            return "200000-500000";
        }
        if (size < 1000000) {
            return "500000-1000000";
        }
        return "1000000+";
    };

    std::map<std::string, uint32_t> histogram;
    for (auto s : leaf_sizes) {
        histogram[get_bucket(s)]++;
    }

    distribution["total_leaf_nodes"].SetInt(total);
    distribution["min_size"].SetInt(min_size);
    distribution["max_size"].SetInt(max_size);
    distribution["avg_size"].SetFloat(static_cast<float>(avg_size));
    distribution["median_size"].SetFloat(static_cast<float>(median_size));
    distribution["std_dev"].SetFloat(static_cast<float>(std_dev));

    JsonType histogram_json;
    for (const auto& [bucket, count] : histogram) {
        histogram_json[bucket].SetInt(count);
    }
    distribution["size_histogram"].SetJson(histogram_json);

    JsonType percentiles_json;
    percentiles_json["p25"].SetInt(percentile(25));
    percentiles_json["p50"].SetInt(percentile(50));
    percentiles_json["p75"].SetInt(percentile(75));
    percentiles_json["p90"].SetInt(percentile(90));
    percentiles_json["p95"].SetInt(percentile(95));
    percentiles_json["p99"].SetInt(percentile(99));
    distribution["percentiles"].SetJson(percentiles_json);

    return distribution;
}

void
PyramidAnalyzer::collect_leaf_sizes(IndexNode* node, Vector<uint32_t>& sizes) {
    if (node == nullptr) {
        return;
    }

    if (node->children_.empty()) {
        uint32_t size = 0;
        if (node->status_ == IndexNode::Status::GRAPH && node->graph_) {
            size = node->graph_->TotalCount();
        } else if (node->status_ == IndexNode::Status::FLAT) {
            size = node->ids_.size();
        }
        if (size > 0) {
            sizes.push_back(size);
        }
        return;
    }

    for (const auto& child : node->children_) {
        collect_leaf_sizes(child.second.get(), sizes);
    }
}

JsonType
PyramidAnalyzer::get_subindex_quality() {
    JsonType quality;

    subindex_stats_.clear();
    analyze_subindexes(pyramid_->root_.get(), "");

    uint32_t graph_count = 0;
    uint32_t flat_count = 0;
    uint32_t problematic_count = 0;
    uint64_t total_size_in_graph = 0;
    double total_weighted_recall = 0.0;
    uint64_t total_weight = 0;

    JsonType subindexes_json;
    for (const auto& stats : subindex_stats_) {
        JsonType subindex_json;
        subindex_json["path"].SetString(stats.path);
        subindex_json["size"].SetInt(stats.size);
        subindex_json["status"].SetString(stats.status == IndexNode::Status::GRAPH ? "GRAPH"
                                                                                   : "FLAT");

        if (stats.status == IndexNode::Status::GRAPH) {
            graph_count++;
            total_size_in_graph += stats.size;
        } else {
            flat_count++;
        }

        subindexes_json[stats.path].SetJson(subindex_json);
    }

    quality["total_subindexes"].SetInt(subindex_stats_.size());
    quality["graph_subindexes"].SetInt(graph_count);
    quality["flat_subindexes"].SetInt(flat_count);
    quality["total_vectors_in_graph"].SetInt(total_size_in_graph);

    JsonType summary;
    summary["avg_subindex_size"].SetFloat(subindex_stats_.empty()
                                              ? 0.0F
                                              : static_cast<float>(total_count_) /
                                                    static_cast<float>(subindex_stats_.size()));
    summary["vectors_in_graph_ratio"].SetFloat(total_count_ > 0
                                                   ? static_cast<float>(total_size_in_graph) /
                                                         static_cast<float>(total_count_)
                                                   : 0.0F);
    quality["summary"].SetJson(summary);

    return quality;
}

void
PyramidAnalyzer::analyze_subindexes(IndexNode* node, const std::string& path) {
    if (node == nullptr) {
        return;
    }

    if (node->children_.empty()) {
        SubIndexStats stats(allocator_);
        stats.path = path;
        stats.status = node->status_;

        if (node->status_ == IndexNode::Status::GRAPH && node->graph_) {
            stats.size = node->graph_->TotalCount();
            collect_subindex_ids(node, stats.ids);
        } else if (node->status_ == IndexNode::Status::FLAT) {
            stats.size = node->ids_.size();
            stats.ids = node->ids_;
        } else {
            stats.size = 0;
        }

        subindex_stats_.push_back(stats);
        return;
    }

    for (const auto& child : node->children_) {
        analyze_subindexes(child.second.get(), path + "/" + child.first);
    }
}

void
PyramidAnalyzer::collect_subindex_ids(IndexNode* node, Vector<InnerIdType>& ids) {
    if (node == nullptr || node->graph_ == nullptr) {
        return;
    }

    auto total = node->graph_->TotalCount();
    ids.clear();
    ids.reserve(total);

    for (InnerIdType i = 0; i < total; ++i) {
        if (node->graph_->CheckIdExists(i)) {
            ids.push_back(i);
        }
    }
}

float
PyramidAnalyzer::calculate_subindex_recall(const Vector<InnerIdType>& subindex_ids) {
    if (subindex_ids.empty() || ground_truth_.empty()) {
        return 0.0F;
    }

    float total_recall = 0.0F;
    uint32_t valid_count = 0;

    for (const auto& sample_id : sample_ids_) {
        if (ground_truth_.find(sample_id) == ground_truth_.end()) {
            continue;
        }

        const auto& gt = ground_truth_.at(sample_id);
        if (gt->Size() == 0) {
            continue;
        }

        UnorderedSet<InnerIdType> gt_set(allocator_);
        const auto* gt_data = gt->GetData();
        for (uint32_t i = 0; i < gt->Size(); ++i) {
            gt_set.insert(gt_data[i].second);
        }

        uint32_t hit_count = 0;
        uint32_t check_count =
            std::min(static_cast<uint32_t>(gt->Size()), static_cast<uint32_t>(subindex_ids.size()));

        for (uint32_t i = 0; i < check_count; ++i) {
            if (gt_set.find(subindex_ids[i]) != gt_set.end()) {
                hit_count++;
            }
        }

        total_recall += static_cast<float>(hit_count) / static_cast<float>(gt->Size());
        valid_count++;
    }

    return valid_count > 0 ? total_recall / static_cast<float>(valid_count) : 0.0F;
}

float
PyramidAnalyzer::calculate_weighted_recall() {
    if (subindex_stats_.empty()) {
        return 0.0F;
    }

    double total_weighted_recall = 0.0;
    uint64_t total_weight = 0;

    for (const auto& stats : subindex_stats_) {
        if (stats.size == 0 || stats.status != IndexNode::Status::GRAPH) {
            continue;
        }
        total_weighted_recall += stats.recall * static_cast<float>(stats.size);
        total_weight += stats.size;
    }

    return total_weight > 0
               ? static_cast<float>(total_weighted_recall / static_cast<double>(total_weight))
               : 0.0F;
}

void
PyramidAnalyzer::sample_global() {
    if (sample_size_ == 0 || total_count_ == 0) {
        return;
    }

    sample_ids_.clear();
    sample_datas_.clear();

    auto actual_sample_size = std::min(sample_size_, static_cast<uint32_t>(total_count_));
    sample_ids_.resize(actual_sample_size);
    std::iota(sample_ids_.begin(), sample_ids_.end(), 0);

    std::random_device rd;
    std::mt19937 rng(rd());
    std::shuffle(sample_ids_.begin(), sample_ids_.end(), rng);

    if (actual_sample_size < static_cast<uint32_t>(total_count_)) {
        sample_ids_.resize(actual_sample_size);
    }

    sample_datas_.resize(static_cast<size_t>(actual_sample_size) * dim_);

    for (uint32_t i = 0; i < actual_sample_size; ++i) {
        InnerIdType sample_id = sample_ids_[i];
        pyramid_->GetVectorByInnerId(sample_id,
                                     sample_datas_.data() + static_cast<size_t>(i) * dim_);
    }
}

void
PyramidAnalyzer::calculate_groundtruth(const Vector<float>& sample_datas,
                                       const Vector<InnerIdType>& sample_ids,
                                       UnorderedMap<InnerIdType, DistHeapPtr>& ground_truth,
                                       uint32_t sample_size) {
    if (not ground_truth.empty()) {
        return;
    }

    Vector<float> distances_array(this->total_count_, allocator_);
    Vector<InnerIdType> ids_array(this->total_count_, allocator_);
    std::iota(ids_array.begin(), ids_array.end(), 0);

    auto codes = pyramid_->use_reorder_ ? pyramid_->precise_codes_ : pyramid_->base_codes_;

    for (uint32_t i = 0; i < sample_size; ++i) {
        if (i % 10 == 0) {
            logger::info("[calculate_groundtruth] Processing sample {} of {}", i, sample_size);
        }

        auto comp = codes->FactoryComputer(sample_datas.data() + static_cast<size_t>(i) * dim_);
        codes->Query(distances_array.data(), comp, ids_array.data(), this->total_count_);

        DistHeapPtr gt = std::make_shared<StandardHeap<true, false>>(allocator_, -1);
        for (uint32_t j = 0; j < this->total_count_; ++j) {
            float dist = distances_array[j];
            if (gt->Size() < topk_) {
                gt->Push({dist, j});
            } else if (dist < gt->Top().first) {
                gt->Push({dist, j});
                gt->Pop();
            }
        }
        ground_truth.insert({sample_ids[i], gt});
    }

    logger::info("[calculate_groundtruth] Completed for {} samples", sample_size);
}

float
PyramidAnalyzer::get_avg_distance() {
    if (ground_truth_.empty() || sample_ids_.empty()) {
        return 0.0F;
    }

    return get_avg_distance_from_groundtruth(sample_ids_, ground_truth_);
}

float
PyramidAnalyzer::get_quantization_error(const std::string& search_param) {
    if (not pyramid_->use_reorder_ || search_result_.empty()) {
        return 0.0F;
    }

    return std::get<0>(
        calculate_quantization_result(sample_datas_, sample_ids_, search_result_, sample_size_));
}

float
PyramidAnalyzer::get_quantization_inversion_ratio(const std::string& search_param) {
    if (not pyramid_->use_reorder_ || search_result_.empty()) {
        return 0.0F;
    }

    return std::get<1>(
        calculate_quantization_result(sample_datas_, sample_ids_, search_result_, sample_size_));
}

std::tuple<float, float>
PyramidAnalyzer::calculate_quantization_result(
    const Vector<float>& sample_datas,
    const Vector<InnerIdType>& sample_ids,
    const UnorderedMap<InnerIdType, Vector<LabelType>>& search_result,
    uint32_t sample_size) {
    if (not pyramid_->use_reorder_) {
        return {0.0F, 0.0F};
    }

    float total_quantization_error = 0.0F;
    float total_quantization_inversion_count_rate = 0.0F;

    for (uint32_t i = 0; i < sample_size; ++i) {
        auto id = sample_ids[i];
        if (search_result.find(id) == search_result.end()) {
            continue;
        }

        const auto& result = search_result.at(id);
        if (result.empty()) {
            continue;
        }

        auto result_size = std::min(static_cast<uint32_t>(result.size()), topk_);
        Vector<LabelType> topk_labels(allocator_);
        topk_labels.assign(result.begin(), result.begin() + result_size);

        auto base_result =
            pyramid_->CalDistanceById(sample_datas.data() + static_cast<size_t>(i) * dim_,
                                      topk_labels.data(),
                                      static_cast<int64_t>(topk_labels.size()),
                                      false);
        const auto* base_distance = base_result->GetDistances();

        auto precise_result =
            pyramid_->CalDistanceById(sample_datas.data() + static_cast<size_t>(i) * dim_,
                                      topk_labels.data(),
                                      static_cast<int64_t>(topk_labels.size()),
                                      true);
        const auto* precise_distance = precise_result->GetDistances();

        uint32_t inversion_count = 0;
        for (uint32_t j = 0; j < result_size; ++j) {
            for (uint32_t k = j + 1; k < result_size; ++k) {
                if ((base_distance[j] - base_distance[k]) *
                        (precise_distance[j] - precise_distance[k]) <
                    0) {
                    inversion_count++;
                }
            }
        }

        total_quantization_inversion_count_rate +=
            static_cast<float>(inversion_count) /
            (static_cast<float>(result_size) * static_cast<float>(result_size - 1) / 2.0F);
    }

    return {total_quantization_error / static_cast<float>(sample_size),
            total_quantization_inversion_count_rate / static_cast<float>(sample_size)};
}

float
PyramidAnalyzer::calculate_search_result(
    const Vector<float>& sample_datas,
    const Vector<InnerIdType>& sample_ids,
    UnorderedMap<InnerIdType, Vector<LabelType>>& search_result,
    const std::string& search_param,
    uint32_t sample_size) {
    float total_time = 0.0F;
    auto start_time = std::chrono::steady_clock::now();

    for (uint32_t i = 0; i < sample_size; ++i) {
        if (i % 1 == 0) {
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::steady_clock::now() - start_time)
                                  .count();
            float progress = static_cast<float>(i) / static_cast<float>(sample_size) * 100.0F;
            float avg_time_ms = (i > 0) ? total_time / static_cast<float>(i) : 0.0F;
            float eta_ms = (i > 0) ? avg_time_ms * static_cast<float>(sample_size - i) : 0.0F;
            logger::info(
                "[calculate_search_result] Progress: {}/{} ({:.1f}%), "
                "elapsed={}ms, avg_query_time={:.2f}ms, ETA={:.0f}ms",
                i,
                sample_size,
                progress,
                elapsed_ms,
                avg_time_ms,
                eta_ms);
        }

        auto query = Dataset::Make();
        query->Dim(dim_)->NumElements(1)->Owner(false)->Float32Vectors(
            sample_datas.data() + static_cast<size_t>(i) * dim_);

        double single_query_time = 0.0;
        DatasetPtr result = nullptr;
        {
            Timer t(single_query_time);
            result = pyramid_->KnnSearch(query, topk_, search_param, nullptr);
        }

        if (result != nullptr) {
            auto result_size = result->GetDim();
            const auto* ids = result->GetIds();
            Vector<LabelType> result_labels(allocator_);
            result_labels.resize(result_size);
            std::memcpy(result_labels.data(), ids, result_size * sizeof(LabelType));
            search_result.insert({sample_ids[i], result_labels});
        }

        total_time += static_cast<float>(single_query_time);
    }

    auto total_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - start_time)
                                .count();
    logger::info(
        "[calculate_search_result] Completed {}/{} queries, total_time={}ms, "
        "avg_time={:.2f}ms",
        sample_size,
        sample_size,
        total_elapsed_ms,
        total_time / static_cast<float>(sample_size));

    search_time_ms_ = total_time / static_cast<float>(sample_size);
    return search_time_ms_;
}

float
PyramidAnalyzer::get_search_recall(
    uint32_t sample_size,
    const Vector<InnerIdType>& sample_ids,
    const UnorderedMap<InnerIdType, DistHeapPtr>& ground_truth,
    const UnorderedMap<InnerIdType, Vector<LabelType>>& search_result) {
    float total_recall = 0.0F;
    uint32_t valid_count = 0;

    for (uint32_t i = 0; i < sample_size; ++i) {
        auto id = sample_ids[i];
        if (ground_truth.find(id) == ground_truth.end() ||
            search_result.find(id) == search_result.end()) {
            continue;
        }

        const auto& gt = ground_truth.at(id);
        const auto& sr = search_result.at(id);

        if (gt->Size() == 0) {
            continue;
        }

        UnorderedSet<LabelType> gt_set(allocator_);
        const auto* gt_data = gt->GetData();
        for (uint32_t j = 0; j < gt->Size(); ++j) {
            gt_set.insert(pyramid_->label_table_->GetLabelById(gt_data[j].second));
        }

        uint32_t hit_count = 0;
        for (const auto& label : sr) {
            if (gt_set.find(label) != gt_set.end()) {
                hit_count++;
            }
        }

        total_recall += static_cast<float>(hit_count) / static_cast<float>(gt->Size());
        valid_count++;
    }

    return valid_count > 0 ? total_recall / static_cast<float>(valid_count) : 0.0F;
}

float
PyramidAnalyzer::get_avg_distance_from_groundtruth(
    const Vector<InnerIdType>& sample_ids,
    const UnorderedMap<InnerIdType, DistHeapPtr>& ground_truth) {
    float dist_sum = 0.0F;
    uint32_t dist_count = 0;

    for (const auto& id : sample_ids) {
        if (ground_truth.find(id) == ground_truth.end()) {
            continue;
        }
        const auto& result = ground_truth.at(id);
        const auto* data = result->GetData();
        for (uint32_t i = 0; i < result->Size(); ++i) {
            dist_sum += data[i].first;
            dist_count++;
        }
    }

    return dist_count > 0 ? dist_sum / static_cast<float>(dist_count) : 0.0F;
}

Vector<InnerIdType>
PyramidAnalyzer::collect_node_ids(const IndexNode* node) {
    Vector<InnerIdType> ids(allocator_);
    if (node == nullptr) {
        return ids;
    }

    if (node->status_ == IndexNode::Status::FLAT) {
        ids = node->ids_;
    } else if (node->status_ == IndexNode::Status::GRAPH && node->graph_ != nullptr) {
        try {
            ids = node->graph_->GetIds();
        } catch (const std::exception& e) {
            auto total = node->graph_->TotalCount();
            ids.reserve(total);
            for (InnerIdType i = 0; i < total; ++i) {
                if (node->graph_->CheckIdExists(i)) {
                    ids.push_back(i);
                }
            }
        }
    }

    return ids;
}

DistHeapPtr
PyramidAnalyzer::calculate_node_groundtruth(const IndexNode* node,
                                            const float* query,
                                            const Vector<InnerIdType>& node_ids) {
    auto gt = std::make_shared<StandardHeap<true, false>>(allocator_, -1);

    if (node_ids.empty() || query == nullptr) {
        return gt;
    }

    auto codes = pyramid_->use_reorder_ ? pyramid_->precise_codes_ : pyramid_->base_codes_;
    if (codes == nullptr) {
        return gt;
    }

    Vector<float> distances(node_ids.size(), allocator_);
    auto computer = codes->FactoryComputer(query);
    codes->Query(distances.data(), computer, node_ids.data(), node_ids.size());

    for (size_t i = 0; i < node_ids.size(); ++i) {
        float dist = distances[i];
        if (gt->Size() < static_cast<int64_t>(topk_)) {
            gt->Push(dist, node_ids[i]);
        } else if (dist < gt->Top().first) {
            gt->Push(dist, node_ids[i]);
            gt->Pop();
        }
    }

    return gt;
}

DistHeapPtr
PyramidAnalyzer::search_single_node(const IndexNode* node,
                                    const float* query,
                                    const std::string& search_param_str) {
    if (node == nullptr || query == nullptr) {
        return std::make_shared<StandardHeap<true, false>>(allocator_, -1);
    }

    if (node->status_ == IndexNode::Status::FLAT) {
        auto result = std::make_shared<StandardHeap<true, false>>(allocator_, -1);
        auto node_ids = node->ids_;
        if (node_ids.empty()) {
            return result;
        }

        auto codes = pyramid_->use_reorder_ ? pyramid_->precise_codes_ : pyramid_->base_codes_;
        Vector<float> distances(node_ids.size(), allocator_);
        auto computer = codes->FactoryComputer(query);
        codes->Query(distances.data(), computer, node_ids.data(), node_ids.size());

        auto ef = static_cast<int64_t>(topk_);
        for (size_t i = 0; i < node_ids.size(); ++i) {
            float dist = distances[i];
            if (result->Size() < ef) {
                result->Push(dist, node_ids[i]);
            } else if (dist < result->Top().first) {
                result->Push(dist, node_ids[i]);
                result->Pop();
            }
        }

        return result;
    }

    if (node->status_ == IndexNode::Status::GRAPH && node->graph_ != nullptr) {
        JsonType params;
        try {
            params = JsonType::Parse(search_param_str);
        } catch (const std::exception& e) {
            logger::error("[search_single_node] Failed to parse search_param: {}", e.what());
            return std::make_shared<StandardHeap<true, false>>(allocator_, -1);
        }

        uint64_t ef_search = 100;
        if (params.Contains("ef_search")) {
            ef_search = static_cast<uint64_t>(params["ef_search"].GetInt());
        } else if (params.Contains("pyramid") && params["pyramid"].Contains("ef_search")) {
            ef_search = static_cast<uint64_t>(params["pyramid"]["ef_search"].GetInt());
        } else if (params.Contains("pyramid") && params["pyramid"].Contains("pyramid") &&
                   params["pyramid"]["pyramid"].Contains("ef_search")) {
            ef_search = static_cast<uint64_t>(params["pyramid"]["pyramid"]["ef_search"].GetInt());
        }

        auto query_dataset = Dataset::Make();
        query_dataset->Dim(dim_)->NumElements(1)->Owner(false)->Float32Vectors(query);

        InnerSearchParam inner_param;
        inner_param.topk = static_cast<int64_t>(topk_);
        inner_param.ef = ef_search;
        inner_param.search_mode = KNN_SEARCH;

        auto vl = pyramid_->pool_->TakeOne();

        SearchStatistics stats;
        QueryContext ctx{.stats = &stats};

        DistHeapPtr result;
        try {
            result = pyramid_->search_node(
                node, vl, inner_param, query_dataset, pyramid_->base_codes_, ctx, ef_search);

            if (pyramid_->use_reorder_ && result != nullptr && !result->Empty()) {
                result = pyramid_->reorder_->Reorder(result, query, inner_param.topk, ctx);
            }
        } catch (const std::exception& e) {
            logger::error("[search_single_node] search_node threw exception: {}", e.what());
            result = std::make_shared<StandardHeap<true, false>>(allocator_, -1);
        }

        pyramid_->pool_->ReturnOne(vl);

        return result;
    }

    return std::make_shared<StandardHeap<true, false>>(allocator_, -1);
}

float
PyramidAnalyzer::calculate_node_recall(const IndexNode* node,
                                       const float* queries,
                                       uint32_t query_count,
                                       const std::string& search_param_str) {
    if (node == nullptr || queries == nullptr || query_count == 0) {
        return 0.0F;
    }

    auto node_ids = collect_node_ids(node);
    if (node_ids.empty()) {
        return 0.0F;
    }

    float total_recall = 0.0F;
    uint32_t valid_count = 0;

    for (uint32_t q = 0; q < query_count; ++q) {
        const float* query = queries + static_cast<size_t>(q) * dim_;

        auto gt = calculate_node_groundtruth(node, query, node_ids);
        auto search_result = search_single_node(node, query, search_param_str);

        if (gt->Size() == 0 || search_result->Size() == 0) {
            continue;
        }

        UnorderedSet<InnerIdType> gt_set(allocator_);
        const auto* gt_data = gt->GetData();
        for (uint32_t i = 0; i < gt->Size(); ++i) {
            gt_set.insert(gt_data[i].second);
        }

        uint32_t hit_count = 0;
        const auto* sr_data = search_result->GetData();
        uint32_t check_count = std::min(static_cast<uint32_t>(search_result->Size()), topk_);
        for (uint32_t i = 0; i < check_count; ++i) {
            if (gt_set.find(sr_data[i].second) != gt_set.end()) {
                hit_count++;
            }
        }

        float recall = static_cast<float>(hit_count) /
                       static_cast<float>(std::min(static_cast<uint32_t>(gt->Size()), topk_));
        total_recall += recall;
        valid_count++;
    }

    return valid_count > 0 ? total_recall / static_cast<float>(valid_count) : 0.0F;
}

PyramidAnalyzer::GraphQualityAnalysis
PyramidAnalyzer::get_node_degree_distribution(const IndexNode* node,
                                              const Vector<InnerIdType>& node_ids) {
    GraphQualityAnalysis result;
    if (node == nullptr || node->graph_ == nullptr || node_ids.empty()) {
        return result;
    }

    auto graph = node->graph_;
    auto total = static_cast<uint32_t>(node_ids.size());
    if (total == 0) {
        return result;
    }

    UnorderedMap<InnerIdType, uint32_t> in_degrees(allocator_);
    UnorderedMap<InnerIdType, uint32_t> out_degrees(allocator_);

    for (auto id : node_ids) {
        in_degrees[id] = 0;
        out_degrees[id] = 0;
    }

    UnorderedSet<InnerIdType> id_set(allocator_);
    for (auto id : node_ids) {
        id_set.insert(id);
    }

    for (auto id : node_ids) {
        Vector<InnerIdType> neighbors(allocator_);
        graph->GetNeighbors(id, neighbors);
        out_degrees[id] = static_cast<uint32_t>(neighbors.size());

        for (auto nb : neighbors) {
            if (id_set.find(nb) != id_set.end()) {
                in_degrees[nb]++;
            }
        }
    }

    double total_degree = 0.0;
    for (auto id : node_ids) {
        uint32_t out_deg = out_degrees[id];
        uint32_t in_deg = in_degrees[id];

        if (out_deg == 0) {
            result.zero_out_degree_count++;
        }
        if (in_deg == 0) {
            result.zero_in_degree_count++;
        }
        total_degree += out_deg;
        result.max_out_degree = std::max(result.max_out_degree, out_deg);
        result.max_in_degree = std::max(result.max_in_degree, in_deg);
    }

    result.avg_degree = static_cast<float>(total_degree / static_cast<double>(total));

    return result;
}

float
PyramidAnalyzer::get_node_neighbor_recall(const IndexNode* node,
                                          const Vector<InnerIdType>& node_ids) {
    if (node == nullptr || node->graph_ == nullptr || node_ids.empty()) {
        return 0.0F;
    }

    auto graph = node->graph_;
    auto codes = pyramid_->use_reorder_ ? pyramid_->precise_codes_ : pyramid_->base_codes_;
    if (codes == nullptr) {
        return 0.0F;
    }

    UnorderedSet<InnerIdType> id_set(allocator_);
    for (auto id : node_ids) {
        id_set.insert(id);
    }

    float total_neighbor_recall = 0.0F;
    uint32_t valid_count = 0;

    uint32_t sample_count =
        std::min(static_cast<uint32_t>(100), static_cast<uint32_t>(node_ids.size()));

    std::random_device rd;
    std::mt19937 rng(rd());
    Vector<InnerIdType> sample_indices(node_ids.begin(), node_ids.end(), allocator_);
    std::shuffle(sample_indices.begin(), sample_indices.end(), rng);
    sample_indices.resize(sample_count);

    for (auto sample_id : sample_indices) {
        Vector<InnerIdType> neighbors(allocator_);
        graph->GetNeighbors(sample_id, neighbors);

        if (neighbors.empty()) {
            continue;
        }

        Vector<InnerIdType> valid_neighbors(allocator_);
        for (auto nb : neighbors) {
            if (id_set.find(nb) != id_set.end()) {
                valid_neighbors.push_back(nb);
            }
        }

        if (valid_neighbors.empty()) {
            continue;
        }

        bool need_release = false;
        const auto* code = codes->GetCodesById(sample_id, need_release);
        if (code == nullptr) {
            continue;
        }

        Vector<float> query(dim_, allocator_);
        codes->Decode(code, query.data());
        if (need_release) {
            codes->Release(code);
        }

        Vector<float> distances(node_ids.size(), allocator_);
        auto computer = codes->FactoryComputer(query.data());
        codes->Query(distances.data(), computer, node_ids.data(), node_ids.size());

        DistHeapPtr gt = std::make_shared<StandardHeap<true, false>>(allocator_, -1);
        for (size_t i = 0; i < node_ids.size(); ++i) {
            if (gt->Size() < static_cast<int64_t>(valid_neighbors.size())) {
                gt->Push(distances[i], node_ids[i]);
            } else if (distances[i] < gt->Top().first) {
                gt->Push(distances[i], node_ids[i]);
                gt->Pop();
            }
        }

        UnorderedSet<InnerIdType> gt_set(allocator_);
        const auto* gt_data = gt->GetData();
        for (uint32_t i = 0; i < gt->Size(); ++i) {
            gt_set.insert(gt_data[i].second);
        }

        uint32_t hit_count = 0;
        for (auto nb : valid_neighbors) {
            if (gt_set.find(nb) != gt_set.end()) {
                hit_count++;
            }
        }

        total_neighbor_recall +=
            static_cast<float>(hit_count) / static_cast<float>(valid_neighbors.size());
        valid_count++;
    }

    return valid_count > 0 ? total_neighbor_recall / static_cast<float>(valid_count) : 0.0F;
}

PyramidAnalyzer::GraphQualityAnalysis
PyramidAnalyzer::get_node_connectivity(const IndexNode* node, const Vector<InnerIdType>& node_ids) {
    GraphQualityAnalysis result;
    if (node == nullptr || node->graph_ == nullptr || node_ids.empty()) {
        return result;
    }

    auto graph = node->graph_;
    auto total = static_cast<uint32_t>(node_ids.size());

    UnorderedSet<InnerIdType> id_set(allocator_);
    for (auto id : node_ids) {
        id_set.insert(id);
    }

    UnorderedMap<InnerIdType, bool> visited(allocator_);
    for (auto id : node_ids) {
        visited[id] = false;
    }

    Vector<uint32_t> component_sizes(allocator_);

    for (auto start_id : node_ids) {
        if (visited[start_id]) {
            continue;
        }

        uint32_t component_size = 0;
        std::queue<InnerIdType> q;
        q.push(start_id);
        visited[start_id] = true;

        while (!q.empty()) {
            auto curr = q.front();
            q.pop();
            component_size++;

            Vector<InnerIdType> neighbors(allocator_);
            graph->GetNeighbors(curr, neighbors);

            for (auto nb : neighbors) {
                if (id_set.find(nb) != id_set.end() && !visited[nb]) {
                    visited[nb] = true;
                    q.push(nb);
                }
            }
        }

        component_sizes.push_back(component_size);
    }

    result.component_count = static_cast<uint32_t>(component_sizes.size());
    result.max_component_size = *std::max_element(component_sizes.begin(), component_sizes.end());
    result.singleton_count = static_cast<uint32_t>(std::count_if(
        component_sizes.begin(), component_sizes.end(), [](uint32_t s) { return s == 1; }));
    result.connectivity_ratio =
        static_cast<float>(result.max_component_size) / static_cast<float>(total);

    return result;
}

PyramidAnalyzer::GraphQualityAnalysis
PyramidAnalyzer::analyze_entry_point(const IndexNode* node, const Vector<InnerIdType>& node_ids) {
    GraphQualityAnalysis result;
    if (node == nullptr || node->graph_ == nullptr || node_ids.empty()) {
        return result;
    }

    auto graph = node->graph_;
    auto total = static_cast<uint32_t>(node_ids.size());

    UnorderedSet<InnerIdType> id_set(allocator_);
    for (auto id : node_ids) {
        id_set.insert(id);
    }

    InnerIdType entry = node->entry_point_;
    result.entry_point = entry;

    if (id_set.find(entry) == id_set.end()) {
        result.unreachable_count = total;
        return result;
    }

    UnorderedMap<InnerIdType, uint32_t> distances(allocator_);
    for (auto id : node_ids) {
        distances[id] = UINT32_MAX;
    }

    std::queue<InnerIdType> q;
    q.push(entry);
    distances[entry] = 0;

    uint32_t max_distance = 0;
    double total_distance = 0.0;
    uint32_t reachable_count = 1;

    while (!q.empty()) {
        auto curr = q.front();
        q.pop();

        Vector<InnerIdType> neighbors(allocator_);
        graph->GetNeighbors(curr, neighbors);

        for (auto nb : neighbors) {
            if (id_set.find(nb) != id_set.end() && distances[nb] == UINT32_MAX) {
                distances[nb] = distances[curr] + 1;
                max_distance = std::max(max_distance, distances[nb]);
                total_distance += distances[nb];
                reachable_count++;
                q.push(nb);
            }
        }
    }

    result.max_distance_from_entry = max_distance;
    result.unreachable_count = total - reachable_count;
    result.avg_distance_from_entry =
        reachable_count > 1
            ? static_cast<float>(total_distance / static_cast<double>(reachable_count - 1))
            : 0.0F;

    return result;
}

JsonType
PyramidAnalyzer::analyze_node_graph_quality(const IndexNode* node,
                                            const std::string& path,
                                            float recall,
                                            const Vector<InnerIdType>& node_ids) {
    JsonType analysis;
    analysis["path"].SetString(path);
    analysis["recall"].SetFloat(recall);
    analysis["size"].SetInt(node->graph_ ? node->graph_->TotalCount() : 0);

    logger::info("[analyze_node_graph_quality] Analyzing node: path={}, recall={:.4f}, size={}",
                 path,
                 recall,
                 node->graph_ ? node->graph_->TotalCount() : 0);

    auto degree_result = get_node_degree_distribution(node, node_ids);
    logger::info("[analyze_node_graph_quality] Degree: avg={:.2f}, zero_out={}, zero_in={}",
                 degree_result.avg_degree,
                 degree_result.zero_out_degree_count,
                 degree_result.zero_in_degree_count);

    JsonType degree_json;
    degree_json["avg_degree"].SetFloat(degree_result.avg_degree);
    degree_json["zero_out_degree_count"].SetInt(degree_result.zero_out_degree_count);
    degree_json["zero_in_degree_count"].SetInt(degree_result.zero_in_degree_count);
    degree_json["max_out_degree"].SetInt(degree_result.max_out_degree);
    degree_json["max_in_degree"].SetInt(degree_result.max_in_degree);
    analysis["degree"].SetJson(degree_json);

    float neighbor_recall = get_node_neighbor_recall(node, node_ids);
    logger::info("[analyze_node_graph_quality] Neighbor recall: {:.4f}", neighbor_recall);
    analysis["neighbor_recall"].SetFloat(neighbor_recall);

    auto connectivity_result = get_node_connectivity(node, node_ids);
    logger::info(
        "[analyze_node_graph_quality] Connectivity: components={}, max_size={}, singleton={}, "
        "ratio={:.4f}",
        connectivity_result.component_count,
        connectivity_result.max_component_size,
        connectivity_result.singleton_count,
        connectivity_result.connectivity_ratio);

    JsonType connectivity_json;
    connectivity_json["component_count"].SetInt(connectivity_result.component_count);
    connectivity_json["max_component_size"].SetInt(connectivity_result.max_component_size);
    connectivity_json["singleton_count"].SetInt(connectivity_result.singleton_count);
    connectivity_json["connectivity_ratio"].SetFloat(connectivity_result.connectivity_ratio);
    analysis["connectivity"].SetJson(connectivity_json);

    auto entry_result = analyze_entry_point(node, node_ids);
    logger::info(
        "[analyze_node_graph_quality] Entry point: id={}, max_dist={}, unreachable={}, "
        "avg_dist={:.2f}",
        entry_result.entry_point,
        entry_result.max_distance_from_entry,
        entry_result.unreachable_count,
        entry_result.avg_distance_from_entry);

    JsonType entry_json;
    entry_json["entry_point"].SetInt(entry_result.entry_point);
    entry_json["max_distance_from_entry"].SetInt(entry_result.max_distance_from_entry);
    entry_json["unreachable_count"].SetInt(entry_result.unreachable_count);
    entry_json["avg_distance_from_entry"].SetFloat(entry_result.avg_distance_from_entry);
    analysis["entry_point_analysis"].SetJson(entry_json);

    return analysis;
}

JsonType
PyramidAnalyzer::get_graph_node_recall_stats(const std::string& search_param_str) {
    JsonType stats;

    low_recall_nodes_.clear();

    float total_weighted_recall = 0.0F;
    uint64_t total_size = 0;
    uint32_t node_count = 0;
    uint32_t skipped_count = 0;

    std::function<void(IndexNode*, const std::string&)> traverse = [&](IndexNode* node,
                                                                       const std::string& path) {
        if (node == nullptr) {
            return;
        }

        if (node->children_.empty() && node->status_ == IndexNode::Status::GRAPH) {
            uint32_t size = node->graph_ ? node->graph_->TotalCount() : 0;

            if (size < 10000) {
                skipped_count++;
            } else {
                node_count++;

                if (sample_datas_.empty()) {
                    sample_global();
                }

                if (!sample_datas_.empty()) {
                    float recall = calculate_node_recall(
                        node, sample_datas_.data(), sample_ids_.size(), search_param_str);

                    logger::info(
                        "[get_graph_node_recall_stats] Node: path={}, size={}, recall={:.4f}",
                        path,
                        size,
                        recall);

                    total_weighted_recall += recall * static_cast<float>(size);
                    total_size += size;

                    if (recall < 0.8F) {
                        auto node_ids = collect_node_ids(node);
                        float duplicate_ratio = 0.0F;
                        bool entry_point_duplicate = false;
                        uint32_t entry_point_group_size = 0;

                        if (!node_ids.empty()) {
                            duplicate_ratio = get_node_duplicate_ratio(node, node_ids);
                            if (node->status_ == IndexNode::Status::GRAPH && node->graph_) {
                                entry_point_duplicate = check_entry_point_duplicate(
                                    node, node_ids, entry_point_group_size);
                            }
                        }

                        LowRecallNodeInfo info;
                        info.path = path;
                        info.size = size;
                        info.recall = recall;
                        info.duplicate_ratio = duplicate_ratio;
                        info.entry_point_duplicate = entry_point_duplicate;
                        info.entry_point_group_size = entry_point_group_size;
                        low_recall_nodes_.push_back(info);
                    }
                }
            }
        }

        for (const auto& child : node->children_) {
            traverse(child.second.get(), path + "/" + child.first);
        }
    };

    traverse(pyramid_->root_.get(), "");

    float weighted_recall =
        total_size > 0 ? total_weighted_recall / static_cast<float>(total_size) : 0.0F;

    stats["weighted_recall"].SetFloat(weighted_recall);
    stats["node_count"].SetInt(node_count);
    stats["total_size"].SetInt(total_size);
    stats["skipped_node_count"].SetInt(skipped_count);

    if (!low_recall_nodes_.empty()) {
        JsonType low_recall_nodes_json;
        for (const auto& info : low_recall_nodes_) {
            JsonType node_info;
            node_info["size"].SetInt(info.size);
            node_info["recall"].SetFloat(info.recall);
            node_info["duplicate_ratio"].SetFloat(info.duplicate_ratio);
            node_info["entry_point_duplicate"].SetBool(info.entry_point_duplicate);
            node_info["entry_point_group_size"].SetInt(info.entry_point_group_size);
            low_recall_nodes_json[info.path].SetJson(node_info);
        }
        stats["low_recall_nodes"].SetJson(low_recall_nodes_json);
    }

    logger::info(
        "[get_graph_node_recall_stats] Final: recall={:.4f}, nodes={}, skipped={}, "
        "low_recall_nodes={}",
        weighted_recall,
        node_count,
        skipped_count,
        low_recall_nodes_.size());

    return stats;
}

float
PyramidAnalyzer::GetDuplicateRatio() {
    if (sample_datas_.empty()) {
        sample_global();
    }
    if (sample_datas_.empty()) {
        return 0.0F;
    }

    uint64_t total_duplicate_count = 0;
    uint64_t total_vector_count = 0;

    std::function<void(IndexNode*)> traverse = [&](IndexNode* node) {
        if (node == nullptr) {
            return;
        }

        if (node->children_.empty()) {
            auto node_ids = collect_node_ids(node);
            if (node_ids.empty()) {
                return;
            }

            float duplicate_ratio = get_node_duplicate_ratio(node, node_ids);
            auto duplicate_count =
                static_cast<uint32_t>(duplicate_ratio * static_cast<float>(node_ids.size()));

            total_duplicate_count += duplicate_count;
            total_vector_count += node_ids.size();
        }

        for (const auto& child : node->children_) {
            traverse(child.second.get());
        }
    };

    traverse(pyramid_->root_.get());

    return total_vector_count > 0
               ? static_cast<float>(total_duplicate_count) / static_cast<float>(total_vector_count)
               : 0.0F;
}

float
PyramidAnalyzer::get_node_duplicate_ratio(const IndexNode* node,
                                          const Vector<InnerIdType>& node_ids) {
    if (node_ids.empty() || sample_datas_.empty()) {
        return 0.0F;
    }

    constexpr float epsilon = 2e-6F;
    auto codes = pyramid_->base_codes_;

    Vector<Vector<InnerIdType>> groups(allocator_);
    groups.emplace_back(node_ids.begin(), node_ids.end(), allocator_);

    auto query_count = static_cast<uint32_t>(sample_ids_.size());
    for (uint32_t q = 0; q < query_count && !groups.empty(); ++q) {
        Vector<Vector<InnerIdType>> new_groups(allocator_);
        auto comp = codes->FactoryComputer(sample_datas_.data() + static_cast<size_t>(q) * dim_);

        for (auto& group : groups) {
            if (group.empty()) {
                continue;
            }

            Vector<float> dists(group.size(), allocator_);
            codes->Query(dists.data(), comp, group.data(), group.size());

            Vector<std::pair<float, InnerIdType>> sorted(group.size(), allocator_);
            for (size_t i = 0; i < group.size(); ++i) {
                sorted[i] = {dists[i], group[i]};
            }
            std::sort(sorted.begin(), sorted.end());

            Vector<InnerIdType> sub(allocator_);
            sub.push_back(sorted[0].second);
            float first_dist = sorted[0].first;

            for (size_t i = 1; i < sorted.size(); ++i) {
                if (sorted[i].first - first_dist <= epsilon) {
                    sub.push_back(sorted[i].second);
                } else {
                    if (sub.size() > 1) {
                        new_groups.push_back(std::move(sub));
                    }
                    sub.clear();
                    sub.push_back(sorted[i].second);
                    first_dist = sorted[i].first;
                }
            }
            if (sub.size() > 1) {
                new_groups.push_back(std::move(sub));
            }
        }
        groups = std::move(new_groups);
    }

    uint64_t duplicate_count = 0;
    for (const auto& group : groups) {
        duplicate_count += group.size() - 1;
    }

    return static_cast<float>(duplicate_count) / static_cast<float>(node_ids.size());
}

bool
PyramidAnalyzer::check_entry_point_duplicate(const IndexNode* node,
                                             const Vector<InnerIdType>& node_ids,
                                             uint32_t& duplicate_group_size) {
    duplicate_group_size = 0;
    if (node_ids.empty() || node->graph_ == nullptr) {
        return false;
    }

    InnerIdType entry_id = node->entry_point_;

    UnorderedSet<InnerIdType> id_set(allocator_);
    for (auto id : node_ids) {
        id_set.insert(id);
    }
    if (id_set.find(entry_id) == id_set.end()) {
        return false;
    }

    Vector<float> entry_vector(dim_, allocator_);
    pyramid_->GetVectorByInnerId(entry_id, entry_vector.data());

    auto codes = pyramid_->base_codes_;
    auto comp = codes->FactoryComputer(entry_vector.data());
    Vector<float> dists(node_ids.size(), allocator_);
    codes->Query(dists.data(), comp, node_ids.data(), node_ids.size());

    constexpr float epsilon = 2e-6F;
    uint32_t count = 0;
    for (size_t i = 0; i < node_ids.size(); ++i) {
        if (dists[i] <= epsilon) {
            count++;
        }
    }

    if (count > 1) {
        duplicate_group_size = count;
        return true;
    }

    return false;
}

}  // namespace vsag
