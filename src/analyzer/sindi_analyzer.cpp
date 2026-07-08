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

#include "sindi_analyzer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "simd/fp16_simd.h"
#include "utils/sparse_vector_transform.h"

namespace vsag {
namespace {

constexpr int64_t K_ANALYZE_DEFAULT_TOPK = 10;
constexpr uint64_t K_ANALYZE_BASE_SAMPLE_LIMIT = 10;
constexpr uint64_t K_ANALYZE_DOC_SAMPLE_LIMIT = 100000;
constexpr float K_ANALYZE_EPSILON = 1e-6F;

bool
is_sq8_value_quantization(SparseValueQuantizationType type) {
    return type == SparseValueQuantizationType::SQ8;
}

struct analyze_options {
    std::string base_path;
    std::string groundtruth_path;
    std::string save_groundtruth_path;
};

JsonType
make_skip_json(const std::string& reason) {
    JsonType json;
    json["skipped_reason"].SetString(reason);
    return json;
}

float
calculate_percentile(const std::vector<float>& sorted_values, float percentile) {
    if (sorted_values.empty()) {
        return 0.0F;
    }
    float index = percentile * static_cast<float>(sorted_values.size() - 1);
    auto lower = static_cast<size_t>(std::floor(index));
    auto upper = static_cast<size_t>(std::ceil(index));
    if (lower == upper) {
        return sorted_values[lower];
    }
    float weight = index - static_cast<float>(lower);
    return sorted_values[lower] * (1.0F - weight) + sorted_values[upper] * weight;
}

JsonType
make_distribution_stats_from_sorted(const std::vector<float>& sorted_values) {
    JsonType json;
    if (sorted_values.empty()) {
        json["mean"].SetFloat(0.0F);
        json["max"].SetFloat(0.0F);
        json["p95"].SetFloat(0.0F);
        json["p99"].SetFloat(0.0F);
        return json;
    }

    float sum = std::accumulate(sorted_values.begin(), sorted_values.end(), 0.0F);
    json["mean"].SetFloat(sum / static_cast<float>(sorted_values.size()));
    json["max"].SetFloat(sorted_values.back());
    json["p95"].SetFloat(calculate_percentile(sorted_values, 0.95F));
    json["p99"].SetFloat(calculate_percentile(sorted_values, 0.99F));
    return json;
}

JsonType
make_distribution_stats(const std::vector<float>& values) {
    auto sorted_values = values;
    std::sort(sorted_values.begin(), sorted_values.end());
    return make_distribution_stats_from_sorted(sorted_values);
}

std::streamsize
to_stream_size(size_t bytes) {
    return static_cast<std::streamsize>(bytes);
}

void
release_sparse_vector(Allocator* allocator, SparseVector* vector) {
    if (vector == nullptr || allocator == nullptr) {
        return;
    }
    if (vector->ids_ != nullptr) {
        allocator->Deallocate(vector->ids_);
        vector->ids_ = nullptr;
    }
    if (vector->vals_ != nullptr) {
        allocator->Deallocate(vector->vals_);
        vector->vals_ = nullptr;
    }
    vector->len_ = 0;
}

void
copy_sparse_vector(const SparseVector& source, SparseVector* target, Allocator* allocator) {
    target->len_ = source.len_;
    target->ids_ = nullptr;
    target->vals_ = nullptr;
    if (source.len_ == 0 || allocator == nullptr) {
        return;
    }
    auto bytes = sizeof(uint32_t) * static_cast<size_t>(source.len_);
    target->ids_ = static_cast<uint32_t*>(allocator->Allocate(bytes));
    std::copy(source.ids_, source.ids_ + source.len_, target->ids_);
    target->vals_ =
        static_cast<float*>(allocator->Allocate(sizeof(float) * static_cast<size_t>(source.len_)));
    std::copy(source.vals_, source.vals_ + source.len_, target->vals_);
}

float
get_sparse_distance(const SparseVector& query, const SparseVector& base) {
    float dot = 0.0F;
    uint32_t query_idx = 0;
    uint32_t base_idx = 0;
    while (query_idx < query.len_ && base_idx < base.len_) {
        if (query.ids_[query_idx] < base.ids_[base_idx]) {
            ++query_idx;
        } else if (query.ids_[query_idx] > base.ids_[base_idx]) {
            ++base_idx;
        } else {
            dot += query.vals_[query_idx] * base.vals_[base_idx];
            ++query_idx;
            ++base_idx;
        }
    }
    return 1.0F - dot;
}

DatasetPtr
make_single_query_dataset(const SparseVector* query) {
    auto dataset = Dataset::Make();
    dataset->NumElements(1)->SparseVectors(query)->Owner(false);
    return dataset;
}

void
set_skip_metrics(JsonType& stats, const std::string& reason) {
    stats["recall_base"].SetJson(make_skip_json(reason));
    stats["doc_prune_bias_mean"].SetJson(make_skip_json(reason));
    stats["doc_prune_inversion_count_rate"].SetJson(make_skip_json(reason));
    stats["doc_prune_recall"].SetJson(make_skip_json(reason));
    stats["quantization_bias_ratio"].SetJson(make_skip_json(reason));
    stats["quantization_inversion_count_rate"].SetJson(make_skip_json(reason));
    stats["quantization_recall"].SetJson(make_skip_json(reason));
}

int64_t
resolve_candidate_count(const SINDISearchParameter& search_param, int64_t topk) {
    return std::max<int64_t>(search_param.n_candidate, topk);
}

}  // namespace

void
SINDIAnalyzer::get_pruned_sparse_vector_by_inner_id(InnerIdType inner_id,
                                                    SparseVector* data,
                                                    Allocator* specified_allocator) const {
    std::shared_lock rlock(sindi_->global_mutex_);
    Allocator* allocator =
        specified_allocator != nullptr ? specified_allocator : sindi_->allocator_;
    auto cur_window = inner_id / sindi_->window_size_;
    auto window_start_id = cur_window * sindi_->window_size_;
    auto term_list = sindi_->window_term_list_[cur_window];
    term_list->GetSparseVector(inner_id - window_start_id, data, allocator);
    if (sindi_->remap_term_ids_ && sindi_->term_id_mapper_) {
        for (uint32_t i = 0; i < data->len_; ++i) {
            data->ids_[i] = sindi_->term_id_mapper_->ReverseMap(data->ids_[i]);
        }
    }
}

std::vector<std::pair<float, int64_t>>
SINDIAnalyzer::collect_coarse_candidates(const SparseVector& query,
                                         const SINDISearchParameter& search_param,
                                         int64_t candidate_count) const {
    std::shared_lock rlock(sindi_->global_mutex_);
    std::vector<std::pair<float, int64_t>> coarse_candidates;
    if (candidate_count <= 0 || query.len_ == 0) {
        return coarse_candidates;
    }

    SparseVector effective_query = query;
    Vector<uint32_t> tmp_ids(sindi_->allocator_);
    Vector<float> tmp_vals(sindi_->allocator_);
    if (sindi_->remap_term_ids_) {
        effective_query = sindi_->remap_sparse_vector_for_query(query, tmp_ids, tmp_vals);
        if (effective_query.len_ == 0) {
            return coarse_candidates;
        }
    }

    InnerSearchParam inner_param;
    inner_param.ef = candidate_count;
    inner_param.topk = candidate_count;
    MaxHeap heap(sindi_->allocator_);
    auto computer =
        std::make_shared<SparseTermComputer>(effective_query, search_param, sindi_->allocator_);
    const bool use_term_lists_heap_insert = sindi_->UseTermListsHeapInsert(search_param);

    Vector<float> dists(sindi_->window_size_, 0.0F, sindi_->allocator_);
    for (int64_t cur = 0; cur < static_cast<int64_t>(sindi_->window_term_list_.size()); ++cur) {
        auto window_start_id = static_cast<uint32_t>(cur * sindi_->window_size_);
        auto term_list = sindi_->window_term_list_[cur];
        term_list->Query(dists.data(), computer);
        if (use_term_lists_heap_insert) {
            term_list->InsertHeapByTermLists<KNN_SEARCH, PURE>(
                dists.data(), computer, heap, inner_param, window_start_id);
        } else {
            term_list->InsertHeapByDists<KNN_SEARCH, PURE>(
                dists.data(), dists.size(), heap, inner_param, window_start_id);
        }
    }

    coarse_candidates.resize(heap.size());
    for (int64_t idx = static_cast<int64_t>(heap.size()) - 1; idx >= 0; --idx) {
        coarse_candidates[idx] = {1.0F + heap.top().first,
                                  sindi_->label_table_->GetLabelById(heap.top().second)};
        heap.pop();
    }
    return coarse_candidates;
}

std::vector<std::pair<float, int64_t>>
SINDIAnalyzer::collect_doc_prune_candidates(const SparseVector& query,
                                            const SINDISearchParameter& search_param,
                                            int64_t candidate_count) const {
    std::shared_lock rlock(sindi_->global_mutex_);
    std::vector<std::pair<float, int64_t>> doc_prune_candidates;
    if (candidate_count <= 0 || query.len_ == 0) {
        return doc_prune_candidates;
    }

    SparseVector effective_query = query;
    Vector<uint32_t> tmp_ids(sindi_->allocator_);
    Vector<float> tmp_vals(sindi_->allocator_);
    if (sindi_->remap_term_ids_) {
        effective_query = sindi_->remap_sparse_vector_for_query(query, tmp_ids, tmp_vals);
        if (effective_query.len_ == 0) {
            return doc_prune_candidates;
        }
    }

    InnerSearchParam inner_param;
    inner_param.ef = candidate_count;
    inner_param.topk = candidate_count;
    MaxHeap heap(sindi_->allocator_);
    auto computer =
        std::make_shared<SparseTermComputer>(effective_query, search_param, sindi_->allocator_);
    const bool use_term_lists_heap_insert = sindi_->UseTermListsHeapInsert(search_param);

    Vector<float> dists(sindi_->window_size_, 0.0F, sindi_->allocator_);

    for (int64_t cur = 0; cur < static_cast<int64_t>(sindi_->window_term_list_.size()); ++cur) {
        auto window_start_id = static_cast<uint32_t>(cur * sindi_->window_size_);
        auto term_list = sindi_->window_term_list_[cur];
        if (term_list == nullptr) {
            continue;
        }
        std::fill(dists.begin(), dists.end(), 0.0F);

        while (computer->HasNextTerm()) {
            auto term_idx = computer->NextTermIter();
            auto term = computer->GetTerm(term_idx);
            if (term >= term_list->term_sizes_.size() || term_list->term_sizes_[term] == 0) {
                continue;
            }
            auto term_size = static_cast<uint32_t>(
                static_cast<float>(term_list->term_sizes_[term]) * computer->term_retain_ratio_);
            if (term_size == 0) {
                continue;
            }

            if (is_sq8_value_quantization(term_list->sparse_value_quant_type_)) {
                computer->ScanForAccumulateSQ8(term_idx,
                                               term_list->term_ids_[term]->data(),
                                               term_list->term_datas_[term]->data(),
                                               term_size,
                                               dists.data());
            } else if (term_list->sparse_value_quant_type_ == SparseValueQuantizationType::FP16) {
                computer->ScanForAccumulateFP16Bytes(term_idx,
                                                     term_list->term_ids_[term]->data(),
                                                     term_list->term_datas_[term]->data(),
                                                     term_size,
                                                     dists.data());
            } else {
                computer->ScanForAccumulateFloatBytes(term_idx,
                                                      term_list->term_ids_[term]->data(),
                                                      term_list->term_datas_[term]->data(),
                                                      term_size,
                                                      dists.data());
            }
        }
        computer->ResetTerm();

        if (use_term_lists_heap_insert) {
            term_list->InsertHeapByTermLists<KNN_SEARCH, PURE>(
                dists.data(), computer, heap, inner_param, window_start_id);
        } else {
            term_list->InsertHeapByDists<KNN_SEARCH, PURE>(
                dists.data(), dists.size(), heap, inner_param, window_start_id);
        }
    }

    doc_prune_candidates.resize(heap.size());
    for (int64_t idx = static_cast<int64_t>(heap.size()) - 1; idx >= 0; --idx) {
        doc_prune_candidates[idx] = {1.0F + heap.top().first,
                                     sindi_->label_table_->GetLabelById(heap.top().second)};
        heap.pop();
    }
    return doc_prune_candidates;
}

std::vector<std::pair<float, int64_t>>
SINDIAnalyzer::rerank_candidates(const SparseVector& query,
                                 const std::vector<std::pair<float, int64_t>>& coarse_candidates,
                                 int64_t topk) const {
    std::vector<std::pair<float, int64_t>> reranked;
    if (not sindi_->use_reorder_ || coarse_candidates.empty() || topk <= 0) {
        return reranked;
    }
    reranked.reserve(coarse_candidates.size());
    for (const auto& coarse : coarse_candidates) {
        auto exact_dist = calculate_exact_distance_by_label(query, coarse.second);
        reranked.emplace_back(exact_dist, coarse.second);
    }
    std::sort(reranked.begin(), reranked.end(), [](const auto& left, const auto& right) {
        if (std::abs(left.first - right.first) > K_ANALYZE_EPSILON) {
            return left.first < right.first;
        }
        return left.second < right.second;
    });
    if (static_cast<int64_t>(reranked.size()) > topk) {
        reranked.resize(static_cast<size_t>(topk));
    }
    return reranked;
}

bool
SINDIAnalyzer::get_original_sparse_vector_by_inner_id(InnerIdType inner_id,
                                                      SparseVector* data,
                                                      Allocator* specified_allocator,
                                                      const DatasetPtr& base_dataset) const {
    Allocator* allocator =
        specified_allocator != nullptr ? specified_allocator : sindi_->allocator_;
    if (allocator == nullptr) {
        return false;
    }
    if (sindi_->use_reorder_) {
        sindi_->GetSparseVectorByInnerId(inner_id, data, allocator);
        return true;
    }
    if (base_dataset == nullptr || base_dataset->GetSparseVectors() == nullptr ||
        inner_id >= base_dataset->GetNumElements()) {
        return false;
    }
    copy_sparse_vector(base_dataset->GetSparseVectors()[inner_id], data, allocator);
    return true;
}

float
SINDIAnalyzer::calculate_exact_distance_by_label(const SparseVector& query,
                                                 int64_t label,
                                                 const DatasetPtr& base_dataset) const {
    auto inner_id = sindi_->label_table_->GetIdByLabel(label);
    SparseVector original{};
    if (not get_original_sparse_vector_by_inner_id(
            inner_id, &original, sindi_->allocator_, base_dataset)) {
        return std::numeric_limits<float>::max();
    }
    auto distance = get_sparse_distance(query, original);
    release_sparse_vector(sindi_->allocator_, &original);
    return distance;
}

float
SINDIAnalyzer::calculate_pruned_distance_by_label(const SparseVector& query, int64_t label) const {
    auto inner_id = sindi_->label_table_->GetIdByLabel(label);
    SparseVector pruned{};
    get_pruned_sparse_vector_by_inner_id(inner_id, &pruned, sindi_->allocator_);
    auto distance = get_sparse_distance(query, pruned);
    release_sparse_vector(sindi_->allocator_, &pruned);
    return distance;
}

DatasetPtr
SINDIAnalyzer::calculate_ground_truth(const DatasetPtr& query_dataset,
                                      int64_t topk,
                                      const DatasetPtr& base_dataset) const {
    if (query_dataset == nullptr || query_dataset->GetSparseVectors() == nullptr || topk <= 0) {
        return nullptr;
    }
    if (not sindi_->use_reorder_ &&
        (base_dataset == nullptr || base_dataset->GetSparseVectors() == nullptr)) {
        return nullptr;
    }

    auto query_count = query_dataset->GetNumElements();
    auto base_count = base_dataset == nullptr ? sindi_->cur_element_count_.load()
                                              : base_dataset->GetNumElements();
    auto effective_topk = std::min<int64_t>(topk, base_count);
    if (effective_topk <= 0) {
        return nullptr;
    }

    auto* ids = static_cast<int64_t*>(sindi_->allocator_->Allocate(
        sizeof(int64_t) * static_cast<size_t>(query_count) * effective_topk));
    auto* distances = static_cast<float*>(sindi_->allocator_->Allocate(
        sizeof(float) * static_cast<size_t>(query_count) * effective_topk));

    for (int64_t query_idx = 0; query_idx < query_count; ++query_idx) {
        std::priority_queue<std::pair<float, int64_t>> heap;
        const auto& query = query_dataset->GetSparseVectors()[query_idx];
        for (int64_t base_idx = 0; base_idx < base_count; ++base_idx) {
            SparseVector original{};
            const SparseVector* base_vector = nullptr;
            if (base_dataset != nullptr && base_dataset->GetSparseVectors() != nullptr) {
                base_vector = base_dataset->GetSparseVectors() + base_idx;
            } else {
                sindi_->GetSparseVectorByInnerId(base_idx, &original, sindi_->allocator_);
                base_vector = &original;
            }

            auto label = base_dataset != nullptr && base_dataset->GetIds() != nullptr
                             ? base_dataset->GetIds()[base_idx]
                             : sindi_->label_table_->GetLabelById(base_idx);
            heap.emplace(get_sparse_distance(query, *base_vector), label);
            if (static_cast<int64_t>(heap.size()) > effective_topk) {
                heap.pop();
            }
            release_sparse_vector(sindi_->allocator_, &original);
        }

        for (int64_t rank = effective_topk - 1; rank >= 0; --rank) {
            auto offset = static_cast<size_t>(query_idx) * effective_topk + rank;
            ids[offset] = heap.top().second;
            distances[offset] = heap.top().first;
            heap.pop();
        }
    }

    auto ground_truth = Dataset::Make();
    ground_truth->NumElements(query_count)
        ->Dim(effective_topk)
        ->Ids(ids)
        ->Distances(distances)
        ->Owner(true, sindi_->allocator_);
    return ground_truth;
}

namespace {

JsonType
parse_sindi_search_json(const std::string& params_str) {
    JsonType json;
    if (params_str.empty() || params_str == "default") {
        json[INDEX_SINDI].SetJson(JsonType());
        return json;
    }
    json = JsonType::Parse(params_str);
    if (not json.Contains(INDEX_SINDI)) {
        json[INDEX_SINDI].SetJson(JsonType());
    }
    return json;
}

analyze_options
parse_analyze_options(const JsonType& json) {
    analyze_options options;
    if (not json.Contains("analyze")) {
        return options;
    }
    auto analyze = json["analyze"];
    if (analyze.Contains("base_path")) {
        options.base_path = analyze["base_path"].GetString();
    }
    if (analyze.Contains("groundtruth_path")) {
        options.groundtruth_path = analyze["groundtruth_path"].GetString();
    }
    if (analyze.Contains("save_groundtruth_path")) {
        options.save_groundtruth_path = analyze["save_groundtruth_path"].GetString();
    }
    return options;
}

DatasetPtr
load_sparse_dataset(const std::string& file_path) {
    std::fstream in_stream(file_path, std::ios::binary | std::ios::in);
    if (not in_stream.is_open()) {
        logger::warn("Failed to open sparse dataset file: {}", file_path);
        return nullptr;
    }

    int64_t rows = 0;
    int64_t cols = 0;
    int64_t nnz = 0;
    in_stream.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    in_stream.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    in_stream.read(reinterpret_cast<char*>(&nnz), sizeof(nnz));
    if (rows < 0 || cols < 0 || nnz < 0) {
        logger::warn("Invalid sparse dataset header in file: {}", file_path);
        return nullptr;
    }

    std::vector<int64_t> indptr(static_cast<size_t>(rows) + 1);
    std::vector<int32_t> indices(static_cast<size_t>(nnz));
    std::vector<float> values(static_cast<size_t>(nnz));

    in_stream.read(reinterpret_cast<char*>(indptr.data()),
                   to_stream_size(sizeof(int64_t) * (static_cast<size_t>(rows) + 1)));
    in_stream.read(reinterpret_cast<char*>(indices.data()),
                   to_stream_size(sizeof(int32_t) * static_cast<size_t>(nnz)));
    in_stream.read(reinterpret_cast<char*>(values.data()),
                   to_stream_size(sizeof(float) * static_cast<size_t>(nnz)));
    if (not in_stream.good()) {
        logger::warn("Failed to read sparse dataset CSR arrays from file: {}", file_path);
        return nullptr;
    }

    if (indptr[0] != 0 || indptr[static_cast<size_t>(rows)] != nnz) {
        logger::warn("Invalid sparse dataset indptr bounds in file: {}", file_path);
        return nullptr;
    }
    for (int64_t row = 0; row < rows; ++row) {
        if (indptr[static_cast<size_t>(row)] > indptr[static_cast<size_t>(row + 1)] ||
            indptr[static_cast<size_t>(row)] < 0 || indptr[static_cast<size_t>(row + 1)] > nnz) {
            logger::warn("Invalid sparse dataset indptr monotonicity in file: {}", file_path);
            return nullptr;
        }
    }
    for (int64_t offset = 0; offset < nnz; ++offset) {
        if (indices[static_cast<size_t>(offset)] < 0 ||
            indices[static_cast<size_t>(offset)] >= cols) {
            logger::warn("Invalid sparse dataset column index in file: {}", file_path);
            return nullptr;
        }
    }

    auto* sparse_vectors = new SparseVector[static_cast<size_t>(rows)];

    for (int64_t row = 0; row < rows; ++row) {
        auto begin = indptr[static_cast<size_t>(row)];
        auto end = indptr[static_cast<size_t>(row + 1)];
        auto len = end - begin;
        sparse_vectors[row].len_ = static_cast<uint32_t>(len);
        sparse_vectors[row].ids_ = new uint32_t[static_cast<size_t>(len)];
        sparse_vectors[row].vals_ = new float[static_cast<size_t>(len)];
        for (int64_t offset = 0; offset < len; ++offset) {
            auto value_offset = static_cast<size_t>(begin + offset);
            sparse_vectors[row].ids_[offset] = static_cast<uint32_t>(indices[value_offset]);
            sparse_vectors[row].vals_[offset] = values[value_offset];
        }
    }

    auto dataset = Dataset::Make();
    dataset->SparseVectors(sparse_vectors)->Owner(true)->NumElements(rows)->Dim(cols);
    return dataset;
}

void
check_base_dataset_for_analyze(const DatasetPtr& base_dataset, int64_t expected_count) {
    CHECK_ARGUMENT(base_dataset != nullptr, "SINDI analyze requires valid sparse base dataset");
    CHECK_ARGUMENT(base_dataset->GetSparseVectors() != nullptr,
                   "SINDI analyze requires valid sparse base dataset");
    CHECK_ARGUMENT(base_dataset->GetNumElements() == expected_count,
                   "SINDI analyze base dataset count mismatch with index element count");
}

DatasetPtr
load_ground_truth(const std::string& file_path) {
    std::fstream in_stream(file_path, std::ios::binary | std::ios::in);
    if (not in_stream.is_open()) {
        logger::warn("Failed to open ground truth file: {}", file_path);
        return nullptr;
    }

    uint32_t rows = 0;
    uint32_t dim = 0;
    in_stream.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    in_stream.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    if (not in_stream.good()) {
        logger::warn("Failed to read ground truth header from file: {}", file_path);
        return nullptr;
    }
    if (dim != 0 && rows > std::numeric_limits<size_t>::max() / dim) {
        logger::warn("Invalid ground truth shape in file: {}", file_path);
        return nullptr;
    }
    auto total = static_cast<size_t>(rows) * dim;
    std::vector<int32_t> ids32(total);
    std::vector<float> distance_buffer(total);
    in_stream.read(reinterpret_cast<char*>(ids32.data()), to_stream_size(sizeof(int32_t) * total));
    in_stream.read(reinterpret_cast<char*>(distance_buffer.data()),
                   to_stream_size(sizeof(float) * total));
    if (not in_stream.good()) {
        logger::warn("Failed to read ground truth payload from file: {}", file_path);
        return nullptr;
    }

    auto* ids64 = new int64_t[total];
    auto* distances = new float[total];
    for (size_t i = 0; i < total; ++i) {
        ids64[i] = ids32[i];
        distances[i] = distance_buffer[i];
    }

    auto dataset = Dataset::Make();
    dataset->NumElements(rows)->Dim(dim)->Ids(ids64)->Distances(distances)->Owner(true);
    return dataset;
}

bool
save_ground_truth(const std::string& file_path, const DatasetPtr& ground_truth) {
    if (file_path.empty() || ground_truth == nullptr) {
        return false;
    }
    std::fstream out_stream(file_path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (not out_stream.is_open()) {
        logger::warn("Failed to write ground truth file: {}", file_path);
        return false;
    }
    if (ground_truth->GetNumElements() < 0 || ground_truth->GetDim() < 0 ||
        ground_truth->GetNumElements() > std::numeric_limits<uint32_t>::max() ||
        ground_truth->GetDim() > std::numeric_limits<uint32_t>::max()) {
        logger::warn("Ground truth shape exceeds file format limits for file: {}", file_path);
        return false;
    }
    auto rows = static_cast<uint32_t>(ground_truth->GetNumElements());
    auto dim = static_cast<uint32_t>(ground_truth->GetDim());
    auto total = static_cast<size_t>(rows) * dim;
    if (ground_truth->GetIds() == nullptr || ground_truth->GetDistances() == nullptr) {
        logger::warn("Invalid ground truth dataset for file: {}", file_path);
        return false;
    }
    std::vector<int32_t> ids(total);
    for (size_t i = 0; i < total; ++i) {
        if (ground_truth->GetIds()[i] < std::numeric_limits<int32_t>::min() ||
            ground_truth->GetIds()[i] > std::numeric_limits<int32_t>::max()) {
            logger::warn("Ground truth id exceeds int32 range for file: {}", file_path);
            return false;
        }
        ids[i] = static_cast<int32_t>(ground_truth->GetIds()[i]);
    }
    out_stream.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    out_stream.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out_stream.write(reinterpret_cast<const char*>(ids.data()),
                     to_stream_size(sizeof(int32_t) * total));
    out_stream.write(reinterpret_cast<const char*>(ground_truth->GetDistances()),
                     to_stream_size(sizeof(float) * total));
    return out_stream.good();
}

float
calculate_recall(const int64_t* result_ids,
                 int64_t result_dim,
                 const int64_t* gt_ids,
                 int64_t topk) {
    if (topk == 0) {
        return 0.0F;
    }
    std::unordered_set<int64_t> gt_set;
    gt_set.reserve(static_cast<size_t>(topk));
    for (int64_t i = 0; i < topk; ++i) {
        gt_set.insert(gt_ids[i]);
    }
    int64_t hit_count = 0;
    for (int64_t i = 0; i < result_dim; ++i) {
        if (gt_set.count(result_ids[i]) != 0) {
            ++hit_count;
        }
    }
    return static_cast<float>(hit_count) / static_cast<float>(topk);
}

float
calculate_candidate_recall(const std::vector<std::pair<float, int64_t>>& candidates,
                           const int64_t* gt_ids,
                           int64_t topk,
                           int64_t candidate_limit) {
    auto count = static_cast<int64_t>(candidates.size());
    if (candidate_limit >= 0) {
        count = std::min(count, candidate_limit);
    }
    std::vector<int64_t> candidate_ids;
    candidate_ids.reserve(static_cast<size_t>(count));
    for (int64_t idx = 0; idx < count; ++idx) {
        candidate_ids.push_back(candidates[idx].second);
    }
    return calculate_recall(candidate_ids.data(), count, gt_ids, topk);
}

}  // namespace

JsonType
SINDIAnalyzer::get_active_term_count_stats() const {
    std::vector<float> active_means;
    std::vector<float> active_counts;
    active_means.reserve(sindi_->window_term_list_.size());
    active_counts.reserve(sindi_->window_term_list_.size());

    for (const auto& window : sindi_->window_term_list_) {
        if (window == nullptr) {
            continue;
        }
        float active_count = 0.0F;
        for (uint32_t term_size : window->term_sizes_) {
            if (term_size > 0) {
                active_count += 1.0F;
            }
        }
        active_counts.push_back(active_count);
        auto denominator = static_cast<float>(std::max(1U, window->term_capacity_));
        active_means.push_back(active_count / denominator);
    }

    JsonType stats;
    if (active_means.empty()) {
        stats["mean"].SetFloat(0.0F);
        stats["min"].SetFloat(0.0F);
        stats["max"].SetFloat(0.0F);
        stats["avg_count"].SetFloat(0.0F);
        return stats;
    }

    float ratio_sum = std::accumulate(active_means.begin(), active_means.end(), 0.0F);
    float count_sum = std::accumulate(active_counts.begin(), active_counts.end(), 0.0F);
    stats["mean"].SetFloat(ratio_sum / static_cast<float>(active_means.size()));
    stats["min"].SetFloat(*std::min_element(active_means.begin(), active_means.end()));
    stats["max"].SetFloat(*std::max_element(active_means.begin(), active_means.end()));
    stats["avg_count"].SetFloat(count_sum / static_cast<float>(active_counts.size()));
    return stats;
}

JsonType
SINDIAnalyzer::calculate_recall_stats(const DatasetPtr& query_dataset,
                                      const DatasetPtr& ground_truth,
                                      const SINDISearchParameter& search_param,
                                      int64_t topk) const {
    JsonType stats;
    if (query_dataset == nullptr || query_dataset->GetSparseVectors() == nullptr ||
        ground_truth == nullptr || ground_truth->GetIds() == nullptr || topk <= 0) {
        auto skip = make_skip_json("ground truth unavailable");
        stats["recall_query"].SetJson(skip);
        stats["mean_latency_ms"].SetJson(skip);
        stats["doc_prune_recall"].SetJson(skip);
        stats["quantization_recall"].SetJson(skip);
        return stats;
    }

    auto query_count = query_dataset->GetNumElements();
    float recall_sum = 0.0F;
    float doc_prune_recall_sum = 0.0F;
    float quantization_recall_sum = 0.0F;
    double total_latency_ms = 0.0;
    auto search_param_str = search_param.ToJson().Dump();
    SINDISearchParameter no_query_prune_param = search_param;
    no_query_prune_param.query_prune_ratio = 0.0F;

    for (int64_t query_idx = 0; query_idx < query_count; ++query_idx) {
        auto single_query =
            make_single_query_dataset(query_dataset->GetSparseVectors() + query_idx);
        auto start = std::chrono::high_resolution_clock::now();
        auto search_result = sindi_->KnnSearch(single_query, topk, search_param_str, nullptr);
        auto end = std::chrono::high_resolution_clock::now();
        total_latency_ms += std::chrono::duration<double, std::milli>(end - start).count();
        recall_sum += calculate_recall(search_result->GetIds(),
                                       search_result->GetDim(),
                                       ground_truth->GetIds() + query_idx * ground_truth->GetDim(),
                                       topk);

        const auto& query = query_dataset->GetSparseVectors()[query_idx];
        auto candidate_count = resolve_candidate_count(no_query_prune_param, topk);
        auto doc_prune_candidates =
            collect_doc_prune_candidates(query, no_query_prune_param, candidate_count);
        auto doc_prune_recall =
            calculate_candidate_recall(doc_prune_candidates,
                                       ground_truth->GetIds() + query_idx * ground_truth->GetDim(),
                                       topk,
                                       -1);
        doc_prune_recall_sum += doc_prune_recall;
        if (is_sq8_value_quantization(sindi_->sparse_value_quant_type_)) {
            auto quantization_candidates =
                collect_coarse_candidates(query, no_query_prune_param, candidate_count);
            quantization_recall_sum += calculate_candidate_recall(
                quantization_candidates,
                ground_truth->GetIds() + query_idx * ground_truth->GetDim(),
                topk,
                -1);
        }
    }

    auto denominator = static_cast<float>(std::max<int64_t>(1, query_count));
    stats["recall_query"].SetFloat(recall_sum / denominator);
    stats["doc_prune_recall"].SetFloat(doc_prune_recall_sum / denominator);
    if (is_sq8_value_quantization(sindi_->sparse_value_quant_type_)) {
        stats["quantization_recall"].SetFloat(quantization_recall_sum / denominator);
    } else {
        stats["quantization_recall"].SetJson(make_skip_json("quantization disabled"));
    }
    stats["mean_latency_ms"].SetFloat(static_cast<float>(total_latency_ms) / denominator);
    return stats;
}

JsonType
SINDIAnalyzer::calculate_postings_scanned_stats(const DatasetPtr& query_dataset,
                                                const SINDISearchParameter& search_param) const {
    if (query_dataset == nullptr || query_dataset->GetSparseVectors() == nullptr) {
        return make_skip_json("query dataset unavailable");
    }

    float pruned_term_sum = 0.0F;
    float term_hit_sum = 0.0F;
    float posting_hit_mean_sum = 0.0F;
    for (int64_t query_idx = 0; query_idx < query_dataset->GetNumElements(); ++query_idx) {
        const auto& query = query_dataset->GetSparseVectors()[query_idx];
        Vector<std::pair<uint32_t, float>> sorted_terms(sindi_->allocator_);
        sort_sparse_vector(query, sorted_terms);
        auto pruned_len = static_cast<uint32_t>((1.0F - search_param.query_prune_ratio) *
                                                static_cast<float>(query.len_));
        if (pruned_len == 0 && query.len_ > 0) {
            pruned_len = 1;
        }
        pruned_term_sum += static_cast<float>(pruned_len);

        uint32_t hit_term_count = 0;
        for (uint32_t term_idx = 0; term_idx < pruned_len && term_idx < sorted_terms.size();
             ++term_idx) {
            auto term = sorted_terms[term_idx].first;
            if (sindi_->remap_term_ids_) {
                auto compact = sindi_->term_id_mapper_->TryMap(term);
                if (not compact.has_value()) {
                    continue;
                }
                term = compact.value();
            }
            bool has_posting = false;
            for (const auto& window : sindi_->window_term_list_) {
                if (window != nullptr && term < window->term_sizes_.size() &&
                    window->term_sizes_[term] > 0) {
                    has_posting = true;
                    break;
                }
            }
            if (has_posting) {
                ++hit_term_count;
            }
        }
        term_hit_sum += static_cast<float>(hit_term_count);
        posting_hit_mean_sum +=
            pruned_len == 0 ? 0.0F
                            : static_cast<float>(hit_term_count) / static_cast<float>(pruned_len);
    }

    JsonType stats;
    auto denominator = static_cast<float>(std::max<int64_t>(1, query_dataset->GetNumElements()));
    stats["query_term_count_after_prune_mean"].SetFloat(pruned_term_sum / denominator);
    stats["query_term_with_posting_mean"].SetFloat(term_hit_sum / denominator);
    stats["posting_hit_mean"].SetFloat(posting_hit_mean_sum / denominator);
    return stats;
}

JsonType
SINDIAnalyzer::get_base_search_stats(const std::string& search_param,
                                     const DatasetPtr& base_dataset) const {
    JsonType stats;
    if (not sindi_->use_reorder_ &&
        (base_dataset == nullptr || base_dataset->GetSparseVectors() == nullptr)) {
        set_skip_metrics(
            stats,
            "requires original base vectors; provide --base_path or rebuild with use_reorder=true");
        return stats;
    }

    std::vector<InnerIdType> sample_ids(static_cast<size_t>(sindi_->cur_element_count_));
    std::iota(sample_ids.begin(), sample_ids.end(), 0);
    std::random_device rd;
    std::mt19937 rng(rd());
    std::shuffle(sample_ids.begin(), sample_ids.end(), rng);

    auto sample_count = std::min<uint64_t>(sample_ids.size(), K_ANALYZE_BASE_SAMPLE_LIMIT);
    if (sample_count == 0) {
        set_skip_metrics(stats, "empty index");
        return stats;
    }

    auto* queries = static_cast<SparseVector*>(
        sindi_->allocator_->Allocate(sizeof(SparseVector) * sample_count));
    for (uint64_t idx = 0; idx < sample_count; ++idx) {
        new (queries + idx) SparseVector{};
        get_original_sparse_vector_by_inner_id(
            sample_ids[idx], queries + idx, sindi_->allocator_, base_dataset);
    }
    auto query_dataset = Dataset::Make();
    query_dataset->SparseVectors(queries)
        ->NumElements(static_cast<int64_t>(sample_count))
        ->Owner(true, sindi_->allocator_)
        ->Dim(sindi_->dim_);

    auto search_json = parse_sindi_search_json(search_param);
    SINDISearchParameter parsed_search_param;
    parsed_search_param.FromJson(search_json);
    auto ground_truth = calculate_ground_truth(query_dataset, K_ANALYZE_DEFAULT_TOPK, base_dataset);
    auto recall_stats = calculate_recall_stats(
        query_dataset, ground_truth, parsed_search_param, K_ANALYZE_DEFAULT_TOPK);
    if (recall_stats.Contains("recall_query")) {
        stats["recall_base"].SetJson(recall_stats["recall_query"]);
    }
    if (recall_stats.Contains("doc_prune_recall")) {
        stats["doc_prune_recall"].SetJson(recall_stats["doc_prune_recall"]);
    }
    if (recall_stats.Contains("quantization_recall")) {
        stats["quantization_recall"].SetJson(recall_stats["quantization_recall"]);
    }

    auto quality_stats = calculate_distance_quality_stats(
        query_dataset, parsed_search_param, base_dataset, K_ANALYZE_DEFAULT_TOPK);
    for (const auto& key : {"doc_prune_bias_mean",
                            "doc_prune_inversion_count_rate",
                            "quantization_bias_ratio",
                            "quantization_inversion_count_rate"}) {
        if (quality_stats.Contains(key)) {
            stats[key].SetJson(quality_stats[key]);
        }
    }
    return stats;
}

JsonType
SINDIAnalyzer::get_posting_length_distribution_stats() const {
    std::vector<float> posting_lengths;
    for (const auto& window : sindi_->window_term_list_) {
        if (window == nullptr) {
            continue;
        }
        for (uint32_t term_size : window->term_sizes_) {
            if (term_size > 0) {
                posting_lengths.push_back(static_cast<float>(term_size));
            }
        }
    }

    if (posting_lengths.empty()) {
        return make_skip_json("no non-empty posting lists");
    }

    std::sort(posting_lengths.begin(), posting_lengths.end());
    JsonType posting_stats = make_distribution_stats_from_sorted(posting_lengths);
    float threshold = calculate_percentile(posting_lengths, 0.99F);
    auto long_tail_count = std::count_if(posting_lengths.begin(),
                                         posting_lengths.end(),
                                         [&](float value) { return value > threshold; });
    posting_stats["long_tail_threshold"].SetFloat(threshold);
    posting_stats["long_tail_mean"].SetFloat(static_cast<float>(long_tail_count) /
                                             static_cast<float>(posting_lengths.size()));
    return posting_stats;
}

JsonType
SINDIAnalyzer::get_quantization_range_stats() const {
    if (not is_sq8_value_quantization(sindi_->sparse_value_quant_type_)) {
        return make_skip_json("quantization disabled");
    }
    JsonType stats;
    stats["min_val"].SetFloat(sindi_->quantization_params_->min_val);
    stats["max_val"].SetFloat(sindi_->quantization_params_->max_val);
    stats["diff"].SetFloat(sindi_->quantization_params_->diff);
    return stats;
}

JsonType
SINDIAnalyzer::get_mean_doc_retained(const DatasetPtr& base_dataset) const {
    if (not sindi_->use_reorder_ &&
        (base_dataset == nullptr || base_dataset->GetSparseVectors() == nullptr)) {
        return make_skip_json(
            "requires original base vectors; provide --base_path or rebuild with use_reorder=true");
    }

    std::vector<int64_t> sample_ids(static_cast<size_t>(sindi_->cur_element_count_));
    std::iota(sample_ids.begin(), sample_ids.end(), 0);
    std::mt19937 rng(47);
    std::shuffle(sample_ids.begin(), sample_ids.end(), rng);

    auto retained_sample_count = std::min<uint64_t>(sample_ids.size(), K_ANALYZE_DOC_SAMPLE_LIMIT);
    float retained_ratio_sum = 0.0F;
    uint64_t valid_count = 0;
    for (uint64_t idx = 0; idx < retained_sample_count; ++idx) {
        SparseVector original{};
        SparseVector pruned{};
        if (not get_original_sparse_vector_by_inner_id(
                sample_ids[idx], &original, sindi_->allocator_, base_dataset)) {
            continue;
        }
        get_pruned_sparse_vector_by_inner_id(sample_ids[idx], &pruned, sindi_->allocator_);
        if (original.len_ > 0) {
            retained_ratio_sum +=
                static_cast<float>(pruned.len_) / static_cast<float>(original.len_);
            ++valid_count;
        }
        release_sparse_vector(sindi_->allocator_, &original);
        release_sparse_vector(sindi_->allocator_, &pruned);
    }

    JsonType stats;
    stats["mean"].SetFloat(valid_count == 0 ? 0.0F
                                            : retained_ratio_sum / static_cast<float>(valid_count));
    return stats;
}

JsonType
SINDIAnalyzer::calculate_distance_quality_stats(const DatasetPtr& query_dataset,
                                                const SINDISearchParameter& search_param,
                                                const DatasetPtr& base_dataset,
                                                int64_t topk) const {
    JsonType stats;
    if (query_dataset == nullptr || query_dataset->GetSparseVectors() == nullptr || topk <= 0) {
        auto skip = make_skip_json("query dataset unavailable");
        stats["doc_prune_bias_mean"].SetJson(skip);
        stats["doc_prune_inversion_count_rate"].SetJson(skip);
        if (is_sq8_value_quantization(sindi_->sparse_value_quant_type_)) {
            stats["quantization_bias_ratio"].SetJson(skip);
            stats["quantization_inversion_count_rate"].SetJson(skip);
        }
        return stats;
    }
    if (not sindi_->use_reorder_ &&
        (base_dataset == nullptr || base_dataset->GetSparseVectors() == nullptr)) {
        auto skip = make_skip_json(
            "requires original base vectors; provide --base_path or rebuild with use_reorder=true");
        stats["doc_prune_bias_mean"].SetJson(skip);
        stats["doc_prune_inversion_count_rate"].SetJson(skip);
        if (is_sq8_value_quantization(sindi_->sparse_value_quant_type_)) {
            stats["quantization_bias_ratio"].SetJson(skip);
            stats["quantization_inversion_count_rate"].SetJson(skip);
        }
        return stats;
    }

    float doc_bias_sum = 0.0F;
    float doc_inversion_rate_sum = 0.0F;
    float quant_bias_sum = 0.0F;
    float quant_inversion_rate_sum = 0.0F;
    auto query_count = query_dataset->GetNumElements();
    auto sampled_query_count = std::min<int64_t>(
        query_count, std::max<int64_t>(1, static_cast<int64_t>(base_sample_size_)));
    SINDISearchParameter no_query_prune_param = search_param;
    no_query_prune_param.query_prune_ratio = 0.0F;
    auto candidate_count = resolve_candidate_count(no_query_prune_param, topk);
    int64_t valid_doc_query_count = 0;
    int64_t valid_quant_query_count = 0;

    for (int64_t query_idx = 0; query_idx < sampled_query_count; ++query_idx) {
        const auto& query = query_dataset->GetSparseVectors()[query_idx];
        auto doc_candidates =
            collect_doc_prune_candidates(query, no_query_prune_param, candidate_count);
        auto doc_count = static_cast<int64_t>(doc_candidates.size());
        if (doc_count > 0) {
            ++valid_doc_query_count;
            std::vector<float> exact_distances(static_cast<size_t>(doc_count));
            std::vector<float> pruned_distances(static_cast<size_t>(doc_count));
            float query_doc_bias = 0.0F;
            for (int64_t rank = 0; rank < doc_count; ++rank) {
                exact_distances[rank] = calculate_exact_distance_by_label(
                    query, doc_candidates[rank].second, base_dataset);
                pruned_distances[rank] =
                    calculate_pruned_distance_by_label(query, doc_candidates[rank].second);
                query_doc_bias += std::abs(pruned_distances[rank] - exact_distances[rank]) /
                                  std::max(std::abs(exact_distances[rank]), K_ANALYZE_EPSILON);
            }
            doc_bias_sum += query_doc_bias / static_cast<float>(doc_count);

            int64_t doc_pair_count = 0;
            int64_t doc_inversion_count = 0;
            for (int64_t left = 0; left < doc_count; ++left) {
                for (int64_t right = left + 1; right < doc_count; ++right) {
                    auto exact_diff = exact_distances[left] - exact_distances[right];
                    auto pruned_diff = pruned_distances[left] - pruned_distances[right];
                    if (std::abs(exact_diff) > K_ANALYZE_EPSILON &&
                        std::abs(pruned_diff) > K_ANALYZE_EPSILON) {
                        ++doc_pair_count;
                        if ((exact_diff < 0.0F) != (pruned_diff < 0.0F)) {
                            ++doc_inversion_count;
                        }
                    }
                }
            }
            doc_inversion_rate_sum +=
                doc_pair_count == 0
                    ? 0.0F
                    : static_cast<float>(doc_inversion_count) / static_cast<float>(doc_pair_count);
        }

        if (not is_sq8_value_quantization(sindi_->sparse_value_quant_type_)) {
            continue;
        }
        auto quant_candidates =
            collect_coarse_candidates(query, no_query_prune_param, candidate_count);
        auto quant_count = static_cast<int64_t>(quant_candidates.size());
        if (quant_count <= 0) {
            continue;
        }
        ++valid_quant_query_count;
        std::vector<float> pruned_distances(static_cast<size_t>(quant_count));
        std::vector<float> quantized_distances(static_cast<size_t>(quant_count));
        float query_quant_bias = 0.0F;
        for (int64_t rank = 0; rank < quant_count; ++rank) {
            pruned_distances[rank] =
                calculate_pruned_distance_by_label(query, quant_candidates[rank].second);
            quantized_distances[rank] = quant_candidates[rank].first;
            query_quant_bias += std::abs(quantized_distances[rank] - pruned_distances[rank]) /
                                std::max(std::abs(pruned_distances[rank]), K_ANALYZE_EPSILON);
        }
        quant_bias_sum += query_quant_bias / static_cast<float>(quant_count);

        int64_t quant_pair_count = 0;
        int64_t quant_inversion_count = 0;
        for (int64_t left = 0; left < quant_count; ++left) {
            for (int64_t right = left + 1; right < quant_count; ++right) {
                auto pruned_diff = pruned_distances[left] - pruned_distances[right];
                auto quant_diff = quantized_distances[left] - quantized_distances[right];
                if (std::abs(pruned_diff) > K_ANALYZE_EPSILON &&
                    std::abs(quant_diff) > K_ANALYZE_EPSILON) {
                    ++quant_pair_count;
                    if ((pruned_diff < 0.0F) != (quant_diff < 0.0F)) {
                        ++quant_inversion_count;
                    }
                }
            }
        }
        quant_inversion_rate_sum +=
            quant_pair_count == 0
                ? 0.0F
                : static_cast<float>(quant_inversion_count) / static_cast<float>(quant_pair_count);
    }

    auto doc_denominator = static_cast<float>(std::max<int64_t>(1, valid_doc_query_count));
    stats["doc_prune_bias_mean"].SetFloat(doc_bias_sum / doc_denominator);
    stats["doc_prune_inversion_count_rate"].SetFloat(doc_inversion_rate_sum / doc_denominator);
    if (is_sq8_value_quantization(sindi_->sparse_value_quant_type_)) {
        auto quant_denominator = static_cast<float>(std::max<int64_t>(1, valid_quant_query_count));
        stats["quantization_bias_ratio"].SetFloat(quant_bias_sum / quant_denominator);
        stats["quantization_inversion_count_rate"].SetFloat(quant_inversion_rate_sum /
                                                            quant_denominator);
    }
    return stats;
}

JsonType
SINDIAnalyzer::GetStats() {
    auto search_json = parse_sindi_search_json(search_params_);
    auto analyze_options = parse_analyze_options(search_json);
    DatasetPtr base_dataset = nullptr;
    if (not analyze_options.base_path.empty()) {
        base_dataset = load_sparse_dataset(analyze_options.base_path);
        check_base_dataset_for_analyze(base_dataset, sindi_->cur_element_count_);
    }

    JsonType stats;
    stats["total_count"].SetUint64(static_cast<uint64_t>(sindi_->cur_element_count_));
    const auto window_count = sindi_->immutable_data_ == nullptr
                                  ? sindi_->window_term_list_.size()
                                  : sindi_->immutable_data_->windows.size();
    stats["window_count"].SetUint64(static_cast<uint64_t>(window_count));
    stats["active_term_count"].SetJson(get_active_term_count_stats());
    stats["posting_length_distribution"].SetJson(get_posting_length_distribution_stats());
    if (is_sq8_value_quantization(sindi_->sparse_value_quant_type_)) {
        stats["quantization_range"].SetJson(get_quantization_range_stats());
    }
    stats["mean_doc_retained"].SetJson(get_mean_doc_retained(base_dataset));

    auto default_search_json = parse_sindi_search_json("default");
    default_search_json[INDEX_SINDI][SPARSE_QUERY_PRUNE_RATIO].SetFloat(0.0F);
    default_search_json[INDEX_SINDI][SPARSE_TERM_PRUNE_RATIO].SetFloat(0.0F);
    default_search_json[INDEX_SINDI][SPARSE_N_CANDIDATE].SetInt(500);
    auto default_search_param = default_search_json.Dump();

    auto base_search_stats = get_base_search_stats(default_search_param, base_dataset);
    for (const auto& key : {"recall_base",
                            "doc_prune_recall",
                            "quantization_recall",
                            "doc_prune_bias_mean",
                            "doc_prune_inversion_count_rate",
                            "quantization_bias_ratio",
                            "quantization_inversion_count_rate"}) {
        if (base_search_stats.Contains(key)) {
            stats[key].SetJson(base_search_stats[key]);
        }
    }

    return stats;
}

JsonType
SINDIAnalyzer::AnalyzeIndexBySearch(const SearchRequest& request) {
    JsonType stats;
    auto search_json = parse_sindi_search_json(request.params_str_);
    auto analyze_options = parse_analyze_options(search_json);
    SINDISearchParameter search_param;
    search_param.FromJson(search_json);

    auto query_dataset = request.query_;
    CHECK_ARGUMENT(query_dataset != nullptr, "query dataset is null");
    CHECK_ARGUMENT(query_dataset->GetSparseVectors() != nullptr,
                   "SINDI analyze requires sparse query dataset");

    auto effective_topk = std::min<int64_t>(request.topk_, sindi_->cur_element_count_);
    if (effective_topk <= 0) {
        return stats;
    }

    DatasetPtr base_dataset = nullptr;
    if (not analyze_options.base_path.empty()) {
        base_dataset = load_sparse_dataset(analyze_options.base_path);
        check_base_dataset_for_analyze(base_dataset, sindi_->cur_element_count_);
    }

    DatasetPtr ground_truth = nullptr;
    bool has_explicit_groundtruth = not analyze_options.groundtruth_path.empty();
    std::string ground_truth_skip_reason =
        "ground truth unavailable; provide --groundtruth_path or --base_path";
    if (has_explicit_groundtruth) {
        ground_truth = load_ground_truth(analyze_options.groundtruth_path);
    }
    if (ground_truth == nullptr && not has_explicit_groundtruth) {
        ground_truth = calculate_ground_truth(query_dataset, effective_topk, base_dataset);
        if (ground_truth == nullptr && not sindi_->use_reorder_ &&
            (base_dataset == nullptr || base_dataset->GetSparseVectors() == nullptr)) {
            ground_truth_skip_reason =
                "ground truth auto-generation skipped because original base vectors are "
                "unavailable; provide --base_path or --groundtruth_path or rebuild with "
                "use_reorder=true";
        }
        if (ground_truth != nullptr) {
            save_ground_truth(analyze_options.save_groundtruth_path, ground_truth);
        }
    }

    bool has_ground_truth = ground_truth != nullptr &&
                            ground_truth->GetNumElements() == query_dataset->GetNumElements() &&
                            ground_truth->GetDim() >= effective_topk;
    if (not has_ground_truth) {
        ground_truth = nullptr;
    }

    float coarse_recall_sum = 0.0F;
    float rerank_recall_sum = 0.0F;
    std::vector<float> last_topk_ranks;

    if (has_ground_truth) {
        auto recall_stats =
            calculate_recall_stats(query_dataset, ground_truth, search_param, effective_topk);
        for (const auto& key :
             {"recall_query", "doc_prune_recall", "quantization_recall", "mean_latency_ms"}) {
            if (recall_stats.Contains(key)) {
                stats[key].SetJson(recall_stats[key]);
            }
        }
        stats["time_cost_query"].SetJson(recall_stats["mean_latency_ms"]);

        auto quality_stats = calculate_distance_quality_stats(
            query_dataset, search_param, base_dataset, effective_topk);
        for (const auto& key : {"doc_prune_bias_mean",
                                "doc_prune_inversion_count_rate",
                                "quantization_bias_ratio",
                                "quantization_inversion_count_rate"}) {
            if (quality_stats.Contains(key)) {
                stats[key].SetJson(quality_stats[key]);
            }
        }

        for (int64_t query_idx = 0; query_idx < query_dataset->GetNumElements(); ++query_idx) {
            const auto& query = query_dataset->GetSparseVectors()[query_idx];
            auto candidate_count = resolve_candidate_count(search_param, effective_topk);
            auto coarse_candidates =
                collect_coarse_candidates(query, search_param, candidate_count);
            if (not coarse_candidates.empty()) {
                coarse_recall_sum += calculate_candidate_recall(
                    coarse_candidates,
                    ground_truth->GetIds() + query_idx * ground_truth->GetDim(),
                    effective_topk,
                    effective_topk);
                auto reranked = rerank_candidates(query, coarse_candidates, effective_topk);
                std::vector<int64_t> reranked_ids;
                reranked_ids.reserve(reranked.size());
                std::unordered_map<int64_t, int64_t> coarse_rank;
                for (size_t rank = 0; rank < coarse_candidates.size(); ++rank) {
                    coarse_rank[coarse_candidates[rank].second] = static_cast<int64_t>(rank) + 1;
                }
                int64_t last_rank = 0;
                for (const auto& item : reranked) {
                    reranked_ids.push_back(item.second);
                    last_rank = std::max(last_rank, coarse_rank[item.second]);
                }
                rerank_recall_sum +=
                    calculate_recall(reranked_ids.data(),
                                     static_cast<int64_t>(reranked_ids.size()),
                                     ground_truth->GetIds() + query_idx * ground_truth->GetDim(),
                                     effective_topk);
                last_topk_ranks.push_back(static_cast<float>(last_rank));
            }
        }
    } else {
        auto skip = make_skip_json(ground_truth_skip_reason);
        stats["recall_query"].SetJson(skip);
        stats["mean_latency_ms"].SetJson(skip);
        stats["time_cost_query"].SetJson(skip);
        stats["doc_prune_recall"].SetJson(skip);
        stats["quantization_recall"].SetJson(skip);
        stats["doc_prune_bias_mean"].SetJson(skip);
        stats["doc_prune_inversion_count_rate"].SetJson(skip);
        if (is_sq8_value_quantization(sindi_->sparse_value_quant_type_)) {
            stats["quantization_bias_ratio"].SetJson(skip);
            stats["quantization_inversion_count_rate"].SetJson(skip);
        }
    }

    stats["postings_scanned"].SetJson(
        calculate_postings_scanned_stats(query_dataset, search_param));

    if (sindi_->use_reorder_ && has_ground_truth && not last_topk_ranks.empty()) {
        JsonType reorder_recall;
        reorder_recall["before_reorder_recall_k_at_k"].SetFloat(
            coarse_recall_sum / static_cast<float>(last_topk_ranks.size()));
        reorder_recall["after_reorder_recall_k_at_k"].SetFloat(
            rerank_recall_sum / static_cast<float>(last_topk_ranks.size()));
        stats["reorder_recall"].SetJson(reorder_recall);
        stats["last_topk_rank_in_heap"].SetJson(make_distribution_stats(last_topk_ranks));
    } else if (sindi_->use_reorder_) {
        stats["reorder_recall"].SetJson(
            make_skip_json("ground truth unavailable for reorder analysis"));
        stats["last_topk_rank_in_heap"].SetJson(
            make_skip_json("ground truth unavailable for reorder analysis"));
    }

    return stats;
}

}  // namespace vsag
