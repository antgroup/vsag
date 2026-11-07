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

#include "neighbor_search.h"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <unordered_map>

#include "hdf5_source.h"
#include "vector_math.h"

namespace vsag::dataset_profile {

namespace {

constexpr uint64_t QUERY_GROUP = 8;
constexpr uint64_t ROW_TILE = 128;

bool
heap_order(const Neighbor& a, const Neighbor& b) {
    return neighbor_less(a, b);
}

}  // namespace

void
TopK::Push(double distance, uint64_t index) {
    if (k_ == 0) {
        return;
    }
    const Neighbor candidate{distance, index};
    if (heap_.size() < k_) {
        heap_.push_back(candidate);
        std::push_heap(heap_.begin(), heap_.end(), heap_order);
        return;
    }
    if (neighbor_less(candidate, heap_.front())) {
        std::pop_heap(heap_.begin(), heap_.end(), heap_order);
        heap_.back() = candidate;
        std::push_heap(heap_.begin(), heap_.end(), heap_order);
    }
}

void
TopK::Merge(const TopK& other) {
    for (const auto& neighbor : other.heap_) {
        Push(neighbor.distance, neighbor.index);
    }
}

std::vector<Neighbor>
TopK::Sorted() const {
    std::vector<Neighbor> sorted = heap_;
    std::sort(sorted.begin(), sorted.end(), neighbor_less);
    return sorted;
}

DenseKnnSearch::DenseKnnSearch(const DenseQuerySet& queries, Metric metric, uint64_t k)
    : queries_(queries),
      metric_(metric),
      k_(k),
      metric_heaps_(queries.Size(), TopK(k)),
      euclidean_heaps_(metric == Metric::IP ? queries.Size() : 0, TopK(k)) {
}

void
DenseKnnSearch::Scan(const float* rows,
                     const double* norms,
                     const uint8_t* valid,
                     uint64_t first_row,
                     uint64_t count) {
    const DenseQuerySet& queries = queries_;
    const Metric metric = metric_;
    std::vector<TopK>& metric_heaps = metric_heaps_;
    std::vector<TopK>& euclidean_heaps = euclidean_heaps_;
    const uint64_t query_count = queries.Size();
    const uint64_t dim = queries.dim;
    const auto groups = static_cast<int64_t>((query_count + QUERY_GROUP - 1) / QUERY_GROUP);
    // Each query belongs to one group, so every heap is updated by exactly one thread.
    // clang-format off
#pragma omp parallel for schedule(dynamic, 1) default(none) \
    shared(queries, metric, metric_heaps, euclidean_heaps, rows, norms, valid, first_row, count, \
           query_count, dim, groups)
    // clang-format on
    for (int64_t group = 0; group < groups; ++group) {
        const uint64_t query_begin = static_cast<uint64_t>(group) * QUERY_GROUP;
        const uint64_t query_end = std::min(query_count, query_begin + QUERY_GROUP);
        for (uint64_t tile = 0; tile < count; tile += ROW_TILE) {
            const uint64_t tile_end = std::min(count, tile + ROW_TILE);
            for (uint64_t q = query_begin; q < query_end; ++q) {
                const float* query = queries.vectors.data() + q * dim;
                const uint64_t self = queries.self_rows[q];
                for (uint64_t r = tile; r < tile_end; ++r) {
                    const uint64_t row = first_row + r;
                    if (valid[r] == 0 || row == self) {
                        continue;
                    }
                    const float* candidate = rows + r * dim;
                    if (metric == Metric::L2) {
                        metric_heaps[q].Push(l2_sqr(query, candidate, dim), row);
                    } else if (metric == Metric::IP) {
                        double ip = 0.0;
                        double l2 = 0.0;
                        inner_product_and_l2_sqr(query, candidate, dim, ip, l2);
                        metric_heaps[q].Push(1.0 - ip, row);
                        euclidean_heaps[q].Push(l2, row);
                    } else {
                        const double denominator = queries.norms[q] * norms[r];
                        const double ip = inner_product(query, candidate, dim);
                        metric_heaps[q].Push(1.0 - ip / denominator, row);
                    }
                }
            }
        }
    }
}

std::vector<std::vector<Neighbor>>
DenseKnnSearch::MetricNeighbors() const {
    std::vector<std::vector<Neighbor>> result(metric_heaps_.size());
    for (uint64_t q = 0; q < metric_heaps_.size(); ++q) {
        result[q] = metric_heaps_[q].Sorted();
    }
    return result;
}

std::vector<std::vector<double>>
DenseKnnSearch::GeometricDistances() const {
    std::vector<std::vector<double>> result(metric_heaps_.size());
    for (uint64_t q = 0; q < metric_heaps_.size(); ++q) {
        if (metric_ == Metric::IP) {
            for (const auto& neighbor : euclidean_heaps_[q].Sorted()) {
                result[q].push_back(std::sqrt(std::max(0.0, neighbor.distance)));
            }
        } else {
            for (const auto& neighbor : metric_heaps_[q].Sorted()) {
                double geometric = 0.0;
                geometric_from_metric(metric_, neighbor.distance, geometric);
                result[q].push_back(geometric);
            }
        }
    }
    return result;
}

double
dense_metric_distance(
    const DenseQuerySet& queries, uint64_t q, const float* row, double row_norm, Metric metric) {
    const float* query = queries.vectors.data() + q * queries.dim;
    if (metric == Metric::L2) {
        return l2_sqr(query, row, queries.dim);
    }
    if (metric == Metric::IP) {
        double ip = 0.0;
        double l2 = 0.0;
        inner_product_and_l2_sqr(query, row, queries.dim, ip, l2);
        return 1.0 - ip;
    }
    const double denominator = queries.norms[q] * row_norm;
    const double ip = inner_product(query, row, queries.dim);
    return 1.0 - ip / denominator;
}

SparseKnnSearch::SparseKnnSearch(const SparseQuerySet& queries, Metric metric, uint64_t k)
    : queries_(queries), metric_(metric), k_(k) {
    std::unordered_map<uint32_t, uint32_t> slots;
    for (uint64_t q = 0; q < queries.Size(); ++q) {
        for (uint64_t e = queries.offsets[q]; e < queries.offsets[q + 1]; ++e) {
            auto [iterator, inserted] =
                slots.emplace(queries.ids[e], static_cast<uint32_t>(postings_.size()));
            if (inserted) {
                postings_.emplace_back();
                term_ids_.push_back(queries.ids[e]);
            }
            postings_[iterator->second].emplace_back(static_cast<uint32_t>(q), queries.values[e]);
        }
    }
    // Sort the term dictionary so lookups are a binary search with no hashing on the hot path.
    std::vector<uint32_t> order(term_ids_.size());
    for (uint32_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [this](uint32_t a, uint32_t b) {
        return term_ids_[a] < term_ids_[b];
    });
    std::vector<uint32_t> sorted_terms(order.size());
    std::vector<std::vector<std::pair<uint32_t, float>>> sorted_postings(order.size());
    for (uint64_t i = 0; i < order.size(); ++i) {
        sorted_terms[i] = term_ids_[order[i]];
        sorted_postings[i] = std::move(postings_[order[i]]);
    }
    term_ids_ = std::move(sorted_terms);
    postings_ = std::move(sorted_postings);
    const auto threads = static_cast<uint64_t>(std::max(1, omp_get_max_threads()));
    thread_heaps_.assign(threads, std::vector<TopK>(queries.Size(), TopK(k)));
    overlap_counts_.assign(queries.Size() * threads, 0);
    disjoint_rows_.assign(queries.Size(), {});
    if (k > 0) {
        for (uint64_t q = 0; q < queries.Size(); ++q) {
            open_queries_.push_back(static_cast<uint32_t>(q));
        }
    }
    thread_disjoint_.assign(threads, {});
}

void
SparseKnnSearch::Scan(const SparseBatch& batch,
                      const std::vector<double>& row_norms,
                      const std::vector<uint8_t>& valid) {
    const SparseQuerySet& queries = queries_;
    const Metric metric = metric_;
    const uint64_t k = k_;
    const std::vector<uint32_t>& term_ids = term_ids_;
    const std::vector<std::vector<std::pair<uint32_t, float>>>& postings = postings_;
    const std::vector<uint32_t>& open_queries = open_queries_;
    std::vector<std::vector<TopK>>& thread_heaps = thread_heaps_;
    std::vector<uint64_t>& overlap_counts = overlap_counts_;
    std::vector<std::vector<std::vector<uint64_t>>>& thread_disjoint = thread_disjoint_;
    const uint64_t query_count = queries.Size();
    const auto rows = static_cast<int64_t>(batch.Rows());
    const auto threads = static_cast<int>(thread_heaps.size());
    for (auto& lists : thread_disjoint) {
        lists.assign(open_queries.size(), {});
    }
    // Heaps, counters and disjoint-row lists are per thread; results are merged in a total
    // order afterwards.
    // clang-format off
#pragma omp parallel num_threads(threads) default(none) \
    shared(queries, metric, k, term_ids, postings, open_queries, thread_heaps, overlap_counts, \
           thread_disjoint, query_count, rows, batch, row_norms, valid)
    // clang-format on
    {
        const auto thread = static_cast<uint64_t>(omp_get_thread_num());
        std::vector<double> scores(query_count, 0.0);
        std::vector<uint8_t> touched_flags(query_count, 0);
        std::vector<uint32_t> touched;
        std::vector<std::vector<uint64_t>>& disjoint = thread_disjoint[thread];
#pragma omp for schedule(static)
        for (int64_t r = 0; r < rows; ++r) {
            if (valid[r] == 0) {
                continue;
            }
            const uint64_t row = batch.first_row + static_cast<uint64_t>(r);
            touched.clear();
            const uint64_t begin = batch.offsets[r];
            const uint64_t end = batch.offsets[r + 1];
            auto term_cursor = term_ids.begin();
            for (uint64_t e = begin; e < end; ++e) {
                term_cursor = std::lower_bound(term_cursor, term_ids.end(), batch.ids[e]);
                if (term_cursor == term_ids.end()) {
                    break;
                }
                if (*term_cursor != batch.ids[e]) {
                    continue;
                }
                const auto slot = static_cast<uint64_t>(term_cursor - term_ids.begin());
                const double row_value = batch.values[e];
                for (const auto& [query, query_value] : postings[slot]) {
                    if (touched_flags[query] == 0) {
                        touched_flags[query] = 1;
                        touched.push_back(query);
                    }
                    scores[query] += static_cast<double>(query_value) * row_value;
                }
            }
            // This row shares no term with the untouched open queries: distance exactly 1.
            // Each list is a bounded max-heap, so it keeps the k smallest rows in any order.
            for (uint64_t i = 0; i < open_queries.size(); ++i) {
                const uint32_t query = open_queries[i];
                if (touched_flags[query] != 0 || queries.self_rows[query] == row) {
                    continue;
                }
                std::vector<uint64_t>& heap = disjoint[i];
                if (heap.size() < k) {
                    heap.push_back(row);
                    std::push_heap(heap.begin(), heap.end());
                } else if (row < heap.front()) {
                    std::pop_heap(heap.begin(), heap.end());
                    heap.back() = row;
                    std::push_heap(heap.begin(), heap.end());
                }
            }
            for (const uint32_t query : touched) {
                if (queries.self_rows[query] != row) {
                    double distance = 1.0 - scores[query];
                    if (metric == Metric::COSINE) {
                        const double denominator = queries.norms[query] * row_norms[r];
                        distance = denominator > 0.0 ? 1.0 - scores[query] / denominator : 1.0;
                    }
                    thread_heaps[thread][query].Push(distance, row);
                    ++overlap_counts[thread * query_count + query];
                }
                scores[query] = 0.0;
                touched_flags[query] = 0;
            }
        }
    }
    merge_disjoint_rows();
}

void
SparseKnnSearch::merge_disjoint_rows() {
    std::vector<uint32_t> still_open;
    for (uint64_t i = 0; i < open_queries_.size(); ++i) {
        const uint32_t query = open_queries_[i];
        std::vector<uint64_t>& rows = disjoint_rows_[query];
        for (const auto& lists : thread_disjoint_) {
            rows.insert(rows.end(), lists[i].begin(), lists[i].end());
        }
        std::sort(rows.begin(), rows.end());
        rows.resize(std::min<uint64_t>(rows.size(), k_));
        if (rows.size() < k_) {
            still_open.push_back(query);
        }
    }
    open_queries_ = std::move(still_open);
}

std::vector<std::vector<Neighbor>>
SparseKnnSearch::MetricNeighbors(std::vector<std::vector<uint8_t>>* shares_term) const {
    std::vector<std::vector<Neighbor>> result(queries_.Size());
    if (shares_term != nullptr) {
        shares_term->assign(queries_.Size(), {});
    }
    for (uint64_t q = 0; q < queries_.Size(); ++q) {
        TopK merged(k_);
        for (const auto& heaps : thread_heaps_) {
            merged.Merge(heaps[q]);
        }
        for (const uint64_t row : disjoint_rows_[q]) {
            merged.Push(1.0, row);
        }
        result[q] = merged.Sorted();
        while (result[q].size() < k_) {
            result[q].push_back(Neighbor{1.0, NO_ROW});
        }
        std::sort(result[q].begin(), result[q].end(), neighbor_less);
        if (shares_term != nullptr) {
            const std::vector<uint64_t>& disjoint = disjoint_rows_[q];
            for (const auto& neighbor : result[q]) {
                const bool shared =
                    neighbor.index != NO_ROW &&
                    !std::binary_search(disjoint.begin(), disjoint.end(), neighbor.index);
                (*shares_term)[q].push_back(shared ? 1 : 0);
            }
        }
    }
    return result;
}

uint64_t
SparseKnnSearch::QueriesWithFewOverlaps() const {
    const uint64_t query_count = queries_.Size();
    uint64_t few = 0;
    for (uint64_t q = 0; q < query_count; ++q) {
        uint64_t total = 0;
        for (uint64_t t = 0; t < thread_heaps_.size(); ++t) {
            total += overlap_counts_[t * query_count + q];
        }
        if (total < k_) {
            ++few;
        }
    }
    return few;
}

double
lid_mle(const std::vector<double>& ascending_distances) {
    std::vector<double> positive;
    positive.reserve(ascending_distances.size());
    for (const double distance : ascending_distances) {
        if (distance > 0.0 && std::isfinite(distance)) {
            positive.push_back(distance);
        }
    }
    if (positive.size() < 2) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double farthest = positive.back();
    double sum = 0.0;
    for (uint64_t j = 0; j + 1 < positive.size(); ++j) {
        sum += std::log(farthest / positive[j]);
    }
    if (sum <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return static_cast<double>(positive.size() - 1) / sum;
}

double
two_nn_dimension(const std::vector<double>& ratios) {
    double sum = 0.0;
    uint64_t count = 0;
    for (const double ratio : ratios) {
        if (ratio > 1.0 && std::isfinite(ratio)) {
            sum += std::log(ratio);
            ++count;
        }
    }
    if (count < 2 || sum <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return static_cast<double>(count) / sum;
}

double
sparse_dot(const uint32_t* ids_a,
           const float* values_a,
           uint64_t length_a,
           const uint32_t* ids_b,
           const float* values_b,
           uint64_t length_b) {
    double dot = 0.0;
    uint64_t i = 0;
    uint64_t j = 0;
    while (i < length_a && j < length_b) {
        if (ids_a[i] == ids_b[j]) {
            dot += static_cast<double>(values_a[i]) * static_cast<double>(values_b[j]);
            ++i;
            ++j;
        } else if (ids_a[i] < ids_b[j]) {
            ++i;
        } else {
            ++j;
        }
    }
    return dot;
}

}  // namespace vsag::dataset_profile
