
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

#include "simq.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "index_feature_list.h"
#include "inner_string_params.h"
#include "metric_type.h"
#include "storage/serialization.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "typing.h"
#include "utils/util_functions.h"

namespace vsag {

// ─────────────────────────────────────────────────────────────────────────────
// Internal clustering helper
// ─────────────────────────────────────────────────────────────────────────────

namespace {

struct ClusterMemberEntry {
    InnerIdType vec_id;
    float       distance;
};

class HNSWDynamicClustering {
public:
    HNSWDynamicClustering(float init_cluster_ratio,
                          int64_t max_cluster_size,
                          int64_t split_start_idx,
                          int64_t random_seed,
                          vsag::Allocator* allocator)
        : init_cluster_ratio_(init_cluster_ratio),
          max_cluster_size_(static_cast<int>(max_cluster_size)),
          split_start_idx_(static_cast<int>(split_start_idx)),
          random_seed_(static_cast<int>(random_seed)),
          allocator_(allocator) {}

    ~HNSWDynamicClustering() {
        delete hnsw_;
        delete space_;
    }

    void
    fit(const float* vecs, int64_t num_vecs, int64_t dim);

    std::vector<int>                                         cluster_centers_;
    std::unordered_map<int, std::vector<ClusterMemberEntry>> clusters_;
    std::vector<int>                                         vec_to_cluster_;

private:
    void
    build_hnsw(const std::vector<int>& center_ids, int64_t dim);

    int
    find_nearest_cluster(int vec_id) const;

    float
    ip_distance(int v1, int v2) const;

    void
    sorted_insert(std::vector<ClusterMemberEntry>& members, InnerIdType vec_id, float dist);

    void
    split_cluster(int old_center_id, int64_t dim);

    float            init_cluster_ratio_;
    int              max_cluster_size_;
    int              split_start_idx_;
    int              random_seed_;
    vsag::Allocator* allocator_{nullptr};

    const float* vecs_{nullptr};
    int64_t      num_vecs_{0};
    int64_t      dim_{0};

    hnswlib::InnerProductSpace* space_{nullptr};
    hnswlib::HierarchicalNSW*   hnsw_{nullptr};

    std::vector<int> hnsw_id_to_vec_;
    std::vector<int> vec_to_hnsw_id_;
    int              next_hnsw_id_{0};
};

void
HNSWDynamicClustering::build_hnsw(const std::vector<int>& center_ids, int64_t dim) {
    delete hnsw_;
    hnsw_ = nullptr;
    delete space_;
    space_ = nullptr;

    space_ = new hnswlib::InnerProductSpace(static_cast<uint64_t>(dim),
                                            vsag::DataTypes::DATA_TYPE_FLOAT);
    hnsw_ = new hnswlib::HierarchicalNSW(
        space_, static_cast<uint64_t>(center_ids.size()), allocator_, 16, 100);
    hnsw_->init_memory_space();

    hnsw_id_to_vec_.clear();
    vec_to_hnsw_id_.assign(static_cast<uint64_t>(num_vecs_), -1);
    next_hnsw_id_ = 0;

    for (int cid : center_ids) {
        hnsw_->addPoint(vecs_ + cid * dim_, static_cast<hnswlib::LabelType>(next_hnsw_id_));
        hnsw_id_to_vec_.push_back(cid);
        vec_to_hnsw_id_[cid] = next_hnsw_id_;
        ++next_hnsw_id_;
    }
}

int
HNSWDynamicClustering::find_nearest_cluster(int vec_id) const {
    auto result = hnsw_->searchKnn(vecs_ + vec_id * dim_, 1, 50);
    return hnsw_id_to_vec_[static_cast<int>(result.top().second)];
}

float
HNSWDynamicClustering::ip_distance(int v1, int v2) const {
    const float* a = vecs_ + v1 * dim_;
    const float* b = vecs_ + v2 * dim_;
    return space_->get_dist_func()(a, b, space_->get_dist_func_param());
}

void
HNSWDynamicClustering::sorted_insert(std::vector<ClusterMemberEntry>& members,
                                     InnerIdType vec_id,
                                     float dist) {
    auto it = std::lower_bound(members.begin(), members.end(), dist,
                               [](const ClusterMemberEntry& e, float val) {
                                   return e.distance < val;
                               });
    members.insert(it, {vec_id, dist});
}

void
HNSWDynamicClustering::split_cluster(int old_center_id, int64_t /*dim*/) {
    auto& cluster = clusters_[old_center_id];

    int new_center_id = static_cast<int>(cluster.back().vec_id);

    auto split_it = cluster.begin() + (split_start_idx_ - 1);
    std::vector<ClusterMemberEntry> to_move(split_it, cluster.end());
    cluster.erase(split_it, cluster.end());

    std::vector<ClusterMemberEntry> new_cluster;
    new_cluster.push_back({static_cast<InnerIdType>(new_center_id), 0.0f});
    vec_to_cluster_[new_center_id] = new_center_id;

    for (auto& m : to_move) {
        if (static_cast<int>(m.vec_id) == new_center_id) continue;
        float d = ip_distance(static_cast<int>(m.vec_id), new_center_id);
        sorted_insert(new_cluster, m.vec_id, d);
        vec_to_cluster_[m.vec_id] = new_center_id;
    }

    clusters_[new_center_id] = std::move(new_cluster);
    cluster_centers_.push_back(new_center_id);

    if (hnsw_ != nullptr) {
        if (static_cast<uint64_t>(next_hnsw_id_) >= hnsw_->getMaxElements()) {
            hnsw_->resizeIndex(hnsw_->getMaxElements() * 2);
        }
        hnsw_->addPoint(vecs_ + new_center_id * dim_,
                        static_cast<hnswlib::LabelType>(next_hnsw_id_));
        if (new_center_id >= static_cast<int>(vec_to_hnsw_id_.size()))
            vec_to_hnsw_id_.resize(new_center_id + 1, -1);
        hnsw_id_to_vec_.push_back(new_center_id);
        vec_to_hnsw_id_[new_center_id] = next_hnsw_id_;
        ++next_hnsw_id_;
    }
}

void
HNSWDynamicClustering::fit(const float* vecs, int64_t num_vecs, int64_t dim) {
    vecs_     = vecs;
    num_vecs_ = num_vecs;
    dim_      = dim;

    vec_to_cluster_.assign(num_vecs, -1);

    int num_init = std::max(1, static_cast<int>(num_vecs * init_cluster_ratio_));
    std::vector<int> all_indices(num_vecs);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    std::mt19937 rng(random_seed_);
    std::shuffle(all_indices.begin(), all_indices.end(), rng);

    std::vector<int> init_centers(all_indices.begin(), all_indices.begin() + num_init);

    cluster_centers_ = init_centers;
    for (int cid : init_centers) {
        clusters_[cid] = {{static_cast<InnerIdType>(cid), 0.0f}};
        vec_to_cluster_[cid] = cid;
    }

    build_hnsw(init_centers, dim);

    for (auto it = all_indices.begin() + num_init; it != all_indices.end(); ++it) {
        int vid = *it;
        int nearest = find_nearest_cluster(vid);
        float dist  = ip_distance(vid, nearest);

        sorted_insert(clusters_[nearest], static_cast<InnerIdType>(vid), dist);
        vec_to_cluster_[vid] = nearest;

        if (static_cast<int>(clusters_[nearest].size()) > max_cluster_size_) {
            split_cluster(nearest, dim);
        }
    }
}

}  // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// HNSW serialization helpers
// ─────────────────────────────────────────────────────────────────────────────

static std::string
save_hnsw_to_string(hnswlib::HierarchicalNSW* hnsw) {
    std::ostringstream oss(std::ios::out | std::ios::binary);
    IOStreamWriter writer(oss);
    hnsw->saveIndex(writer);
    return oss.str();
}

static hnswlib::HierarchicalNSW*
load_hnsw_from_string(const std::string& bytes,
                      hnswlib::SpaceInterface* space,
                      vsag::Allocator* allocator) {
    std::istringstream iss(bytes, std::ios::in | std::ios::binary);
    IOStreamReader reader(iss);
    auto* hnsw = new hnswlib::HierarchicalNSW(space, 0, allocator);
    hnsw->loadIndex(reader, space);
    return hnsw;
}

// ─────────────────────────────────────────────────────────────────────────────
// Constructor / destructor
// ─────────────────────────────────────────────────────────────────────────────

SIMQ::SIMQ(const SIMQParameterPtr& param, const IndexCommonParam& common_param)
    : InnerIndexInterface(param, common_param),
      cluster_offsets_(allocator_),
      cluster_data_(allocator_),
      vec_to_cluster_(allocator_) {
    mv_codes_ = FlattenInterface::MakeInstance(param->base_codes_param, common_param);
    init_cluster_ratio_ = param->init_cluster_ratio;
    max_cluster_size_   = param->max_cluster_size;
    split_start_idx_    = param->split_start_idx;
    random_seed_        = param->random_seed;
    default_coarse_k_   = param->coarse_k;
    default_rerank_k_   = param->rerank_k;
    this->has_raw_vector_ = true;
}

SIMQ::~SIMQ() {
    delete rep_hnsw_;
    delete rep_space_;
}

// ─────────────────────────────────────────────────────────────────────────────
// Build
// ─────────────────────────────────────────────────────────────────────────────

std::vector<int64_t>
SIMQ::Build(const DatasetPtr& data) {
    std::unique_lock lock(global_mutex_);

    const MultiVector* mvs = data->GetMultiVectors();
    CHECK_ARGUMENT(mvs != nullptr, "simq build: data.multi_vectors is nullptr");

    int64_t mv_dim = data->GetMultiVectorDim();
    CHECK_ARGUMENT(mv_dim == dim_,
                   fmt::format("simq build: multi_vector_dim({}) != index dim({})", mv_dim, dim_));

    int64_t num_docs = data->GetNumElements();
    const int64_t* labels = data->GetIds();
    CHECK_ARGUMENT(labels != nullptr, "simq build: labels (ids) is nullptr");

    // Count total token vectors for clustering
    uint64_t total_vecs = 0;
    for (int64_t i = 0; i < num_docs; ++i) total_vecs += mvs[i].len_;
    CHECK_ARGUMENT(total_vecs > 0, "simq build: total number of vectors must be > 0");

    // Build flat token array for clustering (clustering needs contiguous float*)
    Vector<float> flat(total_vecs * static_cast<uint64_t>(mv_dim), allocator_);
    Vector<InnerIdType> vec_to_doc(total_vecs, allocator_);

    uint64_t vec_off = 0;
    for (int64_t i = 0; i < num_docs; ++i) {
        uint64_t n = static_cast<uint64_t>(mvs[i].len_) * static_cast<uint64_t>(mv_dim);
        if (n > 0) {
            CHECK_ARGUMENT(mvs[i].vectors_ != nullptr,
                           fmt::format("simq build: vectors for doc {} is nullptr", i));
            std::memcpy(flat.data() + vec_off * static_cast<uint64_t>(mv_dim),
                        mvs[i].vectors_,
                        n * sizeof(float));
        }
        for (uint32_t t = 0; t < mvs[i].len_; ++t)
            vec_to_doc[vec_off + t] = static_cast<InnerIdType>(i);
        vec_off += mvs[i].len_;
    }

    total_count_ = static_cast<uint64_t>(num_docs);

    // Store multi-vector documents via MultiVectorDataCell
    mv_codes_->Train(flat.data(), total_vecs);
    mv_codes_->Resize(static_cast<InnerIdType>(num_docs));
    mv_codes_->BatchInsertVector(mvs, static_cast<InnerIdType>(num_docs), nullptr);

    for (int64_t i = 0; i < num_docs; ++i)
        this->label_table_->Insert(static_cast<InnerIdType>(i), labels[i]);

    run_clustering(flat.data(), vec_to_doc, static_cast<int64_t>(total_vecs), mv_dim);
    build_rep_hnsw(flat.data(), mv_dim);

    { Vector<InnerIdType> tmp(allocator_); tmp.swap(vec_to_cluster_); }

    return {};
}

void
SIMQ::run_clustering(const float* flat_vecs,
                     const Vector<InnerIdType>& vec_to_doc,
                     int64_t num_vecs,
                     int64_t dim) {
    HNSWDynamicClustering clustering(
        init_cluster_ratio_, max_cluster_size_, split_start_idx_, random_seed_, allocator_);
    clustering.fit(flat_vecs, num_vecs, dim);

    int64_t nc = static_cast<int64_t>(clustering.cluster_centers_.size());
    num_clusters_ = nc;

    std::vector<int> center_to_idx(num_vecs, -1);
    for (int idx = 0; idx < nc; ++idx)
        center_to_idx[clustering.cluster_centers_[idx]] = idx;

    // Build flat inverted index: cluster i owns [cluster_offsets_[i], cluster_offsets_[i+1])
    cluster_offsets_.resize(static_cast<uint64_t>(nc) + 1);
    uint64_t total_members = 0;
    for (int idx = 0; idx < nc; ++idx) {
        cluster_offsets_[idx] = static_cast<InnerIdType>(total_members);
        int cid = clustering.cluster_centers_[idx];
        total_members += clustering.clusters_.count(cid) ? clustering.clusters_[cid].size() : 0;
    }
    cluster_offsets_[nc] = static_cast<InnerIdType>(total_members);

    // Store doc IDs (not token IDs) in cluster_data_ for direct rerank lookup
    cluster_data_.resize(total_members);
    for (int idx = 0; idx < nc; ++idx) {
        int cid = clustering.cluster_centers_[idx];
        auto it = clustering.clusters_.find(cid);
        if (it == clustering.clusters_.end()) continue;
        uint64_t off = cluster_offsets_[idx];
        for (auto& m : it->second)
            cluster_data_[off++] = vec_to_doc[m.vec_id];
    }

    vec_to_cluster_.resize(static_cast<uint64_t>(num_vecs));
    for (int64_t v = 0; v < num_vecs; ++v) {
        int cid = clustering.vec_to_cluster_[v];
        vec_to_cluster_[v] = static_cast<InnerIdType>(center_to_idx[cid]);
    }
}

void
SIMQ::build_rep_hnsw(const float* flat_vecs, int64_t dim) {
    std::vector<int> representative_vids(num_clusters_, 0);

    std::vector<std::vector<int>> cluster_token_members(num_clusters_);
    for (int64_t v = 0; v < static_cast<int64_t>(vec_to_cluster_.size()); ++v)
        cluster_token_members[vec_to_cluster_[v]].push_back(static_cast<int>(v));

    for (int64_t idx = 0; idx < num_clusters_; ++idx) {
        auto& members = cluster_token_members[idx];
        if (members.empty()) { representative_vids[idx] = 0; continue; }

        std::vector<float> mean(dim, 0.0f);
        for (int vid : members) {
            const float* v = flat_vecs + vid * dim;
            for (int d = 0; d < dim; ++d) mean[d] += v[d];
        }
        float best_dot = -1e30f;
        int   best_vid = members[0];
        for (int vid : members) {
            const float* v = flat_vecs + vid * dim;
            float dot = 0.0f;
            for (int d = 0; d < dim; ++d) dot += v[d] * mean[d];
            if (dot > best_dot) { best_dot = dot; best_vid = vid; }
        }
        representative_vids[idx] = best_vid;
    }

    delete rep_hnsw_;
    rep_hnsw_ = nullptr;
    delete rep_space_;
    rep_space_ = nullptr;
    rep_space_ = new hnswlib::InnerProductSpace(static_cast<uint64_t>(dim),
                                                vsag::DataTypes::DATA_TYPE_FLOAT);
    rep_hnsw_ = new hnswlib::HierarchicalNSW(
        rep_space_, static_cast<uint64_t>(num_clusters_), allocator_, 16, 100);
    rep_hnsw_->init_memory_space();

    for (int64_t idx = 0; idx < num_clusters_; ++idx)
        rep_hnsw_->addPoint(flat_vecs + representative_vids[idx] * dim,
                            static_cast<hnswlib::LabelType>(idx));
}

// ─────────────────────────────────────────────────────────────────────────────
// Add
// ─────────────────────────────────────────────────────────────────────────────

std::vector<int64_t>
SIMQ::Add(const DatasetPtr& /*data*/, AddMode /*mode*/) {
    throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                        "simq: incremental Add is not supported; use Build");
}

// ─────────────────────────────────────────────────────────────────────────────
// Search helpers
// ─────────────────────────────────────────────────────────────────────────────

std::vector<std::pair<InnerIdType, float>>
SIMQ::coarse_search(const float* query_tokens,
                    uint32_t query_token_count,
                    int64_t coarse_k) const {
    // All buffers are local — safe for concurrent searches under shared_lock.
    std::unordered_map<InnerIdType, float> score_map;
    score_map.reserve(static_cast<uint64_t>(coarse_k) * static_cast<uint64_t>(max_cluster_size_));
    std::unordered_set<InnerIdType> seen_this_token;
    seen_this_token.reserve(static_cast<uint64_t>(coarse_k) * static_cast<uint64_t>(max_cluster_size_));

    for (uint32_t ti = 0; ti < query_token_count; ++ti) {
        const float* qt = query_tokens + ti * dim_;

        int64_t actual_coarse_k = std::min(coarse_k, num_clusters_);
        auto result = rep_hnsw_->searchKnn(qt, static_cast<uint64_t>(actual_coarse_k), 50);

        std::vector<std::pair<float, InnerIdType>> cscores;
        cscores.reserve(result.size());
        while (!result.empty()) {
            float cscore = 1.0f - result.top().first;
            auto  cidx   = static_cast<InnerIdType>(result.top().second);
            cscores.push_back({cscore, cidx});
            result.pop();
        }
        std::reverse(cscores.begin(), cscores.end());

        seen_this_token.clear();
        for (auto& [cscore, cidx] : cscores) {
            if (cidx >= static_cast<InnerIdType>(num_clusters_)) continue;
            uint64_t beg = cluster_offsets_[cidx];
            uint64_t end = cluster_offsets_[cidx + 1];
            for (uint64_t j = beg; j < end; ++j) {
                InnerIdType doc_id = cluster_data_[j];
                if (!seen_this_token.insert(doc_id).second) continue;
                score_map[doc_id] += cscore;
            }
        }
    }

    std::vector<std::pair<InnerIdType, float>> ranked(score_map.begin(), score_map.end());
    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    return ranked;
}

// ─────────────────────────────────────────────────────────────────────────────
// KnnSearch
// ─────────────────────────────────────────────────────────────────────────────

DatasetPtr
SIMQ::KnnSearch(const DatasetPtr& query,
                int64_t k,
                const std::string& parameters,
                const FilterPtr& filter) const {
    std::shared_lock lock(global_mutex_);

    if (total_count_ == 0 || rep_hnsw_ == nullptr) {
        return Dataset::Make();
    }

    CHECK_ARGUMENT(query->GetNumElements() > 0, "simq search: query.num_elements must be > 0");
    const MultiVector* query_mvs = query->GetMultiVectors();
    CHECK_ARGUMENT(query_mvs != nullptr, "simq search: query.multi_vectors is nullptr");

    auto sp = SIMQSearchParameters::FromJson(parameters);
    int64_t coarse_k = sp.coarse_k > 0 ? sp.coarse_k : default_coarse_k_;
    int64_t rerank_k = sp.rerank_k > 0 ? sp.rerank_k : default_rerank_k_;
    rerank_k = std::min(rerank_k, static_cast<int64_t>(total_count_));
    k        = std::min(k, static_cast<int64_t>(total_count_));

    auto coarse_results = coarse_search(
        query_mvs[0].vectors_, query_mvs[0].len_, coarse_k);
    if (static_cast<int64_t>(coarse_results.size()) > rerank_k)
        coarse_results.resize(rerank_k);

    // Exact MaxSim rerank via MultiVectorDataCell
    auto computer = mv_codes_->FactoryComputer(&query_mvs[0]);
    std::vector<std::pair<float, InnerIdType>> reranked;
    reranked.reserve(coarse_results.size());
    for (auto& [doc_id, _] : coarse_results) {
        if (filter != nullptr &&
            !filter->CheckValid(this->label_table_->GetLabelById(doc_id)))
            continue;
        float dist = 0.0f;
        mv_codes_->Query(&dist, computer, &doc_id, 1);
        reranked.push_back({dist, doc_id});
    }
    std::sort(reranked.begin(), reranked.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    int64_t result_count = std::min(k, static_cast<int64_t>(reranked.size()));
    auto [result_ds, dists, ids] = create_fast_dataset(result_count, allocator_);
    for (int64_t i = 0; i < result_count; ++i) {
        dists[i] = reranked[i].first;
        ids[i]   = this->label_table_->GetLabelById(reranked[i].second);
    }
    return std::move(result_ds);
}

// ─────────────────────────────────────────────────────────────────────────────
// RangeSearch
// ─────────────────────────────────────────────────────────────────────────────

DatasetPtr
SIMQ::RangeSearch(const DatasetPtr& query,
                  float radius,
                  const std::string& parameters,
                  const FilterPtr& filter,
                  int64_t limited_size) const {
    std::shared_lock lock(global_mutex_);

    if (total_count_ == 0 || rep_hnsw_ == nullptr) {
        return Dataset::Make();
    }

    CHECK_ARGUMENT(query->GetNumElements() > 0, "simq range search: query.num_elements must be > 0");
    const MultiVector* query_mvs = query->GetMultiVectors();
    CHECK_ARGUMENT(query_mvs != nullptr, "simq range search: query.multi_vectors is nullptr");

    auto sp = SIMQSearchParameters::FromJson(parameters);
    int64_t coarse_k = sp.coarse_k > 0 ? sp.coarse_k : default_coarse_k_;
    int64_t rerank_k = sp.rerank_k > 0 ? sp.rerank_k : default_rerank_k_;
    rerank_k = std::min(rerank_k, static_cast<int64_t>(total_count_));

    auto coarse_results = coarse_search(
        query_mvs[0].vectors_, query_mvs[0].len_, coarse_k);
    if (static_cast<int64_t>(coarse_results.size()) > rerank_k)
        coarse_results.resize(rerank_k);

    auto computer = mv_codes_->FactoryComputer(&query_mvs[0]);
    std::vector<std::pair<float, InnerIdType>> in_range;
    for (auto& [doc_id, _] : coarse_results) {
        if (filter != nullptr &&
            !filter->CheckValid(this->label_table_->GetLabelById(doc_id)))
            continue;
        float dist = 0.0f;
        mv_codes_->Query(&dist, computer, &doc_id, 1);
        if (dist <= radius)
            in_range.push_back({dist, doc_id});
    }

    if (limited_size >= 0 && static_cast<int64_t>(in_range.size()) > limited_size) {
        std::nth_element(in_range.begin(),
                         in_range.begin() + limited_size,
                         in_range.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });
        in_range.resize(limited_size);
    }
    std::sort(in_range.begin(), in_range.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    auto [result_ds, dists, ids] =
        create_fast_dataset(static_cast<int64_t>(in_range.size()), allocator_);
    for (uint64_t i = 0; i < in_range.size(); ++i) {
        dists[i] = in_range[i].first;
        ids[i]   = this->label_table_->GetLabelById(in_range[i].second);
    }
    return std::move(result_ds);
}

// ─────────────────────────────────────────────────────────────────────────────
// Serialize / Deserialize
// ─────────────────────────────────────────────────────────────────────────────

void
SIMQ::serialize_rep_hnsw(StreamWriter& writer) const {
    std::string bytes = save_hnsw_to_string(rep_hnsw_);
    auto sz = static_cast<uint64_t>(bytes.size());
    StreamWriter::WriteObj(writer, sz);
    writer.Write(bytes.data(), sz);
}

void
SIMQ::deserialize_rep_hnsw(StreamReader& reader) {
    uint64_t sz{0};
    StreamReader::ReadObj(reader, sz);
    std::string bytes(sz, '\0');
    reader.Read(bytes.data(), sz);

    delete rep_hnsw_;
    rep_hnsw_ = nullptr;
    delete rep_space_;
    rep_space_ = nullptr;
    rep_space_ = new hnswlib::InnerProductSpace(static_cast<uint64_t>(dim_),
                                                vsag::DataTypes::DATA_TYPE_FLOAT);
    rep_hnsw_ = load_hnsw_from_string(bytes, rep_space_, allocator_);
}

void
SIMQ::Serialize(StreamWriter& writer) const {
    std::shared_lock lock(global_mutex_);
    if (rep_hnsw_ == nullptr) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "simq: cannot serialize an unbuilt index");
    }
    StreamWriter::WriteObj(writer, total_count_);
    StreamWriter::WriteObj(writer, num_clusters_);

    StreamWriter::WriteVector(writer, cluster_offsets_);
    StreamWriter::WriteVector(writer, cluster_data_);

    serialize_rep_hnsw(writer);

    mv_codes_->Serialize(writer);
    this->label_table_->Serialize(writer);

    JsonType info;
    info["dim"].SetInt(dim_);
    info["total_count"].SetInt(total_count_);
    info[INDEX_PARAM].SetString(this->create_param_ptr_->ToString());
    write_index_footer(writer, info);
}

void
SIMQ::Deserialize(StreamReader& reader) {
    std::unique_lock lock(global_mutex_);

    JsonType info;
    if (!read_index_footer(reader, info)) {
        throw VsagException(ErrorType::READ_ERROR, "simq: failed to read index footer");
    }

    BufferStreamReader buf_reader(&reader, std::numeric_limits<uint64_t>::max(), allocator_);

    dim_ = info["dim"].GetInt();

    StreamReader::ReadObj(buf_reader, total_count_);
    StreamReader::ReadObj(buf_reader, num_clusters_);

    StreamReader::ReadVector(buf_reader, cluster_offsets_);
    StreamReader::ReadVector(buf_reader, cluster_data_);

    deserialize_rep_hnsw(buf_reader);

    mv_codes_->Deserialize(buf_reader);
    this->label_table_->Deserialize(buf_reader);
}

// ─────────────────────────────────────────────────────────────────────────────
// InitFeatures
// ─────────────────────────────────────────────────────────────────────────────

void
SIMQ::InitFeatures() {
    index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_BUILD,
        IndexFeature::SUPPORT_KNN_SEARCH,
        IndexFeature::SUPPORT_KNN_SEARCH_WITH_ID_FILTER,
        IndexFeature::SUPPORT_RANGE_SEARCH,
        IndexFeature::SUPPORT_RANGE_SEARCH_WITH_ID_FILTER,
        IndexFeature::SUPPORT_DESERIALIZE_BINARY_SET,
        IndexFeature::SUPPORT_DESERIALIZE_FILE,
        IndexFeature::SUPPORT_DESERIALIZE_READER_SET,
        IndexFeature::SUPPORT_SERIALIZE_BINARY_SET,
        IndexFeature::SUPPORT_SERIALIZE_FILE,
        IndexFeature::SUPPORT_SERIALIZE_WRITE_FUNC,
        IndexFeature::SUPPORT_GET_MEMORY_USAGE,
        IndexFeature::SUPPORT_ESTIMATE_MEMORY,
        IndexFeature::SUPPORT_CHECK_ID_EXIST,
        IndexFeature::SUPPORT_SEARCH_CONCURRENT,
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// External parameter mapping
// ─────────────────────────────────────────────────────────────────────────────

static const std::string SIMQ_PARAMS_TEMPLATE =
    R"(
    {
        "{TYPE_KEY}": "{INDEX_SIMQ}",
        "{BASE_CODES_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{TYPE_KEY}": "{IO_TYPE_VALUE_MEMORY_IO}",
                "{IO_FILE_PATH_KEY}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "{CODES_TYPE_KEY}": "multi_vector"
        }
    })";

ParamPtr
SIMQ::CheckAndMappingExternalParam(const JsonType& external_param,
                                   const IndexCommonParam& common_param) {
    const ConstParamMap external_mapping = {
        {BRUTE_FORCE_BASE_IO_TYPE,   {BASE_CODES_KEY, IO_PARAMS_KEY, TYPE_KEY}},
        {BRUTE_FORCE_BASE_FILE_PATH, {BASE_CODES_KEY, IO_PARAMS_KEY, IO_FILE_PATH_KEY}},
        {"init_cluster_ratio", {"init_cluster_ratio"}},
        {"max_cluster_size",   {"max_cluster_size"}},
        {"split_start_idx",    {"split_start_idx"}},
        {"random_seed",        {"random_seed"}},
        {"coarse_k",           {"coarse_k"}},
        {"rerank_k",           {"rerank_k"}},
    };

    if (common_param.data_type_ == DataTypes::DATA_TYPE_INT8) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("simq does not support {} datatype", DATATYPE_INT8));
    }
    if (common_param.metric_ != MetricType::METRIC_TYPE_IP &&
        common_param.metric_ != MetricType::METRIC_TYPE_COSINE) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "simq only supports ip or cosine metric types");
    }

    std::string str = format_map(SIMQ_PARAMS_TEMPLATE, DEFAULT_MAP);
    auto inner_json = JsonType::Parse(str);
    mapping_external_param_to_inner(external_param, external_mapping, inner_json);

    auto simq_param = std::make_shared<SIMQParameter>();
    simq_param->FromJson(inner_json);
    return simq_param;
}

}  // namespace vsag
