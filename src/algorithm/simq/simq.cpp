
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
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "dataset_impl.h"
#include "index_feature_list.h"
#include "inner_string_params.h"
#include "metric_type.h"
#include "query_context.h"
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

struct cluster_member_entry {
    InnerIdType vec_id;
    float distance;
};

std::string
dump_simq_statistics(const SearchStatistics& stats,
                     uint64_t coarse_dist_cmp,
                     uint64_t coarse_probe_count,
                     uint64_t coarse_candidate_count,
                     uint64_t rerank_candidate_count,
                     uint64_t filtered_candidate_count,
                     uint64_t result_count,
                     bool limited_size_applied) {
    auto json = JsonType::Parse(stats.Dump());
    json["simq_coarse_dist_cmp"].SetUint64(coarse_dist_cmp);
    json["simq_coarse_probe_count"].SetUint64(coarse_probe_count);
    json["simq_coarse_candidate_count"].SetUint64(coarse_candidate_count);
    json["simq_rerank_candidate_count"].SetUint64(rerank_candidate_count);
    json["simq_filtered_candidate_count"].SetUint64(filtered_candidate_count);
    json["simq_result_count"].SetUint64(result_count);
    json["simq_limited_size_applied"].SetBool(limited_size_applied);
    return json.Dump();
}

uint64_t
read_dist_cmp(const DatasetPtr& result_ds) {
    if (result_ds == nullptr) {
        return 0;
    }
    auto values = result_ds->GetStatistics({"dist_cmp"});
    if (values.empty() || values[0].empty()) {
        return 0;
    }
    return std::strtoull(values[0].c_str(), nullptr, 10);
}

class HGraphDynamicClustering {
public:
    HGraphDynamicClustering(float init_cluster_ratio,
                            int64_t max_cluster_size,
                            int64_t split_start_idx,
                            int64_t random_seed,
                            IndexCommonParam common_param)
        : init_cluster_ratio_(init_cluster_ratio),
          max_cluster_size_(static_cast<int>(max_cluster_size)),
          split_start_idx_(static_cast<int>(split_start_idx)),
          random_seed_(static_cast<int>(random_seed)),
          common_param_(std::move(common_param)) {
    }

    ~HGraphDynamicClustering() = default;

    void
    Fit(const float* vecs, int64_t num_vecs, int64_t dim);

    std::vector<int> cluster_centers_;
    std::unordered_map<int, std::vector<cluster_member_entry>> clusters_;
    std::vector<int> vec_to_cluster_;

private:
    void
    build_hgraph(const std::vector<int>& center_ids, int64_t dim);

    int
    find_nearest_cluster(int vec_id) const;

    float
    ip_distance(int v1, int v2) const;

    static void
    sorted_insert(std::vector<cluster_member_entry>& members, InnerIdType vec_id, float dist);

    void
    split_cluster(int old_center_id, int64_t dim);

    float init_cluster_ratio_;
    int max_cluster_size_;
    int split_start_idx_;
    int random_seed_;
    IndexCommonParam common_param_;

    const float* vecs_{nullptr};
    int64_t num_vecs_{0};
    int64_t dim_{0};

    std::shared_ptr<HGraph> hgraph_{nullptr};
};

void
HGraphDynamicClustering::build_hgraph(const std::vector<int>& center_ids, int64_t dim) {
    IndexCommonParam cp = common_param_;
    cp.metric_ = MetricType::METRIC_TYPE_IP;
    cp.data_type_ = DataTypes::DATA_TYPE_FLOAT;
    cp.dim_ = dim;

    auto param = HGraph::CheckAndMappingExternalParam(JsonType::Parse("{}"), cp);
    hgraph_ = std::make_shared<HGraph>(param, cp);

    auto n = static_cast<int64_t>(center_ids.size());
    std::vector<float> vecs(static_cast<uint64_t>(n) * static_cast<uint64_t>(dim));
    std::vector<int64_t> labels(static_cast<uint64_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        std::memcpy(vecs.data() + i * dim, vecs_ + center_ids[i] * dim, dim * sizeof(float));
        labels[i] = static_cast<int64_t>(center_ids[i]);
    }

    auto ds = Dataset::Make();
    ds->NumElements(n)->Dim(dim)->Float32Vectors(vecs.data())->Ids(labels.data())->Owner(false);
    hgraph_->Build(ds);
}

int
HGraphDynamicClustering::find_nearest_cluster(int vec_id) const {
    auto query_ds = Dataset::Make();
    query_ds->NumElements(1)->Dim(dim_)->Float32Vectors(vecs_ + vec_id * dim_)->Owner(false);
    auto result = hgraph_->KnnSearch(query_ds, 1, R"({"hgraph": {"ef_search": 100}})", nullptr);
    return static_cast<int>(result->GetIds()[0]);
}

float
HGraphDynamicClustering::ip_distance(int v1, int v2) const {
    const float* a = vecs_ + v1 * dim_;
    const float* b = vecs_ + v2 * dim_;
    float dot = 0.0F;
    for (int64_t d = 0; d < dim_; ++d) {
        dot += a[d] * b[d];
    }
    return 1.0F - dot;
}

void
HGraphDynamicClustering::sorted_insert(std::vector<cluster_member_entry>& members,
                                       InnerIdType vec_id,
                                       float dist) {
    auto it = std::lower_bound(
        members.begin(), members.end(), dist, [](const cluster_member_entry& e, float val) {
            return e.distance < val;
        });
    members.insert(it, {vec_id, dist});
}

void
HGraphDynamicClustering::split_cluster(int old_center_id, int64_t /*dim*/) {
    auto& cluster = clusters_[old_center_id];

    int new_center_id = static_cast<int>(cluster.back().vec_id);

    auto split_it = cluster.begin() + (split_start_idx_ - 1);
    std::vector<cluster_member_entry> to_move(split_it, cluster.end());
    cluster.erase(split_it, cluster.end());

    std::vector<cluster_member_entry> new_cluster;
    new_cluster.push_back({static_cast<InnerIdType>(new_center_id), 0.0F});
    vec_to_cluster_[new_center_id] = new_center_id;

    for (auto& m : to_move) {
        if (static_cast<int>(m.vec_id) == new_center_id) {
            continue;
        }
        float d = ip_distance(static_cast<int>(m.vec_id), new_center_id);
        sorted_insert(new_cluster, m.vec_id, d);
        vec_to_cluster_[m.vec_id] = new_center_id;
    }

    clusters_[new_center_id] = std::move(new_cluster);
    cluster_centers_.push_back(new_center_id);

    if (hgraph_ != nullptr) {
        auto label = static_cast<int64_t>(new_center_id);
        auto new_ds = Dataset::Make();
        new_ds->NumElements(1)
            ->Dim(dim_)
            ->Float32Vectors(vecs_ + new_center_id * dim_)
            ->Ids(&label)
            ->Owner(false);
        hgraph_->Add(new_ds);
    }
}

void
HGraphDynamicClustering::Fit(const float* vecs, int64_t num_vecs, int64_t dim) {
    vecs_ = vecs;
    num_vecs_ = num_vecs;
    dim_ = dim;

    vec_to_cluster_.assign(num_vecs, -1);

    auto num_init =
        std::max(1, static_cast<int>(static_cast<float>(num_vecs) * init_cluster_ratio_));
    std::vector<int> all_indices(num_vecs);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    std::mt19937 rng(random_seed_);
    std::shuffle(all_indices.begin(), all_indices.end(), rng);

    std::vector<int> init_centers(all_indices.begin(), all_indices.begin() + num_init);

    cluster_centers_ = init_centers;
    for (int cid : init_centers) {
        clusters_[cid] = {{static_cast<InnerIdType>(cid), 0.0F}};
        vec_to_cluster_[cid] = cid;
    }

    build_hgraph(init_centers, dim);

    for (auto it = all_indices.begin() + num_init; it != all_indices.end(); ++it) {
        int vid = *it;
        int nearest = find_nearest_cluster(vid);
        float dist = ip_distance(vid, nearest);

        sorted_insert(clusters_[nearest], static_cast<InnerIdType>(vid), dist);
        vec_to_cluster_[vid] = nearest;

        if (static_cast<int>(clusters_[nearest].size()) > max_cluster_size_) {
            split_cluster(nearest, dim);
        }
    }
}

}  // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// Constructor / destructor
// ─────────────────────────────────────────────────────────────────────────────

SIMQ::SIMQ(const SIMQParameterPtr& param, const IndexCommonParam& common_param)
    : InnerIndexInterface(param, common_param),
      common_param_(common_param),
      cluster_lists_(allocator_),
      vec_to_cluster_(allocator_),
      token_to_doc_(allocator_),
      token_to_offset_(allocator_),
      token_to_dist_(allocator_),
      cluster_token_counts_(allocator_) {
    mv_codes_ = FlattenInterface::MakeInstance(param->base_codes_param, common_param);
    init_cluster_ratio_ = param->init_cluster_ratio;
    max_cluster_size_ = param->max_cluster_size;
    split_start_idx_ = param->split_start_idx;
    random_seed_ = param->random_seed;
    default_coarse_k_ = param->coarse_k;
    default_rerank_k_ = param->rerank_k;
    this->has_raw_vector_ = true;
}

SIMQ::~SIMQ() = default;

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
    for (int64_t i = 0; i < num_docs; ++i) {
        total_vecs += mvs[i].len_;
    }
    CHECK_ARGUMENT(total_vecs > 0, "simq build: total number of vectors must be > 0");

    // Build flat token array for clustering (clustering needs contiguous float*)
    Vector<float> flat(total_vecs * static_cast<uint64_t>(mv_dim), allocator_);
    Vector<InnerIdType> vec_to_doc(total_vecs, allocator_);

    token_to_doc_.resize(total_vecs);
    token_to_offset_.resize(total_vecs);
    token_to_dist_.resize(total_vecs, 0.0F);

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
        for (uint32_t t = 0; t < mvs[i].len_; ++t) {
            vec_to_doc[vec_off + t] = static_cast<InnerIdType>(i);
            token_to_doc_[vec_off + t] = static_cast<InnerIdType>(i);
            token_to_offset_[vec_off + t] = t;
        }
        vec_off += mvs[i].len_;
    }

    total_count_ = static_cast<uint64_t>(num_docs);

    // Store multi-vector documents via MultiVectorDataCell
    mv_codes_->Train(flat.data(), total_vecs);
    mv_codes_->Resize(static_cast<InnerIdType>(num_docs));
    mv_codes_->BatchInsertVector(mvs, static_cast<InnerIdType>(num_docs), nullptr);

    for (int64_t i = 0; i < num_docs; ++i) {
        this->label_table_->Insert(static_cast<InnerIdType>(i), labels[i]);
    }

    run_clustering(flat.data(), vec_to_doc, static_cast<int64_t>(total_vecs), mv_dim);
    build_rep_hgraph(flat.data(), mv_dim);

    return {};
}

void
SIMQ::run_clustering(const float* flat_vecs,
                     const Vector<InnerIdType>& vec_to_doc,
                     int64_t num_vecs,
                     int64_t dim) {
    HGraphDynamicClustering clustering(
        init_cluster_ratio_, max_cluster_size_, split_start_idx_, random_seed_, common_param_);
    clustering.Fit(flat_vecs, num_vecs, dim);

    auto nc = static_cast<int64_t>(clustering.cluster_centers_.size());
    num_clusters_ = nc;

    std::unordered_map<int, int> center_to_idx;
    center_to_idx.reserve(static_cast<uint64_t>(nc));
    for (int idx = 0; idx < nc; ++idx) {
        center_to_idx[clustering.cluster_centers_[idx]] = idx;
    }

    // Build per-cluster unique doc sets; a doc with multiple tokens maps to one entry per cluster
    std::vector<std::unordered_set<InnerIdType>> cluster_doc_sets(static_cast<uint64_t>(nc));
    for (int64_t v = 0; v < num_vecs; ++v) {
        int cid = clustering.vec_to_cluster_[v];
        cluster_doc_sets[static_cast<uint64_t>(center_to_idx.at(cid))].insert(vec_to_doc[v]);
    }

    // Build per-cluster doc-ID lists
    cluster_lists_.resize(static_cast<uint64_t>(nc), Vector<InnerIdType>(allocator_));
    for (int idx = 0; idx < nc; ++idx) {
        for (InnerIdType doc_id : cluster_doc_sets[static_cast<uint64_t>(idx)]) {
            cluster_lists_[static_cast<uint64_t>(idx)].push_back(doc_id);
        }
    }

    // Build a lookup: vec_id → distance to its cluster center
    std::vector<float> vec_to_dist(static_cast<uint64_t>(num_vecs), 0.0F);
    for (auto& [cid, members] : clustering.clusters_) {
        for (auto& m : members) {
            vec_to_dist[m.vec_id] = m.distance;
        }
    }

    vec_to_cluster_.resize(static_cast<uint64_t>(num_vecs));
    cluster_token_counts_.assign(static_cast<uint64_t>(nc), 0);
    for (int64_t v = 0; v < num_vecs; ++v) {
        int cid = clustering.vec_to_cluster_[v];
        auto idx = static_cast<InnerIdType>(center_to_idx.at(cid));
        vec_to_cluster_[v] = idx;
        token_to_dist_[v] = vec_to_dist[v];
        ++cluster_token_counts_[idx];
    }
}

void
SIMQ::build_rep_hgraph(const float* flat_vecs, int64_t dim) {
    // Build per-cluster token member lists
    std::vector<std::vector<int>> cluster_token_members(static_cast<uint64_t>(num_clusters_));
    for (int64_t v = 0; v < static_cast<int64_t>(vec_to_cluster_.size()); ++v) {
        cluster_token_members[vec_to_cluster_[v]].push_back(static_cast<int>(v));
    }

    // For each cluster pick the token vector closest to the cluster centroid
    std::vector<float> rep_vecs(static_cast<uint64_t>(num_clusters_) * static_cast<uint64_t>(dim));
    std::vector<int64_t> labels(static_cast<uint64_t>(num_clusters_));

    for (int64_t idx = 0; idx < num_clusters_; ++idx) {
        auto& members = cluster_token_members[static_cast<uint64_t>(idx)];
        auto* dst = rep_vecs.data() + idx * dim;
        // Label is the sequential cluster index so coarse_search IDs map directly
        labels[static_cast<uint64_t>(idx)] = idx;

        if (members.empty()) {
            std::memset(dst, 0, static_cast<uint64_t>(dim) * sizeof(float));
            continue;
        }

        std::vector<float> mean(static_cast<uint64_t>(dim), 0.0F);
        for (int vid : members) {
            const auto* v = flat_vecs + vid * dim;
            for (int d = 0; d < dim; ++d) {
                mean[static_cast<uint64_t>(d)] += v[d];
            }
        }
        float best_dot = -1e30F;
        int best_vid = members[0];
        for (int vid : members) {
            const auto* v = flat_vecs + vid * dim;
            float dot = 0.0F;
            for (int d = 0; d < dim; ++d) {
                dot += v[d] * mean[static_cast<uint64_t>(d)];
            }
            if (dot > best_dot) {
                best_dot = dot;
                best_vid = vid;
            }
        }
        std::memcpy(dst, flat_vecs + best_vid * dim, static_cast<uint64_t>(dim) * sizeof(float));
    }

    IndexCommonParam cp = common_param_;
    cp.metric_ = MetricType::METRIC_TYPE_IP;
    cp.data_type_ = DataTypes::DATA_TYPE_FLOAT;
    cp.dim_ = dim;

    auto param = HGraph::CheckAndMappingExternalParam(JsonType::Parse("{}"), cp);
    rep_hgraph_ = std::make_shared<HGraph>(param, cp);

    auto ds = Dataset::Make();
    ds->NumElements(num_clusters_)
        ->Dim(dim)
        ->Float32Vectors(rep_vecs.data())
        ->Ids(labels.data())
        ->Owner(false);
    rep_hgraph_->Build(ds);
}

// ─────────────────────────────────────────────────────────────────────────────
// Add
// ─────────────────────────────────────────────────────────────────────────────

std::vector<int64_t>
SIMQ::Add(const DatasetPtr& data) {
    std::unique_lock lock(global_mutex_);

    if (rep_hgraph_ == nullptr) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "simq: must call Build before Add");
    }

    const MultiVector* mvs = data->GetMultiVectors();
    CHECK_ARGUMENT(mvs != nullptr, "simq add: data.multi_vectors is nullptr");

    int64_t num_docs = data->GetNumElements();
    const int64_t* labels = data->GetIds();
    CHECK_ARGUMENT(labels != nullptr, "simq add: labels (ids) is nullptr");

    for (int64_t i = 0; i < num_docs; ++i) {
        auto inner_id = static_cast<InnerIdType>(total_count_);

        mv_codes_->Resize(inner_id + 1);
        mv_codes_->InsertVector(&mvs[i], inner_id);
        this->label_table_->Insert(inner_id, labels[i]);

        std::unordered_set<InnerIdType> clusters_seen;
        for (uint32_t t = 0; t < mvs[i].len_; ++t) {
            const auto* token_vec = mvs[i].vectors_ + t * static_cast<uint64_t>(dim_);

            auto query_ds = Dataset::Make();
            query_ds->NumElements(1)->Dim(dim_)->Float32Vectors(token_vec)->Owner(false);
            auto result_ds =
                rep_hgraph_->KnnSearch(query_ds, 1, R"({"hgraph": {"ef_search": 100}})", nullptr);

            auto cluster_idx = static_cast<InnerIdType>(result_ds->GetIds()[0]);
            float token_dist = result_ds->GetDistances()[0];

            vec_to_cluster_.push_back(cluster_idx);
            token_to_doc_.push_back(inner_id);
            token_to_offset_.push_back(t);
            token_to_dist_.push_back(token_dist);

            if (clusters_seen.insert(cluster_idx).second) {
                cluster_lists_[cluster_idx].push_back(inner_id);
            }

            ++cluster_token_counts_[cluster_idx];
            if (static_cast<int64_t>(cluster_token_counts_[cluster_idx]) > max_cluster_size_) {
                split_cluster_incremental(cluster_idx);
            }
        }

        ++total_count_;
    }

    return {};
}

void
SIMQ::split_cluster_incremental(InnerIdType cluster_idx) {
    // Collect all global token IDs that belong to this cluster
    std::vector<InnerIdType> cluster_tokens;
    for (uint64_t ti = 0; ti < vec_to_cluster_.size(); ++ti) {
        if (vec_to_cluster_[ti] == cluster_idx) {
            cluster_tokens.push_back(static_cast<InnerIdType>(ti));
        }
    }

    uint64_t n = cluster_tokens.size();
    if (n < 2) {
        return;
    }

    // Sort tokens by stored distance to cluster representative (ascending = closer first)
    std::sort(cluster_tokens.begin(), cluster_tokens.end(), [this](InnerIdType a, InnerIdType b) {
        return token_to_dist_[a] < token_to_dist_[b];
    });

    // Median split: first half (closer) stays in old cluster,
    //               second half (farther) moves to new cluster.
    uint64_t half = n / 2;
    auto new_cluster_idx = static_cast<InnerIdType>(num_clusters_);

    std::unordered_set<InnerIdType> old_docs;
    std::unordered_set<InnerIdType> new_docs;
    for (uint64_t rank = 0; rank < n; ++rank) {
        InnerIdType tid = cluster_tokens[rank];
        if (rank < half) {
            old_docs.insert(token_to_doc_[tid]);
        } else {
            vec_to_cluster_[tid] = new_cluster_idx;
            new_docs.insert(token_to_doc_[tid]);
        }
    }

    // Rebuild cluster_lists_ for old cluster; add entry for new cluster
    cluster_lists_[cluster_idx].clear();
    for (InnerIdType doc_id : old_docs) {
        cluster_lists_[cluster_idx].push_back(doc_id);
    }

    cluster_lists_.push_back(Vector<InnerIdType>(allocator_));
    for (InnerIdType doc_id : new_docs) {
        cluster_lists_.back().push_back(doc_id);
    }

    // Update token counts directly from known split sizes
    cluster_token_counts_[cluster_idx] = half;
    cluster_token_counts_.push_back(n - half);

    // New cluster representative: the boundary token (closest to old center among new half)
    // Fetch its vector to register in rep_hgraph_
    InnerIdType rep_tid = cluster_tokens[half];
    InnerIdType rep_doc = token_to_doc_[rep_tid];
    uint32_t rep_offset = token_to_offset_[rep_tid];
    bool need_release = false;
    const auto* codes = mv_codes_->GetCodesById(rep_doc, need_release);
    const auto* all_toks = reinterpret_cast<const float*>(codes + sizeof(uint32_t));
    std::vector<float> new_rep_vec(all_toks + rep_offset * static_cast<uint64_t>(dim_),
                                   all_toks + (rep_offset + 1) * static_cast<uint64_t>(dim_));
    if (need_release) {
        mv_codes_->Release(codes);
    }

    auto new_label = static_cast<int64_t>(new_cluster_idx);
    auto new_ds = Dataset::Make();
    new_ds->NumElements(1)
        ->Dim(dim_)
        ->Float32Vectors(new_rep_vec.data())
        ->Ids(&new_label)
        ->Owner(false);
    rep_hgraph_->Add(new_ds);

    // Update token_to_dist_ for tokens moved to new cluster so future splits
    // sort by distance to the new representative, not the old one.
    const auto udim = static_cast<uint64_t>(dim_);
    for (uint64_t rank = half; rank < n; ++rank) {
        InnerIdType tid = cluster_tokens[rank];
        InnerIdType doc_id = token_to_doc_[tid];
        uint32_t offset = token_to_offset_[tid];
        bool nr = false;
        const auto* c = mv_codes_->GetCodesById(doc_id, nr);
        const auto* tv = reinterpret_cast<const float*>(c + sizeof(uint32_t)) + offset * udim;
        float dot = 0.0F;
        for (uint64_t d = 0; d < udim; ++d) {
            dot += tv[d] * new_rep_vec[d];
        }
        token_to_dist_[tid] = 1.0F - dot;
        if (nr) {
            mv_codes_->Release(c);
        }
    }

    ++num_clusters_;
}

// ─────────────────────────────────────────────────────────────────────────────
// Search helpers
// ─────────────────────────────────────────────────────────────────────────────

std::vector<std::pair<InnerIdType, float>>
SIMQ::coarse_search(const float* query_tokens,
                    uint32_t query_token_count,
                    int64_t coarse_k,
                    uint64_t* coarse_dist_cmp,
                    uint64_t* coarse_probe_count) const {
    // All buffers are local — safe for concurrent searches under shared_lock.
    std::unordered_map<InnerIdType, float> score_map;
    score_map.reserve(static_cast<uint64_t>(coarse_k) * static_cast<uint64_t>(max_cluster_size_));
    std::unordered_set<InnerIdType> seen_this_token;
    seen_this_token.reserve(static_cast<uint64_t>(coarse_k) *
                            static_cast<uint64_t>(max_cluster_size_));

    for (uint32_t ti = 0; ti < query_token_count; ++ti) {
        const auto* qt = query_tokens + ti * dim_;

        int64_t actual_coarse_k = std::min(coarse_k, num_clusters_);
        if (actual_coarse_k <= 0) {
            continue;
        }
        if (coarse_probe_count != nullptr) {
            *coarse_probe_count += static_cast<uint64_t>(actual_coarse_k);
        }

        auto query_ds = Dataset::Make();
        query_ds->NumElements(1)->Dim(dim_)->Float32Vectors(qt)->Owner(false);
        auto result_ds = rep_hgraph_->KnnSearch(
            query_ds, actual_coarse_k, R"({"hgraph": {"ef_search": 100}})", nullptr);
        if (coarse_dist_cmp != nullptr) {
            *coarse_dist_cmp += read_dist_cmp(result_ds);
        }

        int64_t nres = result_ds->GetDim();
        const auto* rdists = result_ds->GetDistances();
        const int64_t* rids = result_ds->GetIds();

        std::vector<std::pair<float, InnerIdType>> cscores;
        cscores.reserve(static_cast<uint64_t>(nres));
        for (int64_t ri = 0; ri < nres; ++ri) {
            float cscore = 1.0F - rdists[ri];
            auto cidx = static_cast<InnerIdType>(rids[ri]);
            cscores.emplace_back(cscore, cidx);
        }
        std::sort(cscores.begin(), cscores.end(), [](const auto& a, const auto& b) {
            return a.first > b.first;
        });

        seen_this_token.clear();
        for (auto& [cscore, cidx] : cscores) {
            if (cidx >= static_cast<InnerIdType>(num_clusters_)) {
                continue;
            }
            for (InnerIdType doc_id : cluster_lists_[cidx]) {
                if (!seen_this_token.insert(doc_id).second) {
                    continue;
                }
                score_map[doc_id] += cscore;
            }
        }
    }

    std::vector<std::pair<InnerIdType, float>> ranked(score_map.begin(), score_map.end());
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });
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
    SearchStatistics stats;

    if (total_count_ == 0 || rep_hgraph_ == nullptr) {
        auto result = Dataset::Make();
        result->Statistics(dump_simq_statistics(stats, 0, 0, 0, 0, 0, 0, false));
        return result;
    }

    CHECK_ARGUMENT(query->GetNumElements() > 0, "simq search: query.num_elements must be > 0");
    const MultiVector* query_mvs = query->GetMultiVectors();
    CHECK_ARGUMENT(query_mvs != nullptr, "simq search: query.multi_vectors is nullptr");
    CHECK_ARGUMENT(query_mvs[0].len_ > 0, "simq search: query multi_vector length must be > 0");
    CHECK_ARGUMENT(query_mvs[0].vectors_ != nullptr,
                   "simq search: query multi_vector vectors is nullptr");

    auto sp = SIMQSearchParameters::FromJson(parameters);
    int64_t coarse_k = sp.coarse_k > 0 ? sp.coarse_k : default_coarse_k_;
    int64_t rerank_k = sp.rerank_k > 0 ? sp.rerank_k : default_rerank_k_;
    rerank_k = std::min(rerank_k, static_cast<int64_t>(total_count_));
    k = std::min(k, static_cast<int64_t>(total_count_));

    uint64_t coarse_dist_cmp = 0;
    uint64_t coarse_probe_count = 0;
    auto coarse_results = coarse_search(
        query_mvs[0].vectors_, query_mvs[0].len_, coarse_k, &coarse_dist_cmp, &coarse_probe_count);
    uint64_t coarse_candidate_count = coarse_results.size();
    if (static_cast<int64_t>(coarse_results.size()) > rerank_k) {
        coarse_results.resize(rerank_k);
    }
    uint64_t rerank_candidate_count = coarse_results.size();

    // Exact MaxSim rerank via MultiVectorDataCell
    auto computer = mv_codes_->FactoryComputer(&query_mvs[0]);
    std::vector<std::pair<float, InnerIdType>> reranked;
    reranked.reserve(coarse_results.size());
    uint64_t filtered_candidate_count = 0;
    for (auto& [doc_id, _] : coarse_results) {
        if (filter != nullptr && !filter->CheckValid(this->label_table_->GetLabelById(doc_id))) {
            ++filtered_candidate_count;
            continue;
        }
        float dist = 0.0F;
        mv_codes_->Query(&dist, computer, &doc_id, 1);
        ++stats.dist_cmp;
        reranked.emplace_back(dist, doc_id);
    }
    std::sort(reranked.begin(), reranked.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    int64_t result_count = std::min(k, static_cast<int64_t>(reranked.size()));
    auto [result_ds, dists, ids] = create_fast_dataset(result_count, allocator_);
    for (int64_t i = 0; i < result_count; ++i) {
        dists[i] = reranked[i].first;
        ids[i] = this->label_table_->GetLabelById(reranked[i].second);
    }
    result_ds->Statistics(dump_simq_statistics(stats,
                                               coarse_dist_cmp,
                                               coarse_probe_count,
                                               coarse_candidate_count,
                                               rerank_candidate_count,
                                               filtered_candidate_count,
                                               static_cast<uint64_t>(result_count),
                                               false));
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
    SearchStatistics stats;

    if (total_count_ == 0 || rep_hgraph_ == nullptr) {
        auto result = Dataset::Make();
        result->Statistics(dump_simq_statistics(stats, 0, 0, 0, 0, 0, 0, false));
        return result;
    }

    CHECK_ARGUMENT(query->GetNumElements() > 0,
                   "simq range search: query.num_elements must be > 0");
    const MultiVector* query_mvs = query->GetMultiVectors();
    CHECK_ARGUMENT(query_mvs != nullptr, "simq range search: query.multi_vectors is nullptr");
    CHECK_ARGUMENT(query_mvs[0].len_ > 0,
                   "simq range search: query multi_vector length must be > 0");
    CHECK_ARGUMENT(query_mvs[0].vectors_ != nullptr,
                   "simq range search: query multi_vector vectors is nullptr");

    auto sp = SIMQSearchParameters::FromJson(parameters);
    int64_t coarse_k = sp.coarse_k > 0 ? sp.coarse_k : default_coarse_k_;
    int64_t rerank_k = sp.rerank_k > 0 ? sp.rerank_k : default_rerank_k_;
    rerank_k = std::min(rerank_k, static_cast<int64_t>(total_count_));

    uint64_t coarse_dist_cmp = 0;
    uint64_t coarse_probe_count = 0;
    auto coarse_results = coarse_search(
        query_mvs[0].vectors_, query_mvs[0].len_, coarse_k, &coarse_dist_cmp, &coarse_probe_count);
    uint64_t coarse_candidate_count = coarse_results.size();
    if (static_cast<int64_t>(coarse_results.size()) > rerank_k) {
        coarse_results.resize(rerank_k);
    }
    uint64_t rerank_candidate_count = coarse_results.size();

    auto computer = mv_codes_->FactoryComputer(&query_mvs[0]);
    std::vector<std::pair<float, InnerIdType>> in_range;
    uint64_t filtered_candidate_count = 0;
    for (auto& [doc_id, _] : coarse_results) {
        if (filter != nullptr && !filter->CheckValid(this->label_table_->GetLabelById(doc_id))) {
            ++filtered_candidate_count;
            continue;
        }
        float dist = 0.0F;
        mv_codes_->Query(&dist, computer, &doc_id, 1);
        ++stats.dist_cmp;
        if (dist <= radius) {
            in_range.emplace_back(dist, doc_id);
        }
    }

    bool limited_size_applied = false;
    if (limited_size >= 0 && static_cast<int64_t>(in_range.size()) > limited_size) {
        limited_size_applied = true;
        std::nth_element(in_range.begin(),
                         in_range.begin() + limited_size,
                         in_range.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });
        in_range.resize(limited_size);
    }
    std::sort(in_range.begin(), in_range.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    auto [result_ds, dists, ids] =
        create_fast_dataset(static_cast<int64_t>(in_range.size()), allocator_);
    for (uint64_t i = 0; i < in_range.size(); ++i) {
        dists[i] = in_range[i].first;
        ids[i] = this->label_table_->GetLabelById(in_range[i].second);
    }
    result_ds->Statistics(dump_simq_statistics(stats,
                                               coarse_dist_cmp,
                                               coarse_probe_count,
                                               coarse_candidate_count,
                                               rerank_candidate_count,
                                               filtered_candidate_count,
                                               static_cast<uint64_t>(in_range.size()),
                                               limited_size_applied));
    return std::move(result_ds);
}

// ─────────────────────────────────────────────────────────────────────────────
// Serialize / Deserialize
// ─────────────────────────────────────────────────────────────────────────────

void
SIMQ::serialize_rep_hgraph(StreamWriter& writer) const {
    // Serialize HGraph to a temp buffer, then write [size][data] so the
    // nested HGraph footer is properly bounded during deserialization.
    std::stringstream ss;
    IOStreamWriter tmp_writer(ss);
    rep_hgraph_->Serialize(tmp_writer);
    std::string blob = ss.str();
    auto blob_size = static_cast<uint64_t>(blob.size());
    StreamWriter::WriteObj(writer, blob_size);
    writer.Write(blob.data(), blob_size);
}

void
SIMQ::deserialize_rep_hgraph(StreamReader& reader) {
    uint64_t blob_size = 0;
    StreamReader::ReadObj(reader, blob_size);

    IndexCommonParam cp = common_param_;
    cp.metric_ = MetricType::METRIC_TYPE_IP;
    cp.data_type_ = DataTypes::DATA_TYPE_FLOAT;
    cp.dim_ = dim_;

    auto param = HGraph::CheckAndMappingExternalParam(JsonType::Parse("{}"), cp);
    rep_hgraph_ = std::make_shared<HGraph>(param, cp);

    // Use SliceStreamReader so HGraph's footer seeks within its own data only.
    SliceStreamReader slice(&reader, blob_size);
    rep_hgraph_->Deserialize(slice);
}

void
SIMQ::Serialize(StreamWriter& writer) const {
    std::shared_lock lock(global_mutex_);
    if (rep_hgraph_ == nullptr) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "simq: cannot serialize an unbuilt index");
    }
    uint64_t total_count_val = total_count_.load();
    StreamWriter::WriteObj(writer, total_count_val);
    StreamWriter::WriteObj(writer, num_clusters_);

    auto n_clusters = static_cast<uint64_t>(cluster_lists_.size());
    StreamWriter::WriteObj(writer, n_clusters);
    for (const auto& list : cluster_lists_) {
        StreamWriter::WriteVector(writer, list);
    }

    StreamWriter::WriteVector(writer, vec_to_cluster_);
    StreamWriter::WriteVector(writer, token_to_doc_);
    StreamWriter::WriteVector(writer, token_to_offset_);
    StreamWriter::WriteVector(writer, token_to_dist_);
    StreamWriter::WriteVector(writer, cluster_token_counts_);

    serialize_rep_hgraph(writer);

    mv_codes_->Serialize(writer);
    this->label_table_->Serialize(writer);

    JsonType info;
    info["dim"].SetInt(dim_);
    info["total_count"].SetInt(total_count_.load());
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

    if (info.Contains(INDEX_PARAM) && info[INDEX_PARAM].IsString()) {
        auto inner = JsonType::Parse(info[INDEX_PARAM].GetString());
        SIMQParameter tmp_param;
        tmp_param.FromJson(inner);
        default_coarse_k_ = tmp_param.coarse_k;
        default_rerank_k_ = tmp_param.rerank_k;
        max_cluster_size_ = tmp_param.max_cluster_size;
        split_start_idx_ = tmp_param.split_start_idx;
        random_seed_ = tmp_param.random_seed;
        init_cluster_ratio_ = tmp_param.init_cluster_ratio;
    }

    uint64_t total_count_val = 0;
    StreamReader::ReadObj(buf_reader, total_count_val);
    total_count_.store(total_count_val);
    StreamReader::ReadObj(buf_reader, num_clusters_);

    uint64_t n_clusters = 0;
    StreamReader::ReadObj(buf_reader, n_clusters);
    cluster_lists_.resize(n_clusters, Vector<InnerIdType>(allocator_));
    for (auto& list : cluster_lists_) {
        StreamReader::ReadVector(buf_reader, list);
    }

    StreamReader::ReadVector(buf_reader, vec_to_cluster_);
    StreamReader::ReadVector(buf_reader, token_to_doc_);
    StreamReader::ReadVector(buf_reader, token_to_offset_);
    StreamReader::ReadVector(buf_reader, token_to_dist_);
    StreamReader::ReadVector(buf_reader, cluster_token_counts_);

    deserialize_rep_hgraph(buf_reader);

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
        IndexFeature::SUPPORT_ADD_AFTER_BUILD,
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
                "{TYPE_KEY}": "{IO_TYPE_VALUE_ASYNC_IO}",
                "{IO_FILE_PATH_KEY}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "{CODES_TYPE_KEY}": "multi_vector"
        }
    })";

ParamPtr
SIMQ::CheckAndMappingExternalParam(const JsonType& external_param,
                                   const IndexCommonParam& common_param) {
    const ConstParamMap external_mapping = {
        {BRUTE_FORCE_BASE_IO_TYPE, {BASE_CODES_KEY, IO_PARAMS_KEY, TYPE_KEY}},
        {BRUTE_FORCE_BASE_FILE_PATH, {BASE_CODES_KEY, IO_PARAMS_KEY, IO_FILE_PATH_KEY}},
        {"init_cluster_ratio", {"init_cluster_ratio"}},
        {"max_cluster_size", {"max_cluster_size"}},
        {"split_start_idx", {"split_start_idx"}},
        {"random_seed", {"random_seed"}},
        {"coarse_k", {"coarse_k"}},
        {"rerank_k", {"rerank_k"}},
    };

    if (common_param.data_type_ != DataTypes::DATA_TYPE_FLOAT) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "simq only supports float32 datatype");
    }
    if (common_param.metric_ != MetricType::METRIC_TYPE_IP) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "simq only supports ip metric type");
    }

    std::string str = format_map(SIMQ_PARAMS_TEMPLATE, DEFAULT_MAP);
    auto inner_json = JsonType::Parse(str);
    mapping_external_param_to_inner(external_param, external_mapping, inner_json);

    auto simq_param = std::make_shared<SIMQParameter>();
    simq_param->FromJson(inner_json);
    return simq_param;
}

}  // namespace vsag
