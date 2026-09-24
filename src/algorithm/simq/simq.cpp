
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
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "datacell/multi_vector_datacell_parameter.h"
#include "dataset_impl.h"
#include "impl/logger/logger.h"
#include "impl/thread_pool/safe_thread_pool.h"
#include "index_feature_list.h"
#include "inner_string_params.h"
#include "metric_type.h"
#include "query_context.h"
#include "simq_utils.h"
#include "storage/serialization.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "typing.h"
#include "utils/search_threshold.h"
#include "utils/timer.h"
#include "utils/util_functions.h"

namespace vsag {

static void
wait_all_futures(std::vector<std::future<void>>& futures);

namespace {

// Keep optional access local instead of coupling its dataflow to the search loops.
bool
matches_search_threshold(float distance, const std::optional<float>& threshold) {
    return not threshold.has_value() or (std::isfinite(distance) and distance <= threshold.value());
}

struct ClusterMemberEntry {
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
                     bool limited_size_applied,
                     double coarse_ms,
                     double query_ms,
                     double sort_ms,
                     uint32_t mv_io_ms,
                     uint32_t mv_compute_ms,
                     uint32_t mv_candidates) {
    auto json = JsonType::Parse(stats.Dump());
    json["simq_coarse_dist_cmp"].SetUint64(coarse_dist_cmp);
    json["simq_coarse_probe_count"].SetUint64(coarse_probe_count);
    json["simq_coarse_candidate_count"].SetUint64(coarse_candidate_count);
    json["simq_rerank_candidate_count"].SetUint64(rerank_candidate_count);
    json["simq_filtered_candidate_count"].SetUint64(filtered_candidate_count);
    json["simq_result_count"].SetUint64(result_count);
    json["simq_limited_size_applied"].SetBool(limited_size_applied);
    json["simq_coarse_ms"].SetDouble(coarse_ms);
    json["simq_query_ms"].SetDouble(query_ms);
    json["simq_sort_ms"].SetDouble(sort_ms);
    json["simq_mv_io_ms"].SetInt(static_cast<int>(mv_io_ms));
    json["simq_mv_compute_ms"].SetInt(static_cast<int>(mv_compute_ms));
    json["simq_mv_candidates"].SetInt(static_cast<int>(mv_candidates));

    // Standard distance evaluation tracking (compatible with upstream PR #2545)
    uint64_t routing_dist_cmp = coarse_dist_cmp;
    uint64_t rerank_dist_cmp = stats.dist_cmp.load(std::memory_order_relaxed);
    uint64_t total_dist_cmp = routing_dist_cmp + rerank_dist_cmp;

    auto phase_json = JsonType::Parse("{}");
    phase_json["routing"].SetUint64(routing_dist_cmp);
    phase_json["rerank"].SetUint64(rerank_dist_cmp);
    json["distance_evaluations_by_phase"].SetJson(phase_json);

    json["distance_evaluations"].SetUint64(total_dist_cmp);

    auto backend_json = JsonType::Parse("{}");
    backend_json["fp32"].SetUint64(total_dist_cmp);
    json["distance_evaluations_by_backend"].SetJson(backend_json);

    json["complete"].SetBool(true);

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

// Both HGraphs own independent quantizer models. Only decoded vectors cross
// graph boundaries; encoded pair distances always use the document model.
JsonType
simq_graph_parameters(const std::string& quantization, int64_t threads) {
    auto param = JsonType::Parse(
        R"({"max_degree":32,"ef_construction":50,"use_reorder":false,"build_by_base":true})");
    param["base_quantization_type"].SetString(quantization);
    param["build_thread_count"].SetInt(threads);
    return param;
}

// Bounded FP32 staging for graph APIs, not a full representative-vector copy.
void
insert_simq_representatives(const std::shared_ptr<HGraph>& graph,
                            const std::vector<int>& ids,
                            const uint8_t* codes,
                            const FlattenInterfacePtr& storage,
                            int64_t dim) {
    constexpr uint64_t batch_limit = 4096;
    const auto code_size = storage->GetQuantizerCodeSize();
    for (uint64_t offset = 0; offset < ids.size(); offset += batch_limit) {
        const uint64_t count = std::min(batch_limit, ids.size() - offset);
        std::vector<float> scratch(count * dim);
        std::vector<int64_t> labels(count);
        for (uint64_t i = 0; i < count; ++i) {
            labels[i] = static_cast<int64_t>(static_cast<uint64_t>(ids[offset + i]));
            storage->Decode(codes + labels[i] * code_size, scratch.data() + i * dim);
        }
        auto ds = Dataset::Make()
                      ->NumElements(static_cast<int64_t>(count))
                      ->Dim(dim)
                      ->Float32Vectors(scratch.data())
                      ->Ids(labels.data())
                      ->Owner(false);
        if (offset == 0) {
            graph->Build(ds);
        } else {
            graph->Add(ds);
        }
    }
}

class HGraphDynamicClustering {
public:
    HGraphDynamicClustering(float init_cluster_ratio,
                            int64_t max_cluster_size,
                            int64_t split_start_idx,
                            int64_t random_seed,
                            int64_t build_thread_count,
                            IndexCommonParam common_param,
                            std::shared_ptr<SafeThreadPool> thread_pool,
                            FlattenInterfacePtr storage,
                            std::string quantization)
        : init_cluster_ratio_(init_cluster_ratio),
          max_cluster_size_(static_cast<int>(max_cluster_size)),
          split_start_idx_(static_cast<int>(split_start_idx)),
          random_seed_(static_cast<int>(random_seed)),
          build_thread_count_(build_thread_count),
          common_param_(std::move(common_param)),
          thread_pool_(std::move(thread_pool)),
          storage_(std::move(storage)),
          quantization_(std::move(quantization)) {
        token_codes_ = std::dynamic_pointer_cast<TokenCodeInterface>(storage_);
        CHECK_ARGUMENT(token_codes_ != nullptr, "simq token-code capability missing");
    }

    ~HGraphDynamicClustering() = default;

    void
    Fit(const uint8_t* codes, int64_t num_vecs, int64_t dim);

    std::vector<int> cluster_centers_;
    std::unordered_map<int, std::vector<ClusterMemberEntry>> clusters_;
    std::vector<int> vec_to_cluster_;

private:
    void
    build_hgraph(const std::vector<int>& center_ids, int64_t dim);

    int
    find_nearest_cluster(int vec_id) const;

    float
    ip_distance(int v1, int v2) const;

    static void
    sorted_insert(std::vector<ClusterMemberEntry>& members, InnerIdType vec_id, float dist);

    void
    split_cluster(int old_center_id, int64_t dim);

    float init_cluster_ratio_;
    int max_cluster_size_;
    int split_start_idx_;
    int random_seed_;
    int64_t build_thread_count_;
    IndexCommonParam common_param_;
    std::shared_ptr<SafeThreadPool> thread_pool_;

    FlattenInterfacePtr storage_;
    std::string quantization_;
    std::shared_ptr<TokenCodeInterface> token_codes_;
    const uint8_t* codes_{nullptr};
    uint64_t code_size_{0};
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

    auto param = HGraph::CheckAndMappingExternalParam(
        simq_graph_parameters(quantization_, build_thread_count_), cp);
    hgraph_ = std::make_shared<HGraph>(param, cp);
    insert_simq_representatives(hgraph_, center_ids, codes_, storage_, dim);
}

int
HGraphDynamicClustering::find_nearest_cluster(int vec_id) const {
    if (vec_id < 0 || vec_id >= num_vecs_) {
        return cluster_centers_.empty() ? 0 : cluster_centers_[0];
    }

    thread_local std::vector<float> query;
    query.resize(static_cast<uint64_t>(dim_));
    storage_->Decode(codes_ + static_cast<uint64_t>(vec_id) * code_size_, query.data());
    auto query_ds = Dataset::Make();
    query_ds->NumElements(1)->Dim(dim_)->Float32Vectors(query.data())->Owner(false);
    auto result = hgraph_->KnnSearch(query_ds, 1, R"({"hgraph": {"ef_search": 100}})", nullptr);

    if (!result || result->GetIds() == nullptr || result->GetDim() == 0) {
        return cluster_centers_.empty() ? 0 : cluster_centers_[0];
    }

    int nearest_id = static_cast<int>(result->GetIds()[0]);

    bool found = false;
    for (int cid : cluster_centers_) {
        if (cid == nearest_id) {
            found = true;
            break;
        }
    }

    if (!found) {
        return cluster_centers_.empty() ? 0 : cluster_centers_[0];
    }

    return nearest_id;
}

float
HGraphDynamicClustering::ip_distance(int v1, int v2) const {
    return token_codes_->ComputeTokenCodes(codes_ + v1 * code_size_, codes_ + v2 * code_size_);
}

void
HGraphDynamicClustering::sorted_insert(std::vector<ClusterMemberEntry>& members,
                                       InnerIdType vec_id,
                                       float dist) {
    auto it = std::lower_bound(
        members.begin(), members.end(), dist, [](const ClusterMemberEntry& e, float val) {
            return e.distance < val;
        });
    members.insert(it, {vec_id, dist});
}

void
HGraphDynamicClustering::split_cluster(int old_center_id, int64_t /*dim*/) {
    auto it = clusters_.find(old_center_id);
    if (it == clusters_.end()) {
        return;  // Cluster not found
    }
    auto& cluster = it->second;

    if (cluster.empty() || static_cast<int>(cluster.size()) < split_start_idx_) {
        return;  // Not enough elements to split
    }

    // IP self-distance is not necessarily zero (especially with quantization).
    // Never reuse a live graph center as the key of another partition.
    int new_center_id = -1;
    for (auto member = cluster.rbegin();
         member != std::make_reverse_iterator(cluster.begin() + (split_start_idx_ - 1));
         ++member) {
        if (clusters_.count(static_cast<int>(member->vec_id)) == 0) {
            new_center_id = static_cast<int>(member->vec_id);
            break;
        }
    }

    if (new_center_id < 0) {
        new_center_id = static_cast<int>(cluster.back().vec_id);
    }
    CHECK_ARGUMENT(new_center_id < num_vecs_, "SIMQ split center ID out of bounds");

    auto split_it = cluster.begin() + (split_start_idx_ - 1);
    std::vector<ClusterMemberEntry> to_move(split_it, cluster.end());
    cluster.erase(split_it, cluster.end());

    std::vector<ClusterMemberEntry> new_cluster;
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
        std::vector<float> new_center(dim_);
        storage_->Decode(codes_ + new_center_id * code_size_, new_center.data());
        auto label = static_cast<int64_t>(new_center_id);
        auto new_ds = Dataset::Make();
        new_ds->NumElements(1)
            ->Dim(dim_)
            ->Float32Vectors(new_center.data())
            ->Ids(&label)
            ->Owner(false);
        hgraph_->Add(new_ds);
    }
}

void
HGraphDynamicClustering::Fit(const uint8_t* codes, int64_t num_vecs, int64_t dim) {
    codes_ = codes;
    code_size_ = storage_->GetQuantizerCodeSize();
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

    const int64_t batch_size = 10000;  // Process 10k tokens per batch
    const int64_t num_threads = std::max<int64_t>(1, build_thread_count_);

    auto remaining_it = all_indices.begin() + num_init;

    while (remaining_it != all_indices.end()) {
        auto batch_end = remaining_it;
        int64_t count = 0;
        while (batch_end != all_indices.end() && count < batch_size) {
            ++batch_end;
            ++count;
        }

        if (count == 0) {
            break;
        }

        std::vector<std::pair<int, int>> batch_assignments(count);  // (vid, nearest_cid)

        if (num_threads > 1 && count > 100 && thread_pool_) {
            std::vector<std::future<void>> futures;
            const int64_t chunk_size = (count + num_threads - 1) / num_threads;

            for (int64_t t = 0; t < num_threads; ++t) {
                const int64_t start = t * chunk_size;
                const int64_t end = std::min(start + chunk_size, count);
                if (start >= count) {
                    break;
                }

                futures.push_back(thread_pool_->GeneralEnqueue([&, t, start, end]() {
                    for (int64_t i = start; i < end; ++i) {
                        int vid = *(remaining_it + i);
                        int nearest = find_nearest_cluster(vid);
                        batch_assignments[i] = {vid, nearest};
                    }
                }));
            }

            for (auto& f : futures) {
                f.get();
            }
        } else {
            for (int64_t i = 0; i < count; ++i) {
                int vid = *(remaining_it + i);
                int nearest = find_nearest_cluster(vid);
                batch_assignments[i] = {vid, nearest};
            }
        }

        for (const auto& [vid, nearest] : batch_assignments) {
            if (vid < 0 || vid >= num_vecs_) {
                continue;  // Invalid vector ID
            }
            if (nearest < 0 || nearest >= num_vecs_) {
                continue;  // Invalid cluster ID
            }
            if (clusters_.find(nearest) == clusters_.end()) {
                continue;  // Invalid cluster ID
            }

            float dist = ip_distance(vid, nearest);
            sorted_insert(clusters_[nearest], static_cast<InnerIdType>(vid), dist);
            vec_to_cluster_[vid] = nearest;

            if (static_cast<int>(clusters_[nearest].size()) > max_cluster_size_) {
                split_cluster(nearest, dim);
            }
        }

        remaining_it = batch_end;
    }
}

}  // anonymous namespace

SIMQ::SIMQ(const SIMQParameterPtr& param, const IndexCommonParam& common_param)
    : InnerIndexInterface(param, common_param),
      representative_codes_(common_param.allocator_.get()),
      common_param_(common_param),
      cluster_lists_(allocator_),
      vec_to_cluster_(allocator_),
      token_to_doc_(allocator_),
      token_to_offset_(allocator_),
      token_to_dist_(allocator_),
      cluster_token_counts_(allocator_) {
    mv_codes_ = FlattenInterface::MakeInstance(param->base_codes_param, common_param);
    token_codes_ = std::dynamic_pointer_cast<TokenCodeInterface>(mv_codes_);
    CHECK_ARGUMENT(token_codes_ != nullptr, "simq token-code capability missing");
    quantization_type_ = param->quantization_type;
    representative_quantization_type_ = quantization_type_;
    init_cluster_ratio_ = param->init_cluster_ratio;
    max_cluster_size_ = param->max_cluster_size;
    split_start_idx_ = param->split_start_idx;
    random_seed_ = param->random_seed;
    default_coarse_k_ = param->coarse_k;
    default_rerank_k_ = param->rerank_k;
    split_delay_seconds_ = param->split_delay_seconds;
    this->has_raw_vector_ = true;
}

SIMQ::~SIMQ() = default;

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

    uint64_t total_vecs = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(num_docs); ++i) {
        total_vecs += mvs[i].len_;
    }
    CHECK_ARGUMENT(total_vecs > 0, "simq build: total number of vectors must be > 0");

    CHECK_ARGUMENT(total_vecs <= static_cast<uint64_t>(std::numeric_limits<int>::max()),
                   "simq token count exceeds clustering ID capacity");
    Vector<InnerIdType> vec_to_doc(total_vecs, allocator_);

    token_to_doc_.resize(total_vecs);
    token_to_offset_.resize(total_vecs);
    token_to_dist_.resize(total_vecs, 0.0F);

    uint64_t vec_off = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(num_docs); ++i) {
        uint64_t n = static_cast<uint64_t>(mvs[i].len_) * static_cast<uint64_t>(mv_dim);
        if (n > 0) {
            CHECK_ARGUMENT(mvs[i].vectors_ != nullptr,
                           fmt::format("simq build: vectors for doc {} is nullptr", i));
        }
        for (uint32_t t = 0; t < mvs[i].len_; ++t) {
            vec_to_doc[vec_off + t] = static_cast<InnerIdType>(i);
            token_to_doc_[vec_off + t] = static_cast<InnerIdType>(i);
            token_to_offset_[vec_off + t] = t;
        }
        vec_off += mvs[i].len_;
    }

    total_count_ = static_cast<uint64_t>(num_docs);

    // Deterministic bounded sample; never allocate total_tokens * dim FP32.
    CHECK_ARGUMENT(mv_dim > 0, "SIMQ training dimension must be positive");
    CHECK_ARGUMENT(total_vecs <= std::numeric_limits<uint64_t>::max() /
                                     (static_cast<uint64_t>(mv_dim) * sizeof(float)),
                   "SIMQ training size overflow");
    const uint64_t sample_count = std::min<uint64_t>(total_vecs, 4096);
    {
        Vector<float> sample(sample_count * mv_dim, allocator_);
        std::mt19937_64 rng(random_seed_);
        std::uniform_int_distribution<uint64_t> pick(0, total_vecs - 1);
        for (uint64_t i = 0; i < sample_count; ++i) {
            const auto tid = sample_count == total_vecs ? i : pick(rng);
            const auto* token = mvs[token_to_doc_[tid]].vectors_ +
                                static_cast<uint64_t>(token_to_offset_[tid]) * mv_dim;
            std::memcpy(sample.data() + i * mv_dim, token, mv_dim * sizeof(float));
        }
        mv_codes_->Train(sample.data(), sample_count);
    }
    mv_codes_->Resize(static_cast<InnerIdType>(num_docs));
    mv_codes_->BatchInsertVector(mvs, static_cast<InnerIdType>(num_docs), nullptr);

    for (int64_t i = 0; i < static_cast<int64_t>(num_docs); ++i) {
        this->label_table_->Insert(static_cast<InnerIdType>(i), labels[i]);
    }

    const auto code_size = mv_codes_->GetQuantizerCodeSize();
    CHECK_ARGUMENT(code_size != 0, "SIMQ token code size must be nonzero");
    CHECK_ARGUMENT(total_vecs <= std::numeric_limits<uint64_t>::max() / code_size,
                   "SIMQ token code size overflow");
    Vector<uint8_t> token_codes(total_vecs * code_size, allocator_);
    for (uint64_t tid = 0; tid < total_vecs; ++tid) {
        const auto* token = mvs[token_to_doc_[tid]].vectors_ +
                            static_cast<uint64_t>(token_to_offset_[tid]) * mv_dim;
        token_codes_->EncodeToken(token, token_codes.data() + tid * code_size);
    }
    run_clustering(token_codes.data(), vec_to_doc, static_cast<int64_t>(total_vecs), mv_dim);
    build_rep_hgraph(token_codes.data(), mv_dim);

    return {};
}

void
SIMQ::run_clustering(const uint8_t* token_codes,
                     const Vector<InnerIdType>& vec_to_doc,
                     int64_t num_vecs,
                     int64_t dim) {
    HGraphDynamicClustering clustering(init_cluster_ratio_,
                                       max_cluster_size_,
                                       split_start_idx_,
                                       random_seed_,
                                       static_cast<int64_t>(build_thread_count_),
                                       common_param_,
                                       this->thread_pool_,
                                       mv_codes_,
                                       quantization_type_);
    clustering.Fit(token_codes, num_vecs, dim);

    auto nc = static_cast<int64_t>(clustering.cluster_centers_.size());
    num_clusters_ = nc;

    std::unordered_map<int, int> center_to_idx;
    center_to_idx.reserve(static_cast<uint64_t>(nc));
    for (int idx = 0; idx < nc; ++idx) {
        center_to_idx[clustering.cluster_centers_[idx]] = idx;
    }

    std::vector<std::unordered_set<InnerIdType>> cluster_doc_sets(static_cast<uint64_t>(nc));
    for (int64_t v = 0; v < num_vecs; ++v) {
        int cid = clustering.vec_to_cluster_[v];
        cluster_doc_sets[static_cast<uint64_t>(center_to_idx.at(cid))].insert(vec_to_doc[v]);
    }

    cluster_lists_.resize(static_cast<uint64_t>(nc), Vector<InnerIdType>(allocator_));
    for (int idx = 0; idx < nc; ++idx) {
        for (InnerIdType doc_id : cluster_doc_sets[static_cast<uint64_t>(idx)]) {
            cluster_lists_[static_cast<uint64_t>(idx)].push_back(doc_id);
        }
    }

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
SIMQ::build_rep_hgraph(const uint8_t* token_codes, int64_t dim) {
    std::vector<std::vector<InnerIdType>> members(num_clusters_);
    for (uint64_t tid = 0; tid < vec_to_cluster_.size(); ++tid) {
        members[vec_to_cluster_[tid]].push_back(static_cast<InnerIdType>(tid));
    }
    const auto code_size = mv_codes_->GetQuantizerCodeSize();
    std::vector<InnerIdType> representatives(num_clusters_);
    representative_codes_.resize(static_cast<uint64_t>(num_clusters_) * code_size);
    std::vector<float> mean(dim);
    std::vector<float> decoded(dim);
    std::vector<uint8_t> mean_code(code_size);
    for (int64_t c = 0; c < num_clusters_; ++c) {
        CHECK_ARGUMENT(not members[c].empty(), "simq empty partition");
        std::fill(mean.begin(), mean.end(), 0.0F);
        for (const auto tid : members[c]) {
            mv_codes_->Decode(token_codes + tid * code_size, decoded.data());
            for (int64_t d = 0; d < dim; ++d) {
                mean[d] += decoded[d];
            }
        }
        for (auto& value : mean) {
            value /= static_cast<float>(members[c].size());
        }
        token_codes_->EncodeToken(mean.data(), mean_code.data());
        float best = std::numeric_limits<float>::max();
        representatives[c] = members[c][0];
        for (const auto tid : members[c]) {
            const auto distance =
                token_codes_->ComputeTokenCodes(mean_code.data(), token_codes + tid * code_size);
            if (distance < best) {
                best = distance;
                representatives[c] = tid;
            }
        }
    }
    for (int64_t c = 0; c < num_clusters_; ++c) {
        std::memcpy(representative_codes_.data() + c * code_size,
                    token_codes + representatives[c] * code_size,
                    code_size);
    }
    // Split ordering must use one model and the FINAL representative.
    for (uint64_t tid = 0; tid < vec_to_cluster_.size(); ++tid) {
        token_to_dist_[tid] = token_codes_->ComputeTokenCodes(
            token_codes + tid * code_size,
            token_codes + representatives[vec_to_cluster_[tid]] * code_size);
    }
    IndexCommonParam cp = common_param_;
    cp.metric_ = MetricType::METRIC_TYPE_IP;
    cp.data_type_ = DataTypes::DATA_TYPE_FLOAT;
    cp.dim_ = dim;
    auto param = HGraph::CheckAndMappingExternalParam(
        simq_graph_parameters(quantization_type_, static_cast<int64_t>(build_thread_count_)), cp);
    rep_hgraph_ = std::make_shared<HGraph>(param, cp);
    constexpr uint64_t batch_limit = 4096;
    for (uint64_t offset = 0; offset < representatives.size(); offset += batch_limit) {
        const uint64_t count = std::min(batch_limit, representatives.size() - offset);
        std::vector<float> scratch(count * dim);
        std::vector<int64_t> labels(count);
        for (uint64_t i = 0; i < count; ++i) {
            labels[i] = static_cast<int64_t>(static_cast<uint64_t>(offset + i));
            mv_codes_->Decode(token_codes + representatives[offset + i] * code_size,
                              scratch.data() + i * dim);
        }
        auto ds = Dataset::Make()
                      ->NumElements(static_cast<int64_t>(count))
                      ->Dim(dim)
                      ->Float32Vectors(scratch.data())
                      ->Ids(labels.data())
                      ->Owner(false);
        if (offset == 0) {
            rep_hgraph_->Build(ds);
        } else {
            rep_hgraph_->Add(ds);
        }
    }
}

static void
wait_all_futures(std::vector<std::future<void>>& futures) {
    std::exception_ptr first_exception = nullptr;
    for (auto& future : futures) {
        if (not future.valid()) {
            continue;
        }
        try {
            future.get();
        } catch (...) {
            if (not first_exception) {
                first_exception = std::current_exception();
            }
        }
    }
    if (first_exception) {
        std::rethrow_exception(first_exception);
    }
}

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

    CHECK_ARGUMENT(data->GetMultiVectorDim() == dim_, "simq add dimension mismatch");
    uint64_t old_token_count = vec_to_cluster_.size();

    Vector<uint64_t> doc_token_offsets(num_docs + 1, allocator_);
    doc_token_offsets[0] = old_token_count;
    uint64_t total_new_tokens = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(num_docs); ++i) {
        total_new_tokens += mvs[i].len_;
        doc_token_offsets[i + 1] = old_token_count + total_new_tokens;
    }

    auto base_inner_id = static_cast<InnerIdType>(total_count_);

    uint64_t new_token_count = old_token_count + total_new_tokens;
    vec_to_cluster_.resize(new_token_count);
    token_to_doc_.resize(new_token_count);
    token_to_offset_.resize(new_token_count);
    token_to_dist_.resize(new_token_count, 0.0F);

    mv_codes_->Resize(base_inner_id + static_cast<InnerIdType>(num_docs));

    // mv_codes_ and label_table_ have internal locks; inserting serially here
    // avoids contention during the parallel phase.  With MemoryIO this is
    // essentially free (memcpy).
    for (int64_t i = 0; i < static_cast<int64_t>(num_docs); ++i) {
        auto inner_id = static_cast<InnerIdType>(base_inner_id + i);
        mv_codes_->InsertVector(&mvs[i], inner_id);
        this->label_table_->Insert(inner_id, labels[i]);
    }

    // Each thread handles one doc: searches all its tokens on rep_hgraph_
    // and writes directly to pre-allocated token slots (disjoint ranges, no
    // cross-thread data race on the per-token vectors).
    // Cluster-level structures (cluster_lists_, cluster_token_counts_) are
    // collected per-thread and merged in Phase 4.
    struct PerThreadClusterData {
        // cluster_idx → list of inner_ids that touch it (unique per thread)
        std::unordered_map<InnerIdType, std::vector<InnerIdType>> cluster_docs;
        // cluster_idx → token count contribution
        std::unordered_map<InnerIdType, uint64_t> cluster_token_contrib;
    };

    const auto udim = static_cast<uint64_t>(dim_);
    bool use_parallel = this->thread_pool_ != nullptr and num_docs > 1;

    add_completed_docs_.store(0, std::memory_order_relaxed);
    add_completed_tokens_.store(0, std::memory_order_relaxed);
    add_total_docs_ = static_cast<uint64_t>(num_docs);
    add_total_tokens_ = total_new_tokens;
    last_reported_pct_ = -1;

    if (use_parallel) {
        Vector<PerThreadClusterData> per_thread(num_docs, allocator_);
        std::vector<std::future<void>> futures;
        futures.reserve(num_docs);

        for (int64_t i = 0; i < static_cast<int64_t>(num_docs); ++i) {
            futures.emplace_back(this->thread_pool_->GeneralEnqueue(
                [this, i, mvs, &per_thread, &doc_token_offsets, base_inner_id, udim]() {
                    auto inner_id = static_cast<InnerIdType>(base_inner_id + i);
                    uint64_t tok_off = doc_token_offsets[i];
                    auto& td = per_thread[i];

                    std::unordered_set<InnerIdType> clusters_seen;
                    for (uint32_t t = 0; t < mvs[i].len_; ++t) {
                        thread_local std::vector<uint8_t> token_code;
                        thread_local std::vector<float> token_scratch;
                        token_code.resize(mv_codes_->GetQuantizerCodeSize());
                        token_scratch.resize(udim);
                        token_codes_->EncodeToken(mvs[i].vectors_ + t * udim, token_code.data());
                        mv_codes_->Decode(token_code.data(), token_scratch.data());
                        const auto* token_vec = token_scratch.data();
                        auto query_ds = Dataset::Make();
                        query_ds->NumElements(1)
                            ->Dim(static_cast<int64_t>(udim))
                            ->Float32Vectors(token_vec)
                            ->Owner(false);
                        auto result_ds = rep_hgraph_->KnnSearch(
                            query_ds, 1, R"({"hgraph": {"ef_search": 100}})", nullptr);

                        auto cluster_idx = static_cast<InnerIdType>(result_ds->GetIds()[0]);
                        float token_dist = token_codes_->ComputeTokenCodes(
                            token_code.data(),
                            representative_codes_.data() +
                                static_cast<uint64_t>(cluster_idx) * token_code.size());

                        // Write to pre-allocated token slot (no race:
                        // each thread owns a disjoint token range)
                        uint64_t tid = tok_off + t;
                        vec_to_cluster_[tid] = cluster_idx;
                        token_to_doc_[tid] = inner_id;
                        token_to_offset_[tid] = t;
                        token_to_dist_[tid] = token_dist;

                        if (clusters_seen.insert(cluster_idx).second) {
                            td.cluster_docs[cluster_idx].push_back(inner_id);
                        }
                        td.cluster_token_contrib[cluster_idx]++;
                    }

                    add_completed_docs_.fetch_add(1, std::memory_order_relaxed);
                    add_completed_tokens_.fetch_add(mvs[i].len_, std::memory_order_relaxed);

                    uint64_t completed = add_completed_docs_.load(std::memory_order_relaxed);
                    int pct = static_cast<int>(100.0 * static_cast<double>(completed) /
                                               static_cast<double>(add_total_docs_));
                    if (pct > last_reported_pct_) {
                        last_reported_pct_ = pct;
                        logger::info("[SIMQ Add] Progress: {}% ({}/{} docs, {}/{} tokens)",
                                     pct,
                                     completed,
                                     add_total_docs_,
                                     add_completed_tokens_.load(std::memory_order_relaxed),
                                     add_total_tokens_);
                    }
                }));
        }

        wait_all_futures(futures);

        for (int64_t i = 0; i < static_cast<int64_t>(num_docs); ++i) {
            auto& td = per_thread[i];
            for (auto& [cluster_idx, doc_ids] : td.cluster_docs) {
                auto& list = cluster_lists_[cluster_idx];
                list.insert(list.end(), doc_ids.begin(), doc_ids.end());
            }
            for (auto& [cluster_idx, count] : td.cluster_token_contrib) {
                cluster_token_counts_[cluster_idx] += count;
                if (static_cast<int64_t>(cluster_token_counts_[cluster_idx]) > max_cluster_size_) {
                    pending_splits_.insert(cluster_idx);
                }
            }
        }
    } else {
        for (int64_t i = 0; i < static_cast<int64_t>(num_docs); ++i) {
            auto inner_id = static_cast<InnerIdType>(base_inner_id + i);
            uint64_t tok_off = doc_token_offsets[i];

            std::unordered_set<InnerIdType> clusters_seen;
            for (uint32_t t = 0; t < mvs[i].len_; ++t) {
                std::vector<uint8_t> token_code(mv_codes_->GetQuantizerCodeSize());
                std::vector<float> token_scratch(udim);
                token_codes_->EncodeToken(mvs[i].vectors_ + t * udim, token_code.data());
                mv_codes_->Decode(token_code.data(), token_scratch.data());
                const auto* token_vec = token_scratch.data();

                auto query_ds = Dataset::Make();
                query_ds->NumElements(1)
                    ->Dim(static_cast<int64_t>(udim))
                    ->Float32Vectors(token_vec)
                    ->Owner(false);
                auto result_ds = rep_hgraph_->KnnSearch(
                    query_ds, 1, R"({"hgraph": {"ef_search": 100}})", nullptr);

                auto cluster_idx = static_cast<InnerIdType>(result_ds->GetIds()[0]);
                float token_dist = token_codes_->ComputeTokenCodes(
                    token_code.data(),
                    representative_codes_.data() +
                        static_cast<uint64_t>(cluster_idx) * token_code.size());

                uint64_t tid = tok_off + t;
                vec_to_cluster_[tid] = cluster_idx;
                token_to_doc_[tid] = inner_id;
                token_to_offset_[tid] = t;
                token_to_dist_[tid] = token_dist;

                if (clusters_seen.insert(cluster_idx).second) {
                    cluster_lists_[cluster_idx].push_back(inner_id);
                }

                ++cluster_token_counts_[cluster_idx];
                if (static_cast<int64_t>(cluster_token_counts_[cluster_idx]) > max_cluster_size_) {
                    pending_splits_.insert(cluster_idx);
                }
            }

            add_completed_docs_.fetch_add(1, std::memory_order_relaxed);
            add_completed_tokens_.fetch_add(mvs[i].len_, std::memory_order_relaxed);

            uint64_t completed = add_completed_docs_.load(std::memory_order_relaxed);
            int pct = static_cast<int>(100.0 * static_cast<double>(completed) /
                                       static_cast<double>(add_total_docs_));
            if (pct > last_reported_pct_) {
                last_reported_pct_ = pct;
                logger::info("[SIMQ Add] Progress: {}% ({}/{} docs, {}/{} tokens)",
                             pct,
                             completed,
                             add_total_docs_,
                             add_completed_tokens_.load(std::memory_order_relaxed),
                             add_total_tokens_);
            }
        }
    }

    total_count_ = base_inner_id + static_cast<uint64_t>(num_docs);

    flush_pending_splits();

    return {};
}

void
SIMQ::flush_pending_splits() {
    // Three-phase parallel split:
    // Phase 1 (serial): Determine which clusters to split, collect tokens in one pass O(N)
    // Phase 2 (parallel): Execute splits concurrently (inter-cluster parallelism)
    // Phase 3 (serial): Finalize counters and check for re-split

    auto now = std::chrono::steady_clock::now();
    const bool immediate = split_delay_seconds_ <= 0.0;
    const auto delay = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::duration<double>(split_delay_seconds_));

    std::unordered_set<InnerIdType> clusters_to_split;
    std::unordered_map<InnerIdType, int64_t> cluster_to_task_idx;
    std::vector<SplitTask> tasks;

    std::unordered_set<InnerIdType> deferred;
    for (InnerIdType cluster_idx : pending_splits_) {
        if (cluster_idx >= static_cast<InnerIdType>(cluster_token_counts_.size())) {
            pending_split_first_overflow_.erase(cluster_idx);
            continue;
        }
        if (static_cast<int64_t>(cluster_token_counts_[cluster_idx]) <= max_cluster_size_) {
            pending_split_first_overflow_.erase(cluster_idx);
            continue;
        }

        auto ts_it = pending_split_first_overflow_.find(cluster_idx);
        if (ts_it == pending_split_first_overflow_.end()) {
            pending_split_first_overflow_[cluster_idx] = now;
            ts_it = pending_split_first_overflow_.find(cluster_idx);
        }

        if (immediate or (now - ts_it->second) >= delay) {
            clusters_to_split.insert(cluster_idx);
            cluster_to_task_idx[cluster_idx] = static_cast<int64_t>(tasks.size());

            SplitTask task;
            task.cluster_idx = cluster_idx;
            tasks.push_back(std::move(task));
        } else {
            deferred.insert(cluster_idx);
        }
    }

    if (clusters_to_split.empty()) {
        pending_splits_ = std::move(deferred);
        return;
    }

    // One-pass token collection: O(N) instead of O(K*N)
    const uint64_t total_tokens = vec_to_cluster_.size();
    for (uint64_t ti = 0; ti < total_tokens; ++ti) {
        InnerIdType cluster_idx = vec_to_cluster_[ti];
        auto it = clusters_to_split.find(cluster_idx);
        if (it != clusters_to_split.end()) {
            int64_t task_idx = cluster_to_task_idx[cluster_idx];
            tasks[task_idx].tokens.push_back(static_cast<InnerIdType>(ti));
        }
    }

    for (auto& task : tasks) {
        uint64_t n = task.tokens.size();
        if (n < 2) {
            continue;  // Nothing to split
        }

        std::sort(task.tokens.begin(), task.tokens.end(), [this](InnerIdType a, InnerIdType b) {
            return token_to_dist_[a] < token_to_dist_[b];
        });

        task.half = n / 2;

        for (uint64_t rank = 0; rank < n; ++rank) {
            InnerIdType tid = task.tokens[rank];
            if (rank < task.half) {
                task.old_docs.insert(token_to_doc_[tid]);
            } else {
                task.new_docs.insert(token_to_doc_[tid]);
            }
        }
    }

    tasks.erase(
        std::remove_if(
            tasks.begin(), tasks.end(), [](const SplitTask& t) { return t.tokens.size() < 2; }),
        tasks.end());

    // Re-assign contiguous new cluster indices after filtering.
    for (size_t i = 0; i < tasks.size(); ++i) {
        tasks[i].new_cluster_idx = static_cast<InnerIdType>(num_clusters_ + i);
    }

    pending_splits_ = std::move(deferred);
    if (not tasks.empty()) {
        prepare_and_execute_splits(tasks);
    }
}

void
SIMQ::prepare_and_execute_splits(std::vector<SplitTask>& tasks) {
    // Use push_back to add new slots (resize doesn't work with AllocatorWrapper)
    const auto new_cluster_count = static_cast<int64_t>(tasks.size());
    for (int64_t i = 0; i < new_cluster_count; ++i) {
        cluster_lists_.push_back(Vector<InnerIdType>(allocator_));
        cluster_token_counts_.push_back(0);
        representative_codes_.resize(static_cast<uint64_t>(num_clusters_ + new_cluster_count) *
                                     mv_codes_->GetQuantizerCodeSize());
    }

    if (this->thread_pool_) {
        std::vector<std::future<void>> futures;
        futures.reserve(tasks.size());

        for (const auto& task : tasks) {
            const SplitTask* task_ptr = &task;
            futures.push_back(this->thread_pool_->GeneralEnqueue(
                [this, task_ptr]() { execute_split_parallel(*task_ptr); }));
        }

        wait_all_futures(futures);
    } else {
        for (const auto& task : tasks) {
            execute_split_parallel(task);
        }
    }

    num_clusters_ += new_cluster_count;

    for (const auto& task : tasks) {
        pending_split_first_overflow_.erase(task.cluster_idx);

        if (cluster_token_counts_[task.cluster_idx] > static_cast<uint64_t>(max_cluster_size_)) {
            pending_splits_.insert(task.cluster_idx);
        }
        if (cluster_token_counts_[task.new_cluster_idx] >
            static_cast<uint64_t>(max_cluster_size_)) {
            pending_splits_.insert(task.new_cluster_idx);
        }
    }
}

void
SIMQ::execute_split_parallel(const SplitTask& task) {
    for (uint64_t rank = task.half; rank < task.tokens.size(); ++rank) {
        InnerIdType tid = task.tokens[rank];
        vec_to_cluster_[tid] = task.new_cluster_idx;
    }

    cluster_lists_[task.cluster_idx].clear();
    for (InnerIdType doc_id : task.old_docs) {
        cluster_lists_[task.cluster_idx].push_back(doc_id);
    }

    cluster_lists_[task.new_cluster_idx].clear();
    for (InnerIdType doc_id : task.new_docs) {
        cluster_lists_[task.new_cluster_idx].push_back(doc_id);
    }

    cluster_token_counts_[task.cluster_idx] = task.half;
    cluster_token_counts_[task.new_cluster_idx] = task.tokens.size() - task.half;

    // Use the farthest token (from old center) in the new half as new center
    InnerIdType rep_tid = task.tokens[task.tokens.size() - 1];
    InnerIdType rep_doc = token_to_doc_[rep_tid];
    uint32_t rep_offset = token_to_offset_[rep_tid];

    const auto udim = static_cast<uint64_t>(dim_);
    const uint64_t code_size_per_token = mv_codes_->GetQuantizerCodeSize();
    auto codes = mv_codes_->AcquireCodesById(rep_doc);
    CHECK_ARGUMENT(codes, "failed to read simq representative vector");

    std::vector<uint8_t> rep_code(code_size_per_token);
    std::memcpy(
        rep_code.data(),
        codes.Data() + sizeof(uint32_t) + static_cast<uint64_t>(rep_offset) * code_size_per_token,
        code_size_per_token);
    std::memcpy(representative_codes_.data() +
                    static_cast<uint64_t>(task.new_cluster_idx) * code_size_per_token,
                rep_code.data(),
                code_size_per_token);
    std::vector<float> new_rep_vec(udim);
    mv_codes_->Decode(
        codes.Data() + sizeof(uint32_t) + static_cast<uint64_t>(rep_offset) * code_size_per_token,
        new_rep_vec.data());

    auto new_label = static_cast<int64_t>(task.new_cluster_idx);
    auto new_ds = Dataset::Make();
    new_ds->NumElements(1)
        ->Dim(dim_)
        ->Float32Vectors(new_rep_vec.data())
        ->Ids(&new_label)
        ->Owner(false);
    {
        std::lock_guard<std::mutex> lock(rep_hgraph_mutex_);
        rep_hgraph_->Add(new_ds);
    }

    for (uint64_t rank = task.half; rank < task.tokens.size(); ++rank) {
        const auto tid = task.tokens[rank];
        auto token_codes = mv_codes_->AcquireCodesById(token_to_doc_[tid]);
        CHECK_ARGUMENT(token_codes, "failed to read simq split vector");
        token_to_dist_[tid] = token_codes_->ComputeTokenCodes(
            rep_code.data(),
            token_codes.Data() + sizeof(uint32_t) +
                static_cast<uint64_t>(token_to_offset_[tid]) * code_size_per_token);
    }
}

std::vector<std::pair<InnerIdType, float>>
SIMQ::coarse_search(const float* query_tokens,
                    uint32_t query_token_count,
                    int64_t coarse_k,
                    uint64_t* coarse_dist_cmp,
                    uint64_t* coarse_probe_count) const {
    // Flat-array fast-path replacing the previous unordered_map + unordered_set
    // pair, which dominated coarse-search latency. Buffers are reused across
    // queries via mutable member state and lazily grown to fit total_count_ on
    // first call after Build/Add/Deserialize.
    const auto n_docs = static_cast<size_t>(total_count_);
    if (coarse_score_buf_.size() < n_docs) {
        coarse_score_buf_.assign(n_docs, 0.0F);
        coarse_seen_buf_.assign(n_docs, false);
    }
    coarse_dirty_.clear();
    std::vector<bool> touched(n_docs, false);

    // Each query token's search is independent. We do all KnnSearch calls in
    // parallel, then sequentially propagate scores (which is fast O(k) per token).
    struct TokenSearchResult {
        std::vector<std::pair<float, InnerIdType>> cscores;
        int64_t actual_coarse_k{0};
        uint64_t dist_cmp{0};
    };
    std::vector<TokenSearchResult> token_results(query_token_count);

    if (this->thread_pool_ && query_token_count > 1) {
        std::vector<std::future<void>> futures;
        futures.reserve(query_token_count);

        for (uint32_t ti = 0; ti < query_token_count; ++ti) {
            futures.push_back(this->thread_pool_->GeneralEnqueue([&, ti]() {
                const auto* qt = query_tokens + ti * dim_;
                auto& result = token_results[ti];

                result.actual_coarse_k = std::min(coarse_k, num_clusters_);
                if (result.actual_coarse_k <= 0) {
                    return;
                }

                auto query_ds = Dataset::Make();
                query_ds->NumElements(1)->Dim(dim_)->Float32Vectors(qt)->Owner(false);
                auto result_ds = rep_hgraph_->KnnSearch(
                    query_ds, result.actual_coarse_k, R"({"hgraph": {"ef_search": 100}})", nullptr);

                if (coarse_dist_cmp != nullptr) {
                    result.dist_cmp = read_dist_cmp(result_ds);
                }

                int64_t nres = result_ds->GetDim();
                const auto* rdists = result_ds->GetDistances();
                const int64_t* rids = result_ds->GetIds();

                result.cscores.reserve(static_cast<uint64_t>(nres));
                for (int64_t ri = 0; ri < nres; ++ri) {
                    float cscore = 1.0F - rdists[ri];
                    auto cidx = static_cast<InnerIdType>(rids[ri]);
                    result.cscores.emplace_back(cscore, cidx);
                }
                std::sort(result.cscores.begin(),
                          result.cscores.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
            }));
        }
        wait_all_futures(futures);
    } else {
        for (uint32_t ti = 0; ti < query_token_count; ++ti) {
            const auto* qt = query_tokens + ti * dim_;
            auto& result = token_results[ti];

            result.actual_coarse_k = std::min(coarse_k, num_clusters_);
            if (result.actual_coarse_k <= 0) {
                continue;
            }

            auto query_ds = Dataset::Make();
            query_ds->NumElements(1)->Dim(dim_)->Float32Vectors(qt)->Owner(false);
            auto result_ds = rep_hgraph_->KnnSearch(
                query_ds, result.actual_coarse_k, R"({"hgraph": {"ef_search": 100}})", nullptr);

            if (coarse_dist_cmp != nullptr) {
                result.dist_cmp = read_dist_cmp(result_ds);
            }

            int64_t nres = result_ds->GetDim();
            const auto* rdists = result_ds->GetDistances();
            const int64_t* rids = result_ds->GetIds();

            result.cscores.reserve(static_cast<uint64_t>(nres));
            for (int64_t ri = 0; ri < nres; ++ri) {
                float cscore = 1.0F - rdists[ri];
                auto cidx = static_cast<InnerIdType>(rids[ri]);
                result.cscores.emplace_back(cscore, cidx);
            }
            std::sort(result.cscores.begin(),
                      result.cscores.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
        }
    }

    for (uint32_t ti = 0; ti < query_token_count; ++ti) {
        const auto& result = token_results[ti];
        if (result.actual_coarse_k <= 0) {
            continue;
        }
        if (coarse_probe_count != nullptr) {
            *coarse_probe_count += static_cast<uint64_t>(result.actual_coarse_k);
        }
        if (coarse_dist_cmp != nullptr) {
            *coarse_dist_cmp += result.dist_cmp;
        }

        coarse_seen_dirty_.clear();
        for (const auto& [cscore, cidx] : result.cscores) {
            if (cidx >= static_cast<InnerIdType>(num_clusters_)) {
                continue;
            }
            for (InnerIdType doc_id : cluster_lists_[cidx]) {
                if (coarse_seen_buf_[doc_id]) {
                    continue;
                }
                coarse_seen_buf_[doc_id] = true;
                coarse_seen_dirty_.push_back(doc_id);
                if (not touched[doc_id]) {
                    touched[doc_id] = true;
                    coarse_dirty_.push_back(doc_id);
                }
                coarse_score_buf_[doc_id] += cscore;
            }
        }
        for (InnerIdType doc_id : coarse_seen_dirty_) {
            coarse_seen_buf_[doc_id] = false;
        }
    }

    std::vector<std::pair<InnerIdType, float>> ranked;
    ranked.reserve(coarse_dirty_.size());
    for (InnerIdType doc_id : coarse_dirty_) {
        ranked.emplace_back(doc_id, coarse_score_buf_[doc_id]);
        coarse_score_buf_[doc_id] = 0.0F;  // reset for next query
    }
    coarse_dirty_.clear();

    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });
    return ranked;
}

DatasetPtr
SIMQ::KnnSearch(const DatasetPtr& query,
                int64_t k,
                const std::string& parameters,
                const FilterPtr& filter) const {
    std::unique_lock lock(global_mutex_);
    SearchStatistics stats;

    if (total_count_ == 0 || rep_hgraph_ == nullptr) {
        auto result = Dataset::Make();
        result->Statistics(
            dump_simq_statistics(stats, 0, 0, 0, 0, 0, 0, false, 0.0, 0.0, 0.0, 0, 0, 0));
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
    const auto threshold = ParseSearchThreshold(parameters);
    std::shared_ptr<Timer> timer;
    if (sp.enable_time_record) {
        timer = std::make_shared<Timer>();
        timer->SetThreshold(sp.timeout_ms);
    }

    uint64_t coarse_dist_cmp = 0;
    uint64_t coarse_probe_count = 0;
    auto t_coarse_start = std::chrono::steady_clock::now();
    auto coarse_results = coarse_search(
        query_mvs[0].vectors_, query_mvs[0].len_, coarse_k, &coarse_dist_cmp, &coarse_probe_count);
    double coarse_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_coarse_start)
            .count();
    if (timer and timer->CheckOvertime()) {
        stats.is_timeout.store(true, std::memory_order_relaxed);
    }
    uint64_t coarse_candidate_count = coarse_results.size();
    if (stats.is_timeout.load(std::memory_order_relaxed)) {
        coarse_results.clear();
    } else if (static_cast<int64_t>(coarse_results.size()) > rerank_k) {
        coarse_results.resize(rerank_k);
    }
    uint64_t rerank_candidate_count = coarse_results.size();

    auto computer = mv_codes_->FactoryComputer(&query_mvs[0]);
    std::vector<std::pair<float, InnerIdType>> reranked;
    reranked.reserve(coarse_results.size());
    uint64_t filtered_candidate_count = 0;

    std::vector<InnerIdType> batch_ids;
    batch_ids.reserve(coarse_results.size());
    for (auto& [doc_id, _] : coarse_results) {
        if (timer and timer->CheckOvertime()) {
            stats.is_timeout.store(true, std::memory_order_relaxed);
            break;
        }
        if (filter != nullptr && !filter->CheckValid(this->label_table_->GetLabelById(doc_id))) {
            ++filtered_candidate_count;
            continue;
        }
        batch_ids.push_back(doc_id);
    }

    // Batched Query calls (enable MultiRead in MultiVectorDataCell)
    auto t_query_start = std::chrono::steady_clock::now();
    uint32_t mv_io_ms = 0;
    uint32_t mv_compute_ms = 0;
    uint32_t mv_candidates = 0;
    if (!batch_ids.empty()) {
        std::vector<float> batch_dists(batch_ids.size());
        // Use QueryContext so MultiVectorDataCell can report fine-grained timing
        // back through SearchStatistics.
        QueryContext query_context{.stats = &stats,
                                   .distance_phase = DistanceEvaluationPhase::RERANK};
        // Preserve the single MultiRead batch when no timeout was requested.
        // With a deadline, finish at most one small batch before checking again.
        // Bound work between checks while retaining batched IO.
        constexpr size_t timeout_rerank_batch_size = 32;
        const size_t batch_size = timer ? timeout_rerank_batch_size : batch_ids.size();
        for (size_t offset = 0; offset < batch_ids.size(); offset += batch_size) {
            if (timer and timer->CheckOvertime()) {
                stats.is_timeout.store(true, std::memory_order_relaxed);
                break;
            }
            const auto count = std::min(batch_size, batch_ids.size() - offset);
            mv_codes_->Query(batch_dists.data() + offset,
                             computer,
                             batch_ids.data() + offset,
                             static_cast<InnerIdType>(count),
                             &query_context);
            stats.dist_cmp.fetch_add(static_cast<uint32_t>(count), std::memory_order_relaxed);
            for (size_t i = offset; i < offset + count; ++i) {
                reranked.emplace_back(batch_dists[i], batch_ids[i]);
            }
        }
        mv_io_ms = stats.mv_io_time_ms.load(std::memory_order_relaxed);
        mv_compute_ms = stats.mv_compute_time_ms.load(std::memory_order_relaxed);
        mv_candidates = stats.mv_candidate_count.load(std::memory_order_relaxed);
    }
    double query_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_query_start)
            .count();

    auto t_sort_start = std::chrono::steady_clock::now();
    std::sort(reranked.begin(), reranked.end(), simq_distance_less);
    double sort_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_sort_start)
            .count();

    int64_t result_count = 0;
    for (const auto& [distance, _] : reranked) {
        if (result_count >= k) {
            break;
        }
        if (matches_search_threshold(distance, threshold)) {
            ++result_count;
        }
    }
    auto [result_ds, dists, ids] = create_fast_dataset(result_count, allocator_);
    int64_t result_index = 0;
    for (const auto& [distance, inner_id] : reranked) {
        if (result_index >= result_count) {
            break;
        }
        if (not matches_search_threshold(distance, threshold)) {
            continue;
        }
        dists[result_index] = distance;
        ids[result_index] = this->label_table_->GetLabelById(inner_id);
        ++result_index;
    }
    bool limited_size_applied = false;
    if (timer and timer->CheckOvertime()) {
        stats.is_timeout.store(true, std::memory_order_relaxed);
    }
    result_ds->Statistics(dump_simq_statistics(stats,
                                               coarse_dist_cmp,
                                               coarse_probe_count,
                                               coarse_candidate_count,
                                               rerank_candidate_count,
                                               filtered_candidate_count,
                                               static_cast<uint64_t>(result_count),
                                               limited_size_applied,
                                               coarse_ms,
                                               query_ms,
                                               sort_ms,
                                               mv_io_ms,
                                               mv_compute_ms,
                                               mv_candidates));
    return std::move(result_ds);
}

DatasetPtr
SIMQ::RangeSearch(const DatasetPtr& query,
                  float radius,
                  const std::string& parameters,
                  const FilterPtr& filter,
                  int64_t limited_size) const {
    std::unique_lock lock(global_mutex_);
    SearchStatistics stats;

    if (total_count_ == 0 || rep_hgraph_ == nullptr) {
        auto result = Dataset::Make();
        result->Statistics(
            dump_simq_statistics(stats, 0, 0, 0, 0, 0, 0, false, 0.0, 0.0, 0.0, 0, 0, 0));
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
    std::shared_ptr<Timer> timer;
    if (sp.enable_time_record) {
        timer = std::make_shared<Timer>();
        timer->SetThreshold(sp.timeout_ms);
    }

    uint64_t coarse_dist_cmp = 0;
    uint64_t coarse_probe_count = 0;
    auto t_coarse_start = std::chrono::steady_clock::now();
    auto coarse_results = coarse_search(
        query_mvs[0].vectors_, query_mvs[0].len_, coarse_k, &coarse_dist_cmp, &coarse_probe_count);
    double coarse_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_coarse_start)
            .count();
    if (timer and timer->CheckOvertime()) {
        stats.is_timeout.store(true, std::memory_order_relaxed);
    }
    uint64_t coarse_candidate_count = coarse_results.size();
    if (stats.is_timeout.load(std::memory_order_relaxed)) {
        coarse_results.clear();
    } else if (static_cast<int64_t>(coarse_results.size()) > rerank_k) {
        coarse_results.resize(rerank_k);
    }
    uint64_t rerank_candidate_count = coarse_results.size();

    auto computer = mv_codes_->FactoryComputer(&query_mvs[0]);
    std::vector<std::pair<float, InnerIdType>> in_range;
    uint64_t filtered_candidate_count = 0;
    auto t_query_start = std::chrono::steady_clock::now();
    for (auto& [doc_id, _] : coarse_results) {
        if (timer and timer->CheckOvertime()) {
            stats.is_timeout.store(true, std::memory_order_relaxed);
            break;
        }
        if (filter != nullptr && !filter->CheckValid(this->label_table_->GetLabelById(doc_id))) {
            ++filtered_candidate_count;
            continue;
        }
        float dist = 0.0F;
        mv_codes_->Query(&dist, computer, &doc_id, 1);
        stats.dist_cmp.fetch_add(1, std::memory_order_relaxed);
        if (std::isfinite(dist) and dist <= radius) {
            in_range.emplace_back(dist, doc_id);
        }
    }
    double query_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_query_start)
            .count();

    bool limited_size_applied = false;
    if (limited_size >= 0 && static_cast<int64_t>(in_range.size()) > limited_size) {
        limited_size_applied = true;
        std::nth_element(
            in_range.begin(), in_range.begin() + limited_size, in_range.end(), simq_distance_less);
        in_range.resize(limited_size);
    }
    auto t_sort_start = std::chrono::steady_clock::now();
    std::sort(in_range.begin(), in_range.end(), simq_distance_less);
    double sort_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_sort_start)
            .count();

    auto [result_ds, dists, ids] =
        create_fast_dataset(static_cast<int64_t>(in_range.size()), allocator_);
    for (uint64_t i = 0; i < in_range.size(); ++i) {
        dists[i] = in_range[i].first;
        ids[i] = this->label_table_->GetLabelById(in_range[i].second);
    }
    if (timer and timer->CheckOvertime()) {
        stats.is_timeout.store(true, std::memory_order_relaxed);
    }
    result_ds->Statistics(dump_simq_statistics(stats,
                                               coarse_dist_cmp,
                                               coarse_probe_count,
                                               coarse_candidate_count,
                                               rerank_candidate_count,
                                               filtered_candidate_count,
                                               static_cast<uint64_t>(in_range.size()),
                                               limited_size_applied,
                                               coarse_ms,
                                               query_ms,
                                               sort_ms,
                                               0,
                                               0,
                                               0));
    return std::move(result_ds);
}

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

    auto param = HGraph::CheckAndMappingExternalParam(
        simq_graph_parameters(representative_quantization_type_,
                              static_cast<int64_t>(build_thread_count_)),
        cp);
    rep_hgraph_ = std::make_shared<HGraph>(param, cp);

    // Use SliceStreamReader so HGraph's footer seeks within its own data only.
    SliceStreamReader slice(&reader, blob_size);
    rep_hgraph_->Deserialize(slice);
}

float
SIMQ::CalcDistanceById(const DatasetPtr& query, int64_t id, bool calculate_precise_distance) const {
    CHECK_ARGUMENT(query != nullptr, "distance query must not be null");
    CHECK_ARGUMENT(query->GetNumElements() == 1, "single-ID distance requires one query");
    CHECK_ARGUMENT(query->GetMultiVectorDim() == dim_, "query multi-vector dimension mismatch");
    const auto* vectors = query->GetMultiVectors();
    CHECK_ARGUMENT(vectors != nullptr, "query must contain multi-vectors");
    const bool valid_vector = vectors[0].len_ > 0 && vectors[0].vectors_ != nullptr;
    CHECK_ARGUMENT(valid_vector, "query multi-vector must contain token vectors");
    std::shared_lock lock(global_mutex_);
    const auto [valid, inner_id] = label_table_->TryGetIdByLabel(id);
    if (not valid) {
        return -1.0F;
    }
    // SIMQ has one stored representation; both precision modes use the rerank backend.
    auto computer = mv_codes_->FactoryComputer(vectors);
    float distance = 0.0F;
    mv_codes_->Query(&distance, computer, &inner_id, 1);
    return distance;
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
    StreamWriter::WriteVector(writer, representative_codes_);

    JsonType info;
    info["simq_format_version"].SetInt(1);
    info["representative_quantization_type"].SetString(representative_quantization_type_);
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

    const auto format_version =
        info.Contains("simq_format_version") ? info["simq_format_version"].GetInt() : 0;
    if (format_version != 0 and format_version != 1) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "unsupported SIMQ format version");
    }
    BufferStreamReader buf_reader(&reader, std::numeric_limits<uint64_t>::max(), allocator_);

    CHECK_ARGUMENT(dim_ == info["dim"].GetInt(), "SIMQ deserialize dimension mismatch");
    dim_ = info["dim"].GetInt();

    if (info.Contains(INDEX_PARAM) && info[INDEX_PARAM].IsString()) {
        auto inner = JsonType::Parse(info[INDEX_PARAM].GetString());
        SIMQParameter tmp_param;
        tmp_param.FromJson(inner);
        CHECK_ARGUMENT(quantization_type_ == tmp_param.quantization_type,
                       "SIMQ deserialize quantization must match serialized index");
        quantization_type_ = tmp_param.quantization_type;
        default_coarse_k_ = tmp_param.coarse_k;
        default_rerank_k_ = tmp_param.rerank_k;
        max_cluster_size_ = tmp_param.max_cluster_size;
        split_start_idx_ = tmp_param.split_start_idx;
        random_seed_ = tmp_param.random_seed;
        init_cluster_ratio_ = tmp_param.init_cluster_ratio;
        split_delay_seconds_ = tmp_param.split_delay_seconds;
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
    // Reconstruct pending maintenance after load.
    pending_splits_.clear();
    pending_split_first_overflow_.clear();
    for (uint64_t c = 0; c < cluster_token_counts_.size(); ++c) {
        if (cluster_token_counts_[c] > static_cast<uint64_t>(max_cluster_size_)) {
            pending_splits_.insert(static_cast<InnerIdType>(c));
        }
    }

    CHECK_ARGUMENT(n_clusters == static_cast<uint64_t>(num_clusters_),
                   "SIMQ cluster count mismatch");
    CHECK_ARGUMENT(cluster_token_counts_.size() == n_clusters, "SIMQ cluster metadata mismatch");
    CHECK_ARGUMENT(vec_to_cluster_.size() == token_to_doc_.size(), "SIMQ doc metadata mismatch");
    CHECK_ARGUMENT(vec_to_cluster_.size() == token_to_offset_.size(),
                   "SIMQ offset metadata mismatch");
    CHECK_ARGUMENT(vec_to_cluster_.size() == token_to_dist_.size(),
                   "SIMQ distance metadata mismatch");
    for (uint64_t tid = 0; tid < vec_to_cluster_.size(); ++tid) {
        CHECK_ARGUMENT(vec_to_cluster_[tid] < n_clusters, "SIMQ cluster ID out of bounds");
        CHECK_ARGUMENT(token_to_doc_[tid] < total_count_val, "SIMQ document ID out of bounds");
    }
    if (format_version == 1) {
        CHECK_ARGUMENT(info.Contains("representative_quantization_type"),
                       "SIMQ version 1 requires representative quantization metadata");
    }
    representative_quantization_type_ = info.Contains("representative_quantization_type")
                                            ? info["representative_quantization_type"].GetString()
                                            : "fp32";
    deserialize_rep_hgraph(buf_reader);

    mv_codes_->Deserialize(buf_reader);
    this->label_table_->Deserialize(buf_reader);
    const auto code_size = mv_codes_->GetQuantizerCodeSize();
    CHECK_ARGUMENT(num_clusters_ >= 0, "invalid SIMQ cluster count");
    CHECK_ARGUMENT(code_size != 0, "invalid SIMQ representative code size");
    CHECK_ARGUMENT(
        static_cast<uint64_t>(num_clusters_) <= std::numeric_limits<uint64_t>::max() / code_size,
        "SIMQ representative code size overflow");
    const auto expected_size = static_cast<uint64_t>(num_clusters_) * code_size;
    if (format_version == 1) {
        uint64_t stored_size = 0;
        StreamReader::ReadObj(buf_reader, stored_size);
        CHECK_ARGUMENT(stored_size == expected_size, "SIMQ representative code count mismatch");
        representative_codes_.resize(expected_size);
        buf_reader.Read(reinterpret_cast<char*>(representative_codes_.data()), expected_size);
    } else {
        // Old graphs stored FP32 representatives; migrate each independently.
        // Encoding under the restored document model gives one consistent metric
        // for later Add/split without changing the loaded search graph.
        representative_codes_.resize(expected_size);
        for (int64_t c = 0; c < num_clusters_; ++c) {
            const int64_t label = c;
            auto recovered = rep_hgraph_->GetVectorByIds(&label, 1, nullptr);
            CHECK_ARGUMENT(recovered != nullptr, "SIMQ legacy representative recovery failed");
            CHECK_ARGUMENT(recovered->GetFloat32Vectors() != nullptr,
                           "SIMQ legacy representative vector missing");
            token_codes_->EncodeToken(recovered->GetFloat32Vectors(),
                                      representative_codes_.data() + c * code_size);
        }
        for (uint64_t tid = 0; tid < vec_to_cluster_.size(); ++tid) {
            auto codes = mv_codes_->AcquireCodesById(token_to_doc_[tid]);
            CHECK_ARGUMENT(codes, "SIMQ legacy migration token read failed");
            uint32_t token_count = 0;
            std::memcpy(&token_count, codes.Data(), sizeof(token_count));
            CHECK_ARGUMENT(token_to_offset_[tid] < token_count,
                           "SIMQ legacy token offset out of bounds");
            token_to_dist_[tid] = token_codes_->ComputeTokenCodes(
                codes.Data() + sizeof(uint32_t) +
                    static_cast<uint64_t>(token_to_offset_[tid]) * code_size,
                representative_codes_.data() +
                    static_cast<uint64_t>(vec_to_cluster_[tid]) * code_size);
        }
    }
}

void
SIMQ::InitFeatures() {
    index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_CAL_DISTANCE_BY_ID,
        IndexFeature::SUPPORT_BATCH_CALC_DISTANCE_BY_ID,
        IndexFeature::SUPPORT_BUILD,
        IndexFeature::SUPPORT_ADD_AFTER_BUILD,
        IndexFeature::SUPPORT_BATCH_ADD_WITH_MULTI_THREAD,
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
    });
}

JsonType
build_default_simq_param(const JsonType& external_param) {
    const auto io_type = external_param.Contains(BRUTE_FORCE_BASE_IO_TYPE)
                             ? external_param[BRUTE_FORCE_BASE_IO_TYPE].GetString()
                             : IO_TYPE_VALUE_ASYNC_IO;
    JsonType json;
    json[TYPE_KEY].SetString(INDEX_SIMQ);
    json[BASE_CODES_KEY].SetJson(MultiVectorDataCellParameter::CreateDefault(io_type)->ToJson());
    return json;
}

ParamPtr
SIMQ::CheckAndMappingExternalParam(const JsonType& external_param,
                                   const IndexCommonParam& common_param) {
    if (common_param.data_type_ != DataTypes::DATA_TYPE_FLOAT) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "simq only supports float32 datatype");
    }
    if (common_param.metric_ != MetricType::METRIC_TYPE_IP) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "simq only supports ip metric type");
    }

    auto inner_json = build_default_simq_param(external_param);
    for (const auto& [key, ignored] : external_param.GetInnerJson()->items()) {
        (void)ignored;
        auto value = external_param[key];
        if (key == BRUTE_FORCE_BASE_IO_TYPE) {
            inner_json[BASE_CODES_KEY][IO_PARAMS_KEY][TYPE_KEY].SetJson(value);
        } else if (key == BRUTE_FORCE_BASE_FILE_PATH) {
            inner_json[BASE_CODES_KEY][IO_PARAMS_KEY][IO_FILE_PATH_KEY].SetJson(value);
        } else if (key == "init_cluster_ratio" || key == "max_cluster_size" ||
                   key == "split_start_idx" || key == "random_seed" || key == "coarse_k" ||
                   key == "rerank_k" || key == "quantization_type" ||
                   key == BUILD_THREAD_COUNT_KEY || key == "split_delay_seconds") {
            inner_json[key].SetJson(value);
        } else {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                fmt::format("invalid config param: {}", key));
        }
    }

    auto simq_param = std::make_shared<SIMQParameter>();
    simq_param->FromJson(inner_json);
    return simq_param;
}

}  // namespace vsag
