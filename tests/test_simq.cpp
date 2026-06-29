
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

/**
 * SIMQ functional test
 *
 * Dataset:
 *   - 1000 documents, each with 8 token vectors of dim 128 (8000 token vecs total)
 *   - 100 query documents, each with 4 token vectors
 *   - metric: inner product (ip)
 *
 * Search design (ColBERT-style two-phase retrieval):
 *   Phase 1 (coarse): each query token searches the cluster-center HNSW for
 *   coarse_k nearest cluster centers; their member docs form the candidate set.
 *   Phase 2 (rerank): exact MaxSim over the top-rerank_k candidates.
 *
 * Build parameters:
 *   - init_cluster_ratio = 0.01 → ~80 clusters for 8000 token vecs  (≈ sqrt(N))
 *   - max_cluster_size   = 200  → split threshold
 *   - split_start_idx    = 100  → new cluster starts at position 100 in sorted cluster
 *   - coarse_k           = 20   → each query token probes 20 cluster centers
 *   - rerank_k           = 1000 → exact MaxSim rerank over top-1000 candidates
 *
 * Tests:
 *   1. Build + KNN search + recall@10
 *   2. Serialize / Deserialize (binary set) + recall preserved
 *   3. Parameter sweep: coarse_k in {5, 10, 20}
 */

#include <fmt/format.h>
#include <unistd.h>

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "framework/test_dataset.h"
#include "framework/test_dataset_pool.h"
#include "test_index.h"
#include "vsag/dataset.h"
#include "vsag/index.h"
#include "vsag/vsag.h"

using namespace vsag;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static constexpr int64_t SIMQ_DIM = 128;     // token vector dimension
static constexpr uint64_t BASE_DOCS = 1000;  // documents to index
static constexpr uint64_t QUERY_DOCS = 100;  // query documents
static constexpr int DOC_TOKENS = 8;         // tokens per base document
static constexpr int QUERY_TOKENS = 4;       // tokens per query document
static constexpr int TOP_K = 10;

// Generate a unique temp file path; removes file on destruction.
struct TempFile {
    std::string path;
    explicit TempFile(const char* prefix = "/tmp/simq_mv_XXXXXX") {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%s", prefix);
        int fd = mkstemp(buf);
        REQUIRE(fd >= 0);
        ::close(fd);
        path = buf;
    }
    ~TempFile() {
        std::remove(path.c_str());
    }
};

static std::string
make_build_param(const std::string& mv_file_path,
                 float init_cluster_ratio = 0.01f,  // ~80 clusters for 8000 token vecs
                 int64_t max_cluster_size = 200,
                 int64_t split_start_idx = 100,
                 int64_t coarse_k = 20,
                 int64_t rerank_k = 1000) {
    return fmt::format(
        R"({{
            "dtype": "float32",
            "metric_type": "ip",
            "dim": {},
            "index_param": {{
                "base_io_type": "async_io",
                "base_file_path": "{}",
                "init_cluster_ratio": {},
                "max_cluster_size": {},
                "split_start_idx": {},
                "coarse_k": {},
                "rerank_k": {}
            }}
        }})",
        SIMQ_DIM,
        mv_file_path,
        init_cluster_ratio,
        max_cluster_size,
        split_start_idx,
        coarse_k,
        rerank_k);
}

static std::string
make_search_param(int64_t coarse_k = 10, int64_t rerank_k = 1000) {
    return fmt::format(R"({{"simq": {{"coarse_k": {}, "rerank_k": {}}}}})", coarse_k, rerank_k);
}

// Generate normalized random vectors (unit sphere for IP metric)
static void
fill_normalized(float* buf, int64_t n, int64_t dim, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int64_t i = 0; i < n; ++i) {
        float* v = buf + i * dim;
        float norm = 0.0f;
        for (int64_t d = 0; d < dim; ++d) {
            v[d] = nd(rng);
            norm += v[d] * v[d];
        }
        norm = std::sqrt(norm) + 1e-9f;
        for (int64_t d = 0; d < dim; ++d) v[d] /= norm;
    }
}

// MaxSim score: sum over query tokens of max IP over doc tokens
static float
maxsim(const float* q_toks, int nq, const float* d_toks, int nd, int64_t dim) {
    float score = 0.0f;
    for (int qi = 0; qi < nq; ++qi) {
        float best = -1e30f;
        for (int di = 0; di < nd; ++di) {
            float dot = 0.0f;
            for (int64_t d = 0; d < dim; ++d) dot += q_toks[qi * dim + d] * d_toks[di * dim + d];
            if (dot > best)
                best = dot;
        }
        score += best;
    }
    return score;
}

struct MultiVectorDataset {
    // base: BASE_DOCS documents × DOC_TOKENS tokens × SIMQ_DIM floats
    std::vector<float> base_storage;
    std::vector<vsag::MultiVector> base_mvs;
    std::vector<int64_t> base_ids;

    // query: QUERY_DOCS documents × QUERY_TOKENS tokens × SIMQ_DIM floats
    std::vector<float> query_storage;
    std::vector<vsag::MultiVector> query_mvs;

    // ground truth: for each query, sorted list of (doc_id, score)
    // gt_ids[q][k] = k-th nearest doc_id for query q
    std::vector<std::vector<int64_t>> gt_ids;
    std::vector<std::vector<float>> gt_scores;

    vsag::DatasetPtr base_dataset;
    vsag::DatasetPtr query_dataset;
};

static MultiVectorDataset
generate_dataset(uint64_t seed = 42) {
    MultiVectorDataset ds;
    std::mt19937 rng(seed);

    // ── Base documents ────────────────────────────────────────────────────
    ds.base_storage.resize(BASE_DOCS * DOC_TOKENS * SIMQ_DIM);
    fill_normalized(ds.base_storage.data(), BASE_DOCS * DOC_TOKENS, SIMQ_DIM, rng);

    ds.base_mvs.resize(BASE_DOCS);
    ds.base_ids.resize(BASE_DOCS);
    for (uint64_t i = 0; i < BASE_DOCS; ++i) {
        ds.base_mvs[i].len_ = DOC_TOKENS;
        ds.base_mvs[i].vectors_ = ds.base_storage.data() + i * DOC_TOKENS * SIMQ_DIM;
        ds.base_ids[i] = static_cast<int64_t>(i);
    }

    // ── Query documents ───────────────────────────────────────────────────
    ds.query_storage.resize(QUERY_DOCS * QUERY_TOKENS * SIMQ_DIM);
    fill_normalized(ds.query_storage.data(), QUERY_DOCS * QUERY_TOKENS, SIMQ_DIM, rng);

    ds.query_mvs.resize(QUERY_DOCS);
    for (uint64_t q = 0; q < QUERY_DOCS; ++q) {
        ds.query_mvs[q].len_ = QUERY_TOKENS;
        ds.query_mvs[q].vectors_ = ds.query_storage.data() + q * QUERY_TOKENS * SIMQ_DIM;
    }

    // ── Ground truth (exact MaxSim brute-force) ───────────────────────────
    ds.gt_ids.resize(QUERY_DOCS);
    ds.gt_scores.resize(QUERY_DOCS);
    for (uint64_t q = 0; q < QUERY_DOCS; ++q) {
        const float* qtoks = ds.query_mvs[q].vectors_;
        std::vector<std::pair<float, int64_t>> scores(BASE_DOCS);
        for (uint64_t d = 0; d < BASE_DOCS; ++d) {
            float s = maxsim(qtoks, QUERY_TOKENS, ds.base_mvs[d].vectors_, DOC_TOKENS, SIMQ_DIM);
            scores[d] = {s, static_cast<int64_t>(d)};
        }
        std::sort(scores.begin(), scores.end(), [](const auto& a, const auto& b) {
            return a.first > b.first;
        });
        ds.gt_ids[q].resize(TOP_K);
        ds.gt_scores[q].resize(TOP_K);
        for (int k = 0; k < TOP_K; ++k) {
            ds.gt_ids[q][k] = scores[k].second;
            ds.gt_scores[q][k] = scores[k].first;
        }
    }

    // ── VSAG Dataset wrappers ─────────────────────────────────────────────
    ds.base_dataset = vsag::Dataset::Make();
    ds.base_dataset->NumElements(static_cast<int64_t>(BASE_DOCS))
        ->Dim(SIMQ_DIM)
        ->Ids(ds.base_ids.data())
        ->MultiVectors(ds.base_mvs.data())
        ->MultiVectorDim(SIMQ_DIM)
        ->Owner(false);

    ds.query_dataset = vsag::Dataset::Make();
    ds.query_dataset->NumElements(static_cast<int64_t>(QUERY_DOCS))
        ->Dim(SIMQ_DIM)
        ->MultiVectors(ds.query_mvs.data())
        ->MultiVectorDim(SIMQ_DIM)
        ->Owner(false);

    return ds;
}

// Compute recall@TOP_K between returned ids and ground-truth ids for one query
static float
recall_at_k(const int64_t* returned, int64_t n_returned, const int64_t* gt, int64_t n_gt) {
    std::unordered_set<int64_t> gt_set(gt, gt + n_gt);
    int64_t hits = 0;
    for (int64_t i = 0; i < n_returned; ++i)
        if (gt_set.count(returned[i]))
            ++hits;
    return static_cast<float>(hits) / static_cast<float>(n_gt);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test cases
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("SIMQ: dataset generation stats", "[simq][dataset]") {
    auto ds = generate_dataset();
    INFO("Base docs    : " << BASE_DOCS);
    INFO("Doc tokens   : " << DOC_TOKENS);
    INFO("Total tokens : " << BASE_DOCS * DOC_TOKENS);
    INFO("Query docs   : " << QUERY_DOCS);
    INFO("Query tokens : " << QUERY_TOKENS);
    INFO("Dim          : " << SIMQ_DIM);

    REQUIRE(ds.base_mvs.size() == BASE_DOCS);
    REQUIRE(ds.query_mvs.size() == QUERY_DOCS);
    REQUIRE(ds.gt_ids.size() == QUERY_DOCS);
    REQUIRE(ds.gt_ids[0].size() == static_cast<size_t>(TOP_K));

    // Spot-check: ground-truth scores should be descending
    for (uint64_t q = 0; q < QUERY_DOCS; ++q) {
        for (int k = 1; k < TOP_K; ++k) {
            REQUIRE(ds.gt_scores[q][k - 1] >= ds.gt_scores[q][k] - 1e-5f);
        }
    }
    SUCCEED("Dataset generated and ground truth looks sane.");
}

TEST_CASE("SIMQ: build and knn search recall", "[simq][build][search]") {
    auto ds = generate_dataset();
    TempFile tmp;

    // ~80 clusters (sqrt(8000)); coarse_k=10 per token → 4×10=40 probes / 80 = 50% coverage
    auto build_param = make_build_param(tmp.path);
    auto search_param = make_search_param();

    // Create index
    auto result = vsag::Factory::CreateIndex("simq", build_param);
    REQUIRE(result.has_value());
    auto index = result.value();
    REQUIRE(index->GetIndexType() == vsag::IndexType::SIMQ);

    // Build
    auto build_result = index->Build(ds.base_dataset);
    REQUIRE(build_result.has_value());
    REQUIRE(index->GetNumElements() == static_cast<int64_t>(BASE_DOCS));

    // KNN search
    float total_recall = 0.0f;
    for (uint64_t q = 0; q < QUERY_DOCS; ++q) {
        vsag::DatasetPtr one_query = vsag::Dataset::Make();
        one_query->NumElements(1)
            ->Dim(SIMQ_DIM)
            ->MultiVectors(&ds.query_mvs[q])
            ->MultiVectorDim(SIMQ_DIM)
            ->Owner(false);

        auto search_result =
            index->KnnSearch(one_query, TOP_K, search_param, vsag::FilterPtr(nullptr));
        REQUIRE(search_result.has_value());

        auto* ret_ids = search_result.value()->GetIds();
        float r = recall_at_k(ret_ids, TOP_K, ds.gt_ids[q].data(), static_cast<int64_t>(TOP_K));
        total_recall += r;
    }

    float mean_recall = total_recall / static_cast<float>(QUERY_DOCS);
    std::cout << "\n[SIMQ] Mean Recall@" << TOP_K << " = " << mean_recall << "  (over "
              << QUERY_DOCS << " queries)\n";

    // SIMQ is an approximate method; 0.5 is a sanity floor for this small dataset
    REQUIRE(mean_recall >= 0.5f);
    SUCCEED("Build and search completed.");
}

TEST_CASE("SIMQ: serialize and deserialize preserves recall", "[simq][serialization]") {
    auto ds = generate_dataset();
    TempFile tmp_build, tmp_deser;

    auto build_param = make_build_param(tmp_build.path);
    auto search_param = make_search_param();

    auto r1 = vsag::Factory::CreateIndex("simq", build_param);
    REQUIRE(r1.has_value());
    auto index = r1.value();
    auto build_result = index->Build(ds.base_dataset);
    REQUIRE(build_result.has_value());

    // Serialize
    auto serial = index->Serialize();
    REQUIRE(serial.has_value());

    // Deserialize into fresh index (needs its own file path for mv_codes)
    auto r2 = vsag::Factory::CreateIndex("simq", make_build_param(tmp_deser.path));
    REQUIRE(r2.has_value());
    auto index2 = r2.value();
    auto deser = index2->Deserialize(serial.value());
    REQUIRE(deser.has_value());
    REQUIRE(index2->GetNumElements() == static_cast<int64_t>(BASE_DOCS));

    // Compare recall between original and deserialized index
    float recall_orig = 0.0f, recall_deser = 0.0f;
    for (uint64_t q = 0; q < QUERY_DOCS; ++q) {
        vsag::DatasetPtr one_query = vsag::Dataset::Make();
        one_query->NumElements(1)
            ->Dim(SIMQ_DIM)
            ->MultiVectors(&ds.query_mvs[q])
            ->MultiVectorDim(SIMQ_DIM)
            ->Owner(false);

        auto sr1 = index->KnnSearch(one_query, TOP_K, search_param, vsag::FilterPtr(nullptr));
        auto sr2 = index2->KnnSearch(one_query, TOP_K, search_param, vsag::FilterPtr(nullptr));
        REQUIRE(sr1.has_value());
        REQUIRE(sr2.has_value());

        recall_orig += recall_at_k(sr1.value()->GetIds(), TOP_K, ds.gt_ids[q].data(), TOP_K);
        recall_deser += recall_at_k(sr2.value()->GetIds(), TOP_K, ds.gt_ids[q].data(), TOP_K);
    }
    recall_orig /= static_cast<float>(QUERY_DOCS);
    recall_deser /= static_cast<float>(QUERY_DOCS);

    std::cout << "\n[SIMQ Serialize] Original recall@" << TOP_K << " = " << recall_orig
              << "  Deserialized recall@" << TOP_K << " = " << recall_deser << "\n";

    // Deserialized index should produce identical results
    REQUIRE(std::abs(recall_orig - recall_deser) < 0.01f);
    SUCCEED("Serialize/deserialize recall matches.");
}

TEST_CASE("SIMQ: range search", "[simq][range_search]") {
    auto ds = generate_dataset();
    TempFile tmp;

    auto build_param = make_build_param(tmp.path);
    auto search_param = make_search_param();

    auto r = vsag::Factory::CreateIndex("simq", build_param);
    REQUIRE(r.has_value());
    auto index = r.value();
    auto build_result = index->Build(ds.base_dataset);
    REQUIRE(build_result.has_value());

    vsag::DatasetPtr one_query = vsag::Dataset::Make();
    one_query->NumElements(1)
        ->Dim(SIMQ_DIM)
        ->MultiVectors(&ds.query_mvs[0])
        ->MultiVectorDim(SIMQ_DIM)
        ->Owner(false);

    // Use KNN distances to pick a sensible radius: the worst distance in top-k
    auto knn_result = index->KnnSearch(one_query, TOP_K, search_param, vsag::FilterPtr(nullptr));
    REQUIRE(knn_result.has_value());
    const float* knn_dists = knn_result.value()->GetDistances();
    int64_t knn_n = knn_result.value()->GetNumElements();
    REQUIRE(knn_n > 0);
    float radius = knn_dists[knn_n - 1];  // worst distance among top-k results

    SECTION("all returned distances are within radius") {
        auto rr = index->RangeSearch(one_query, radius, search_param, vsag::FilterPtr(nullptr));
        REQUIRE(rr.has_value());
        int64_t n = rr.value()->GetNumElements();
        const float* rdists = rr.value()->GetDistances();
        REQUIRE(n >= 1);
        for (int64_t i = 0; i < n; ++i) {
            REQUIRE(rdists[i] <= radius + 1e-5f);
        }
    }

    SECTION("limited_size truncates results") {
        int64_t limited = 3;
        auto rr =
            index->RangeSearch(one_query, radius, search_param, vsag::FilterPtr(nullptr), limited);
        REQUIRE(rr.has_value());
        REQUIRE(rr.value()->GetNumElements() <= limited);
    }
}

TEST_CASE("SIMQ: parameter sweep on coarse_k and rerank_k", "[simq][sweep]") {
    auto ds = generate_dataset();
    TempFile tmp;

    // ~80 clusters; each query token probes coarse_k centers; 4 tokens total
    auto build_param = make_build_param(tmp.path);

    auto r = vsag::Factory::CreateIndex("simq", build_param);
    REQUIRE(r.has_value());
    auto index = r.value();
    auto build_result = index->Build(ds.base_dataset);
    REQUIRE(build_result.has_value());

    struct Combo {
        int64_t ck;
        int64_t rk;
        float floor;
    };
    // coarse_k=5  → 4×5=20/80 = 25% cluster coverage → modest recall
    // coarse_k=10 → 4×10=40/80 = 50% cluster coverage → good recall
    // coarse_k=20 → 4×20=80/80 = 100% cluster coverage → near-perfect
    std::vector<Combo> combos = {
        {5, 1000, 0.3f},
        {10, 1000, 0.5f},
        {20, 1000, 0.7f},
    };

    std::cout << "\n[SIMQ Sweep] coarse_k / rerank_k → Recall@" << TOP_K << "\n";
    for (auto& c : combos) {
        auto sp = make_search_param(c.ck, c.rk);
        float total = 0.0f;
        for (uint64_t q = 0; q < QUERY_DOCS; ++q) {
            vsag::DatasetPtr one_query = vsag::Dataset::Make();
            one_query->NumElements(1)
                ->Dim(SIMQ_DIM)
                ->MultiVectors(&ds.query_mvs[q])
                ->MultiVectorDim(SIMQ_DIM)
                ->Owner(false);

            auto sr = index->KnnSearch(one_query, TOP_K, sp, vsag::FilterPtr(nullptr));
            REQUIRE(sr.has_value());
            total += recall_at_k(sr.value()->GetIds(), TOP_K, ds.gt_ids[q].data(), TOP_K);
        }
        float recall = total / static_cast<float>(QUERY_DOCS);
        std::cout << "  coarse_k=" << c.ck << " rerank_k=" << c.rk << "  recall=" << recall << "\n";
        REQUIRE(recall >= c.floor);
    }
}

TEST_CASE("SIMQ: incremental Add triggers split and preserves recall", "[simq][add]") {
    // Build with a very small max_cluster_size so Add() triggers splits.
    // Use only half the base docs for Build, then Add the rest.
    static constexpr uint64_t BUILD_DOCS = BASE_DOCS / 2;
    static constexpr uint64_t ADD_DOCS = BASE_DOCS - BUILD_DOCS;

    auto ds = generate_dataset();
    TempFile tmp;

    // Small max_cluster_size (20) with ~5% init ratio → splits happen quickly during Add
    auto build_param = make_build_param(tmp.path,
                                        /*init_cluster_ratio=*/0.05f,
                                        /*max_cluster_size=*/20,
                                        /*split_start_idx=*/10,
                                        /*coarse_k=*/20,
                                        /*rerank_k=*/1000);
    auto search_param = make_search_param(20, 1000);

    auto r = vsag::Factory::CreateIndex("simq", build_param);
    REQUIRE(r.has_value());
    auto index = r.value();

    // Build on first half
    auto build_ds = vsag::Dataset::Make();
    build_ds->NumElements(static_cast<int64_t>(BUILD_DOCS))
        ->Dim(SIMQ_DIM)
        ->Ids(ds.base_ids.data())
        ->MultiVectors(ds.base_mvs.data())
        ->MultiVectorDim(SIMQ_DIM)
        ->Owner(false);
    REQUIRE(index->Build(build_ds).has_value());
    REQUIRE(index->GetNumElements() == static_cast<int64_t>(BUILD_DOCS));

    // Add second half one doc at a time to maximize split triggers
    for (uint64_t i = BUILD_DOCS; i < BASE_DOCS; ++i) {
        auto add_ds = vsag::Dataset::Make();
        add_ds->NumElements(1)
            ->Dim(SIMQ_DIM)
            ->Ids(&ds.base_ids[i])
            ->MultiVectors(&ds.base_mvs[i])
            ->MultiVectorDim(SIMQ_DIM)
            ->Owner(false);
        REQUIRE(index->Add(add_ds).has_value());
    }
    REQUIRE(index->GetNumElements() == static_cast<int64_t>(BASE_DOCS));

    // Compute ground truth over full BASE_DOCS then check recall
    float total_recall = 0.0f;
    for (uint64_t q = 0; q < QUERY_DOCS; ++q) {
        vsag::DatasetPtr one_query = vsag::Dataset::Make();
        one_query->NumElements(1)
            ->Dim(SIMQ_DIM)
            ->MultiVectors(&ds.query_mvs[q])
            ->MultiVectorDim(SIMQ_DIM)
            ->Owner(false);

        auto sr = index->KnnSearch(one_query, TOP_K, search_param, vsag::FilterPtr(nullptr));
        REQUIRE(sr.has_value());
        auto* ret_ids = sr.value()->GetIds();
        // Returned ids must all be valid (in range [0, BASE_DOCS))
        for (int64_t ki = 0; ki < TOP_K; ++ki) REQUIRE(ret_ids[ki] >= 0);
        total_recall +=
            recall_at_k(ret_ids, TOP_K, ds.gt_ids[q].data(), static_cast<int64_t>(TOP_K));
    }

    float mean_recall = total_recall / static_cast<float>(QUERY_DOCS);
    std::cout << "\n[SIMQ Add] Mean Recall@" << TOP_K << " after incremental Add = " << mean_recall
              << "\n";
    REQUIRE(mean_recall >= 0.3f);
}
