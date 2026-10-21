// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <unordered_set>
#include <vector>

#include "datacell/clique_datacell.h"
#include "impl/searcher/mci_epoch_marks.h"

namespace vsag {

struct MCINeighborhoodProbeResult {
    uint64_t cliques{0};
    uint64_t member_slots{0};
    uint64_t considered{0};
    uint64_t satisfied{0};
    double local_ratio{0};
    double enrichment{0};
    const char* prediction{"unknown"};
};

struct MCINeighborhoodProbeScratch {
    MCIEpochMarks nodes;
    std::unordered_set<InnerIdType> seen_cliques;
    std::vector<InnerIdType> cliques;
};

inline void
classify_mci_neighborhood(MCINeighborhoodProbeResult& result,
                          double global_ratio,
                          double negative_ratio,
                          double positive_ratio,
                          double min_expected) {
    if (result.considered == 0 or global_ratio <= 0) {
        return;
    }
    result.local_ratio =
        static_cast<double>(result.satisfied) / static_cast<double>(result.considered);
    result.enrichment = result.local_ratio / global_ratio;
    if (result.enrichment >= positive_ratio and result.satisfied >= 1) {
        result.prediction = "positive";
    } else if (result.enrichment <= negative_ratio and
               static_cast<double>(result.considered) * global_ratio >= min_expected) {
        result.prediction = "negative";
    } else {
        result.prediction = "neutral";
    }
}

// Bounded, deterministic probe of cliques covered by the unfiltered HGraph anchors. Repeated
// members contribute only once. This is a local-yield heuristic, not an unbiased union estimator.
template <typename IsValid>
MCINeighborhoodProbeResult
probe_mci_neighborhood(const CliqueDataCellBaseView& view,
                       uint64_t total,
                       const std::vector<InnerIdType>& anchors,
                       uint64_t max_cliques,
                       uint64_t member_budget,
                       IsValid&& is_valid,
                       MCINeighborhoodProbeScratch& scratch) {
    MCINeighborhoodProbeResult result;
    scratch.nodes.Reset(total);
    scratch.seen_cliques.clear();
    scratch.cliques.clear();
    std::vector<uint64_t> cursors(anchors.size(), 0);
    for (uint64_t i = 0; i < anchors.size(); ++i) {
        if (anchors[i] < total) {
            cursors[i] = view.p_node_to_cid[anchors[i]];
        }
    }
    // Round robin avoids letting one high-incidence anchor consume the entire clique budget.
    const auto slot_limit = max_cliques * std::max<uint64_t>(anchors.size(), 1) * 4;
    uint64_t clique_slots = 0;
    bool more = true;
    while (more and scratch.cliques.size() < max_cliques and clique_slots < slot_limit) {
        more = false;
        for (uint64_t i = 0; i < anchors.size() and scratch.cliques.size() < max_cliques; ++i) {
            if (anchors[i] >= total or cursors[i] >= view.p_node_to_cid[anchors[i] + 1]) {
                continue;
            }
            more = true;
            ++clique_slots;
            const auto cid = view.node_to_cids[cursors[i]++];
            if (cid < view.total_clique_count and scratch.seen_cliques.insert(cid).second) {
                scratch.cliques.push_back(cid);
            }
        }
    }
    result.cliques = scratch.cliques.size();
    if (scratch.cliques.empty()) {
        return result;
    }
    const auto per_clique = (member_budget + scratch.cliques.size() - 1) / scratch.cliques.size();
    for (uint64_t round = 0; round < per_clique and result.member_slots < member_budget; ++round) {
        for (const auto cid : scratch.cliques) {
            const uint64_t begin = view.p_maxc[cid];
            const uint64_t size = view.p_maxc[cid + 1] - begin;
            const auto take = std::min(size, per_clique);
            if (round >= take) {
                continue;
            }
            const auto id = view.maxcs[begin + round * size / take];
            ++result.member_slots;
            if (id < total and scratch.nodes.TryVisit(id)) {
                ++result.considered;
                result.satisfied += is_valid(id) ? 1 : 0;
            }
            if (result.member_slots == member_budget) {
                break;
            }
        }
    }
    return result;
}

}  // namespace vsag
