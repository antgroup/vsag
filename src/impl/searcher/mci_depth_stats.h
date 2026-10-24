// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace vsag {

struct MCIDepthSearchStatistics {
    uint32_t hops_limit{0};
    uint32_t max_hops{0};
    uint32_t max_expanded_hops{0};
    uint64_t expanded_nodes{0};
    uint64_t expanded_cliques{0};
    uint64_t depth_limit_skips{0};
    uint64_t member_checks{0};
    uint32_t member_prefetch_mode{0};
    uint32_t member_prefetch_distance{0};
    uint32_t member_prefetch_block{0};
    uint32_t member_prefetch_locality{0};
    bool filtered_cliques_requested{false};
    bool filtered_cliques_used{false};
    double filtered_id_prepare_ms{0.0};
    double filtered_clique_build_ms{0.0};
    uint64_t filtered_valid_count{0};
    uint64_t filtered_incidences{0};
    uint64_t filtered_nonempty_cliques{0};
    uint64_t filtered_kept_cliques{0};
    uint64_t filtered_members{0};
    uint64_t filtered_capacity_bytes{0};
    uint64_t filtered_source_member_checks{0};
    bool dynamic_clique_scan_requested{false};
    bool dynamic_clique_scan_used{false};
    uint64_t dynamic_selected_seeds{0};
    uint64_t dynamic_scored_points{0};
    uint64_t dynamic_scan_passes{0};
    uint64_t dynamic_point_checks{0};
    uint64_t dynamic_clique_checks{0};
    uint64_t dynamic_count_updates{0};
    uint64_t dynamic_activations{0};
    uint64_t dynamic_replacements{0};
    uint64_t dynamic_active_cliques{0};
    double dynamic_id_prepare_ms{0.0};
    double dynamic_initial_union_ms{0.0};
    double dynamic_scan_ms{0.0};
    uint32_t bounded_scan_pass_limit{0};
    bool bounded_scan_gallop{false};
    double bounded_order_prepare_ms{0.0};
    uint64_t bounded_order_rows{0};
    uint64_t bounded_range_rejects{0};
    uint64_t bounded_gap_rejects{0};
    uint64_t bounded_prefix_skips{0};
    uint64_t bounded_block_skips{0};
    uint64_t bounded_tail_skips{0};
    uint64_t bounded_second_start{0};
    uint64_t bounded_second_initial_end{0};
    uint64_t bounded_binary_comparisons{0};
    uint32_t bounded_initial_min{0};
    uint32_t bounded_initial_max{0};
    uint32_t bounded_final_min{0};
    uint32_t bounded_final_max{0};
    uint64_t hub_clique_threshold{0};
    uint32_t hub_hgraph_hops{0};
    uint64_t hub_candidates{0};
    uint64_t hub_skipped_memberships{0};
    uint64_t hub_expanded_nodes{0};
    uint64_t hub_bridge_expansions{0};
    uint64_t hub_neighbor_checks{0};
    uint64_t hub_distance_evaluations{0};
};

// Request-local state for the synchronous, single-query search and its statistics.
const MCIDepthSearchStatistics&
GetMCIDepthSearchStatistics();

}  // namespace vsag
