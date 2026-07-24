#pragma once

#include <cstdint>
#include <limits>

#include "mci_graph_mce.h"
#include "mci_linear_set.h"

namespace vsag::mci {

struct mce_stats {
    uint64_t roots_seen = 0;
    uint64_t roots_skipped_by_degree = 0;
    uint64_t roots_skipped_no_must = 0;
    uint64_t recursion_calls = 0;
    uint64_t prune_size_bound = 0;
    uint64_t prune_no_must = 0;
    uint64_t saved_cliques = 0;
    uint64_t quota_stop = 0;
    uint64_t remaining_must_stop = 0;

    void
    reset() {
        *this = mce_stats{};
    }

    void
    add(const mce_stats& other) {
        roots_seen += other.roots_seen;
        roots_skipped_by_degree += other.roots_skipped_by_degree;
        roots_skipped_no_must += other.roots_skipped_no_must;
        recursion_calls += other.recursion_calls;
        prune_size_bound += other.prune_size_bound;
        prune_no_must += other.prune_no_must;
        saved_cliques += other.saved_cliques;
        quota_stop += other.quota_stop;
        remaining_must_stop += other.remaining_must_stop;
    }
};

class mce {
private:
    uint32_t cnt = 0;

    bool out_p = false;
    uint32_t now_u;

    linear_set V;
    void
    bk_tomita_rec(uint32_t deep, std::vector<uint32_t>& C, uint32_t P, uint32_t SX, uint32_t EX);

    std::vector<std::vector<uint32_t>> adj;
    std::vector<std::vector<uint32_t>> ed_adj, st_adj;
    std::vector<uint32_t> re_id;
    std::vector<bool> in_c;
    void
    bk_tomita_rec_ccr(uint32_t deep,
                      std::vector<uint32_t>& C,
                      uint32_t SC,
                      uint32_t EC,
                      uint32_t SP,
                      uint32_t SX,
                      uint32_t EX);

    void
    bk_tomita_rec_ccr_v2(uint32_t deep,
                         std::vector<uint32_t>& C,
                         uint32_t SC,
                         uint32_t EC,
                         uint32_t SP,
                         uint32_t SX,
                         uint32_t EX);

    linear_set c_v3, r_v3, p_v3, x_v3;
    uint32_t num_of_non_c_nodes = 0;
    void
    bk_tomita_rec_ccr_v3(
        uint32_t deep, uint32_t ER, uint32_t EC, uint32_t EP, uint32_t SX, uint32_t EX);
    void
    bk_tomita_rec_ccr_v3_must_contain(uint32_t deep,
                                      uint32_t ER,
                                      uint32_t EC,
                                      uint32_t EP,
                                      uint32_t SX,
                                      uint32_t EX,
                                      std::vector<bool>& must_contain_nodes);
    bool
    should_stop_mce() const;
    bool
    state_has_must(uint32_t ER, uint32_t EC, uint32_t EP, std::vector<bool>& must_contain_nodes);
    void
    clear_must_node(uint32_t u, std::vector<bool>& must_contain_nodes);
    bool
    try_save_must_clique(uint32_t ER, uint32_t EC, std::vector<bool>& must_contain_nodes);

    void
    bk_tomita_rec_ccr_v4(
        uint32_t deep, uint32_t ER, uint32_t EC, uint32_t EP, uint32_t SX, uint32_t EX);

    void
    bk_tomita_rec_ccr_v5(
        uint32_t deep, uint32_t ER, uint32_t EC, uint32_t EP, uint32_t SX, uint32_t EX);

public:
    mce(graph_mce&& g, std::vector<std::vector<uint32_t>>* max_cliques_)
        : g(g), max_cliques(max_cliques_) {
        V.resize(g.n);
    }

    void
    set(std::vector<std::vector<uint32_t>>* max_cliques_) {
        max_cliques = max_cliques_;
        max_clique_cnt_save = 0;
    }
    std::vector<std::vector<uint32_t>>* max_cliques;
    uint32_t max_clique_cnt_save = 0;
    uint32_t threshold = 0;
    graph_mce g;

    mce() {
    }

    uint32_t
    run();

    std::vector<bool> choosed_node;
    std::vector<uint32_t> id_s;
    std::vector<uint32_t> degree;
    std::vector<bool> removed;

    std::vector<uint32_t> p_idx;
    std::vector<uint32_t> e_idx;
    std::vector<uint32_t> p_edge;

    std::vector<uint32_t> clique;
    std::vector<uint32_t> nxt_c;
    void
    reserve(uint32_t sz);
    mce_stats stats;
    uint32_t max_clique_cnt_limit = std::numeric_limits<uint32_t>::max();
    uint32_t remaining_must_count = 0;
    bool stop_search = false;
    void
    configure_must_run(uint32_t max_saved_cliques = std::numeric_limits<uint32_t>::max(),
                       uint32_t initial_must_count = 0);
    uint32_t
    run(std::vector<bool>& must_contain_nodes);

    void
    set_output_file() {
        out_p = true;
    };
};

}  // namespace vsag::mci
