#pragma once

#include <algorithm>
#include <limits>
#include <map>

#include "mci_graph_mce.h"
#include "mci_mce.h"

namespace vsag::mci {

template <typename edge_type, typename node_type>
class ccrmce_runner {
public:
    // using CCRMCE::Graph;
    // using CCRMCE::mce;
    std::map<node_type, node_type> mp;
    std::vector<node_type> raw_id;
    std::vector<bool> must_contain_nodes;
    // graph_mce g;
    mce mce_;
    mce_stats last_stats;

    void
    re_id_and_build_csr_graph(std::vector<edge_type>& edge) {
        node_type idx = 0;
        if (raw_id.capacity() < edge.size())
            raw_id.reserve(edge.size());
        raw_id.clear();
        mp.clear();
        for (auto& e : edge) {
            if (mp.find(e.u) == mp.end()) {
                mp[e.u] = idx++;
                raw_id.push_back(e.u);
            }
            if (mp.find(e.v) == mp.end()) {
                mp[e.v] = idx++;
                raw_id.push_back(e.v);
            }
        }

        //change to undirected graph
        graph_mce& g = mce_.g;
        if (g.edges.capacity() < edge.size())
            g.edges.reserve(edge.size());
        g.edges.clear();
        for (auto& e : edge) {
            g.edges.push_back(pair(mp[e.u], mp[e.v]));
        }
        for (auto i = 0; i < g.edges.size(); i++) {
            if (g.edges[i].first > g.edges[i].second) {
                std::swap(g.edges[i].first, g.edges[i].second);
            }
        }
        std::sort(g.edges.begin(), g.edges.end());
        g.m = 0;
        for (auto i = 1; i < g.edges.size(); i++) {
            if (g.edges[i].first == g.edges[i].second)
                continue;
            if (g.edges[i] == g.edges[g.m])
                continue;
            g.m++;
            if (i > g.m) {
                g.edges[g.m] = g.edges[i];
            }
        }
        g.m++;
        g.edges.resize(g.m);
        g.n = idx;
        g.build_csr();
    }

    ccrmce_runner(){};
    node_type
    run(std::vector<edge_type>& edge,
        std::vector<std::vector<node_type>>& max_cliques,
        node_type threshold) {
        if (edge.empty()) {
            mce_.max_clique_cnt_save = 0;
            return 0;
        }
        re_id_and_build_csr_graph(edge);
        mce_.g.change_to_core_order();
        // mce mce(g, &max_cliques);

        mce_.threshold = threshold;
        mce_.set(&max_cliques);
        mce_.run();

        for (auto ii = 0; ii < mce_.max_clique_cnt_save; ii++) {
            auto& max_c = max_cliques[ii];
            for (auto i = 0; i < max_c.size(); i++) {
                max_c[i] = raw_id[max_c[i]];
            }
        }
        return mce_.max_clique_cnt_save;
    }

    void
    reserve(uint32_t sz) {
        mce_.reserve(sz);
        must_contain_nodes.resize(sz);
    }

    node_type
    run(std::vector<edge_type>& edge,
        std::vector<std::vector<node_type>>& max_cliques,
        node_type threshold,
        std::vector<std::atomic<int>>& num_cliques_per_node) {
        if (edge.empty()) {
            mce_.max_clique_cnt_save = 0;
            return 0;
        }
        re_id_and_build_csr_graph(edge);

        mce_.g.change_to_core_order();
        // printf("g:\n");
        // for(uint32_t i = 0; i < mce_.g.n; i++) {
        //     printf("%u:", i);
        //     for(uint32_t j = mce_.g.p_idx2[i]; j < mce_.g.p_idx[i + 1]; j++) {
        //         uint32_t v = mce_.g.p_edge[j];
        //         printf("%u ", v);
        //     }
        //     printf("\n");
        // }

        // mce mce(g, &max_cliques);
        if (must_contain_nodes.size() < mce_.g.n)
            must_contain_nodes.resize(mce_.g.n);
        for (auto u = 0; u < mce_.g.n; u++) {
            must_contain_nodes[u] = false;
        }
        int cnt = 0;
        for (auto u = 0; u < mce_.g.n; u++) {
            int raw_id_u = raw_id[mce_.g.mp[u]];
            if (num_cliques_per_node[raw_id_u].load(std::memory_order_relaxed) == 0) {
                must_contain_nodes[u] = true;
                // printf("mustContain %u\n", u);
                cnt++;
            }
        }

        mce_.threshold = threshold;
        // max_cliques.clear();
        mce_.set(&max_cliques);
        mce_.configure_must_run(std::numeric_limits<uint32_t>::max(), static_cast<uint32_t>(cnt));

        mce_.run(must_contain_nodes);
        last_stats = mce_.stats;

        int cnt2 = 0;
        for (auto u = 0; u < mce_.g.n; u++) {
            if (must_contain_nodes[u]) {
                cnt2++;
            }
        }
        // if(mce_.max_clique_cnt_save>0)
        // printf("valid %d %d, mce_.max_clique_cnt_save %u\n", cnt, cnt2, mce_.max_clique_cnt_save);fflush(stdout);

        for (auto ii = 0; ii < mce_.max_clique_cnt_save; ii++) {
            auto& max_c = max_cliques[ii];

            for (auto i = 0; i < max_c.size(); i++) {
                max_c[i] = raw_id[max_c[i]];
                // printf("%u ", max_c[i]);fflush(stdout);
            }
            // printf("\n");
            // bool is_valid = false;
            // for(auto i = 0; i < max_c.size(); i++) {
            //     if(num_cliques_per_node[ max_c[i] ] == 0) {
            //         is_valid = true; break;
            //     }
            // }
            // if(!is_valid) {
            //     printf("emunerating error\n");fflush(stdout);
            // }
        }

        return mce_.max_clique_cnt_save;
    }

    node_type
    run(std::vector<edge_type>& edge,
        std::vector<std::vector<node_type>>& max_cliques,
        node_type threshold,
        std::vector<std::atomic<int>>& num_cliques_per_node,
        node_type max_saved_cliques) {
        if (edge.empty()) {
            mce_.max_clique_cnt_save = 0;
            return 0;
        }
        re_id_and_build_csr_graph(edge);

        mce_.g.change_to_core_order();

        if (must_contain_nodes.size() < mce_.g.n)
            must_contain_nodes.resize(mce_.g.n);
        for (auto u = 0; u < mce_.g.n; u++) {
            must_contain_nodes[u] = false;
        }
        uint32_t cnt = 0;
        for (auto u = 0; u < mce_.g.n; u++) {
            int raw_id_u = raw_id[mce_.g.mp[u]];
            if (num_cliques_per_node[raw_id_u].load(std::memory_order_relaxed) == 0) {
                must_contain_nodes[u] = true;
                cnt++;
            }
        }

        mce_.threshold = threshold;
        mce_.set(&max_cliques);
        mce_.configure_must_run(static_cast<uint32_t>(max_saved_cliques), cnt);

        mce_.run(must_contain_nodes);
        last_stats = mce_.stats;

        for (auto ii = 0; ii < mce_.max_clique_cnt_save; ii++) {
            auto& max_c = max_cliques[ii];

            for (auto i = 0; i < max_c.size(); i++) {
                max_c[i] = raw_id[max_c[i]];
            }
        }

        return mce_.max_clique_cnt_save;
    }
};

}  // namespace vsag::mci
