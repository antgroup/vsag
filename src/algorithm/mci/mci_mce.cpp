#include "mci_mce.h"

#include "mci_list_linear_heap.h"

// This file contains the mechanically imported MCE reference implementation. Keep its original
// control-flow shape so behavior remains comparable with the upstream algorithm.
// NOLINTBEGIN

namespace vsag::mci {
void
mce::reserve(uint32_t sz) {
    c_v3.resize(sz);
    p_v3.resize(sz);
    r_v3.resize(sz);
    x_v3.resize(sz);
    // choosed_node.resize(sz);
    ed_adj.resize(sz);
    for (uint32_t i = 0; i < sz; i++) ed_adj[i].resize(sz);
    st_adj.resize(sz);
    for (uint32_t i = 0; i < sz; i++) st_adj[i].resize(sz);
    adj.resize(sz);
    for (uint32_t i = 0; i < sz; i++) {
        adj[i].resize(sz);
    }
}

void
mce::configure_must_run(uint32_t max_saved_cliques, uint32_t initial_must_count) {
    max_clique_cnt_limit = max_saved_cliques;
    remaining_must_count = initial_must_count;
    stop_search = false;
    stats.reset();
}

bool
mce::should_stop_mce() const {
    return stop_search || max_clique_cnt_save >= max_clique_cnt_limit || remaining_must_count == 0;
}

bool
mce::state_has_must(uint32_t ER, uint32_t EC, uint32_t EP, std::vector<bool>& must_contain_nodes) {
    if (must_contain_nodes[now_u])
        return true;
    for (uint32_t i = 0; i < EC; i++) {
        uint32_t u = g.p_edge[g.p_idx[now_u] + c_v3[i]];
        if (must_contain_nodes[u])
            return true;
    }
    for (uint32_t i = 0; i < ER; i++) {
        uint32_t u = g.p_edge[g.p_idx[now_u] + r_v3[i]];
        if (must_contain_nodes[u])
            return true;
    }
    for (uint32_t i = 0; i < EP; i++) {
        uint32_t u = g.p_edge[g.p_idx[now_u] + p_v3[i]];
        if (must_contain_nodes[u])
            return true;
    }
    return false;
}

void
mce::clear_must_node(uint32_t u, std::vector<bool>& must_contain_nodes) {
    if (must_contain_nodes[u]) {
        must_contain_nodes[u] = false;
        if (remaining_must_count > 0)
            remaining_must_count--;
    }
}

bool
mce::try_save_must_clique(uint32_t ER, uint32_t EC, std::vector<bool>& must_contain_nodes) {
    if (should_stop_mce())
        return false;
    if (1 + EC + ER < threshold)
        return false;
    if (!state_has_must(ER, EC, 0, must_contain_nodes))
        return false;
    if (max_clique_cnt_save >= max_clique_cnt_limit) {
        stop_search = true;
        stats.quota_stop++;
        return false;
    }

    if (max_clique_cnt_save >= (*max_cliques).size()) {
        (*max_cliques).resize((max_clique_cnt_save + 1) * 2);
    }
    (*max_cliques)[max_clique_cnt_save].clear();
    (*max_cliques)[max_clique_cnt_save].push_back(g.mp[now_u]);
    clear_must_node(now_u, must_contain_nodes);
    for (uint32_t i = 0; i < EC; i++) {
        uint32_t u = g.p_edge[g.p_idx[now_u] + c_v3[i]];
        (*max_cliques)[max_clique_cnt_save].push_back(g.mp[u]);
    }
    for (uint32_t i = 0; i < ER; i++) {
        uint32_t u = g.p_edge[g.p_idx[now_u] + r_v3[i]];
        (*max_cliques)[max_clique_cnt_save].push_back(g.mp[u]);
    }
    for (uint32_t i = 0; i < EC; i++) {
        clear_must_node(g.p_edge[g.p_idx[now_u] + c_v3[i]], must_contain_nodes);
    }
    for (uint32_t i = 0; i < ER; i++) {
        clear_must_node(g.p_edge[g.p_idx[now_u] + r_v3[i]], must_contain_nodes);
    }
    max_clique_cnt_save++;
    stats.saved_cliques++;

    if (max_clique_cnt_save >= max_clique_cnt_limit) {
        stop_search = true;
        stats.quota_stop++;
    }
    if (remaining_must_count == 0) {
        stop_search = true;
        stats.remaining_must_stop++;
    }
    return true;
}

uint32_t
mce::run(std::vector<bool>& must_contain_nodes) {
    c_v3.resize(g.max_d);
    p_v3.resize(g.max_d);
    r_v3.resize(g.max_d);
    x_v3.resize(g.max_d);

    // choosed_node.resize(g.n);
    // for(int i = 0; i < g.n; i++) {
    //     choosed_node[i] = false;
    // }

    cnt = 0;
    max_clique_cnt_save = 0;
    // printf("a0\n");fflush(stdout);
    ed_adj.resize(g.core_number + 1);
    for (uint32_t i = 0; i < g.core_number; i++) ed_adj[i].resize(g.max_d);
    st_adj.resize(g.core_number + 1);
    for (uint32_t i = 0; i < g.core_number; i++) st_adj[i].resize(g.max_d);
    adj.resize(g.max_d);
    for (uint32_t i = 0; i < g.max_d; i++) {
        adj[i].resize(g.core_number);
    }
    // re_id.resize(g.n, g.n);
    if (re_id.size() < g.n)
        re_id.resize(g.n);
    std::fill(re_id.begin(), re_id.begin() + g.n, g.n);

    if (in_c.size() < g.max_d)
        in_c.resize(g.max_d, false);
    std::fill(in_c.begin(), in_c.begin() + g.max_d, false);

    // std::vector<uint32_t> new_clique_nei;
    // printf("a1\n");fflush(stdout);
    list_linear_heap heap(g.core_number, g.core_number);
    heap.reserve(g.core_number, g.core_number);

    id_s.resize(g.core_number + 1);
    degree.resize(g.core_number);
    removed.resize(g.core_number);

    p_idx.resize(g.core_number + 1);
    e_idx.resize(g.core_number + 1);
    p_edge.resize(g.core_number * g.core_number);

    clique.resize(g.core_number);
    nxt_c.resize(g.core_number);

    // std::vector<bool> is_in_x(g.n);
    for (uint32_t u = 0; u < g.n; u++) {
        if (should_stop_mce())
            break;
        stats.roots_seen++;
        now_u = u;
        // if(choosed_node[u]) continue;
        // if(must_contain_nodes[u] == false) continue;
        if (1 + g.p_idx[u + 1] - g.p_idx2[u] < threshold) {
            stats.roots_skipped_by_degree++;
            continue;
        }

        // is_in_x[u] = true;

        bool ok = must_contain_nodes[u];
        if (must_contain_nodes[u] == false)
            for (uint32_t i = g.p_idx2[u]; i < g.p_idx[u + 1]; i++) {
                if (must_contain_nodes[g.p_edge[i]]) {
                    ok = true;
                    break;
                }
            }
        if (!ok) {
            stats.roots_skipped_no_must++;
            continue;
        }

        for (uint32_t i = g.p_idx2[u]; i < g.p_idx[u + 1]; i++) {
            re_id[g.p_edge[i]] = i - g.p_idx2[u] + 1;
        }

        uint32_t n = g.p_idx[u + 1] - g.p_idx2[u];
        for (uint32_t i = 0; i <= n; i++) p_idx[i] = 0;

        for (uint32_t i = g.p_idx2[u]; i < g.p_idx[u + 1]; i++) {
            uint32_t v = g.p_edge[i];

            for (uint32_t j = g.p_idx2[v]; j < g.p_idx[v + 1]; j++) {
                uint32_t w = g.p_edge[j];
                if (re_id[w] < g.n) {
                    p_idx[i - g.p_idx2[u] + 1]++;
                    p_idx[re_id[w]]++;
                }
            }
        }
        for (uint32_t i = 1; i <= n; i++) p_idx[i] += p_idx[i - 1];
        for (uint32_t i = g.p_idx2[u]; i < g.p_idx[u + 1]; i++) {
            uint32_t v = g.p_edge[i];
            for (uint32_t j = g.p_idx2[v]; j < g.p_idx[v + 1]; j++) {
                uint32_t w = g.p_edge[j];
                if (re_id[w] < g.n) {
                    p_edge[p_idx[i - g.p_idx2[u]]++] = re_id[w] - 1;
                    p_edge[p_idx[re_id[w] - 1]++] = i - g.p_idx2[u];
                }
            }
        }
        for (uint32_t i = n; i > 0; i--) p_idx[i] = p_idx[i - 1];
        p_idx[0] = 0;

        clique.clear();
        nxt_c.clear();

        uint32_t queue_n = 0, new_size = 0;
        for (uint32_t i = 0; i < n; i++) removed[i] = false;
        for (uint32_t i = 0; i < n; i++) {
            if (p_idx[i + 1] - p_idx[i] + 2 < threshold)
                id_s[queue_n++] = i;
            degree[i] = p_idx[i + 1] - p_idx[i];
        }
        for (uint32_t i = 0; i < queue_n; i++) {
            uint32_t uu = id_s[i];
            degree[uu] = 0;
            removed[uu] = true;
            for (uint32_t j = p_idx[uu]; j < p_idx[uu + 1]; j++)
                if (degree[p_edge[j]] > 0) {
                    if ((degree[p_edge[j]]--) + 2 == threshold)
                        id_s[queue_n++] = p_edge[j];
                }
        }

        for (uint32_t i = 0; i < n; i++) {
            if (degree[i] + 2 >= threshold)
                id_s[queue_n + (new_size++)] = i;
            else {
                removed[i] = true;
            }
        }

        if (new_size == 0) {
            for (uint32_t i = g.p_idx2[u]; i < g.p_idx[u + 1]; i++) {
                re_id[g.p_edge[i]] = g.n;
            }
            continue;
        } else {
            //list_linear_heap *heap = new list_linear_heap(n, new_size-1);
            heap.init_raw(new_size, new_size - 1, id_s.data() + queue_n, degree.data());

            // printf("b");fflush(stdout);
            for (uint32_t i = 0; i < new_size; i++) {
                uint32_t v, key;

                heap.pop_min(v, key);

                id_s[queue_n + i] = v;
                if (key + i + 1 == new_size) {
                    uint32_t x_size = i + 1;
                    heap.get_ids(id_s.data() + queue_n, x_size);

                    assert(x_size == new_size);
                    // printf("	last key %u, x_size %u\n", key, x_size);
                    for (uint32_t j = i; j < new_size; j++) {
                        clique.push_back(g.p_edge[g.p_idx2[u] + id_s[queue_n + j]]);
                    }

                    for (uint32_t j = 0; j < i; j++) {
                        //gready extend
                        uint32_t tu = g.p_edge[g.p_idx2[u] + id_s[queue_n + j]];
                        nxt_c.push_back(tu);
                    }

                    break;
                }
                removed[v] = 1;
                // printf("%u %u\n", u, key);
                // fflush(stdout);

                for (uint32_t j = p_idx[v]; j < p_idx[v + 1]; j++)
                    if (removed[p_edge[j]] == false) {
                        heap.decrement(p_edge[j], 1);
                    }
            }
            // printf("c\n");fflush(stdout);
        }

        // printf("findClique\n");fflush(stdout);

        for (uint32_t i = g.p_idx[u]; i < g.p_idx[u + 1]; i++) {
            re_id[g.p_edge[i]] = i - g.p_idx[u];
        }
        uint32_t EC = 0;
        uint32_t EX = g.max_d, SX = g.max_d;
        uint32_t ER = 0;
        uint32_t EP = 0;
        for (uint32_t i = g.p_idx[u]; i < g.p_idx2[u]; i++) {
            // if(is_in_x[g.p_edge[i]])
            x_v3.change_to(re_id[g.p_edge[i]], --SX);
            // else
            //             p_v3.change_to(re_id[g.p_edge[i]], EP++);
        }
        for (uint32_t v : nxt_c) {
            p_v3.change_to(re_id[v], EP++);
        }
        for (uint32_t v : clique) {
            c_v3.change_to(re_id[v], EC++);
        }  //X nodes, the neighbors in P
        for (uint32_t i = g.p_idx[u]; i < g.p_idx2[u]; i++) {
            uint32_t v = g.p_edge[i];
            uint32_t v_id = i - g.p_idx[u];
            ed_adj[0][v_id] = 0;
            for (uint32_t j = g.p_idx2[v]; j < g.p_idx[v + 1]; j++) {
                uint32_t w = g.p_edge[j];
                if (re_id[w] == g.n)
                    continue;
                if (p_v3.is_in(re_id[w], 0, EP)) {
                    adj[v_id][ed_adj[0][v_id]++] = re_id[w];
                }
            }
        }
        //P/C nodes, the neighbors in P
        for (uint32_t i = g.p_idx2[u]; i < g.p_idx[u + 1]; i++) {
            uint32_t v_id = i - g.p_idx[u];
            ed_adj[0][v_id] = 0;
        }
        for (uint32_t v : nxt_c) {
            uint32_t v_id = re_id[v];
            for (uint32_t j = g.p_idx2[v]; j < g.p_idx[v + 1]; j++) {
                uint32_t w = g.p_edge[j];
                uint32_t w_id = re_id[w];
                if (re_id[w] == g.n)
                    continue;
                if (p_v3.is_in(w_id, 0, EP)) {
                    adj[v_id][ed_adj[0][v_id]++] = w_id;
                    adj[w_id][ed_adj[0][w_id]++] = re_id[v];
                } else if (c_v3.is_in(w_id, 0, EC)) {
                    adj[w_id][ed_adj[0][w_id]++] = re_id[v];
                }
            }
        }
        for (uint32_t v : clique) {
            uint32_t v_id = re_id[v];
            for (uint32_t j = g.p_idx2[v]; j < g.p_idx[v + 1]; j++) {
                uint32_t w = g.p_edge[j];
                uint32_t w_id = re_id[w];
                if (re_id[w] == g.n)
                    continue;
                if (p_v3.is_in(w_id, 0, EP)) {
                    adj[v_id][ed_adj[0][v_id]++] = w_id;
                }
            }
        }
        //X nodes, the neighbors in C
        for (uint32_t i = g.p_idx[u]; i < g.p_idx2[u]; i++) {
            uint32_t v = g.p_edge[i];
            uint32_t v_id = i - g.p_idx[u];
            st_adj[0][v_id] = g.core_number;

            for (uint32_t j = g.p_idx2[v]; j < g.p_idx[v + 1]; j++) {
                uint32_t w = g.p_edge[j];
                if (re_id[w] == g.n)
                    continue;
                if (c_v3.is_in(re_id[w], 0, EC)) {
                    adj[v_id][--st_adj[0][v_id]] = re_id[w];
                }
            }
        }
        //P/C nodes, the neighbors in C
        for (uint32_t v : nxt_c) {
            uint32_t v_id = re_id[v];
            st_adj[0][v_id] = g.core_number;
        }
        for (uint32_t v : clique) {
            uint32_t v_id = re_id[v];
            st_adj[0][v_id] = g.core_number;
        }
        for (uint32_t v : nxt_c) {
            uint32_t v_id = re_id[v];
            for (uint32_t j = g.p_idx2[v]; j < g.p_idx[v + 1]; j++) {
                uint32_t w = g.p_edge[j];
                if (re_id[w] == g.n)
                    continue;
                if (c_v3.is_in(re_id[w], 0, EC)) {
                    adj[v_id][--st_adj[0][v_id]] = re_id[w];
                }
            }
        }
        for (uint32_t w : clique) {
            uint32_t w_id = re_id[w];
            in_c[w_id] = true;
            for (uint32_t j = g.p_idx2[w]; j < g.p_idx[w + 1]; j++) {
                uint32_t v = g.p_edge[j];
                uint32_t v_id = re_id[v];
                if (v_id == g.n)
                    continue;
                if (p_v3.is_in(v_id, 0, EP)) {
                    adj[v_id][--st_adj[0][v_id]] = w_id;
                }
                // else if(c_v3.is_in(v_id, 0, EC)) {
                //     adj[v_id][--st_adj[0][v_id]] = w_id;
                //     adj[w_id][--st_adj[0][w_id]] = v_id;
                // }
            }
        }
        num_of_non_c_nodes = 1;
        bk_tomita_rec_ccr_v3_must_contain(0, ER, EC, EP, SX, EX, must_contain_nodes);
        if (stop_search)
            break;
        for (uint32_t w : clique) {
            uint32_t w_id = re_id[w];
            in_c[w_id] = false;
        }

        for (uint32_t i = g.p_idx[u]; i < g.p_idx[u + 1]; i++) {
            re_id[g.p_edge[i]] = g.n;
        }
    }
    return cnt;
}

void
mce::bk_tomita_rec_ccr_v3_must_contain(uint32_t deep,
                                       uint32_t ER,
                                       uint32_t EC,
                                       uint32_t EP,
                                       uint32_t SX,
                                       uint32_t EX,
                                       std::vector<bool>& must_contain_nodes) {
    stats.recursion_calls++;
    if (should_stop_mce())
        return;
    if (1 + ER + EC + EP < threshold) {
        stats.prune_size_bound++;
        return;
    }
    if (!state_has_must(ER, EC, EP, must_contain_nodes)) {
        stats.prune_no_must++;
        return;
    }

    bool f = true;
    std::vector<uint32_t>& ed = ed_adj[deep];
    std::vector<uint32_t>& st = st_adj[deep];
    std::vector<uint32_t>& nxt_ed = ed_adj[deep + 1];
    std::vector<uint32_t>& nxt_st = st_adj[deep + 1];

    // if(num_of_non_c_nodes == 0 && deep > 0) f = false;
    if (num_of_non_c_nodes == 0) {
        f = false;
    }
    if (f && EX - SX + EP > 0) {
        for (uint32_t i = 0; i < EP; i++) {
            uint32_t v = p_v3[i];
            if (g.core_number - st[v] == EC) {
                f = false;
                break;
            }
        }
        if (f) {
            for (uint32_t i = SX; i < EX; i++) {
                uint32_t v = x_v3[i];
                if (in_c[v] || g.core_number - st[v] == EC) {
                    f = false;
                    break;
                }
            }
        }
    }
    if (f) {  // if(out_p) {
        // printf("%u ", now_u);
        // for(uint32_t i = 0; i < EC; i++) printf("%u ", g.p_edge[g.p_idx[now_u]+c_v3[i]]);
        // for(uint32_t i = 0; i < ER; i++) printf("%u ", g.p_edge[g.p_idx[now_u]+r_v3[i]]);
        // printf("\n");

        try_save_must_clique(ER, EC, must_contain_nodes);

        // }
        cnt++;
        if (stop_search)
            return;
    }

    bool contain_must = false;
    for (uint32_t i = 0; i < EC; i++) {
        uint32_t u = g.p_edge[g.p_idx[now_u] + c_v3[i]];
        if (must_contain_nodes[u]) {
            contain_must = true;
            break;
        }
    }
    if (contain_must == false)
        for (uint32_t i = 0; i < ER; i++) {
            uint32_t u = g.p_edge[g.p_idx[now_u] + r_v3[i]];
            if (must_contain_nodes[u]) {
                contain_must = true;
                break;
            }
        }
    if (contain_must == false)
        for (uint32_t i = 0; i < EP; i++) {
            uint32_t u = g.p_edge[g.p_idx[now_u] + p_v3[i]];
            if (must_contain_nodes[u]) {
                contain_must = true;
                break;
            }
        }
    if (contain_must == false) {
        stats.prune_no_must++;
        return;
    }

    // printf("C:"); for(uint32_t i = 0; i < EC; i++) printf("%u ", c_v3[i]); printf("\n");
    // printf("R:"); for(uint32_t i = 0; i < ER; i++) printf("%u ", r_v3[i]); printf("\n");
    // printf("P:"); for(uint32_t i = 0; i < EP; i++) printf("%u ", p_v3[i]); printf("\n");

    if (EP == 0)
        return;

    if (EP == 1) {
        uint32_t u = p_v3[0];
        num_of_non_c_nodes = 1;
        r_v3.change_to(u, ER);
        uint32_t new_ec = 0;
        for (uint32_t i = st[u]; i < g.core_number; i++) {
            c_v3.change_to(adj[u][i], new_ec++);
        }
        uint32_t new_ex = SX;
        for (uint32_t i = SX; i < EX; i++) {
            uint32_t v = x_v3[i];

            bool cc = false;
            for (uint32_t j = 0; j < ed[v]; j++) {
                uint32_t w = adj[v][j];
                if (w == u) {
                    cc = true;
                    break;
                }
            }
            if (cc)
                x_v3.change_to(v, new_ex++);
        }
        auto main_nei2 = [&](uint32_t v) {
            uint32_t& nst = st_adj[deep + 1][v];
            uint32_t& ned = ed_adj[deep + 1][v];
            nst = g.core_number;
            ned = 0;
            if (new_ec > 0)
                for (uint32_t j = st[v]; j < nst;) {
                    if (c_v3.is_in(adj[v][j], 0, new_ec)) {
                        std::swap(adj[v][j], adj[v][--nst]);
                    } else
                        j++;
                }
        };

        for (uint32_t i = SX; i < new_ex; i++) {
            uint32_t v = x_v3[i];
            if (in_c[v])
                continue;
            main_nei2(v);
        }

        bk_tomita_rec_ccr_v3_must_contain(
            deep + 1, ER + 1, new_ec, 0, SX, new_ex, must_contain_nodes);
        //r_v3.change_to
        return;
    }

    //remove the X nodes that has no neighbors in P
    for (uint32_t i = SX; i < EX;) {
        uint32_t v = x_v3[i];
        if (ed[v] == 0) {
            x_v3.change_to(v, --EX);
        } else
            i++;
    }

    //remove the C nodes that connects to all nodes in P
    bool update = false;
    for (uint32_t i = 0; i < EC;) {
        uint32_t v = c_v3[i];
        if (ed[v] == EP) {
            c_v3.change_to(v, --EC);
            r_v3.change_to(v, ER++);
            update = true;
            for (uint32_t i = SX; i < EX;) {
                uint32_t vv = x_v3[i];

                if (in_c[vv]) {
                    i++;
                    continue;
                }

                bool cc = false;
                for (uint32_t j = st[vv]; j < g.core_number; j++) {
                    uint32_t w = adj[vv][j];
                    if (w == v) {
                        cc = true;
                        break;
                    }
                }
                if (cc)
                    i++;
                else
                    x_v3.change_to(vv, --EX);
            }
        } else
            i++;
    }
    if (update) {
        auto main_nei_in_c = [&](uint32_t v) {
            uint32_t nst = g.core_number;

            if (EC > 0) {
                for (uint32_t j = st[v]; j < nst;) {
                    if (c_v3.is_in(adj[v][j], 0, EC)) {
                        std::swap(adj[v][j], adj[v][--nst]);
                    } else
                        j++;
                }
            }

            st[v] = nst;
        };

        for (uint32_t i = 0; i < EP; i++) {
            uint32_t v = p_v3[i];
            main_nei_in_c(v);
        }
        for (uint32_t i = SX; i < EX; i++) {
            uint32_t v = x_v3[i];
            if (in_c[v])
                continue;
            main_nei_in_c(v);
        }
        // for(uint32_t i = 0; i < EC; i++) {
        //     uint32_t v = c_v3[i]; main_nei_in_c(v);
        // }
    }

    //remove the P nodes that connects to all PC nodes
    update = false;
    for (uint32_t i = 0, raw_ep = EP; i < EP;) {
        uint32_t v = p_v3[i];
        if (ed[v] == raw_ep - 1 && g.core_number - st[v] == EC) {
            p_v3.change_to(v, --EP);
            r_v3.change_to(v, ER++);
            update = true;
            for (uint32_t i = SX; i < EX;) {
                uint32_t vv = x_v3[i];
                bool cc = false;
                for (uint32_t j = 0; j < ed[vv]; j++) {
                    uint32_t w = adj[vv][j];
                    if (w == v) {
                        cc = true;
                        break;
                    }
                }
                if (cc)
                    i++;
                else
                    x_v3.change_to(vv, --EX);
            }
        } else {
            i++;
        }
    }
    if (update) {
        auto main_nei_in_p = [&](uint32_t v) {
            for (uint32_t j = 0; j < ed[v];) {
                if (p_v3.is_in(adj[v][j], 0, EP)) {
                    j++;
                } else
                    std::swap(adj[v][j], adj[v][--ed[v]]);
            }
        };

        for (uint32_t i = 0; i < EP; i++) {
            uint32_t v = p_v3[i];
            main_nei_in_p(v);
        }
        for (uint32_t i = SX; i < EX; i++) {
            uint32_t v = x_v3[i];
            main_nei_in_p(v);
        }
        for (uint32_t i = 0; i < EC; i++) {
            uint32_t v = c_v3[i];
            main_nei_in_p(v);
        }

        bool f = true;
        for (uint32_t i = 0; i < EP; i++) {
            uint32_t v = p_v3[i];
            if (g.core_number - st[v] == EC) {
                f = false;
                break;
            }
        }
        if (f)
            for (uint32_t i = SX; i < EX; i++) {
                uint32_t v = x_v3[i];
                if (in_c[v] || g.core_number - st[v] == EC) {
                    f = false;
                    break;
                }
            }
        if (f) {  // if(out_p) {
            // printf("%u ", now_u);
            // for(uint32_t i = 0; i < EC; i++) printf("%u ", g.p_edge[g.p_idx[now_u]+c_v3[i]]);
            // for(uint32_t i = 0; i < ER; i++) printf("%u ", g.p_edge[g.p_idx[now_u]+r_v3[i]]);
            // printf("\n");

            try_save_must_clique(ER, EC, must_contain_nodes);
            // }
            cnt++;
            if (stop_search)
                return;
        }

        if (EP == 0)
            return;
    }

    //find the best pivot
    uint32_t pu = p_v3[0], max_d = 0;
    for (uint32_t i = 0; i < EP; i++) {
        uint32_t v = p_v3[i];
        uint32_t d = ed[v] + g.core_number - st[v];
        if (d > max_d) {
            max_d = d;
            pu = v;
        }
    }
    for (uint32_t i = 0; i < EC; i++) {
        uint32_t v = c_v3[i];
        // uint32_t d = ed[v] + g.core_number-st[v];
        uint32_t d = ed[v] + EC - 1;
        if (d > max_d) {
            max_d = d;
            pu = v;
        }
    }
    for (uint32_t i = SX; i < EX; i++) {
        uint32_t v = x_v3[i];
        uint32_t nei_in_c = in_c[v] ? EC - 1 : g.core_number - st[v];
        uint32_t d = ed[v] + nei_in_c;
        if (d > max_d) {
            max_d = d;
            pu = v;
        }
        // continue;
        if (ed[v] == EP - 1 && nei_in_c == EC) {
            //remove the X nodes that has one-non-neighbor in P

            for (uint32_t i = 0, ttt = 0; i < ed[v]; i++) {
                uint32_t tmpv = adj[v][i];
                p_v3.change_to(tmpv, ttt++);
            }
            uint32_t u = p_v3[EP - 1];

            uint32_t new_ep = 0;
            uint32_t new_ex = SX;
            uint32_t new_ec = 0;  //build new P
            for (uint32_t i = 0; i < ed[u]; i++) {
                uint32_t v = adj[u][i];
                if (p_v3.isin(v, 0, EP))
                    p_v3.change_to(v, new_ep++);
            }
            //build new C
            for (uint32_t i = st[u]; i < g.core_number; i++) {
                uint32_t v = adj[u][i];
                if (c_v3.isin(v, 0, EC))
                    c_v3.change_to(v, new_ec++);
            }  //build new X
            num_of_non_c_nodes = 1;
            for (uint32_t i = SX; i < EX; i++) {
                uint32_t v = x_v3[i];
                bool cc = false;
                for (uint32_t j = 0; j < ed[v]; j++) {
                    uint32_t w = adj[v][j];
                    if (w == u) {
                        cc = true;
                        break;
                    }
                }
                if (cc)
                    x_v3.change_to(v, new_ex++);
            }
            // R.push_back(u);

            //maintain the neighbors in P / C
            auto main_nei = [&](uint32_t v) {
                uint32_t& ned = nxt_ed[v];
                // nst = g.core_number;
                // ned = 0;
                // if(EP > 0)
                // for(uint32_t j = 0; j < ed[v]; j++) {
                //     if(p_v3.is_in(adj[v][j], 0, new_ep)) {
                //         std::swap(adj[v][j], adj[v][ned++]);
                //     }
                // }
                // if(EC > 0)
                // for(uint32_t j = st[v]; j < nst; ) {
                //     if(c_v3.is_in(adj[v][j], 0, new_ec)) {
                //         std::swap(adj[v][j], adj[v][--nst]);
                //     }
                //     else j++;
                // }

                ned = ed[v];

                for (uint32_t j = 0; j < ned;) {
                    if (p_v3.is_in(adj[v][j], 0, new_ep)) {
                        j++;
                    } else
                        std::swap(adj[v][j], adj[v][--ned]);
                }

                if (in_c[v])
                    return;

                uint32_t& nst = nxt_st[v];
                nst = st[v];
                for (uint32_t j = st[v]; j < g.core_number; j++) {
                    if (!c_v3.is_in(adj[v][j], 0, new_ec)) {
                        std::swap(adj[v][j], adj[v][nst++]);
                    }
                }
            };

            for (uint32_t i = 0; i < new_ep; i++) {
                uint32_t v = p_v3[i];
                main_nei(v);
            }
            for (uint32_t i = SX; i < new_ex; i++) {
                uint32_t v = x_v3[i];
                main_nei(v);
            }
            for (uint32_t i = 0; i < new_ec; i++) {
                uint32_t v = c_v3[i];
                main_nei(v);
            }

            r_v3.change_to(u, ER);

            bk_tomita_rec_ccr_v3_must_contain(
                deep + 1, ER + 1, new_ec, new_ep, SX, new_ex, must_contain_nodes);

            num_of_non_c_nodes--;

            return;
        }
    }
    std::vector<uint32_t> non_neighbors(EC + EP - max_d + 1);
    uint32_t d = 0;
    uint32_t new_ep = 0;
    for (uint32_t i = 0; i < ed[pu]; i++) {
        uint32_t v = adj[pu][i];
        p_v3.change_to(v, new_ep++);
    }
    for (uint32_t i = new_ep; i < EP; i++) {
        non_neighbors[d++] = p_v3[i];
    }
    uint32_t new_ec = 0;
    if (c_v3.is_in(pu, 0, EC)) {
        new_ec = EC - 1;
        c_v3.change_to(pu, EC - 1);
    } else if (in_c[pu]) {
        new_ec = EC;
    } else {
        for (uint32_t i = st[pu]; i < g.core_number; i++) {
            uint32_t v = adj[pu][i];
            c_v3.change_to(v, new_ec++);
        }
    }
    for (uint32_t i = new_ec, dc = 0; i < EC; i++) {
        uint32_t v = c_v3[i];

        //与pu的邻居相连
        bool f = false;
        for (uint32_t j = 0; j < ed[v]; j++) {
            uint32_t w = adj[v][j];
            if (p_v3.is_in(w, 0, new_ep)) {
                f = true;
                break;
            }
        }
        if (!f) {
            continue;
        }
        // f = false;
        // //包含前面C点的P中的非邻居
        // for(uint32_t j = d-dc; j < d; j++) {
        //     uint32_t pre_cv = non_neighbors[j];

        // }

        non_neighbors[d++] = v;
        dc++;
    }

    for (uint32_t ii = 0; ii < d; ii++) {
        if (should_stop_mce())
            return;
        uint32_t u = non_neighbors[ii];
        uint32_t new_ep = 0;
        uint32_t new_ex = SX;
        uint32_t new_ec = 0;  //build new P
        for (uint32_t i = 0; i < ed[u]; i++) {
            uint32_t v = adj[u][i];
            if (p_v3.isin(v, 0, EP))
                p_v3.change_to(v, new_ep++);
        }
        //build new C
        if (p_v3.is_in(u, 0, EP))
            for (uint32_t i = st[u]; i < g.core_number; i++) {
                uint32_t v = adj[u][i];
                if (c_v3.isin(v, 0, EC))
                    c_v3.change_to(v, new_ec++);
            }
        else {
            c_v3.change_to(u, EC - 1);
            new_ec = EC - 1;
        }  //build new X
        if (p_v3.is_in(u, 0, EP)) {
            num_of_non_c_nodes = 1;
            for (uint32_t i = SX; i < EX; i++) {
                uint32_t v = x_v3[i];
                bool cc = false;
                for (uint32_t j = 0; j < ed[v]; j++) {
                    uint32_t w = adj[v][j];
                    if (w == u) {
                        cc = true;
                        break;
                    }
                }
                if (cc)
                    x_v3.change_to(v, new_ex++);
            }
        } else {
            num_of_non_c_nodes = 0;
            for (uint32_t i = SX; i < EX; i++) {
                uint32_t v = x_v3[i];

                if (in_c[v]) {
                    x_v3.change_to(v, new_ex++);
                    continue;
                }

                bool cc = false;
                for (uint32_t j = st[v]; j < g.core_number; j++) {
                    uint32_t w = adj[v][j];
                    if (w == u) {
                        cc = true;
                        break;
                    }
                }
                if (cc) {
                    x_v3.change_to(v, new_ex++);
                }
            }
        }
        // R.push_back(u);

        //maintain the neighbors in P / C
        auto main_nei = [&](uint32_t v) {
            uint32_t& nst = nxt_st[v];
            uint32_t& ned = nxt_ed[v];
            nst = g.core_number;
            ned = 0;

            if (EP > 0)
                for (uint32_t j = 0; j < ed[v]; j++) {
                    if (p_v3.is_in(adj[v][j], 0, new_ep)) {
                        std::swap(adj[v][j], adj[v][ned++]);
                    }
                }

            if (in_c[v])
                return;

            if (EC > 0)
                for (uint32_t j = st[v]; j < nst;) {
                    if (c_v3.is_in(adj[v][j], 0, new_ec)) {
                        std::swap(adj[v][j], adj[v][--nst]);
                    } else
                        j++;
                }
        };

        for (uint32_t i = 0; i < new_ep; i++) {
            uint32_t v = p_v3[i];
            main_nei(v);
        }
        for (uint32_t i = SX; i < new_ex; i++) {
            uint32_t v = x_v3[i];
            main_nei(v);
        }
        for (uint32_t i = 0; i < new_ec; i++) {
            uint32_t v = c_v3[i];  //main_nei(v);

            uint32_t& ned = nxt_ed[v];
            ned = 0;
            if (EP > 0)
                for (uint32_t j = 0; j < ed[v]; j++) {
                    if (p_v3.is_in(adj[v][j], 0, new_ep)) {
                        std::swap(adj[v][j], adj[v][ned++]);
                    }
                }
        }

        r_v3.change_to(u, ER);

        bk_tomita_rec_ccr_v3_must_contain(
            deep + 1, ER + 1, new_ec, new_ep, SX, new_ex, must_contain_nodes);
        if (stop_search)
            return;

        x_v3.change_to(u, --SX);
        if (p_v3.is_in(u, 0, EP)) {
            num_of_non_c_nodes--;
            p_v3.change_to(u, --EP);
        } else {
            --EC;

            // c_v3.change_to(u, --EC);
        }
    }

    for (uint32_t ii = 0; ii < d; ii++) {
        x_v3.change_to(non_neighbors[ii], SX++);
        // if(in_c[non_neighbors[ii]]) {
        //     c_v3.change_to(non_neighbors[ii], EC++);
        // }
    }
}

uint32_t
mce::run() {
    // printf("mce::runCoreCliqueRemovalV3, core %u\n", g.core_number);fflush(stdout);
    c_v3.resize(g.max_d);
    p_v3.resize(g.max_d);
    r_v3.resize(g.max_d);
    x_v3.resize(g.max_d);

    cnt = 0;
    // printf("a0\n");fflush(stdout);
    ed_adj.resize(g.core_number + 1);
    for (uint32_t i = 0; i < g.core_number; i++) ed_adj[i].resize(g.max_d);
    st_adj.resize(g.core_number + 1);
    for (uint32_t i = 0; i < g.core_number; i++) st_adj[i].resize(g.max_d);
    adj.resize(g.max_d);
    for (uint32_t i = 0; i < g.max_d; i++) {
        adj[i].resize(g.core_number);
    }
    // re_id.resize(g.n, g.n);
    if (re_id.size() < g.n)
        re_id.resize(g.n);
    std::fill(re_id.begin(), re_id.begin() + g.n, g.n);

    if (in_c.size() < g.max_d)
        in_c.resize(g.max_d, false);
    std::fill(in_c.begin(), in_c.begin() + g.max_d, false);

    // std::vector<uint32_t> new_clique_nei;
    // printf("a1\n");fflush(stdout);
    list_linear_heap heap(g.core_number, g.core_number);
    heap.reserve(g.core_number, g.core_number);

    std::vector<uint32_t> id_s(g.core_number + 1);
    std::vector<uint32_t> degree(g.core_number);
    std::vector<bool> removed(g.core_number);

    std::vector<uint32_t> p_idx(g.core_number + 1);
    std::vector<uint32_t> e_idx(g.core_number + 1);
    std::vector<uint32_t> p_edge(g.core_number * g.core_number);

    std::vector<uint32_t> clique(g.core_number);
    std::vector<uint32_t> nxt_c(g.core_number);

    for (uint32_t u = 0; u < g.n; u++) {
        now_u = u;
        if (1 + g.p_idx[u + 1] - g.p_idx2[u] < threshold)
            continue;
        // printf("%u %u\n", u, g.n);fflush(stdout);
        for (uint32_t i = g.p_idx2[u]; i < g.p_idx[u + 1]; i++) {
            re_id[g.p_edge[i]] = i - g.p_idx2[u] + 1;
        }

        uint32_t n = g.p_idx[u + 1] - g.p_idx2[u];
        for (uint32_t i = 0; i <= n; i++) p_idx[i] = 0;

        for (uint32_t i = g.p_idx2[u]; i < g.p_idx[u + 1]; i++) {
            uint32_t v = g.p_edge[i];

            for (uint32_t j = g.p_idx2[v]; j < g.p_idx[v + 1]; j++) {
                uint32_t w = g.p_edge[j];
                if (re_id[w] < g.n) {
                    p_idx[i - g.p_idx2[u] + 1]++;
                    p_idx[re_id[w]]++;
                }
            }
        }
        for (uint32_t i = 1; i <= n; i++) p_idx[i] += p_idx[i - 1];
        for (uint32_t i = g.p_idx2[u]; i < g.p_idx[u + 1]; i++) {
            uint32_t v = g.p_edge[i];
            for (uint32_t j = g.p_idx2[v]; j < g.p_idx[v + 1]; j++) {
                uint32_t w = g.p_edge[j];
                if (re_id[w] < g.n) {
                    p_edge[p_idx[i - g.p_idx2[u]]++] = re_id[w] - 1;
                    p_edge[p_idx[re_id[w] - 1]++] = i - g.p_idx2[u];
                }
            }
        }
        for (uint32_t i = n; i > 0; i--) p_idx[i] = p_idx[i - 1];
        p_idx[0] = 0;

        clique.clear();
        nxt_c.clear();

        uint32_t queue_n = 0, new_size = 0;
        for (uint32_t i = 0; i < n; i++) removed[i] = false;
        for (uint32_t i = 0; i < n; i++) {
            if (p_idx[i + 1] - p_idx[i] + 2 < threshold)
                id_s[queue_n++] = i;
            degree[i] = p_idx[i + 1] - p_idx[i];
        }
        for (uint32_t i = 0; i < queue_n; i++) {
            uint32_t uu = id_s[i];
            degree[uu] = 0;
            removed[uu] = true;
            for (uint32_t j = p_idx[uu]; j < p_idx[uu + 1]; j++)
                if (degree[p_edge[j]] > 0) {
                    if ((degree[p_edge[j]]--) + 2 == threshold)
                        id_s[queue_n++] = p_edge[j];
                }
        }

        for (uint32_t i = 0; i < n; i++) {
            if (degree[i] + 2 >= threshold)
                id_s[queue_n + (new_size++)] = i;
            else {
                removed[i] = true;
            }
        }

        if (new_size == 0) {
            for (uint32_t i = g.p_idx2[u]; i < g.p_idx[u + 1]; i++) {
                re_id[g.p_edge[i]] = g.n;
            }
            continue;
        } else {
            //list_linear_heap *heap = new list_linear_heap(n, new_size-1);
            heap.init_raw(new_size, new_size - 1, id_s.data() + queue_n, degree.data());

            // printf("b");fflush(stdout);
            for (uint32_t i = 0; i < new_size; i++) {
                uint32_t v, key;

                heap.pop_min(v, key);

                id_s[queue_n + i] = v;
                if (key + i + 1 == new_size) {
                    uint32_t x_size = i + 1;
                    heap.get_ids(id_s.data() + queue_n, x_size);

                    assert(x_size == new_size);
                    // printf("	last key %u, x_size %u\n", key, x_size);
                    for (uint32_t j = i; j < new_size; j++) {
                        clique.push_back(g.p_edge[g.p_idx2[u] + id_s[queue_n + j]]);
                    }

                    for (uint32_t j = 0; j < i; j++) {
                        //gready extend
                        uint32_t tu = g.p_edge[g.p_idx2[u] + id_s[queue_n + j]];
                        // bool connect_all = true;
                        // for(uint32_t cu : clique) {
                        //     if(cu < tu) {
                        //         if(!g.connect_out(cu, tu)) {
                        //             connect_all = false; break;
                        //         }
                        //     }
                        //     else if(!g.connect_out(tu, cu)) {
                        //         connect_all = false; break;
                        //     }
                        // }
                        // if(connect_all)  clique.push_back(tu);
                        // else
                        nxt_c.push_back(tu);
                    }

                    break;
                }
                removed[v] = 1;
                // printf("%u %u\n", u, key);
                // fflush(stdout);

                for (uint32_t j = p_idx[v]; j < p_idx[v + 1]; j++)
                    if (removed[p_edge[j]] == false) {
                        heap.decrement(p_edge[j], 1);
                    }
            }
            // printf("c\n");fflush(stdout);
        }

        // printf("findClique\n");fflush(stdout);

        for (uint32_t i = g.p_idx[u]; i < g.p_idx[u + 1]; i++) {
            re_id[g.p_edge[i]] = i - g.p_idx[u];
        }
        uint32_t EC = 0;
        uint32_t EX = g.max_d, SX = g.max_d;
        uint32_t ER = 0;
        uint32_t EP = 0;
        for (uint32_t i = g.p_idx[u]; i < g.p_idx2[u]; i++) {
            x_v3.change_to(re_id[g.p_edge[i]], --SX);
        }
        for (uint32_t v : nxt_c) {
            p_v3.change_to(re_id[v], EP++);
        }
        for (uint32_t v : clique) {
            c_v3.change_to(re_id[v], EC++);
        }  //X nodes, the neighbors in P
        for (uint32_t i = g.p_idx[u]; i < g.p_idx2[u]; i++) {
            uint32_t v = g.p_edge[i];
            uint32_t v_id = i - g.p_idx[u];
            ed_adj[0][v_id] = 0;
            for (uint32_t j = g.p_idx2[v]; j < g.p_idx[v + 1]; j++) {
                uint32_t w = g.p_edge[j];
                if (re_id[w] == g.n)
                    continue;
                if (p_v3.is_in(re_id[w], 0, EP)) {
                    adj[v_id][ed_adj[0][v_id]++] = re_id[w];
                }
            }
        }
        //P/C nodes, the neighbors in P
        for (uint32_t i = g.p_idx2[u]; i < g.p_idx[u + 1]; i++) {
            uint32_t v_id = i - g.p_idx[u];
            ed_adj[0][v_id] = 0;
        }
        for (uint32_t v : nxt_c) {
            uint32_t v_id = re_id[v];
            for (uint32_t j = g.p_idx2[v]; j < g.p_idx[v + 1]; j++) {
                uint32_t w = g.p_edge[j];
                uint32_t w_id = re_id[w];
                if (re_id[w] == g.n)
                    continue;
                if (p_v3.is_in(w_id, 0, EP)) {
                    adj[v_id][ed_adj[0][v_id]++] = w_id;
                    adj[w_id][ed_adj[0][w_id]++] = re_id[v];
                } else if (c_v3.is_in(w_id, 0, EC)) {
                    adj[w_id][ed_adj[0][w_id]++] = re_id[v];
                }
            }
        }
        for (uint32_t v : clique) {
            uint32_t v_id = re_id[v];
            for (uint32_t j = g.p_idx2[v]; j < g.p_idx[v + 1]; j++) {
                uint32_t w = g.p_edge[j];
                uint32_t w_id = re_id[w];
                if (re_id[w] == g.n)
                    continue;
                if (p_v3.is_in(w_id, 0, EP)) {
                    adj[v_id][ed_adj[0][v_id]++] = w_id;
                }
            }
        }
        //X nodes, the neighbors in C
        for (uint32_t i = g.p_idx[u]; i < g.p_idx2[u]; i++) {
            uint32_t v = g.p_edge[i];
            uint32_t v_id = i - g.p_idx[u];
            st_adj[0][v_id] = g.core_number;

            for (uint32_t j = g.p_idx2[v]; j < g.p_idx[v + 1]; j++) {
                uint32_t w = g.p_edge[j];
                if (re_id[w] == g.n)
                    continue;
                if (c_v3.is_in(re_id[w], 0, EC)) {
                    adj[v_id][--st_adj[0][v_id]] = re_id[w];
                }
            }
        }
        //P/C nodes, the neighbors in C
        for (uint32_t v : nxt_c) {
            uint32_t v_id = re_id[v];
            st_adj[0][v_id] = g.core_number;
        }
        for (uint32_t v : clique) {
            uint32_t v_id = re_id[v];
            st_adj[0][v_id] = g.core_number;
        }
        for (uint32_t v : nxt_c) {
            uint32_t v_id = re_id[v];
            for (uint32_t j = g.p_idx2[v]; j < g.p_idx[v + 1]; j++) {
                uint32_t w = g.p_edge[j];
                if (re_id[w] == g.n)
                    continue;
                if (c_v3.is_in(re_id[w], 0, EC)) {
                    adj[v_id][--st_adj[0][v_id]] = re_id[w];
                }
            }
        }
        for (uint32_t w : clique) {
            uint32_t w_id = re_id[w];
            in_c[w_id] = true;
            for (uint32_t j = g.p_idx2[w]; j < g.p_idx[w + 1]; j++) {
                uint32_t v = g.p_edge[j];
                uint32_t v_id = re_id[v];
                if (v_id == g.n)
                    continue;
                if (p_v3.is_in(v_id, 0, EP)) {
                    adj[v_id][--st_adj[0][v_id]] = w_id;
                }
                // else if(c_v3.is_in(v_id, 0, EC)) {
                //     adj[v_id][--st_adj[0][v_id]] = w_id;
                //     adj[w_id][--st_adj[0][w_id]] = v_id;
                // }
            }
        }
        num_of_non_c_nodes = 1;
        bk_tomita_rec_ccr_v3(0, ER, EC, EP, SX, EX);
        for (uint32_t w : clique) {
            uint32_t w_id = re_id[w];
            in_c[w_id] = false;
        }

        for (uint32_t i = g.p_idx[u]; i < g.p_idx[u + 1]; i++) {
            re_id[g.p_edge[i]] = g.n;
        }
    }
    return cnt;
}
void
mce::bk_tomita_rec_ccr_v3(
    uint32_t deep, uint32_t ER, uint32_t EC, uint32_t EP, uint32_t SX, uint32_t EX) {
    bool f = true;
    std::vector<uint32_t>& ed = ed_adj[deep];
    std::vector<uint32_t>& st = st_adj[deep];
    std::vector<uint32_t>& nxt_ed = ed_adj[deep + 1];
    std::vector<uint32_t>& nxt_st = st_adj[deep + 1];

    // if(num_of_non_c_nodes == 0 && deep > 0) f = false;
    if (num_of_non_c_nodes == 0) {
        f = false;
    }
    if (f && EX - SX + EP > 0) {
        for (uint32_t i = 0; i < EP; i++) {
            uint32_t v = p_v3[i];
            if (g.core_number - st[v] == EC) {
                f = false;
                break;
            }
        }
        if (f) {
            for (uint32_t i = SX; i < EX; i++) {
                uint32_t v = x_v3[i];
                if (in_c[v] || g.core_number - st[v] == EC) {
                    f = false;
                    break;
                }
            }
        }
    }
    if (f) {  // if(out_p) {
        // printf("%u ", now_u);
        // for(uint32_t i = 0; i < EC; i++) printf("%u ", g.p_edge[g.p_idx[now_u]+c_v3[i]]);
        // for(uint32_t i = 0; i < ER; i++) printf("%u ", g.p_edge[g.p_idx[now_u]+r_v3[i]]);
        // printf("\n");
        if (1 + EC + ER >= threshold) {
            if (max_clique_cnt_save >= (*max_cliques).size()) {
                (*max_cliques).resize((max_clique_cnt_save + 1) * 2);
            }
            (*max_cliques)[max_clique_cnt_save].clear();
            (*max_cliques)[max_clique_cnt_save].push_back(g.mp[now_u]);
            for (uint32_t i = 0; i < EC; i++) {
                uint32_t u = g.p_edge[g.p_idx[now_u] + c_v3[i]];
                (*max_cliques)[max_clique_cnt_save].push_back(g.mp[u]);
            }
            for (uint32_t i = 0; i < ER; i++) {
                uint32_t u = g.p_edge[g.p_idx[now_u] + r_v3[i]];
                (*max_cliques)[max_clique_cnt_save].push_back(g.mp[u]);
            }
            max_clique_cnt_save++;
        }
        // }
        cnt++;
    }

    if (EP == 0)
        return;

    if (EP == 1) {
        uint32_t u = p_v3[0];
        num_of_non_c_nodes = 1;
        r_v3.change_to(u, ER);
        uint32_t new_ec = 0;
        for (uint32_t i = st[u]; i < g.core_number; i++) {
            c_v3.change_to(adj[u][i], new_ec++);
        }
        uint32_t new_ex = SX;
        for (uint32_t i = SX; i < EX; i++) {
            uint32_t v = x_v3[i];

            bool cc = false;
            for (uint32_t j = 0; j < ed[v]; j++) {
                uint32_t w = adj[v][j];
                if (w == u) {
                    cc = true;
                    break;
                }
            }
            if (cc)
                x_v3.change_to(v, new_ex++);
        }
        auto main_nei2 = [&](uint32_t v) {
            uint32_t& nst = st_adj[deep + 1][v];
            uint32_t& ned = ed_adj[deep + 1][v];
            nst = g.core_number;
            ned = 0;
            if (new_ec > 0)
                for (uint32_t j = st[v]; j < nst;) {
                    if (c_v3.is_in(adj[v][j], 0, new_ec)) {
                        std::swap(adj[v][j], adj[v][--nst]);
                    } else
                        j++;
                }
        };

        for (uint32_t i = SX; i < new_ex; i++) {
            uint32_t v = x_v3[i];
            if (in_c[v])
                continue;
            main_nei2(v);
        }

        bk_tomita_rec_ccr_v3(deep + 1, ER + 1, new_ec, 0, SX, new_ex);
        //r_v3.change_to
        return;
    }

    //remove the X nodes that has no neighbors in P
    for (uint32_t i = SX; i < EX;) {
        uint32_t v = x_v3[i];
        if (ed[v] == 0) {
            x_v3.change_to(v, --EX);
        } else
            i++;
    }

    //remove the C nodes that connects to all nodes in P
    bool update = false;
    for (uint32_t i = 0; i < EC;) {
        uint32_t v = c_v3[i];
        if (ed[v] == EP) {
            c_v3.change_to(v, --EC);
            r_v3.change_to(v, ER++);
            update = true;
            for (uint32_t i = SX; i < EX;) {
                uint32_t vv = x_v3[i];

                if (in_c[vv]) {
                    i++;
                    continue;
                }

                bool cc = false;
                for (uint32_t j = st[vv]; j < g.core_number; j++) {
                    uint32_t w = adj[vv][j];
                    if (w == v) {
                        cc = true;
                        break;
                    }
                }
                if (cc)
                    i++;
                else
                    x_v3.change_to(vv, --EX);
            }
        } else
            i++;
    }
    if (update) {
        auto main_nei_in_c = [&](uint32_t v) {
            uint32_t nst = g.core_number;

            if (EC > 0) {
                for (uint32_t j = st[v]; j < nst;) {
                    if (c_v3.is_in(adj[v][j], 0, EC)) {
                        std::swap(adj[v][j], adj[v][--nst]);
                    } else
                        j++;
                }
            }

            st[v] = nst;
        };

        for (uint32_t i = 0; i < EP; i++) {
            uint32_t v = p_v3[i];
            main_nei_in_c(v);
        }
        for (uint32_t i = SX; i < EX; i++) {
            uint32_t v = x_v3[i];
            if (in_c[v])
                continue;
            main_nei_in_c(v);
        }
        // for(uint32_t i = 0; i < EC; i++) {
        //     uint32_t v = c_v3[i]; main_nei_in_c(v);
        // }
    }

    //remove the P nodes that connects to all PC nodes
    update = false;
    for (uint32_t i = 0, raw_ep = EP; i < EP;) {
        uint32_t v = p_v3[i];
        if (ed[v] == raw_ep - 1 && g.core_number - st[v] == EC) {
            p_v3.change_to(v, --EP);
            r_v3.change_to(v, ER++);
            update = true;
            for (uint32_t i = SX; i < EX;) {
                uint32_t vv = x_v3[i];
                bool cc = false;
                for (uint32_t j = 0; j < ed[vv]; j++) {
                    uint32_t w = adj[vv][j];
                    if (w == v) {
                        cc = true;
                        break;
                    }
                }
                if (cc)
                    i++;
                else
                    x_v3.change_to(vv, --EX);
            }
        } else {
            i++;
        }
    }
    if (update) {
        auto main_nei_in_p = [&](uint32_t v) {
            for (uint32_t j = 0; j < ed[v];) {
                if (p_v3.is_in(adj[v][j], 0, EP)) {
                    j++;
                } else
                    std::swap(adj[v][j], adj[v][--ed[v]]);
            }
        };

        for (uint32_t i = 0; i < EP; i++) {
            uint32_t v = p_v3[i];
            main_nei_in_p(v);
        }
        for (uint32_t i = SX; i < EX; i++) {
            uint32_t v = x_v3[i];
            main_nei_in_p(v);
        }
        for (uint32_t i = 0; i < EC; i++) {
            uint32_t v = c_v3[i];
            main_nei_in_p(v);
        }

        bool f = true;
        for (uint32_t i = 0; i < EP; i++) {
            uint32_t v = p_v3[i];
            if (g.core_number - st[v] == EC) {
                f = false;
                break;
            }
        }
        if (f)
            for (uint32_t i = SX; i < EX; i++) {
                uint32_t v = x_v3[i];
                if (in_c[v] || g.core_number - st[v] == EC) {
                    f = false;
                    break;
                }
            }
        if (f) {  // if(out_p) {
            // printf("%u ", now_u);
            // for(uint32_t i = 0; i < EC; i++) printf("%u ", g.p_edge[g.p_idx[now_u]+c_v3[i]]);
            // for(uint32_t i = 0; i < ER; i++) printf("%u ", g.p_edge[g.p_idx[now_u]+r_v3[i]]);
            // printf("\n");

            if (1 + EC + ER >= threshold) {
                if (max_clique_cnt_save >= (*max_cliques).size()) {
                    (*max_cliques).resize((max_clique_cnt_save + 1) * 2);
                }
                (*max_cliques)[max_clique_cnt_save].clear();
                (*max_cliques)[max_clique_cnt_save].push_back(g.mp[now_u]);
                for (uint32_t i = 0; i < EC; i++) {
                    uint32_t u = g.p_edge[g.p_idx[now_u] + c_v3[i]];
                    (*max_cliques)[max_clique_cnt_save].push_back(g.mp[u]);
                }
                for (uint32_t i = 0; i < ER; i++) {
                    uint32_t u = g.p_edge[g.p_idx[now_u] + r_v3[i]];
                    (*max_cliques)[max_clique_cnt_save].push_back(g.mp[u]);
                }
                max_clique_cnt_save++;
            }
            // }
            cnt++;
        }

        if (EP == 0)
            return;
    }

    //find the best pivot
    uint32_t pu = p_v3[0], max_d = 0;
    for (uint32_t i = 0; i < EP; i++) {
        uint32_t v = p_v3[i];
        uint32_t d = ed[v] + g.core_number - st[v];
        if (d > max_d) {
            max_d = d;
            pu = v;
        }
    }
    for (uint32_t i = 0; i < EC; i++) {
        uint32_t v = c_v3[i];
        // uint32_t d = ed[v] + g.core_number-st[v];
        uint32_t d = ed[v] + EC - 1;
        if (d > max_d) {
            max_d = d;
            pu = v;
        }
    }
    for (uint32_t i = SX; i < EX; i++) {
        uint32_t v = x_v3[i];
        uint32_t nei_in_c = in_c[v] ? EC - 1 : g.core_number - st[v];
        uint32_t d = ed[v] + nei_in_c;
        if (d > max_d) {
            max_d = d;
            pu = v;
        }
        // continue;
        if (ed[v] == EP - 1 && nei_in_c == EC) {
            //remove the X nodes that has one-non-neighbor in P

            for (uint32_t i = 0, ttt = 0; i < ed[v]; i++) {
                uint32_t tmpv = adj[v][i];
                p_v3.change_to(tmpv, ttt++);
            }
            uint32_t u = p_v3[EP - 1];

            uint32_t new_ep = 0;
            uint32_t new_ex = SX;
            uint32_t new_ec = 0;  //build new P
            for (uint32_t i = 0; i < ed[u]; i++) {
                uint32_t v = adj[u][i];
                if (p_v3.isin(v, 0, EP))
                    p_v3.change_to(v, new_ep++);
            }
            //build new C
            for (uint32_t i = st[u]; i < g.core_number; i++) {
                uint32_t v = adj[u][i];
                if (c_v3.isin(v, 0, EC))
                    c_v3.change_to(v, new_ec++);
            }  //build new X
            num_of_non_c_nodes = 1;
            for (uint32_t i = SX; i < EX; i++) {
                uint32_t v = x_v3[i];
                bool cc = false;
                for (uint32_t j = 0; j < ed[v]; j++) {
                    uint32_t w = adj[v][j];
                    if (w == u) {
                        cc = true;
                        break;
                    }
                }
                if (cc)
                    x_v3.change_to(v, new_ex++);
            }
            // R.push_back(u);

            //maintain the neighbors in P / C
            auto main_nei = [&](uint32_t v) {
                uint32_t& ned = nxt_ed[v];
                // nst = g.core_number;
                // ned = 0;
                // if(EP > 0)
                // for(uint32_t j = 0; j < ed[v]; j++) {
                //     if(p_v3.is_in(adj[v][j], 0, new_ep)) {
                //         std::swap(adj[v][j], adj[v][ned++]);
                //     }
                // }
                // if(EC > 0)
                // for(uint32_t j = st[v]; j < nst; ) {
                //     if(c_v3.is_in(adj[v][j], 0, new_ec)) {
                //         std::swap(adj[v][j], adj[v][--nst]);
                //     }
                //     else j++;
                // }

                ned = ed[v];

                for (uint32_t j = 0; j < ned;) {
                    if (p_v3.is_in(adj[v][j], 0, new_ep)) {
                        j++;
                    } else
                        std::swap(adj[v][j], adj[v][--ned]);
                }

                if (in_c[v])
                    return;

                uint32_t& nst = nxt_st[v];
                nst = st[v];
                for (uint32_t j = st[v]; j < g.core_number; j++) {
                    if (!c_v3.is_in(adj[v][j], 0, new_ec)) {
                        std::swap(adj[v][j], adj[v][nst++]);
                    }
                }
            };

            for (uint32_t i = 0; i < new_ep; i++) {
                uint32_t v = p_v3[i];
                main_nei(v);
            }
            for (uint32_t i = SX; i < new_ex; i++) {
                uint32_t v = x_v3[i];
                main_nei(v);
            }
            for (uint32_t i = 0; i < new_ec; i++) {
                uint32_t v = c_v3[i];
                main_nei(v);
            }

            r_v3.change_to(u, ER);

            bk_tomita_rec_ccr_v3(deep + 1, ER + 1, new_ec, new_ep, SX, new_ex);

            num_of_non_c_nodes--;

            return;
        }
    }
    std::vector<uint32_t> non_neighbors(EC + EP - max_d + 1);
    uint32_t d = 0;
    uint32_t new_ep = 0;
    for (uint32_t i = 0; i < ed[pu]; i++) {
        uint32_t v = adj[pu][i];
        p_v3.change_to(v, new_ep++);
    }
    for (uint32_t i = new_ep; i < EP; i++) {
        non_neighbors[d++] = p_v3[i];
    }
    uint32_t new_ec = 0;
    if (c_v3.is_in(pu, 0, EC)) {
        new_ec = EC - 1;
        c_v3.change_to(pu, EC - 1);
    } else if (in_c[pu]) {
        new_ec = EC;
    } else {
        for (uint32_t i = st[pu]; i < g.core_number; i++) {
            uint32_t v = adj[pu][i];
            c_v3.change_to(v, new_ec++);
        }
    }
    for (uint32_t i = new_ec, dc = 0; i < EC; i++) {
        uint32_t v = c_v3[i];

        //与pu的邻居相连
        bool f = false;
        for (uint32_t j = 0; j < ed[v]; j++) {
            uint32_t w = adj[v][j];
            if (p_v3.is_in(w, 0, new_ep)) {
                f = true;
                break;
            }
        }
        if (!f) {
            continue;
        }
        // f = false;
        // //包含前面C点的P中的非邻居
        // for(uint32_t j = d-dc; j < d; j++) {
        //     uint32_t pre_cv = non_neighbors[j];

        // }

        non_neighbors[d++] = v;
        dc++;
    }

    for (uint32_t ii = 0; ii < d; ii++) {
        uint32_t u = non_neighbors[ii];
        uint32_t new_ep = 0;
        uint32_t new_ex = SX;
        uint32_t new_ec = 0;  //build new P
        for (uint32_t i = 0; i < ed[u]; i++) {
            uint32_t v = adj[u][i];
            if (p_v3.isin(v, 0, EP))
                p_v3.change_to(v, new_ep++);
        }
        //build new C
        if (p_v3.is_in(u, 0, EP))
            for (uint32_t i = st[u]; i < g.core_number; i++) {
                uint32_t v = adj[u][i];
                if (c_v3.isin(v, 0, EC))
                    c_v3.change_to(v, new_ec++);
            }
        else {
            c_v3.change_to(u, EC - 1);
            new_ec = EC - 1;
        }  //build new X
        if (p_v3.is_in(u, 0, EP)) {
            num_of_non_c_nodes = 1;
            for (uint32_t i = SX; i < EX; i++) {
                uint32_t v = x_v3[i];
                bool cc = false;
                for (uint32_t j = 0; j < ed[v]; j++) {
                    uint32_t w = adj[v][j];
                    if (w == u) {
                        cc = true;
                        break;
                    }
                }
                if (cc)
                    x_v3.change_to(v, new_ex++);
            }
        } else {
            num_of_non_c_nodes = 0;
            for (uint32_t i = SX; i < EX; i++) {
                uint32_t v = x_v3[i];

                if (in_c[v]) {
                    x_v3.change_to(v, new_ex++);
                    continue;
                }

                bool cc = false;
                for (uint32_t j = st[v]; j < g.core_number; j++) {
                    uint32_t w = adj[v][j];
                    if (w == u) {
                        cc = true;
                        break;
                    }
                }
                if (cc) {
                    x_v3.change_to(v, new_ex++);
                }
            }
        }
        // R.push_back(u);

        //maintain the neighbors in P / C
        auto main_nei = [&](uint32_t v) {
            uint32_t& nst = nxt_st[v];
            uint32_t& ned = nxt_ed[v];
            nst = g.core_number;
            ned = 0;

            if (EP > 0)
                for (uint32_t j = 0; j < ed[v]; j++) {
                    if (p_v3.is_in(adj[v][j], 0, new_ep)) {
                        std::swap(adj[v][j], adj[v][ned++]);
                    }
                }

            if (in_c[v])
                return;

            if (EC > 0)
                for (uint32_t j = st[v]; j < nst;) {
                    if (c_v3.is_in(adj[v][j], 0, new_ec)) {
                        std::swap(adj[v][j], adj[v][--nst]);
                    } else
                        j++;
                }
        };

        for (uint32_t i = 0; i < new_ep; i++) {
            uint32_t v = p_v3[i];
            main_nei(v);
        }
        for (uint32_t i = SX; i < new_ex; i++) {
            uint32_t v = x_v3[i];
            main_nei(v);
        }
        for (uint32_t i = 0; i < new_ec; i++) {
            uint32_t v = c_v3[i];  //main_nei(v);

            uint32_t& ned = nxt_ed[v];
            ned = 0;
            if (EP > 0)
                for (uint32_t j = 0; j < ed[v]; j++) {
                    if (p_v3.is_in(adj[v][j], 0, new_ep)) {
                        std::swap(adj[v][j], adj[v][ned++]);
                    }
                }
        }

        r_v3.change_to(u, ER);

        bk_tomita_rec_ccr_v3(deep + 1, ER + 1, new_ec, new_ep, SX, new_ex);

        x_v3.change_to(u, --SX);
        if (p_v3.is_in(u, 0, EP)) {
            num_of_non_c_nodes--;
            p_v3.change_to(u, --EP);
        } else {
            --EC;

            // c_v3.change_to(u, --EC);
        }
    }

    for (uint32_t ii = 0; ii < d; ii++) {
        x_v3.change_to(non_neighbors[ii], SX++);
        // if(in_c[non_neighbors[ii]]) {
        //     c_v3.change_to(non_neighbors[ii], EC++);
        // }
    }
}

}  // namespace vsag::mci

// NOLINTEND
