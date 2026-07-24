#include "mci_graph_mce.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "mci_list_linear_heap.h"

// This file contains the mechanically imported MCE graph implementation.
// NOLINTBEGIN

namespace vsag::mci {

void
graph_mce::build_csr() {
    m *= 2;

    p_edge.resize(m);
    if (p_idx.size() == 0)
        p_idx.resize(n + 1);
    else {
        p_idx.resize(n + 1);
        for (uint32_t i = 0; i <= n; i++) p_idx[i] = 0;
    }
    if (p_idx2.size() == 0)
        p_idx2.resize(n);
    else {
        p_idx2.resize(n);
        for (uint32_t i = 0; i < n; i++) p_idx2[i] = 0;
    }

    p_idx[0] = 0;
    for (uint32_t i = 0; i < edges.size(); i++) {
        p_idx[edges[i].first + 1]++;
        p_idx[edges[i].second + 1]++;
    }
    for (uint32_t i = 1; i <= n; i++) p_idx[i] += p_idx[i - 1];

    for (uint32_t i = 0; i < edges.size(); i++) {
        p_edge[p_idx[edges[i].first]++] = edges[i].second;
        p_edge[p_idx[edges[i].second]++] = edges[i].first;
    }

    for (uint32_t i = n; i >= 1; i--) p_idx[i] = p_idx[i - 1];
    p_idx[0] = 0;

    for (uint32_t u = 0; u < n; u++) {
        std::sort(p_edge.begin() + p_idx[u], p_edge.begin() + p_idx[u + 1]);
        max_d = std::max(max_d, p_idx[u + 1] - p_idx[u]);

        p_idx2[u] = p_idx[u];
        while (p_idx2[u] < p_idx[u + 1] && p_edge[p_idx2[u]] < u) p_idx2[u]++;
    }
}

void
graph_mce::change_to_core_order() {
    // printf("a");fflush(stdout);
    list_linear_heap lheap(n, max_d + 1);
    std::vector<uint32_t> ids(n);
    std::vector<uint32_t> keys(n + 1);

    for (uint32_t i = 0; i < n; i++) {
        ids[i] = i;
        keys[i] = p_idx[i + 1] - p_idx[i] + 1;
    }
    lheap.init(n, n, ids.data(), keys.data());
    // printf("b%u %u", n, m);;fflush(stdout);
    mp.resize(n);
    // printf("t");;fflush(stdout);
    mp2.resize(n);
    // printf("t");;fflush(stdout);
    std::vector<uint32_t> p_d_edge(m);
    uint32_t* p_d_idx = keys.data();
    // printf("t");;fflush(stdout);

    // printf("cc");;fflush(stdout);
    for (uint32_t i = 0; i < n; i++) {
        uint32_t v, deg_v;

        if (!lheap.pop_min(v, deg_v)) {
            throw std::runtime_error("mci v3 core ordering failed: empty heap");
        }

        mp[i] = v;
        mp2[v] = i;
        for (uint32_t j = p_idx[v]; j < p_idx[v + 1]; j++) {
            lheap.decrement(p_edge[j]);
        }
    }
    // printf("c\n");fflush(stdout);
    p_d_idx[0] = 0;
    for (uint32_t i = 1; i <= n; i++) {
        uint32_t v = mp[i - 1];
        p_d_idx[i] = p_d_idx[i - 1] + p_idx[v + 1] - p_idx[v];
    }

    for (uint32_t i = 0; i < n; i++) {
        uint32_t k = p_idx[mp[i]];
        for (uint32_t j = p_d_idx[i]; j < p_d_idx[i + 1]; j++) {
            p_d_edge[j] = mp2[p_edge[k++]];
        }
        std::sort(p_d_edge.data() + p_d_idx[i], p_d_edge.data() + p_d_idx[i + 1]);
    }

    memcpy(p_edge.data(), p_d_edge.data(), sizeof(uint32_t) * m);
    memcpy(p_idx.data(), p_d_idx, sizeof(uint32_t) * (n + 1));

    p_idx2.resize(n);
    core_number = 0;
    for (uint32_t i = 0; i < n; i++) {
        p_idx2[i] = p_idx[i + 1];
        for (uint32_t j = p_idx[i]; j < p_idx[i + 1]; j++) {
            if (p_edge[j] > i) {
                p_idx2[i] = j;
                core_number = std::max(core_number, p_idx[i + 1] - j);
                break;
            }
        }
    }
}

uint32_t
graph_mce::degree(uint32_t u) {
    return p_idx[u + 1] - p_idx[u];
}

bool
graph_mce::connect(uint32_t u, uint32_t v) {
    return std::binary_search(p_edge.begin() + p_idx[u], p_edge.begin() + p_idx[u + 1], v);
}
bool
graph_mce::connect2(uint32_t u, uint32_t v) {
    return std::binary_search(p_edge.begin() + p_idx[u], p_edge.begin() + p_idx2[u], v);
}
bool
graph_mce::connect_out(uint32_t u, uint32_t v) {
    return std::binary_search(p_edge.begin() + p_idx2[u], p_edge.begin() + p_idx[u + 1], v);
}

}  // namespace vsag::mci

// NOLINTEND
