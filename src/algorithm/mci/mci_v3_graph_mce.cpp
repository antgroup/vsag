#include "mci_v3_graph_mce.h"

#include <algorithm>
#include <iostream>

#include "mci_v3_list_linear_heap.h"

namespace vsag::mci_v3 {

void
Graphmce::buildCSR() {
    m *= 2;

    pEdge.resize(m);
    if (pIdx.size() == 0)
        pIdx.resize(n + 1);
    else {
        pIdx.resize(n + 1);
        for (ui i = 0; i <= n; i++) pIdx[i] = 0;
    }
    if (pIdx2.size() == 0)
        pIdx2.resize(n);
    else {
        pIdx2.resize(n);
        for (ui i = 0; i < n; i++) pIdx2[i] = 0;
    }

    pIdx[0] = 0;
    for (ui i = 0; i < edges.size(); i++) {
        pIdx[edges[i].first + 1]++;
        pIdx[edges[i].second + 1]++;
    }
    for (ui i = 1; i <= n; i++) pIdx[i] += pIdx[i - 1];

    for (ui i = 0; i < edges.size(); i++) {
        pEdge[pIdx[edges[i].first]++] = edges[i].second;
        pEdge[pIdx[edges[i].second]++] = edges[i].first;
    }

    for (ui i = n; i >= 1; i--) pIdx[i] = pIdx[i - 1];
    pIdx[0] = 0;

    for (ui u = 0; u < n; u++) {
        std::sort(pEdge.begin() + pIdx[u], pEdge.begin() + pIdx[u + 1]);
        maxD = std::max(maxD, pIdx[u + 1] - pIdx[u]);

        pIdx2[u] = pIdx[u];
        while (pIdx2[u] < pIdx[u + 1] && pEdge[pIdx2[u]] < u) pIdx2[u]++;
    }
}

void
Graphmce::changeToCoreOrder() {
    // printf("a");fflush(stdout);
    ListLinearHeap lheap(n, maxD + 1);
    ui* ids = new ui[n];
    ui* keys = new ui[n + 1];

    for (ui i = 0; i < n; i++) {
        ids[i] = i;
        keys[i] = pIdx[i + 1] - pIdx[i] + 1;
    }
    lheap.init(n, n, ids, keys);
    // printf("b%u %u", n, m);;fflush(stdout);
    mp.resize(n);
    // printf("t");;fflush(stdout);
    mp2.resize(n);
    // printf("t");;fflush(stdout);
    ui* pDEdge = new ui[m];
    ui* pDIdx = keys;
    // printf("t");;fflush(stdout);

    // printf("cc");;fflush(stdout);
    for (ui i = 0; i < n; i++) {
        ui v, degV;

        if (!lheap.pop_min(v, degV))
            printf("error\n");

        mp[i] = v;
        mp2[v] = i;
        for (ui j = pIdx[v]; j < pIdx[v + 1]; j++) {
            lheap.decrement(pEdge[j]);
        }
    }
    // printf("c\n");fflush(stdout);
    pDIdx[0] = 0;
    for (ui i = 1; i <= n; i++) {
        ui v = mp[i - 1];
        pDIdx[i] = pDIdx[i - 1] + pIdx[v + 1] - pIdx[v];
    }

    for (ui i = 0; i < n; i++) {
        ui k = pIdx[mp[i]];
        for (ui j = pDIdx[i]; j < pDIdx[i + 1]; j++) {
            pDEdge[j] = mp2[pEdge[k++]];
        }
        std::sort(pDEdge + pDIdx[i], pDEdge + pDIdx[i + 1]);
    }

    memcpy(pEdge.data(), pDEdge, sizeof(ui) * m);
    memcpy(pIdx.data(), pDIdx, sizeof(ui) * (n + 1));

    delete[] ids;
    delete[] keys;
    // delete [] mp2;
    delete[] pDEdge;

    pIdx2.resize(n);
    coreNumber = 0;
    for (ui i = 0; i < n; i++) {
        pIdx2[i] = pIdx[i + 1];
        for (ui j = pIdx[i]; j < pIdx[i + 1]; j++) {
            if (pEdge[j] > i) {
                pIdx2[i] = j;
                coreNumber = std::max(coreNumber, pIdx[i + 1] - j);
                break;
            }
        }
    }
}

ui
Graphmce::degree(ui u) {
    return pIdx[u + 1] - pIdx[u];
}

bool
Graphmce::connect(ui u, ui v) {
    return std::binary_search(pEdge.begin() + pIdx[u], pEdge.begin() + pIdx[u + 1], v);
}
bool
Graphmce::connect2(ui u, ui v) {
    return std::binary_search(pEdge.begin() + pIdx[u], pEdge.begin() + pIdx2[u], v);
}
bool
Graphmce::connectOut(ui u, ui v) {
    return std::binary_search(pEdge.begin() + pIdx2[u], pEdge.begin() + pIdx[u + 1], v);
}

}  // namespace vsag::mci_v3