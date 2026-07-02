#pragma once

#include <limits>
#include <map>

#include "mci_v3_graph_mce.h"
#include "mci_v3_mce.h"

namespace vsag::mci_v3 {

template <typename edgeType, typename nodeType>
class ccrmceRunner {
public:
    // using CCRMCE::Graph;
    // using CCRMCE::MCE;
    std::map<nodeType, nodeType> mp;
    std::vector<nodeType> rawId;
    std::vector<bool> mustContainNodes;
    // Graphmce g;
    MCE mce;
    MCEStats lastStats;

    void
    reIdAndBuildCsrgraph(std::vector<edgeType>& edge) {
        nodeType idx = 0;
        if (rawId.capacity() < edge.size())
            rawId.reserve(edge.size());
        rawId.clear();
        mp.clear();
        for (auto& e : edge) {
            if (mp.find(e.u) == mp.end()) {
                mp[e.u] = idx++;
                rawId.push_back(e.u);
            }
            if (mp.find(e.v) == mp.end()) {
                mp[e.v] = idx++;
                rawId.push_back(e.v);
            }
        }

        //change to undirected graph
        Graphmce& g = mce.g;
        if (g.edges.capacity() < edge.size())
            g.edges.reserve(edge.size());
        g.edges.clear();
        for (auto& e : edge) {
            g.edges.push_back(Pair(mp[e.u], mp[e.v]));
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
        g.buildCSR();
    }

    ccrmceRunner(){};
    nodeType
    run(std::vector<edgeType>& edge,
        std::vector<std::vector<nodeType>>& maxCliques,
        nodeType threshold) {
        reIdAndBuildCsrgraph(edge);
        mce.g.changeToCoreOrder();
        // MCE mce(g, &maxCliques);

        mce.threshold = threshold;
        mce.set(&maxCliques);
        mce.run();

        for (auto ii = 0; ii < mce.maxCliqueCntSave; ii++) {
            auto& maxC = maxCliques[ii];
            for (auto i = 0; i < maxC.size(); i++) {
                maxC[i] = rawId[maxC[i]];
            }
        }
        return mce.maxCliqueCntSave;
    }

    void
    reserve(uint32_t sz) {
        mce.reserve(sz);
        mustContainNodes.resize(sz);
    }

    nodeType
    run(std::vector<edgeType>& edge,
        std::vector<std::vector<nodeType>>& maxCliques,
        nodeType threshold,
        std::vector<std::atomic<int>>& numCliquesPerNode) {
        reIdAndBuildCsrgraph(edge);

        mce.g.changeToCoreOrder();
        // printf("g:\n");
        // for(ui i = 0; i < mce.g.n; i++) {
        //     printf("%u:", i);
        //     for(ui j = mce.g.pIdx2[i]; j < mce.g.pIdx[i + 1]; j++) {
        //         ui v = mce.g.pEdge[j];
        //         printf("%u ", v);
        //     }
        //     printf("\n");
        // }

        // MCE mce(g, &maxCliques);
        if (mustContainNodes.size() < mce.g.n)
            mustContainNodes.resize(mce.g.n);
        for (auto u = 0; u < mce.g.n; u++) {
            mustContainNodes[u] = false;
        }
        int cnt = 0;
        for (auto u = 0; u < mce.g.n; u++) {
            int rawIdu = rawId[mce.g.mp[u]];
            if (numCliquesPerNode[rawIdu].load(std::memory_order_relaxed) == 0) {
                mustContainNodes[u] = true;
                // printf("mustContain %u\n", u);
                cnt++;
            }
        }

        mce.threshold = threshold;
        // maxCliques.clear();
        mce.set(&maxCliques);
        mce.configureMustRun(std::numeric_limits<ui>::max(), static_cast<ui>(cnt));

        mce.run(mustContainNodes);
        lastStats = mce.stats;

        int cnt2 = 0;
        for (auto u = 0; u < mce.g.n; u++) {
            if (mustContainNodes[u]) {
                cnt2++;
            }
        }
        // if(mce.maxCliqueCntSave>0)
        // printf("valid %d %d, mce.maxCliqueCntSave %u\n", cnt, cnt2, mce.maxCliqueCntSave);fflush(stdout);

        for (auto ii = 0; ii < mce.maxCliqueCntSave; ii++) {
            auto& maxC = maxCliques[ii];

            for (auto i = 0; i < maxC.size(); i++) {
                maxC[i] = rawId[maxC[i]];
                // printf("%u ", maxC[i]);fflush(stdout);
            }
            // printf("\n");
            // bool isValid = false;
            // for(auto i = 0; i < maxC.size(); i++) {
            //     if(numCliquesPerNode[ maxC[i] ] == 0) {
            //         isValid = true; break;
            //     }
            // }
            // if(!isValid) {
            //     printf("emunerating error\n");fflush(stdout);
            // }
        }

        return mce.maxCliqueCntSave;
    }

    nodeType
    run(std::vector<edgeType>& edge,
        std::vector<std::vector<nodeType>>& maxCliques,
        nodeType threshold,
        std::vector<std::atomic<int>>& numCliquesPerNode,
        nodeType maxSavedCliques) {
        reIdAndBuildCsrgraph(edge);

        mce.g.changeToCoreOrder();

        if (mustContainNodes.size() < mce.g.n)
            mustContainNodes.resize(mce.g.n);
        for (auto u = 0; u < mce.g.n; u++) {
            mustContainNodes[u] = false;
        }
        ui cnt = 0;
        for (auto u = 0; u < mce.g.n; u++) {
            int rawIdu = rawId[mce.g.mp[u]];
            if (numCliquesPerNode[rawIdu].load(std::memory_order_relaxed) == 0) {
                mustContainNodes[u] = true;
                cnt++;
            }
        }

        mce.threshold = threshold;
        mce.set(&maxCliques);
        mce.configureMustRun(static_cast<ui>(maxSavedCliques), cnt);

        mce.run(mustContainNodes);
        lastStats = mce.stats;

        for (auto ii = 0; ii < mce.maxCliqueCntSave; ii++) {
            auto& maxC = maxCliques[ii];

            for (auto i = 0; i < maxC.size(); i++) {
                maxC[i] = rawId[maxC[i]];
            }
        }

        return mce.maxCliqueCntSave;
    }
};

}  // namespace vsag::mci_v3
