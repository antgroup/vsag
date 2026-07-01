#include "mci_v3_mce.h"

#include "mci_v3_list_linear_heap.h"

namespace vsag::mci_v3 {
// #define DEBUG
#ifdef DEBUG
ui xxx = 0, xx;
#endif

// #define RUNNINGTIME_PER_NODE

#ifdef RUNNINGTIME_PER_NODE
double maxTimePerNode = 0.0;
#endif

// #define FIND_EXAMPLE

#ifdef FIND_EXAMPLE
bool CisnotEmpty = false;
bool filterByFromP = false;
bool removeCnonNeighbors = false;
bool removeByX = false;
#endif

// #define PERCENAEG_OF_C
#ifdef PERCENAEG_OF_C
unsigned long long allCNodes = 0;
unsigned long long allPCNodes = 0;
#endif

void
MCE::reserve(ui sz) {
    Cv3.resize(sz);
    Pv3.resize(sz);
    Rv3.resize(sz);
    Xv3.resize(sz);
    // choosedNode.resize(sz);
    edAdj.resize(sz);
    for (ui i = 0; i < sz; i++) edAdj[i].resize(sz);
    stAdj.resize(sz);
    for (ui i = 0; i < sz; i++) stAdj[i].resize(sz);
    adj.resize(sz);
    for (ui i = 0; i < sz; i++) {
        adj[i].resize(sz);
    }
}

void
MCE::configureMustRun(ui maxSavedCliques, ui initialMustCount) {
    maxCliqueCntLimit = maxSavedCliques;
    remainingMustCount = initialMustCount;
    stopSearch = false;
    stats.reset();
}

bool
MCE::shouldStopMCE() const {
    return stopSearch || maxCliqueCntSave >= maxCliqueCntLimit || remainingMustCount == 0;
}

bool
MCE::stateHasMust(ui ER, ui EC, ui EP, std::vector<bool>& mustContainNodes) {
    if (mustContainNodes[nowU])
        return true;
    for (ui i = 0; i < EC; i++) {
        ui u = g.pEdge[g.pIdx[nowU] + Cv3[i]];
        if (mustContainNodes[u])
            return true;
    }
    for (ui i = 0; i < ER; i++) {
        ui u = g.pEdge[g.pIdx[nowU] + Rv3[i]];
        if (mustContainNodes[u])
            return true;
    }
    for (ui i = 0; i < EP; i++) {
        ui u = g.pEdge[g.pIdx[nowU] + Pv3[i]];
        if (mustContainNodes[u])
            return true;
    }
    return false;
}

void
MCE::clearMustNode(ui u, std::vector<bool>& mustContainNodes) {
    if (mustContainNodes[u]) {
        mustContainNodes[u] = false;
        if (remainingMustCount > 0)
            remainingMustCount--;
    }
}

bool
MCE::trySaveMustClique(ui ER, ui EC, ui EP, std::vector<bool>& mustContainNodes) {
    if (shouldStopMCE())
        return false;
    if (1 + EC + ER < threshold)
        return false;
    if (!stateHasMust(ER, EC, EP, mustContainNodes))
        return false;
    if (maxCliqueCntSave >= maxCliqueCntLimit) {
        stopSearch = true;
        stats.quota_stop++;
        return false;
    }

    if (maxCliqueCntSave >= (*maxCliques).size()) {
        (*maxCliques).resize((maxCliqueCntSave + 1) * 2);
    }
    (*maxCliques)[maxCliqueCntSave].clear();
    (*maxCliques)[maxCliqueCntSave].push_back(g.mp[nowU]);
    clearMustNode(nowU, mustContainNodes);
    for (ui i = 0; i < EC; i++) {
        ui u = g.pEdge[g.pIdx[nowU] + Cv3[i]];
        (*maxCliques)[maxCliqueCntSave].push_back(g.mp[u]);
    }
    for (ui i = 0; i < ER; i++) {
        ui u = g.pEdge[g.pIdx[nowU] + Rv3[i]];
        (*maxCliques)[maxCliqueCntSave].push_back(g.mp[u]);
    }
    for (ui i = 0; i < EP; i++) {
        ui u = g.pEdge[g.pIdx[nowU] + Pv3[i]];
        (*maxCliques)[maxCliqueCntSave].push_back(g.mp[u]);
    }

    for (ui i = 0; i < EC; i++) {
        clearMustNode(g.pEdge[g.pIdx[nowU] + Cv3[i]], mustContainNodes);
    }
    for (ui i = 0; i < ER; i++) {
        clearMustNode(g.pEdge[g.pIdx[nowU] + Rv3[i]], mustContainNodes);
    }
    for (ui i = 0; i < EP; i++) {
        clearMustNode(g.pEdge[g.pIdx[nowU] + Pv3[i]], mustContainNodes);
    }
    maxCliqueCntSave++;
    stats.saved_cliques++;

    if (maxCliqueCntSave >= maxCliqueCntLimit) {
        stopSearch = true;
        stats.quota_stop++;
    }
    if (remainingMustCount == 0) {
        stopSearch = true;
        stats.remaining_must_stop++;
    }
    return true;
}

ui
MCE::run(std::vector<bool>& mustContainNodes) {
    if (Cv3.cap() < g.maxD) {
        printf("alloc %u int4\n", g.maxD);
        fflush(stdout);
    }
    Cv3.resize(g.maxD);
    Pv3.resize(g.maxD);
    Rv3.resize(g.maxD);
    Xv3.resize(g.maxD);

    // choosedNode.resize(g.n);
    // for(int i = 0; i < g.n; i++) {
    //     choosedNode[i] = false;
    // }

    cnt = 0;
    maxCliqueCntSave = 0;
    // printf("a0\n");fflush(stdout);
    edAdj.resize(g.coreNumber + 1);
    for (ui i = 0; i <= g.coreNumber; i++) edAdj[i].resize(g.maxD);
    stAdj.resize(g.coreNumber + 1);
    for (ui i = 0; i <= g.coreNumber; i++) stAdj[i].resize(g.maxD);
    adj.resize(g.maxD);
    for (ui i = 0; i < g.maxD; i++) {
        adj[i].resize(g.coreNumber);
    }
    // reId.resize(g.n, g.n);
    if (reId.capacity() < g.n)
        reId.resize(g.n);
    std::fill(reId.begin(), reId.begin() + g.n, g.n);

    if (inC.capacity() < g.maxD)
        inC.resize(g.maxD, false);
    std::fill(inC.begin(), inC.begin() + g.maxD, false);

    // std::vector<ui> newCliqueNei;
    // printf("a1\n");fflush(stdout);
    ListLinearHeap heap(g.coreNumber, g.coreNumber);
    heap.reserve(g.coreNumber, g.coreNumber);

    id_s.resize(g.coreNumber + 1);
    degree.resize(g.coreNumber);
    removed.resize(g.coreNumber);

    pIdx.resize(g.coreNumber + 1);
    eIdx.resize(g.coreNumber + 1);
    pEdge.resize(g.coreNumber * g.coreNumber);

    clique.resize(g.coreNumber);
    nxtC.resize(g.coreNumber);

    // std::vector<bool> isInX(g.n);
    for (ui u = 0; u < g.n; u++) {
        if (shouldStopMCE())
            break;
        stats.roots_seen++;
        nowU = u;
        // if(choosedNode[u]) continue;
        // if(mustContainNodes[u] == false) continue;
        if (1 + g.pIdx[u + 1] - g.pIdx2[u] < threshold) {
            stats.roots_skipped_by_degree++;
            continue;
        }

        // isInX[u] = true;

        bool ok = mustContainNodes[u];
        if (mustContainNodes[u] == false)
            for (ui i = g.pIdx2[u]; i < g.pIdx[u + 1]; i++) {
                if (mustContainNodes[g.pEdge[i]]) {
                    ok = true;
                    break;
                }
            }
        if (!ok) {
            stats.roots_skipped_no_must++;
            continue;
        }

#ifdef DEBUG
        xx = u;
        xxx = u;
        if (xx == xxx)
            printf("\n    st %u\n", u);
#endif

#ifdef RUNNINGTIME_PER_NODE
        double stTime = clock();
#endif

        for (ui i = g.pIdx2[u]; i < g.pIdx[u + 1]; i++) {
            reId[g.pEdge[i]] = i - g.pIdx2[u] + 1;
        }

        ui n = g.pIdx[u + 1] - g.pIdx2[u];
        for (ui i = 0; i <= n; i++) pIdx[i] = 0;

        for (ui i = g.pIdx2[u]; i < g.pIdx[u + 1]; i++) {
            ui v = g.pEdge[i];

            for (ui j = g.pIdx2[v]; j < g.pIdx[v + 1]; j++) {
                ui w = g.pEdge[j];
                if (reId[w] < g.n) {
                    pIdx[i - g.pIdx2[u] + 1]++;
                    pIdx[reId[w]]++;
                }
            }
        }
        for (ui i = 1; i <= n; i++) pIdx[i] += pIdx[i - 1];
        for (ui i = g.pIdx2[u]; i < g.pIdx[u + 1]; i++) {
            ui v = g.pEdge[i];
            for (ui j = g.pIdx2[v]; j < g.pIdx[v + 1]; j++) {
                ui w = g.pEdge[j];
                if (reId[w] < g.n) {
                    pEdge[pIdx[i - g.pIdx2[u]]++] = reId[w] - 1;
                    pEdge[pIdx[reId[w] - 1]++] = i - g.pIdx2[u];
                }
            }
        }
        for (ui i = n; i > 0; i--) pIdx[i] = pIdx[i - 1];
        pIdx[0] = 0;

        clique.clear();
        nxtC.clear();

        ui queue_n = 0, new_size = 0;
        for (ui i = 0; i < n; i++) removed[i] = false;
        for (ui i = 0; i < n; i++) {
            if (pIdx[i + 1] - pIdx[i] + 2 < threshold)
                id_s[queue_n++] = i;
            degree[i] = pIdx[i + 1] - pIdx[i];
        }
        for (ui i = 0; i < queue_n; i++) {
            ui uu = id_s[i];
            degree[uu] = 0;
            removed[uu] = true;
            for (ui j = pIdx[uu]; j < pIdx[uu + 1]; j++)
                if (degree[pEdge[j]] > 0) {
                    if ((degree[pEdge[j]]--) + 2 == threshold)
                        id_s[queue_n++] = pEdge[j];
                }
        }

        for (ui i = 0; i < n; i++) {
            if (degree[i] + 2 >= threshold)
                id_s[queue_n + (new_size++)] = i;
            else {
                removed[i] = true;
            }
        }

        if (new_size == 0) {
            for (ui i = g.pIdx2[u]; i < g.pIdx[u + 1]; i++) {
                reId[g.pEdge[i]] = g.n;
            }
            continue;
        } else {
#ifdef DEBUG
            if (xx == xxx) {
                printf("queue_n:%u\n", queue_n);  //should be zero by default
                printf("To peel %u:", new_size);
                for (ui i = 0; i < new_size; i++) {
                    printf("%u-%u ", id_s[queue_n + i] + 1, degree[id_s[queue_n + i]]);
                }
                printf("\n");
                fflush(stdout);
            }
#endif

            //ListLinearHeap *heap = new ListLinearHeap(n, new_size-1);
            heap.initRaw(new_size, new_size - 1, id_s.data() + queue_n, degree.data());

            // printf("b");fflush(stdout);
            for (ui i = 0; i < new_size; i++) {
                ui v, key;

                heap.pop_min(v, key);

                id_s[queue_n + i] = v;
                if (key + i + 1 == new_size) {
                    ui x_size = i + 1;
                    heap.get_ids(id_s.data() + queue_n, x_size);

                    assert(x_size == new_size);
                    // printf("	last key %u, x_size %u\n", key, x_size);
                    for (ui j = i; j < new_size; j++) {
                        clique.push_back(g.pEdge[g.pIdx2[u] + id_s[queue_n + j]]);
                    }

                    for (ui j = 0; j < i; j++) {
                        //gready extend
                        ui tu = g.pEdge[g.pIdx2[u] + id_s[queue_n + j]];
                        nxtC.push_back(tu);
                    }

                    break;
                }
                removed[v] = 1;
                // printf("%u %u\n", u, key);
                // fflush(stdout);

                for (ui j = pIdx[v]; j < pIdx[v + 1]; j++)
                    if (removed[pEdge[j]] == false) {
                        heap.decrement(pEdge[j], 1);
                    }
            }
            // printf("c\n");fflush(stdout);
        }

        // printf("findClique\n");fflush(stdout);

        for (ui i = g.pIdx[u]; i < g.pIdx[u + 1]; i++) {
            reId[g.pEdge[i]] = i - g.pIdx[u];
        }

#ifdef DEBUG
        if (xx == xxx) {
            printf("nxtC: ");
            for (auto v : nxtC) printf("%u ", reId[v]);
            printf("\n");
            printf("clique: ");
            for (auto v : clique) printf("%u ", reId[v]);
            printf("\n");
            fflush(stdout);
        }
#endif
        ui EC = 0;
        ui EX = g.maxD, SX = g.maxD;
        ui ER = 0;
        ui EP = 0;
        for (ui i = g.pIdx[u]; i < g.pIdx2[u]; i++) {
            // if(isInX[g.pEdge[i]])
            Xv3.changeTo(reId[g.pEdge[i]], --SX);
            // else
            //             Pv3.changeTo(reId[g.pEdge[i]], EP++);
        }
        for (ui v : nxtC) {
            Pv3.changeTo(reId[v], EP++);
        }
        for (ui v : clique) {
            Cv3.changeTo(reId[v], EC++);
        }
#ifdef DEBUG
        if (xx == xxx) {
            printf("EC %u, EP %u, EX-SX %u\n", EC, EP, EX - SX);
            fflush(stdout);
        }
#endif
        //X nodes, the neighbors in P
        for (ui i = g.pIdx[u]; i < g.pIdx2[u]; i++) {
            ui v = g.pEdge[i];
            ui vId = i - g.pIdx[u];
            edAdj[0][vId] = 0;
            for (ui j = g.pIdx2[v]; j < g.pIdx[v + 1]; j++) {
                ui w = g.pEdge[j];
                if (reId[w] == g.n)
                    continue;
                if (Pv3.isIn(reId[w], 0, EP)) {
                    adj[vId][edAdj[0][vId]++] = reId[w];
                }
            }
        }
        //P/C nodes, the neighbors in P
        for (ui i = g.pIdx2[u]; i < g.pIdx[u + 1]; i++) {
            ui vId = i - g.pIdx[u];
            edAdj[0][vId] = 0;
        }
        for (ui v : nxtC) {
            ui vId = reId[v];
            for (ui j = g.pIdx2[v]; j < g.pIdx[v + 1]; j++) {
                ui w = g.pEdge[j];
                ui wId = reId[w];
                if (reId[w] == g.n)
                    continue;
                if (Pv3.isIn(wId, 0, EP)) {
                    adj[vId][edAdj[0][vId]++] = wId;
                    adj[wId][edAdj[0][wId]++] = reId[v];
                } else if (Cv3.isIn(wId, 0, EC)) {
                    adj[wId][edAdj[0][wId]++] = reId[v];
                }
            }
        }
        for (ui v : clique) {
            ui vId = reId[v];
            for (ui j = g.pIdx2[v]; j < g.pIdx[v + 1]; j++) {
                ui w = g.pEdge[j];
                ui wId = reId[w];
                if (reId[w] == g.n)
                    continue;
                if (Pv3.isIn(wId, 0, EP)) {
                    adj[vId][edAdj[0][vId]++] = wId;
                }
            }
        }
        //X nodes, the neighbors in C
        for (ui i = g.pIdx[u]; i < g.pIdx2[u]; i++) {
            ui v = g.pEdge[i];
            ui vId = i - g.pIdx[u];
            stAdj[0][vId] = g.coreNumber;

            for (ui j = g.pIdx2[v]; j < g.pIdx[v + 1]; j++) {
                ui w = g.pEdge[j];
                if (reId[w] == g.n)
                    continue;
                if (Cv3.isIn(reId[w], 0, EC)) {
                    adj[vId][--stAdj[0][vId]] = reId[w];
                }
            }
        }
        //P/C nodes, the neighbors in C
        for (ui v : nxtC) {
            ui vId = reId[v];
            stAdj[0][vId] = g.coreNumber;
        }
        for (ui v : clique) {
            ui vId = reId[v];
            stAdj[0][vId] = g.coreNumber;
        }
        for (ui v : nxtC) {
            ui vId = reId[v];
            for (ui j = g.pIdx2[v]; j < g.pIdx[v + 1]; j++) {
                ui w = g.pEdge[j];
                if (reId[w] == g.n)
                    continue;
                if (Cv3.isIn(reId[w], 0, EC)) {
                    adj[vId][--stAdj[0][vId]] = reId[w];
                }
            }
        }
        for (ui w : clique) {
            ui wId = reId[w];
            inC[wId] = true;
            for (ui j = g.pIdx2[w]; j < g.pIdx[w + 1]; j++) {
                ui v = g.pEdge[j];
                ui vId = reId[v];
                if (vId == g.n)
                    continue;
                if (Pv3.isIn(vId, 0, EP)) {
                    adj[vId][--stAdj[0][vId]] = wId;
                }
                // else if(Cv3.isIn(vId, 0, EC)) {
                //     adj[vId][--stAdj[0][vId]] = wId;
                //     adj[wId][--stAdj[0][wId]] = vId;
                // }
            }
        }
#ifdef FIND_EXAMPLE
        if (EC)
            CisnotEmpty = true;
        filterByFromP = false;
        removeCnonNeighbors = false;
        removeByX = false;
#endif

#ifdef DEBUG
        ui preA = cnt;
#endif

        numOfNonCNodes = 1;
        BKTomitaRecCCRV3_mustcontain(0, ER, EC, EP, SX, EX, mustContainNodes);
        if (stopSearch)
            break;
#ifdef DEBUG
        // if(xx == xxx)
        printf("local:%u:%u, %u\n", u, cnt - preA, maxCliqueCntSave);
        fflush(stdout);
#endif

#ifdef FIND_EXAMPLE
        if (CisnotEmpty && filterByFromP && removeCnonNeighbors && removeByX) {
            printf("ThisIsIt! %u, %u, %u\n", u, nxtC.size() + clique.size(), EX - SX);
        }
#endif
        for (ui w : clique) {
            ui wId = reId[w];
            inC[wId] = false;
        }

        for (ui i = g.pIdx[u]; i < g.pIdx[u + 1]; i++) {
            reId[g.pEdge[i]] = g.n;
        }

#ifdef RUNNINGTIME_PER_NODE
        double edTime = clock();
        maxTimePerNode = std::max(maxTimePerNode, edTime - stTime);
#endif
    }
#ifdef COUNT_NUM_SEARCH_NODES
    printf("numOfDFSSearchNodes:%llu\n", numOfDFSSearchNodes);
#endif
#ifdef PERCENAEG_OF_C
    printf("allCNodes:%llu\n", allCNodes);
    printf("allPCNodes:%llu\n", allPCNodes);
    printf("percentage:%f\n", 1.0 * allCNodes / allPCNodes);
#endif

#ifdef RUNNINGTIME_PER_NODE
    printf("maxTimePerNode:%fs\n", maxTimePerNode / CLOCKS_PER_SEC);
#endif
    return cnt;
}

void
MCE::BKTomitaRecCCRV3_mustcontain(
    ui deep, ui ER, ui EC, ui EP, ui SX, ui EX, std::vector<bool>& mustContainNodes) {
#ifdef DEBUG
    if (xx == xxx) {
        printf("\n    deep %u\n", deep);
        fflush(stdout);
        printf("C:");
        for (ui i = 0; i < EC; i++) printf("%u ", Cv3[i]);
        printf("\n");
        printf("R:");
        for (ui i = 0; i < ER; i++) printf("%u ", Rv3[i]);
        printf("\n");
        printf("P:");
        for (ui i = 0; i < EP; i++) printf("%u ", Pv3[i]);
        printf("\n");
        printf("X:");
        for (ui i = SX; i < EX; i++) printf("%u ", Xv3[i]);
        printf("\n");
        printf("P: P/C\n");
        for (ui i = 0; i < EP; i++) {
            ui vId = Pv3[i];
            printf("%u: ", vId);
            for (ui j = 0; j < edAdj[deep][vId]; j++) {
                printf("%u ", adj[vId][j]);
            }
            printf(" / ");
            for (ui j = stAdj[deep][vId]; j < g.coreNumber; j++) {
                printf("%u ", adj[vId][j]);
            }
            printf("\n");
        }

        printf("PXsg:\n");
        for (ui i = 0; i < EP; i++) {
            ui vId = Pv3[i];
            printf("%u: ", vId);
            for (ui j = SX; j < EX; j++) {
                if (g.connect(g.pEdge[g.pIdx[xx] + vId], g.pEdge[g.pIdx[xx] + Xv3[j]]))
                    printf("%u ", Xv3[j]);
            }
            printf("\n");
        }

        printf("PPsg:\n");
        for (ui i = 0; i < EP; i++) {
            ui vId = Pv3[i];
            printf("%u: ", vId);
            for (ui j = 0; j < EP; j++) {
                if (g.connect(g.pEdge[g.pIdx[xx] + vId], g.pEdge[g.pIdx[xx] + Pv3[j]]))
                    printf("%u ", Pv3[j]);
            }
            printf("\n");
        }
        printf("CPsg:\n");
        for (ui i = 0; i < EC; i++) {
            ui vId = Cv3[i];
            printf("%u: ", vId);
            for (ui j = 0; j < edAdj[deep][vId]; j++) {
                printf("%u ", adj[vId][j]);
            }
            printf("\n");
        }
        printf("XCsg:\n");
        for (ui i = SX; i < EX; i++) {
            ui vId = Xv3[i];
            printf("%u: ", vId);
            for (ui j = stAdj[deep][vId]; j < g.coreNumber; j++) {
                printf("%u ", adj[vId][j]);
            }
            printf("\n");
        }
        printf("CCsg:\n");
        for (ui i = 0; i < EC; i++) {
            ui vId = Cv3[i];
            printf("%u: ", vId);
            for (ui j = stAdj[deep][vId]; j < g.coreNumber; j++) {
                printf("%u ", adj[vId][j]);
            }
            printf("\n");
        }
    }
#endif

#ifdef COUNT_NUM_SEARCH_NODES
    numOfDFSSearchNodes++;
#endif
    stats.recursion_calls++;
    if (shouldStopMCE())
        return;
    if (1 + ER + EC + EP < threshold) {
        stats.prune_size_bound++;
        return;
    }
    if (!stateHasMust(ER, EC, EP, mustContainNodes)) {
        stats.prune_no_must++;
        return;
    }

    bool f = true;
    std::vector<ui>& ed = edAdj[deep];
    std::vector<ui>& st = stAdj[deep];
    std::vector<ui>& nxtEd = edAdj[deep + 1];
    std::vector<ui>& nxtSt = stAdj[deep + 1];

    // if(numOfNonCNodes == 0 && deep > 0) f = false;
    if (numOfNonCNodes == 0) {
#ifdef FIND_EXAMPLE
        filterByFromP = true;
#endif
        f = false;
    }
    if (f && EX - SX + EP > 0) {
        for (ui i = 0; i < EP; i++) {
            ui v = Pv3[i];
            if (g.coreNumber - st[v] == EC) {
                f = false;
                break;
            }
        }
        if (f) {
            for (ui i = SX; i < EX; i++) {
                ui v = Xv3[i];
                if (inC[v] || g.coreNumber - st[v] == EC) {
                    f = false;
                    break;
                }
            }
#ifdef FIND_EXAMPLE
            if (!f)
                removeByX = true;
#endif
        }
    }
#ifdef DEBUG
    if (xx == xxx) {
        printf("F:%d\n", (int)f);
    }
#endif
    if (f) {
#ifdef PERCENAEG_OF_C
        allCNodes += EC;
        allPCNodes += EC + ER;
#endif
        // if(outP) {
        // printf("%u ", nowU);
        // for(ui i = 0; i < EC; i++) printf("%u ", g.pEdge[g.pIdx[nowU]+Cv3[i]]);
        // for(ui i = 0; i < ER; i++) printf("%u ", g.pEdge[g.pIdx[nowU]+Rv3[i]]);
        // printf("\n");

        trySaveMustClique(ER, EC, EP, mustContainNodes);

        // }
        cnt++;
        if (stopSearch)
            return;
    }

    bool containMust = false;
    for (ui i = 0; i < EC; i++) {
        ui u = g.pEdge[g.pIdx[nowU] + Cv3[i]];
        if (mustContainNodes[u]) {
            containMust = true;
            break;
        }
    }
    if (containMust == false)
        for (ui i = 0; i < ER; i++) {
            ui u = g.pEdge[g.pIdx[nowU] + Rv3[i]];
            if (mustContainNodes[u]) {
                containMust = true;
                break;
            }
        }
    if (containMust == false)
        for (ui i = 0; i < EP; i++) {
            ui u = g.pEdge[g.pIdx[nowU] + Pv3[i]];
            if (mustContainNodes[u]) {
                containMust = true;
                break;
            }
        }
    if (containMust == false) {
        stats.prune_no_must++;
        return;
    }

    // printf("C:"); for(ui i = 0; i < EC; i++) printf("%u ", Cv3[i]); printf("\n");
    // printf("R:"); for(ui i = 0; i < ER; i++) printf("%u ", Rv3[i]); printf("\n");
    // printf("P:"); for(ui i = 0; i < EP; i++) printf("%u ", Pv3[i]); printf("\n");

    if (EP == 0)
        return;

    if (EP == 1) {
        ui u = Pv3[0];
#ifdef DEBUG
        if (xx == xxx) {
            printf("EP==1\n");
        }
#endif
        numOfNonCNodes = 1;
        Rv3.changeTo(u, ER);
        ui newEC = 0;
        for (ui i = st[u]; i < g.coreNumber; i++) {
            Cv3.changeTo(adj[u][i], newEC++);
        }
        ui newEX = SX;
        for (ui i = SX; i < EX; i++) {
            ui v = Xv3[i];

            bool cc = false;
            for (ui j = 0; j < ed[v]; j++) {
                ui w = adj[v][j];
                if (w == u) {
                    cc = true;
                    break;
                }
            }
            if (cc)
                Xv3.changeTo(v, newEX++);
        }
        auto mainNei2 = [&](ui v) {
            ui& nst = stAdj[deep + 1][v];
            ui& ned = edAdj[deep + 1][v];
            nst = g.coreNumber;
            ned = 0;
            if (newEC > 0)
                for (ui j = st[v]; j < nst;) {
                    if (Cv3.isIn(adj[v][j], 0, newEC)) {
                        std::swap(adj[v][j], adj[v][--nst]);
                    } else
                        j++;
                }
        };

        for (ui i = SX; i < newEX; i++) {
            ui v = Xv3[i];
            if (inC[v])
                continue;
            mainNei2(v);
        }

        BKTomitaRecCCRV3_mustcontain(deep + 1, ER + 1, newEC, 0, SX, newEX, mustContainNodes);
        //Rv3.changeTo
        return;
    }

    //remove the X nodes that has no neighbors in P
    for (ui i = SX; i < EX;) {
        ui v = Xv3[i];
        if (ed[v] == 0) {
            Xv3.changeTo(v, --EX);
        } else
            i++;
    }

    //remove the C nodes that connects to all nodes in P
    bool update = false;
    for (ui i = 0; i < EC;) {
        ui v = Cv3[i];
        if (ed[v] == EP) {
            Cv3.changeTo(v, --EC);
            Rv3.changeTo(v, ER++);
            update = true;
#ifdef DEBUG
            if (xx == xxx)
                printf("remove C %u\n", v);
#endif
            for (ui i = SX; i < EX;) {
                ui vv = Xv3[i];

                if (inC[vv]) {
                    i++;
                    continue;
                }

                bool cc = false;
                for (ui j = st[vv]; j < g.coreNumber; j++) {
                    ui w = adj[vv][j];
                    if (w == v) {
                        cc = true;
                        break;
                    }
                }
                if (cc)
                    i++;
                else
                    Xv3.changeTo(vv, --EX);
            }
        } else
            i++;
    }
    if (update) {
        auto mainNeiInC = [&](ui v) {
            ui nst = g.coreNumber;

            if (EC > 0) {
                for (ui j = st[v]; j < nst;) {
                    if (Cv3.isIn(adj[v][j], 0, EC)) {
                        std::swap(adj[v][j], adj[v][--nst]);
                    } else
                        j++;
                }
            }

            st[v] = nst;
        };

        for (ui i = 0; i < EP; i++) {
            ui v = Pv3[i];
            mainNeiInC(v);
        }
        for (ui i = SX; i < EX; i++) {
            ui v = Xv3[i];
            if (inC[v])
                continue;
            mainNeiInC(v);
        }
        // for(ui i = 0; i < EC; i++) {
        //     ui v = Cv3[i]; mainNeiInC(v);
        // }
    }

    //remove the P nodes that connects to all PC nodes
    update = false;
    for (ui i = 0, rawEp = EP; i < EP;) {
        ui v = Pv3[i];
        if (ed[v] == rawEp - 1 && g.coreNumber - st[v] == EC) {
            Pv3.changeTo(v, --EP);
            Rv3.changeTo(v, ER++);
            update = true;
#ifdef DEBUG
            if (xx == xxx)
                printf("remove P %u\n", v);
#endif
            for (ui i = SX; i < EX;) {
                ui vv = Xv3[i];
                bool cc = false;
                for (ui j = 0; j < ed[vv]; j++) {
                    ui w = adj[vv][j];
                    if (w == v) {
                        cc = true;
                        break;
                    }
                }
                if (cc)
                    i++;
                else
                    Xv3.changeTo(vv, --EX);
            }
        } else {
            i++;
        }
    }
    if (update) {
        auto mainNeiInP = [&](ui v) {
            for (ui j = 0; j < ed[v];) {
                if (Pv3.isIn(adj[v][j], 0, EP)) {
                    j++;
                } else
                    std::swap(adj[v][j], adj[v][--ed[v]]);
            }
        };

        for (ui i = 0; i < EP; i++) {
            ui v = Pv3[i];
            mainNeiInP(v);
        }
        for (ui i = SX; i < EX; i++) {
            ui v = Xv3[i];
            mainNeiInP(v);
        }
        for (ui i = 0; i < EC; i++) {
            ui v = Cv3[i];
            mainNeiInP(v);
        }

        bool f = true;
        for (ui i = 0; i < EP; i++) {
            ui v = Pv3[i];
            if (g.coreNumber - st[v] == EC) {
                f = false;
                break;
            }
        }
        if (f)
            for (ui i = SX; i < EX; i++) {
                ui v = Xv3[i];
                if (inC[v] || g.coreNumber - st[v] == EC) {
                    f = false;
                    break;
                }
            }
        if (f) {
#ifdef PERCENAEG_OF_C
            allCNodes += EC;
            allPCNodes += EC + ER;
#endif
            // if(outP) {
            // printf("%u ", nowU);
            // for(ui i = 0; i < EC; i++) printf("%u ", g.pEdge[g.pIdx[nowU]+Cv3[i]]);
            // for(ui i = 0; i < ER; i++) printf("%u ", g.pEdge[g.pIdx[nowU]+Rv3[i]]);
            // printf("\n");

            trySaveMustClique(ER, EC, EP, mustContainNodes);
            // }
            cnt++;
            if (stopSearch)
                return;
        }

        if (EP == 0)
            return;
    }

    //find the best pivot
    ui pu = Pv3[0], maxD = 0;
    for (ui i = 0; i < EP; i++) {
        ui v = Pv3[i];
        ui d = ed[v] + g.coreNumber - st[v];
        if (d > maxD) {
            maxD = d;
            pu = v;
        }
    }
    for (ui i = 0; i < EC; i++) {
        ui v = Cv3[i];
        // ui d = ed[v] + g.coreNumber-st[v];
        ui d = ed[v] + EC - 1;
        if (d > maxD) {
            maxD = d;
            pu = v;
        }
    }
    for (ui i = SX; i < EX; i++) {
        ui v = Xv3[i];
        ui neiInC = inC[v] ? EC - 1 : g.coreNumber - st[v];
        ui d = ed[v] + neiInC;
        if (d > maxD) {
            maxD = d;
            pu = v;
        }
        // continue;
        if (ed[v] == EP - 1 && neiInC == EC) {
            //remove the X nodes that has one-non-neighbor in P

            for (ui i = 0, ttt = 0; i < ed[v]; i++) {
                ui tmpv = adj[v][i];
                Pv3.changeTo(tmpv, ttt++);
            }
            ui u = Pv3[EP - 1];

            ui newEP = 0;
            ui newEX = SX;
            ui newEC = 0;
#ifdef DEBUG
            if (xx == xxx) {
                printf("XoneNonNei:accNonNei %u, deep %u\n", u, deep);
                fflush(stdout);
                printf("Xv3 %u:", deep);
                for (ui i = SX; i < EX; i++) printf("%u ", Xv3[i]);
                printf("\n");
            }
#endif
            //build new P
            for (ui i = 0; i < ed[u]; i++) {
                ui v = adj[u][i];
                if (Pv3.isin(v, 0, EP))
                    Pv3.changeTo(v, newEP++);
            }
            //build new C
            for (ui i = st[u]; i < g.coreNumber; i++) {
                ui v = adj[u][i];
                if (Cv3.isin(v, 0, EC))
                    Cv3.changeTo(v, newEC++);
            }
#ifdef DEBUG
            if (xx == xxx) {
                printf("newEP %u, newEC %u\n", newEP, newEC);
                fflush(stdout);
            }
#endif
            //build new X
            numOfNonCNodes = 1;
            for (ui i = SX; i < EX; i++) {
                ui v = Xv3[i];
                bool cc = false;
                for (ui j = 0; j < ed[v]; j++) {
                    ui w = adj[v][j];
                    if (w == u) {
                        cc = true;
                        break;
                    }
                }
                if (cc)
                    Xv3.changeTo(v, newEX++);
            }
#ifdef DEBUG
            if (xx == xxx) {
                printf("newEX %u\n", newEX - SX);
                fflush(stdout);
            }
#endif

            // R.push_back(u);

            //maintain the neighbors in P / C
            auto mainNei = [&](ui v) {
                ui& ned = nxtEd[v];
                // nst = g.coreNumber;
                // ned = 0;
                // if(EP > 0)
                // for(ui j = 0; j < ed[v]; j++) {
                //     if(Pv3.isIn(adj[v][j], 0, newEP)) {
                //         std::swap(adj[v][j], adj[v][ned++]);
                //     }
                // }
                // if(EC > 0)
                // for(ui j = st[v]; j < nst; ) {
                //     if(Cv3.isIn(adj[v][j], 0, newEC)) {
                //         std::swap(adj[v][j], adj[v][--nst]);
                //     }
                //     else j++;
                // }

                ned = ed[v];

                for (ui j = 0; j < ned;) {
                    if (Pv3.isIn(adj[v][j], 0, newEP)) {
                        j++;
                    } else
                        std::swap(adj[v][j], adj[v][--ned]);
                }

                if (inC[v])
                    return;

                ui& nst = nxtSt[v];
                nst = st[v];
                for (ui j = st[v]; j < g.coreNumber; j++) {
                    if (!Cv3.isIn(adj[v][j], 0, newEC)) {
                        std::swap(adj[v][j], adj[v][nst++]);
                    }
                }
            };

            for (ui i = 0; i < newEP; i++) {
                ui v = Pv3[i];
                mainNei(v);
            }
            for (ui i = SX; i < newEX; i++) {
                ui v = Xv3[i];
                mainNei(v);
            }
            for (ui i = 0; i < newEC; i++) {
                ui v = Cv3[i];
                mainNei(v);
            }

            Rv3.changeTo(u, ER);

            BKTomitaRecCCRV3_mustcontain(
                deep + 1, ER + 1, newEC, newEP, SX, newEX, mustContainNodes);

            numOfNonCNodes--;

            return;
        }
    }
#ifdef DEBUG
    if (xx == xxx) {
        printf("pu:%u,maxD %u, EC %u, EP %u, ed[pu] %u\n", pu, maxD, EC, EP, ed[pu]);
        fflush(stdout);
    }
#endif
    std::vector<ui> nonNeighbors(EC + EP - maxD + 1);
    ui d = 0;
    ui newEP = 0;
    for (ui i = 0; i < ed[pu]; i++) {
        ui v = adj[pu][i];
        Pv3.changeTo(v, newEP++);
    }
#ifdef DEBUG
    if (xx == xxx) {
        printf("EP-newEP:%u,EC+EP - maxD %u\n", EP - newEP, EC + EP - maxD);
        fflush(stdout);
    }
#endif
    for (ui i = newEP; i < EP; i++) {
        nonNeighbors[d++] = Pv3[i];
    }
    ui newEC = 0;
    if (Cv3.isIn(pu, 0, EC)) {
        newEC = EC - 1;
        Cv3.changeTo(pu, EC - 1);
    } else if (inC[pu]) {
        newEC = EC;
    } else {
        for (ui i = st[pu]; i < g.coreNumber; i++) {
            ui v = adj[pu][i];
            Cv3.changeTo(v, newEC++);
        }
    }
    for (ui i = newEC, dc = 0; i < EC; i++) {
        ui v = Cv3[i];

        //与pu的邻居相连
        bool f = false;
        for (ui j = 0; j < ed[v]; j++) {
            ui w = adj[v][j];
            if (Pv3.isIn(w, 0, newEP)) {
                f = true;
                break;
            }
        }
        if (!f) {
#ifdef FIND_EXAMPLE
            printf("removeCnonNeighbors\n");
            removeCnonNeighbors = true;
#endif
            continue;
        }
        // f = false;
        // //包含前面C点的P中的非邻居
        // for(ui j = d-dc; j < d; j++) {
        //     ui preCv = nonNeighbors[j];

        // }

        nonNeighbors[d++] = v;
        dc++;
    }

#ifdef DEBUG
    if (xx == xxx) {
        printf("nonNei:");
        for (ui ii = 0; ii < d; ii++) printf("%u ", nonNeighbors[ii]);
        printf("\n");
        fflush(stdout);
    }
#endif

    for (ui ii = 0; ii < d; ii++) {
        if (shouldStopMCE())
            return;
        ui u = nonNeighbors[ii];
        ui newEP = 0;
        ui newEX = SX;
        ui newEC = 0;
#ifdef DEBUG
        if (xx == xxx) {
            printf("accNonNei %u, deep %u\n", u, deep);
            fflush(stdout);
            printf("Xv3 %u:", deep);
            for (ui i = SX; i < EX; i++) printf("%u ", Xv3[i]);
            printf("\n");
            printf("Cv3 %u:", deep);
            for (ui i = 0; i < EC; i++) printf("%u ", Cv3[i]);
            printf("\n");
        }
#endif
        //build new P
        for (ui i = 0; i < ed[u]; i++) {
            ui v = adj[u][i];
            if (Pv3.isin(v, 0, EP))
                Pv3.changeTo(v, newEP++);
        }
        //build new C
        if (Pv3.isIn(u, 0, EP))
            for (ui i = st[u]; i < g.coreNumber; i++) {
                ui v = adj[u][i];
                if (Cv3.isin(v, 0, EC))
                    Cv3.changeTo(v, newEC++);
            }
        else {
            Cv3.changeTo(u, EC - 1);
            newEC = EC - 1;
        }
#ifdef DEBUG
        if (xx == xxx) {
            printf("newEP %u, newEC %u\n", newEP, newEC);
            fflush(stdout);
        }
#endif
#ifdef DEBUG
        if (xx == xxx) {
            printf("accNonNei %u, deep %u\n", u, deep);
            fflush(stdout);
            printf("Xv3 %u:", deep);
            for (ui i = SX; i < EX; i++) printf("%u ", Xv3[i]);
            printf("\n");
            printf("Cv3 %u:", deep);
            for (ui i = 0; i < EC; i++) printf("%u ", Cv3[i]);
            printf("\n");
        }
#endif
        //build new X
        if (Pv3.isIn(u, 0, EP)) {
            numOfNonCNodes = 1;
            for (ui i = SX; i < EX; i++) {
                ui v = Xv3[i];
                bool cc = false;
                for (ui j = 0; j < ed[v]; j++) {
                    ui w = adj[v][j];
                    if (w == u) {
                        cc = true;
                        break;
                    }
                }
                if (cc)
                    Xv3.changeTo(v, newEX++);
            }
        } else {
            numOfNonCNodes = 0;
            for (ui i = SX; i < EX; i++) {
                ui v = Xv3[i];

                if (inC[v]) {
#ifdef DEBUG
                    if (xx == xxx) {
                        printf("xcv %u\n", v);
                        fflush(stdout);
                    }
#endif
                    Xv3.changeTo(v, newEX++);
                    continue;
                }

                bool cc = false;
                for (ui j = st[v]; j < g.coreNumber; j++) {
                    ui w = adj[v][j];
                    if (w == u) {
                        cc = true;
                        break;
                    }
                }
                if (cc) {
#ifdef DEBUG
                    if (xx == xxx) {
                        printf("xv %u\n", v);
                        fflush(stdout);
                    }
#endif
                    Xv3.changeTo(v, newEX++);
                }
            }
        }
#ifdef DEBUG
        if (xx == xxx) {
            printf("newEX %u\n", newEX - SX);
            fflush(stdout);
        }
#endif

        // R.push_back(u);

        //maintain the neighbors in P / C
        auto mainNei = [&](ui v) {
            ui& nst = nxtSt[v];
            ui& ned = nxtEd[v];
            nst = g.coreNumber;
            ned = 0;

            if (EP > 0)
                for (ui j = 0; j < ed[v]; j++) {
                    if (Pv3.isIn(adj[v][j], 0, newEP)) {
                        std::swap(adj[v][j], adj[v][ned++]);
                    }
                }

            if (inC[v])
                return;

            if (EC > 0)
                for (ui j = st[v]; j < nst;) {
                    if (Cv3.isIn(adj[v][j], 0, newEC)) {
                        std::swap(adj[v][j], adj[v][--nst]);
                    } else
                        j++;
                }
        };

        for (ui i = 0; i < newEP; i++) {
            ui v = Pv3[i];
            mainNei(v);
        }
        for (ui i = SX; i < newEX; i++) {
            ui v = Xv3[i];
            mainNei(v);
        }
        for (ui i = 0; i < newEC; i++) {
            ui v = Cv3[i];  //mainNei(v);

            ui& ned = nxtEd[v];
            ned = 0;
            if (EP > 0)
                for (ui j = 0; j < ed[v]; j++) {
                    if (Pv3.isIn(adj[v][j], 0, newEP)) {
                        std::swap(adj[v][j], adj[v][ned++]);
                    }
                }
        }

        Rv3.changeTo(u, ER);

        BKTomitaRecCCRV3_mustcontain(deep + 1, ER + 1, newEC, newEP, SX, newEX, mustContainNodes);
        if (stopSearch)
            return;

        Xv3.changeTo(u, --SX);
#ifdef DEBUG
        if (xx == xxx) {
            printf("Xv322 %u:", deep);
            for (ui i = SX; i < EX; i++) printf("%u ", Xv3[i]);
            printf("\n");
        }
#endif
        if (Pv3.isIn(u, 0, EP)) {
            numOfNonCNodes--;
            Pv3.changeTo(u, --EP);
        } else {
            --EC;

            // Cv3.changeTo(u, --EC);
        }
    }

    for (ui ii = 0; ii < d; ii++) {
        Xv3.changeTo(nonNeighbors[ii], SX++);
        // if(inC[nonNeighbors[ii]]) {
        //     Cv3.changeTo(nonNeighbors[ii], EC++);
        // }
    }

#ifdef DEBUG
    if (xx == xxx) {
        printf("Xvlast %u:", deep);
        for (ui i = SX; i < EX; i++) printf("%u ", Xv3[i]);
        printf("\n");
        printf("Cvlast %u:", deep);
        for (ui i = 0; i < EC; i++) printf("%u ", Cv3[i]);
        printf("\n\n");
    }
#endif
}

ui
MCE::run() {
    // printf("MCE::runCoreCliqueRemovalV3, core %u\n", g.coreNumber);fflush(stdout);
    Cv3.resize(g.maxD);
    Pv3.resize(g.maxD);
    Rv3.resize(g.maxD);
    Xv3.resize(g.maxD);

#ifdef DEBUG
    g.print();
#endif

    cnt = 0;
    // printf("a0\n");fflush(stdout);
    edAdj.resize(g.coreNumber + 1);
    for (ui i = 0; i <= g.coreNumber; i++) edAdj[i].resize(g.maxD);
    stAdj.resize(g.coreNumber + 1);
    for (ui i = 0; i <= g.coreNumber; i++) stAdj[i].resize(g.maxD);
    adj.resize(g.maxD);
    for (ui i = 0; i < g.maxD; i++) {
        adj[i].resize(g.coreNumber);
    }
    // reId.resize(g.n, g.n);
    if (reId.capacity() < g.n)
        reId.resize(g.n);
    std::fill(reId.begin(), reId.begin() + g.n, g.n);

    if (inC.capacity() < g.maxD)
        inC.resize(g.maxD, false);
    std::fill(inC.begin(), inC.begin() + g.maxD, false);

    // std::vector<ui> newCliqueNei;
    // printf("a1\n");fflush(stdout);
    ListLinearHeap heap(g.coreNumber, g.coreNumber);
    heap.reserve(g.coreNumber, g.coreNumber);

    std::vector<ui> id_s(g.coreNumber + 1);
    std::vector<ui> degree(g.coreNumber);
    std::vector<bool> removed(g.coreNumber);

    std::vector<ui> pIdx(g.coreNumber + 1);
    std::vector<ui> eIdx(g.coreNumber + 1);
    std::vector<ui> pEdge(g.coreNumber * g.coreNumber);

    std::vector<ui> clique(g.coreNumber);
    std::vector<ui> nxtC(g.coreNumber);

    for (ui u = 0; u < g.n; u++) {
        nowU = u;
        if (1 + g.pIdx[u + 1] - g.pIdx2[u] < threshold)
            continue;
// printf("%u %u\n", u, g.n);fflush(stdout);
#ifdef DEBUG
        xx = u;
        if (xx == xxx)
            printf("\n    st %u\n", u);
#endif

#ifdef RUNNINGTIME_PER_NODE
        double stTime = clock();
#endif

        for (ui i = g.pIdx2[u]; i < g.pIdx[u + 1]; i++) {
            reId[g.pEdge[i]] = i - g.pIdx2[u] + 1;
        }

        ui n = g.pIdx[u + 1] - g.pIdx2[u];
        for (ui i = 0; i <= n; i++) pIdx[i] = 0;

        for (ui i = g.pIdx2[u]; i < g.pIdx[u + 1]; i++) {
            ui v = g.pEdge[i];

            for (ui j = g.pIdx2[v]; j < g.pIdx[v + 1]; j++) {
                ui w = g.pEdge[j];
                if (reId[w] < g.n) {
                    pIdx[i - g.pIdx2[u] + 1]++;
                    pIdx[reId[w]]++;
                }
            }
        }
        for (ui i = 1; i <= n; i++) pIdx[i] += pIdx[i - 1];
        for (ui i = g.pIdx2[u]; i < g.pIdx[u + 1]; i++) {
            ui v = g.pEdge[i];
            for (ui j = g.pIdx2[v]; j < g.pIdx[v + 1]; j++) {
                ui w = g.pEdge[j];
                if (reId[w] < g.n) {
                    pEdge[pIdx[i - g.pIdx2[u]]++] = reId[w] - 1;
                    pEdge[pIdx[reId[w] - 1]++] = i - g.pIdx2[u];
                }
            }
        }
        for (ui i = n; i > 0; i--) pIdx[i] = pIdx[i - 1];
        pIdx[0] = 0;

        clique.clear();
        nxtC.clear();

        ui queue_n = 0, new_size = 0;
        for (ui i = 0; i < n; i++) removed[i] = false;
        for (ui i = 0; i < n; i++) {
            if (pIdx[i + 1] - pIdx[i] + 2 < threshold)
                id_s[queue_n++] = i;
            degree[i] = pIdx[i + 1] - pIdx[i];
        }
        for (ui i = 0; i < queue_n; i++) {
            ui uu = id_s[i];
            degree[uu] = 0;
            removed[uu] = true;
            for (ui j = pIdx[uu]; j < pIdx[uu + 1]; j++)
                if (degree[pEdge[j]] > 0) {
                    if ((degree[pEdge[j]]--) + 2 == threshold)
                        id_s[queue_n++] = pEdge[j];
                }
        }

        for (ui i = 0; i < n; i++) {
            if (degree[i] + 2 >= threshold)
                id_s[queue_n + (new_size++)] = i;
            else {
                removed[i] = true;
            }
        }

        if (new_size == 0) {
            for (ui i = g.pIdx2[u]; i < g.pIdx[u + 1]; i++) {
                reId[g.pEdge[i]] = g.n;
            }
            continue;
        } else {
#ifdef DEBUG
            if (xx == xxx) {
                printf("queue_n:%u\n", queue_n);  //should be zero by default
                printf("To peel %u:", new_size);
                for (ui i = 0; i < new_size; i++) {
                    printf("%u-%u ", id_s[queue_n + i] + 1, degree[id_s[queue_n + i]]);
                }
                printf("\n");
                fflush(stdout);
            }
#endif

            //ListLinearHeap *heap = new ListLinearHeap(n, new_size-1);
            heap.initRaw(new_size, new_size - 1, id_s.data() + queue_n, degree.data());

            // printf("b");fflush(stdout);
            for (ui i = 0; i < new_size; i++) {
                ui v, key;

                heap.pop_min(v, key);

                id_s[queue_n + i] = v;
                if (key + i + 1 == new_size) {
                    ui x_size = i + 1;
                    heap.get_ids(id_s.data() + queue_n, x_size);

                    assert(x_size == new_size);
                    // printf("	last key %u, x_size %u\n", key, x_size);
                    for (ui j = i; j < new_size; j++) {
                        clique.push_back(g.pEdge[g.pIdx2[u] + id_s[queue_n + j]]);
                    }

                    for (ui j = 0; j < i; j++) {
                        //gready extend
                        ui tu = g.pEdge[g.pIdx2[u] + id_s[queue_n + j]];
                        // bool connectAll = true;
                        // for(ui cu : clique) {
                        //     if(cu < tu) {
                        //         if(!g.connectOut(cu, tu)) {
                        //             connectAll = false; break;
                        //         }
                        //     }
                        //     else if(!g.connectOut(tu, cu)) {
                        //         connectAll = false; break;
                        //     }
                        // }
                        // if(connectAll)  clique.push_back(tu);
                        // else
                        nxtC.push_back(tu);
                    }

                    break;
                }
                removed[v] = 1;
                // printf("%u %u\n", u, key);
                // fflush(stdout);

                for (ui j = pIdx[v]; j < pIdx[v + 1]; j++)
                    if (removed[pEdge[j]] == false) {
                        heap.decrement(pEdge[j], 1);
                    }
            }
            // printf("c\n");fflush(stdout);
        }

        // printf("findClique\n");fflush(stdout);

        for (ui i = g.pIdx[u]; i < g.pIdx[u + 1]; i++) {
            reId[g.pEdge[i]] = i - g.pIdx[u];
        }

#ifdef DEBUG
        if (xx == xxx) {
            printf("nxtC: ");
            for (auto v : nxtC) printf("%u ", reId[v]);
            printf("\n");
            printf("clique: ");
            for (auto v : clique) printf("%u ", reId[v]);
            printf("\n");
            fflush(stdout);
        }
#endif
        ui EC = 0;
        ui EX = g.maxD, SX = g.maxD;
        ui ER = 0;
        ui EP = 0;
        for (ui i = g.pIdx[u]; i < g.pIdx2[u]; i++) {
            Xv3.changeTo(reId[g.pEdge[i]], --SX);
        }
        for (ui v : nxtC) {
            Pv3.changeTo(reId[v], EP++);
        }
        for (ui v : clique) {
            Cv3.changeTo(reId[v], EC++);
        }
#ifdef DEBUG
        if (xx == xxx) {
            printf("EC %u, EP %u, EX %u\n", EC, EP, EX);
            fflush(stdout);
        }
#endif
        //X nodes, the neighbors in P
        for (ui i = g.pIdx[u]; i < g.pIdx2[u]; i++) {
            ui v = g.pEdge[i];
            ui vId = i - g.pIdx[u];
            edAdj[0][vId] = 0;
            for (ui j = g.pIdx2[v]; j < g.pIdx[v + 1]; j++) {
                ui w = g.pEdge[j];
                if (reId[w] == g.n)
                    continue;
                if (Pv3.isIn(reId[w], 0, EP)) {
                    adj[vId][edAdj[0][vId]++] = reId[w];
                }
            }
        }
        //P/C nodes, the neighbors in P
        for (ui i = g.pIdx2[u]; i < g.pIdx[u + 1]; i++) {
            ui vId = i - g.pIdx[u];
            edAdj[0][vId] = 0;
        }
        for (ui v : nxtC) {
            ui vId = reId[v];
            for (ui j = g.pIdx2[v]; j < g.pIdx[v + 1]; j++) {
                ui w = g.pEdge[j];
                ui wId = reId[w];
                if (reId[w] == g.n)
                    continue;
                if (Pv3.isIn(wId, 0, EP)) {
                    adj[vId][edAdj[0][vId]++] = wId;
                    adj[wId][edAdj[0][wId]++] = reId[v];
                } else if (Cv3.isIn(wId, 0, EC)) {
                    adj[wId][edAdj[0][wId]++] = reId[v];
                }
            }
        }
        for (ui v : clique) {
            ui vId = reId[v];
            for (ui j = g.pIdx2[v]; j < g.pIdx[v + 1]; j++) {
                ui w = g.pEdge[j];
                ui wId = reId[w];
                if (reId[w] == g.n)
                    continue;
                if (Pv3.isIn(wId, 0, EP)) {
                    adj[vId][edAdj[0][vId]++] = wId;
                }
            }
        }
        //X nodes, the neighbors in C
        for (ui i = g.pIdx[u]; i < g.pIdx2[u]; i++) {
            ui v = g.pEdge[i];
            ui vId = i - g.pIdx[u];
            stAdj[0][vId] = g.coreNumber;

            for (ui j = g.pIdx2[v]; j < g.pIdx[v + 1]; j++) {
                ui w = g.pEdge[j];
                if (reId[w] == g.n)
                    continue;
                if (Cv3.isIn(reId[w], 0, EC)) {
                    adj[vId][--stAdj[0][vId]] = reId[w];
                }
            }
        }
        //P/C nodes, the neighbors in C
        for (ui v : nxtC) {
            ui vId = reId[v];
            stAdj[0][vId] = g.coreNumber;
        }
        for (ui v : clique) {
            ui vId = reId[v];
            stAdj[0][vId] = g.coreNumber;
        }
        for (ui v : nxtC) {
            ui vId = reId[v];
            for (ui j = g.pIdx2[v]; j < g.pIdx[v + 1]; j++) {
                ui w = g.pEdge[j];
                if (reId[w] == g.n)
                    continue;
                if (Cv3.isIn(reId[w], 0, EC)) {
                    adj[vId][--stAdj[0][vId]] = reId[w];
                }
            }
        }
        for (ui w : clique) {
            ui wId = reId[w];
            inC[wId] = true;
            for (ui j = g.pIdx2[w]; j < g.pIdx[w + 1]; j++) {
                ui v = g.pEdge[j];
                ui vId = reId[v];
                if (vId == g.n)
                    continue;
                if (Pv3.isIn(vId, 0, EP)) {
                    adj[vId][--stAdj[0][vId]] = wId;
                }
                // else if(Cv3.isIn(vId, 0, EC)) {
                //     adj[vId][--stAdj[0][vId]] = wId;
                //     adj[wId][--stAdj[0][wId]] = vId;
                // }
            }
        }
#ifdef FIND_EXAMPLE
        if (EC)
            CisnotEmpty = true;
        filterByFromP = false;
        removeCnonNeighbors = false;
        removeByX = false;
#endif

#ifdef DEBUG
        ui preA = cnt;
#endif
        numOfNonCNodes = 1;
        BKTomitaRecCCRV3(0, ER, EC, EP, SX, EX);
#ifdef DEBUG
        // if(xx == xxx)
        printf("local:%u:%u, %u\n", u, cnt - preA, maxCliqueCntSave);
        fflush(stdout);
#endif

#ifdef FIND_EXAMPLE
        if (CisnotEmpty && filterByFromP && removeCnonNeighbors && removeByX) {
            printf("ThisIsIt! %u, %u, %u\n", u, nxtC.size() + clique.size(), EX - SX);
        }
#endif
        for (ui w : clique) {
            ui wId = reId[w];
            inC[wId] = false;
        }

        for (ui i = g.pIdx[u]; i < g.pIdx[u + 1]; i++) {
            reId[g.pEdge[i]] = g.n;
        }

#ifdef RUNNINGTIME_PER_NODE
        double edTime = clock();
        maxTimePerNode = std::max(maxTimePerNode, edTime - stTime);
#endif
    }
#ifdef COUNT_NUM_SEARCH_NODES
    printf("numOfDFSSearchNodes:%llu\n", numOfDFSSearchNodes);
#endif
#ifdef PERCENAEG_OF_C
    printf("allCNodes:%llu\n", allCNodes);
    printf("allPCNodes:%llu\n", allPCNodes);
    printf("percentage:%f\n", 1.0 * allCNodes / allPCNodes);
#endif

#ifdef RUNNINGTIME_PER_NODE
    printf("maxTimePerNode:%fs\n", maxTimePerNode / CLOCKS_PER_SEC);
#endif
    return cnt;
}

// #define DEBUG
// ui xx, xxx;

void
MCE::BKTomitaRecCCRV3(ui deep, ui ER, ui EC, ui EP, ui SX, ui EX) {
#ifdef DEBUG
    if (xx == xxx) {
        printf("\n    deep %u\n", deep);
        fflush(stdout);
        printf("C:");
        for (ui i = 0; i < EC; i++) printf("%u ", Cv3[i]);
        printf("\n");
        printf("R:");
        for (ui i = 0; i < ER; i++) printf("%u ", Rv3[i]);
        printf("\n");
        printf("P:");
        for (ui i = 0; i < EP; i++) printf("%u ", Pv3[i]);
        printf("\n");
        printf("X:");
        for (ui i = SX; i < EX; i++) printf("%u ", Xv3[i]);
        printf("\n");
        printf("P: P/C\n");
        for (ui i = 0; i < EP; i++) {
            ui vId = Pv3[i];
            printf("%u: ", vId);
            for (ui j = 0; j < edAdj[deep][vId]; j++) {
                printf("%u ", adj[vId][j]);
            }
            printf(" / ");
            for (ui j = stAdj[deep][vId]; j < g.coreNumber; j++) {
                printf("%u ", adj[vId][j]);
            }
            printf("\n");
        }

        printf("PXsg:\n");
        for (ui i = 0; i < EP; i++) {
            ui vId = Pv3[i];
            printf("%u: ", vId);
            for (ui j = SX; j < EX; j++) {
                if (g.connect(g.pEdge[g.pIdx[xx] + vId], g.pEdge[g.pIdx[xx] + Xv3[j]]))
                    printf("%u ", Xv3[j]);
            }
            printf("\n");
        }

        printf("PPsg:\n");
        for (ui i = 0; i < EP; i++) {
            ui vId = Pv3[i];
            printf("%u: ", vId);
            for (ui j = 0; j < EP; j++) {
                if (g.connect(g.pEdge[g.pIdx[xx] + vId], g.pEdge[g.pIdx[xx] + Pv3[j]]))
                    printf("%u ", Pv3[j]);
            }
            printf("\n");
        }
        printf("CPsg:\n");
        for (ui i = 0; i < EC; i++) {
            ui vId = Cv3[i];
            printf("%u: ", vId);
            for (ui j = 0; j < edAdj[deep][vId]; j++) {
                printf("%u ", adj[vId][j]);
            }
            printf("\n");
        }
        printf("XCsg:\n");
        for (ui i = SX; i < EX; i++) {
            ui vId = Xv3[i];
            printf("%u: ", vId);
            for (ui j = stAdj[deep][vId]; j < g.coreNumber; j++) {
                printf("%u ", adj[vId][j]);
            }
            printf("\n");
        }
        printf("CCsg:\n");
        for (ui i = 0; i < EC; i++) {
            ui vId = Cv3[i];
            printf("%u: ", vId);
            for (ui j = stAdj[deep][vId]; j < g.coreNumber; j++) {
                printf("%u ", adj[vId][j]);
            }
            printf("\n");
        }
    }
#endif

#ifdef COUNT_NUM_SEARCH_NODES
    numOfDFSSearchNodes++;
#endif

    bool f = true;
    std::vector<ui>& ed = edAdj[deep];
    std::vector<ui>& st = stAdj[deep];
    std::vector<ui>& nxtEd = edAdj[deep + 1];
    std::vector<ui>& nxtSt = stAdj[deep + 1];

    // if(numOfNonCNodes == 0 && deep > 0) f = false;
    if (numOfNonCNodes == 0) {
#ifdef FIND_EXAMPLE
        filterByFromP = true;
#endif
        f = false;
    }
    if (f && EX - SX + EP > 0) {
        for (ui i = 0; i < EP; i++) {
            ui v = Pv3[i];
            if (g.coreNumber - st[v] == EC) {
                f = false;
                break;
            }
        }
        if (f) {
            for (ui i = SX; i < EX; i++) {
                ui v = Xv3[i];
                if (inC[v] || g.coreNumber - st[v] == EC) {
                    f = false;
                    break;
                }
            }
#ifdef FIND_EXAMPLE
            if (!f)
                removeByX = true;
#endif
        }
    }
#ifdef DEBUG
    if (xx == xxx) {
        printf("F:%d\n", (int)f);
    }
#endif
    if (f) {
#ifdef PERCENAEG_OF_C
        allCNodes += EC;
        allPCNodes += EC + ER;
#endif
        // if(outP) {
        // printf("%u ", nowU);
        // for(ui i = 0; i < EC; i++) printf("%u ", g.pEdge[g.pIdx[nowU]+Cv3[i]]);
        // for(ui i = 0; i < ER; i++) printf("%u ", g.pEdge[g.pIdx[nowU]+Rv3[i]]);
        // printf("\n");
        if (1 + EC + ER >= threshold) {
            if (maxCliqueCntSave >= (*maxCliques).size()) {
                (*maxCliques).resize((maxCliqueCntSave + 1) * 2);
            }
            (*maxCliques)[maxCliqueCntSave].clear();
            (*maxCliques)[maxCliqueCntSave].push_back(g.mp[nowU]);
            for (ui i = 0; i < EC; i++) {
                ui u = g.pEdge[g.pIdx[nowU] + Cv3[i]];
                (*maxCliques)[maxCliqueCntSave].push_back(g.mp[u]);
            }
            for (ui i = 0; i < ER; i++) {
                ui u = g.pEdge[g.pIdx[nowU] + Rv3[i]];
                (*maxCliques)[maxCliqueCntSave].push_back(g.mp[u]);
            }
            maxCliqueCntSave++;
        }
        // }
        cnt++;
    }

    if (EP == 0)
        return;

    if (EP == 1) {
        ui u = Pv3[0];
#ifdef DEBUG
        if (xx == xxx) {
            printf("EP==1\n");
        }
#endif
        numOfNonCNodes = 1;
        Rv3.changeTo(u, ER);
        ui newEC = 0;
        for (ui i = st[u]; i < g.coreNumber; i++) {
            Cv3.changeTo(adj[u][i], newEC++);
        }
        ui newEX = SX;
        for (ui i = SX; i < EX; i++) {
            ui v = Xv3[i];

            bool cc = false;
            for (ui j = 0; j < ed[v]; j++) {
                ui w = adj[v][j];
                if (w == u) {
                    cc = true;
                    break;
                }
            }
            if (cc)
                Xv3.changeTo(v, newEX++);
        }
        auto mainNei2 = [&](ui v) {
            ui& nst = stAdj[deep + 1][v];
            ui& ned = edAdj[deep + 1][v];
            nst = g.coreNumber;
            ned = 0;
            if (newEC > 0)
                for (ui j = st[v]; j < nst;) {
                    if (Cv3.isIn(adj[v][j], 0, newEC)) {
                        std::swap(adj[v][j], adj[v][--nst]);
                    } else
                        j++;
                }
        };

        for (ui i = SX; i < newEX; i++) {
            ui v = Xv3[i];
            if (inC[v])
                continue;
            mainNei2(v);
        }

        BKTomitaRecCCRV3(deep + 1, ER + 1, newEC, 0, SX, newEX);
        //Rv3.changeTo
        return;
    }

    //remove the X nodes that has no neighbors in P
    for (ui i = SX; i < EX;) {
        ui v = Xv3[i];
        if (ed[v] == 0) {
            Xv3.changeTo(v, --EX);
        } else
            i++;
    }

    //remove the C nodes that connects to all nodes in P
    bool update = false;
    for (ui i = 0; i < EC;) {
        ui v = Cv3[i];
        if (ed[v] == EP) {
            Cv3.changeTo(v, --EC);
            Rv3.changeTo(v, ER++);
            update = true;
#ifdef DEBUG
            if (xx == xxx)
                printf("remove C %u\n", v);
#endif
            for (ui i = SX; i < EX;) {
                ui vv = Xv3[i];

                if (inC[vv]) {
                    i++;
                    continue;
                }

                bool cc = false;
                for (ui j = st[vv]; j < g.coreNumber; j++) {
                    ui w = adj[vv][j];
                    if (w == v) {
                        cc = true;
                        break;
                    }
                }
                if (cc)
                    i++;
                else
                    Xv3.changeTo(vv, --EX);
            }
        } else
            i++;
    }
    if (update) {
        auto mainNeiInC = [&](ui v) {
            ui nst = g.coreNumber;

            if (EC > 0) {
                for (ui j = st[v]; j < nst;) {
                    if (Cv3.isIn(adj[v][j], 0, EC)) {
                        std::swap(adj[v][j], adj[v][--nst]);
                    } else
                        j++;
                }
            }

            st[v] = nst;
        };

        for (ui i = 0; i < EP; i++) {
            ui v = Pv3[i];
            mainNeiInC(v);
        }
        for (ui i = SX; i < EX; i++) {
            ui v = Xv3[i];
            if (inC[v])
                continue;
            mainNeiInC(v);
        }
        // for(ui i = 0; i < EC; i++) {
        //     ui v = Cv3[i]; mainNeiInC(v);
        // }
    }

    //remove the P nodes that connects to all PC nodes
    update = false;
    for (ui i = 0, rawEp = EP; i < EP;) {
        ui v = Pv3[i];
        if (ed[v] == rawEp - 1 && g.coreNumber - st[v] == EC) {
            Pv3.changeTo(v, --EP);
            Rv3.changeTo(v, ER++);
            update = true;
#ifdef DEBUG
            if (xx == xxx)
                printf("remove P %u\n", v);
#endif
            for (ui i = SX; i < EX;) {
                ui vv = Xv3[i];
                bool cc = false;
                for (ui j = 0; j < ed[vv]; j++) {
                    ui w = adj[vv][j];
                    if (w == v) {
                        cc = true;
                        break;
                    }
                }
                if (cc)
                    i++;
                else
                    Xv3.changeTo(vv, --EX);
            }
        } else {
            i++;
        }
    }
    if (update) {
        auto mainNeiInP = [&](ui v) {
            for (ui j = 0; j < ed[v];) {
                if (Pv3.isIn(adj[v][j], 0, EP)) {
                    j++;
                } else
                    std::swap(adj[v][j], adj[v][--ed[v]]);
            }
        };

        for (ui i = 0; i < EP; i++) {
            ui v = Pv3[i];
            mainNeiInP(v);
        }
        for (ui i = SX; i < EX; i++) {
            ui v = Xv3[i];
            mainNeiInP(v);
        }
        for (ui i = 0; i < EC; i++) {
            ui v = Cv3[i];
            mainNeiInP(v);
        }

        bool f = true;
        for (ui i = 0; i < EP; i++) {
            ui v = Pv3[i];
            if (g.coreNumber - st[v] == EC) {
                f = false;
                break;
            }
        }
        if (f)
            for (ui i = SX; i < EX; i++) {
                ui v = Xv3[i];
                if (inC[v] || g.coreNumber - st[v] == EC) {
                    f = false;
                    break;
                }
            }
        if (f) {
#ifdef PERCENAEG_OF_C
            allCNodes += EC;
            allPCNodes += EC + ER;
#endif
            // if(outP) {
            // printf("%u ", nowU);
            // for(ui i = 0; i < EC; i++) printf("%u ", g.pEdge[g.pIdx[nowU]+Cv3[i]]);
            // for(ui i = 0; i < ER; i++) printf("%u ", g.pEdge[g.pIdx[nowU]+Rv3[i]]);
            // printf("\n");

            if (1 + EC + ER >= threshold) {
                if (maxCliqueCntSave >= (*maxCliques).size()) {
                    (*maxCliques).resize((maxCliqueCntSave + 1) * 2);
                }
                (*maxCliques)[maxCliqueCntSave].clear();
                (*maxCliques)[maxCliqueCntSave].push_back(g.mp[nowU]);
                for (ui i = 0; i < EC; i++) {
                    ui u = g.pEdge[g.pIdx[nowU] + Cv3[i]];
                    (*maxCliques)[maxCliqueCntSave].push_back(g.mp[u]);
                }
                for (ui i = 0; i < ER; i++) {
                    ui u = g.pEdge[g.pIdx[nowU] + Rv3[i]];
                    (*maxCliques)[maxCliqueCntSave].push_back(g.mp[u]);
                }
                maxCliqueCntSave++;
            }
            // }
            cnt++;
        }

        if (EP == 0)
            return;
    }

    //find the best pivot
    ui pu = Pv3[0], maxD = 0;
    for (ui i = 0; i < EP; i++) {
        ui v = Pv3[i];
        ui d = ed[v] + g.coreNumber - st[v];
        if (d > maxD) {
            maxD = d;
            pu = v;
        }
    }
    for (ui i = 0; i < EC; i++) {
        ui v = Cv3[i];
        // ui d = ed[v] + g.coreNumber-st[v];
        ui d = ed[v] + EC - 1;
        if (d > maxD) {
            maxD = d;
            pu = v;
        }
    }
    for (ui i = SX; i < EX; i++) {
        ui v = Xv3[i];
        ui neiInC = inC[v] ? EC - 1 : g.coreNumber - st[v];
        ui d = ed[v] + neiInC;
        if (d > maxD) {
            maxD = d;
            pu = v;
        }
        // continue;
        if (ed[v] == EP - 1 && neiInC == EC) {
            //remove the X nodes that has one-non-neighbor in P

            for (ui i = 0, ttt = 0; i < ed[v]; i++) {
                ui tmpv = adj[v][i];
                Pv3.changeTo(tmpv, ttt++);
            }
            ui u = Pv3[EP - 1];

            ui newEP = 0;
            ui newEX = SX;
            ui newEC = 0;
#ifdef DEBUG
            if (xx == xxx) {
                printf("XoneNonNei:accNonNei %u, deep %u\n", u, deep);
                fflush(stdout);
                printf("Xv3 %u:", deep);
                for (ui i = SX; i < EX; i++) printf("%u ", Xv3[i]);
                printf("\n");
            }
#endif
            //build new P
            for (ui i = 0; i < ed[u]; i++) {
                ui v = adj[u][i];
                if (Pv3.isin(v, 0, EP))
                    Pv3.changeTo(v, newEP++);
            }
            //build new C
            for (ui i = st[u]; i < g.coreNumber; i++) {
                ui v = adj[u][i];
                if (Cv3.isin(v, 0, EC))
                    Cv3.changeTo(v, newEC++);
            }
#ifdef DEBUG
            if (xx == xxx) {
                printf("newEP %u, newEC %u\n", newEP, newEC);
                fflush(stdout);
            }
#endif
            //build new X
            numOfNonCNodes = 1;
            for (ui i = SX; i < EX; i++) {
                ui v = Xv3[i];
                bool cc = false;
                for (ui j = 0; j < ed[v]; j++) {
                    ui w = adj[v][j];
                    if (w == u) {
                        cc = true;
                        break;
                    }
                }
                if (cc)
                    Xv3.changeTo(v, newEX++);
            }
#ifdef DEBUG
            if (xx == xxx) {
                printf("newEX %u\n", newEX - SX);
                fflush(stdout);
            }
#endif

            // R.push_back(u);

            //maintain the neighbors in P / C
            auto mainNei = [&](ui v) {
                ui& ned = nxtEd[v];
                // nst = g.coreNumber;
                // ned = 0;
                // if(EP > 0)
                // for(ui j = 0; j < ed[v]; j++) {
                //     if(Pv3.isIn(adj[v][j], 0, newEP)) {
                //         std::swap(adj[v][j], adj[v][ned++]);
                //     }
                // }
                // if(EC > 0)
                // for(ui j = st[v]; j < nst; ) {
                //     if(Cv3.isIn(adj[v][j], 0, newEC)) {
                //         std::swap(adj[v][j], adj[v][--nst]);
                //     }
                //     else j++;
                // }

                ned = ed[v];

                for (ui j = 0; j < ned;) {
                    if (Pv3.isIn(adj[v][j], 0, newEP)) {
                        j++;
                    } else
                        std::swap(adj[v][j], adj[v][--ned]);
                }

                if (inC[v])
                    return;

                ui& nst = nxtSt[v];
                nst = st[v];
                for (ui j = st[v]; j < g.coreNumber; j++) {
                    if (!Cv3.isIn(adj[v][j], 0, newEC)) {
                        std::swap(adj[v][j], adj[v][nst++]);
                    }
                }
            };

            for (ui i = 0; i < newEP; i++) {
                ui v = Pv3[i];
                mainNei(v);
            }
            for (ui i = SX; i < newEX; i++) {
                ui v = Xv3[i];
                mainNei(v);
            }
            for (ui i = 0; i < newEC; i++) {
                ui v = Cv3[i];
                mainNei(v);
            }

            Rv3.changeTo(u, ER);

            BKTomitaRecCCRV3(deep + 1, ER + 1, newEC, newEP, SX, newEX);

            numOfNonCNodes--;

            return;
        }
    }
#ifdef DEBUG
    if (xx == xxx) {
        printf("pu:%u,maxD %u, EC %u, EP %u, ed[pu] %u\n", pu, maxD, EC, EP, ed[pu]);
        fflush(stdout);
    }
#endif
    std::vector<ui> nonNeighbors(EC + EP - maxD + 1);
    ui d = 0;
    ui newEP = 0;
    for (ui i = 0; i < ed[pu]; i++) {
        ui v = adj[pu][i];
        Pv3.changeTo(v, newEP++);
    }
#ifdef DEBUG
    if (xx == xxx) {
        printf("EP-newEP:%u,EC+EP - maxD %u\n", EP - newEP, EC + EP - maxD);
        fflush(stdout);
    }
#endif
    for (ui i = newEP; i < EP; i++) {
        nonNeighbors[d++] = Pv3[i];
    }
    ui newEC = 0;
    if (Cv3.isIn(pu, 0, EC)) {
        newEC = EC - 1;
        Cv3.changeTo(pu, EC - 1);
    } else if (inC[pu]) {
        newEC = EC;
    } else {
        for (ui i = st[pu]; i < g.coreNumber; i++) {
            ui v = adj[pu][i];
            Cv3.changeTo(v, newEC++);
        }
    }
    for (ui i = newEC, dc = 0; i < EC; i++) {
        ui v = Cv3[i];

        //与pu的邻居相连
        bool f = false;
        for (ui j = 0; j < ed[v]; j++) {
            ui w = adj[v][j];
            if (Pv3.isIn(w, 0, newEP)) {
                f = true;
                break;
            }
        }
        if (!f) {
#ifdef FIND_EXAMPLE
            printf("removeCnonNeighbors\n");
            removeCnonNeighbors = true;
#endif
            continue;
        }
        // f = false;
        // //包含前面C点的P中的非邻居
        // for(ui j = d-dc; j < d; j++) {
        //     ui preCv = nonNeighbors[j];

        // }

        nonNeighbors[d++] = v;
        dc++;
    }

#ifdef DEBUG
    if (xx == xxx) {
        printf("nonNei:");
        for (ui ii = 0; ii < d; ii++) printf("%u ", nonNeighbors[ii]);
        printf("\n");
        fflush(stdout);
    }
#endif

    for (ui ii = 0; ii < d; ii++) {
        ui u = nonNeighbors[ii];
        ui newEP = 0;
        ui newEX = SX;
        ui newEC = 0;
#ifdef DEBUG
        if (xx == xxx) {
            printf("accNonNei %u, deep %u\n", u, deep);
            fflush(stdout);
            printf("Xv3 %u:", deep);
            for (ui i = SX; i < EX; i++) printf("%u ", Xv3[i]);
            printf("\n");
            printf("Cv3 %u:", deep);
            for (ui i = 0; i < EC; i++) printf("%u ", Cv3[i]);
            printf("\n");
        }
#endif
        //build new P
        for (ui i = 0; i < ed[u]; i++) {
            ui v = adj[u][i];
            if (Pv3.isin(v, 0, EP))
                Pv3.changeTo(v, newEP++);
        }
        //build new C
        if (Pv3.isIn(u, 0, EP))
            for (ui i = st[u]; i < g.coreNumber; i++) {
                ui v = adj[u][i];
                if (Cv3.isin(v, 0, EC))
                    Cv3.changeTo(v, newEC++);
            }
        else {
            Cv3.changeTo(u, EC - 1);
            newEC = EC - 1;
        }
#ifdef DEBUG
        if (xx == xxx) {
            printf("newEP %u, newEC %u\n", newEP, newEC);
            fflush(stdout);
        }
#endif
#ifdef DEBUG
        if (xx == xxx) {
            printf("accNonNei %u, deep %u\n", u, deep);
            fflush(stdout);
            printf("Xv3 %u:", deep);
            for (ui i = SX; i < EX; i++) printf("%u ", Xv3[i]);
            printf("\n");
            printf("Cv3 %u:", deep);
            for (ui i = 0; i < EC; i++) printf("%u ", Cv3[i]);
            printf("\n");
        }
#endif
        //build new X
        if (Pv3.isIn(u, 0, EP)) {
            numOfNonCNodes = 1;
            for (ui i = SX; i < EX; i++) {
                ui v = Xv3[i];
                bool cc = false;
                for (ui j = 0; j < ed[v]; j++) {
                    ui w = adj[v][j];
                    if (w == u) {
                        cc = true;
                        break;
                    }
                }
                if (cc)
                    Xv3.changeTo(v, newEX++);
            }
        } else {
            numOfNonCNodes = 0;
            for (ui i = SX; i < EX; i++) {
                ui v = Xv3[i];

                if (inC[v]) {
#ifdef DEBUG
                    if (xx == xxx) {
                        printf("xcv %u\n", v);
                        fflush(stdout);
                    }
#endif
                    Xv3.changeTo(v, newEX++);
                    continue;
                }

                bool cc = false;
                for (ui j = st[v]; j < g.coreNumber; j++) {
                    ui w = adj[v][j];
                    if (w == u) {
                        cc = true;
                        break;
                    }
                }
                if (cc) {
#ifdef DEBUG
                    if (xx == xxx) {
                        printf("xv %u\n", v);
                        fflush(stdout);
                    }
#endif
                    Xv3.changeTo(v, newEX++);
                }
            }
        }
#ifdef DEBUG
        if (xx == xxx) {
            printf("newEX %u\n", newEX - SX);
            fflush(stdout);
        }
#endif

        // R.push_back(u);

        //maintain the neighbors in P / C
        auto mainNei = [&](ui v) {
            ui& nst = nxtSt[v];
            ui& ned = nxtEd[v];
            nst = g.coreNumber;
            ned = 0;

            if (EP > 0)
                for (ui j = 0; j < ed[v]; j++) {
                    if (Pv3.isIn(adj[v][j], 0, newEP)) {
                        std::swap(adj[v][j], adj[v][ned++]);
                    }
                }

            if (inC[v])
                return;

            if (EC > 0)
                for (ui j = st[v]; j < nst;) {
                    if (Cv3.isIn(adj[v][j], 0, newEC)) {
                        std::swap(adj[v][j], adj[v][--nst]);
                    } else
                        j++;
                }
        };

        for (ui i = 0; i < newEP; i++) {
            ui v = Pv3[i];
            mainNei(v);
        }
        for (ui i = SX; i < newEX; i++) {
            ui v = Xv3[i];
            mainNei(v);
        }
        for (ui i = 0; i < newEC; i++) {
            ui v = Cv3[i];  //mainNei(v);

            ui& ned = nxtEd[v];
            ned = 0;
            if (EP > 0)
                for (ui j = 0; j < ed[v]; j++) {
                    if (Pv3.isIn(adj[v][j], 0, newEP)) {
                        std::swap(adj[v][j], adj[v][ned++]);
                    }
                }
        }

        Rv3.changeTo(u, ER);

        BKTomitaRecCCRV3(deep + 1, ER + 1, newEC, newEP, SX, newEX);

        Xv3.changeTo(u, --SX);
#ifdef DEBUG
        if (xx == xxx) {
            printf("Xv322 %u:", deep);
            for (ui i = SX; i < EX; i++) printf("%u ", Xv3[i]);
            printf("\n");
        }
#endif
        if (Pv3.isIn(u, 0, EP)) {
            numOfNonCNodes--;
            Pv3.changeTo(u, --EP);
        } else {
            --EC;

            // Cv3.changeTo(u, --EC);
        }
    }

    for (ui ii = 0; ii < d; ii++) {
        Xv3.changeTo(nonNeighbors[ii], SX++);
        // if(inC[nonNeighbors[ii]]) {
        //     Cv3.changeTo(nonNeighbors[ii], EC++);
        // }
    }

#ifdef DEBUG
    if (xx == xxx) {
        printf("Xvlast %u:", deep);
        for (ui i = SX; i < EX; i++) printf("%u ", Xv3[i]);
        printf("\n");
        printf("Cvlast %u:", deep);
        for (ui i = 0; i < EC; i++) printf("%u ", Cv3[i]);
        printf("\n\n");
    }
#endif
}

}  // namespace vsag::mci_v3