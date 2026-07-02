#pragma once

#include <limits>

#include "mci_v3_graph_mce.h"
#include "mci_v3_linear_set.h"
#include "mci_v3_types.h"

namespace vsag::mci_v3 {

struct MCEStats {
    ull roots_seen = 0;
    ull roots_skipped_by_degree = 0;
    ull roots_skipped_no_must = 0;
    ull recursion_calls = 0;
    ull prune_size_bound = 0;
    ull prune_no_must = 0;
    ull saved_cliques = 0;
    ull quota_stop = 0;
    ull remaining_must_stop = 0;

    void
    reset() {
        *this = MCEStats{};
    }

    void
    add(const MCEStats& other) {
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

class MCE {
private:
    ui cnt = 0;

    bool outP = false;
    ui nowU;

    LinearSet V;
    void
    BKTomitaRec(ui deep, std::vector<ui>& C, ui P, ui SX, ui EX);

    std::vector<std::vector<ui>> adj;
    std::vector<std::vector<ui>> edAdj, stAdj;
    std::vector<ui> reId;
    std::vector<bool> inC;
    void
    BKTomitaRecCCR(ui deep, std::vector<ui>& C, ui SC, ui EC, ui SP, ui SX, ui EX);

    void
    BKTomitaRecCCRV2(ui deep, std::vector<ui>& C, ui SC, ui EC, ui SP, ui SX, ui EX);

    LinearSet Cv3, Rv3, Pv3, Xv3;
    ui numOfNonCNodes = 0;
    void
    BKTomitaRecCCRV3(ui deep, ui ER, ui EC, ui EP, ui SX, ui EX);
    void
    BKTomitaRecCCRV3_mustcontain(
        ui deep, ui ER, ui EC, ui EP, ui SX, ui EX, std::vector<bool>& mustContainNodes);
    bool
    shouldStopMCE() const;
    bool
    stateHasMust(ui ER, ui EC, ui EP, std::vector<bool>& mustContainNodes);
    void
    clearMustNode(ui u, std::vector<bool>& mustContainNodes);
    bool
    trySaveMustClique(ui ER, ui EC, ui EP, std::vector<bool>& mustContainNodes);

    void
    BKTomitaRecCCRV4(ui deep, ui ER, ui EC, ui EP, ui SX, ui EX);

    void
    BKTomitaRecCCRV5(ui deep, ui ER, ui EC, ui EP, ui SX, ui EX);
// #define COUNT_NUM_SEARCH_NODES
#ifdef COUNT_NUM_SEARCH_NODES
    ull numOfDFSSearchNodes = 0;
#endif

public:
    MCE(Graphmce&& g, std::vector<std::vector<ui>>* maxCliques_) : g(g), maxCliques(maxCliques_) {
        V.resize(g.n);
    }

    void
    set(std::vector<std::vector<ui>>* maxCliques_) {
        maxCliques = maxCliques_;
        maxCliqueCntSave = 0;
    }
    std::vector<std::vector<ui>>* maxCliques;
    ui maxCliqueCntSave = 0;
    ui threshold = 0;
    Graphmce g;

    MCE() {
    }

    ui
    run();

    std::vector<bool> choosedNode;
    std::vector<ui> id_s;
    std::vector<ui> degree;
    std::vector<bool> removed;

    std::vector<ui> pIdx;
    std::vector<ui> eIdx;
    std::vector<ui> pEdge;

    std::vector<ui> clique;
    std::vector<ui> nxtC;
    void
    reserve(ui sz);
    MCEStats stats;
    ui maxCliqueCntLimit = std::numeric_limits<ui>::max();
    ui remainingMustCount = 0;
    bool stopSearch = false;
    void
    configureMustRun(ui maxSavedCliques = std::numeric_limits<ui>::max(), ui initialMustCount = 0);
    ui
    run(std::vector<bool>& mustContainNodes);

    void
    setOutputFile() {
        outP = true;
    };
};

}  // namespace vsag::mci_v3