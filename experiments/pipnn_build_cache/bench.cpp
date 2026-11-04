// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <vector>
struct Edge {
    float distance;
    uint32_t target;
};
struct Snapshot {
    uint32_t k;
    std::vector<uint32_t> members;
    std::vector<Edge> edges;
};
std::vector<Snapshot>* capture_sink = nullptr;
bool capture_only = false;
#include "builder_exposed.cpp"
#include "datacell/graph_interface_parameter.h"
#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
double
elapsed(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}
void
require(bool ok, const char* msg) {
    if (!ok)
        throw std::runtime_error(msg);
}
std::vector<float>
read_fvecs(const std::string& path, uint32_t n) {
    std::ifstream f(path, std::ios::binary);
    require(bool(f), "open fvecs");
    std::vector<float> v(uint64_t(n) * 128);
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t d = 0;
        f.read((char*)&d, 4);
        require(d == 128, "fvecs dimension");
        f.read((char*)(v.data() + uint64_t(i) * 128), 512);
        require(bool(f), "short fvecs");
    }
    return v;
}
auto
make_graph(const vsag::IndexCommonParam& c, uint32_t n) {
    auto j = vsag::JsonType::Parse(
        "{\"io_params\":{\"type\":\"block_memory_io\"},\"max_degree\":32,\"init_capacity\":" +
        std::to_string(n) + "}");
    auto p = vsag::GraphInterfaceParameter::GetGraphParameterByJson(
        vsag::GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_FLAT, j);
    auto g = vsag::GraphInterface::MakeInstance(p, c);
    g->Resize(n);
    return g;
}
bool
better(uint32_t source, const Edge& a, const Edge& b) {
    if (a.distance != b.distance)
        return a.distance < b.distance;
    auto gap = [&](uint32_t t) { return source > t ? source - t : t - source; };
    return gap(a.target) != gap(b.target) ? gap(a.target) < gap(b.target) : a.target < b.target;
}
void
offer(std::vector<Edge>& edges, uint64_t off, uint32_t k, uint32_t source, Edge e) {
    if (!better(source, e, edges[off + k - 1]))
        return;
    uint32_t pos = k - 1;
    while (pos && better(source, e, edges[off + pos - 1])) {
        edges[off + pos] = edges[off + pos - 1];
        --pos;
    }
    edges[off + pos] = e;
}
uint32_t
leaf_k(uint64_t n) {
    return n > 1 ? std::min<uint64_t>(5, n - 1) : 0;
}
struct Stats {
    uint64_t clean = 0, rows = 0, reused = 0, refill = 0, fullpairs = 0, rectdots = 0,
             kfallback = 0;
};
Snapshot
incremental(vsag::PiPNNPipeline& p,
            const Snapshot& old,
            const std::vector<uint8_t>& live,
            Stats& stats) {
    Snapshot s;
    s.k = 0;
    std::vector<uint32_t> old_rows;
    for (uint32_t i = 0; i < old.members.size(); ++i)
        if (live[old.members[i]]) {
            s.members.push_back(old.members[i]);
            old_rows.push_back(i);
        }
    const uint32_t n = s.members.size();
    s.k = leaf_k(n);
    uint32_t k = s.k;
    stats.fullpairs += uint64_t(n) * (n ? n - 1 : 0) / 2;
    stats.rows += n;
    if (n < 2)
        return s;
    if (n == old.members.size())
        ++stats.clean;
    if (k != old.k) {
        ++stats.kfallback;
        std::vector<Snapshot> temp;
        capture_sink = &temp;
        capture_only = true;
        vsag::Leaf leaf(p.allocator_);
        leaf.assign(s.members.begin(), s.members.end());
        p.build_leaf(leaf);
        capture_sink = nullptr;
        capture_only = false;
        require(temp.size() == 1, "fallback capture");
        stats.rectdots += uint64_t(n) * (n - 1) / 2;
        stats.refill += n;
        return std::move(temp[0]);
    }
    s.edges.resize(uint64_t(n) * k);
    std::vector<uint32_t> affected;
    for (uint32_t i = 0; i < n; ++i) {
        bool valid = true;
        uint64_t off = uint64_t(old_rows[i]) * k;
        for (uint32_t j = 0; j < k; ++j) valid = valid && live[old.edges[off + j].target];
        if (valid) {
            std::copy_n(old.edges.data() + off, k, s.edges.data() + uint64_t(i) * k);
            ++stats.reused;
        } else
            affected.push_back(i);
    }
    stats.refill += affected.size();
    stats.rectdots += uint64_t(affected.size()) * n;
    if (!affected.empty()) {
        std::vector<float> matrix(uint64_t(n) * 128), a(uint64_t(affected.size()) * 128),
            dots(uint64_t(affected.size()) * n);
        for (uint32_t i = 0; i < n; ++i)
            std::copy_n(p.vector_by_local_id(s.members[i]), 128, matrix.data() + uint64_t(i) * 128);
        for (uint32_t i = 0; i < affected.size(); ++i)
            std::copy_n(
                matrix.data() + uint64_t(affected[i]) * 128, 128, a.data() + uint64_t(i) * 128);
        vsag::BlasFunction::Sgemm(vsag::BlasFunction::RowMajor,
                                  vsag::BlasFunction::NoTrans,
                                  vsag::BlasFunction::Trans,
                                  affected.size(),
                                  n,
                                  128,
                                  1,
                                  a.data(),
                                  128,
                                  matrix.data(),
                                  128,
                                  0,
                                  dots.data(),
                                  n);
        for (uint32_t row = 0; row < affected.size(); ++row) {
            uint32_t i = affected[row], source = s.members[i];
            uint64_t off = uint64_t(i) * k;
            for (uint32_t j = 0; j < k; ++j)
                s.edges[off + j] = {std::numeric_limits<float>::infinity(), UINT32_MAX};
            for (uint32_t j = 0; j < n; ++j)
                if (i != j) {
                    uint32_t target = s.members[j];
                    float dot = dots[uint64_t(row) * n + j];
                    float d = std::max(0.0f, p.norms_[source] + p.norms_[target] - dot - dot);
                    offer(s.edges, off, k, source, {d, target});
                }
        }
    }
    return s;
}
void
emit(vsag::PiPNNPipeline& p, const Snapshot& s) {
    for (uint32_t i = 0; i < s.members.size(); ++i)
        for (uint32_t j = 0; j < s.k; ++j) {
            auto e = s.edges[uint64_t(i) * s.k + j];
            p.insert_bidirectional(s.members[i], e.target, e.distance);
        }
}
void
write_cache(const std::string& path, const std::vector<Snapshot>& ss) {
    std::ofstream f(path, std::ios::binary);
    uint64_t n = ss.size();
    f.write((char*)&n, 8);
    for (auto& s : ss) {
        uint64_t z = s.members.size();
        f.write((char*)&z, 8);
        f.write((char*)&s.k, 4);
        f.write((char*)s.members.data(), z * 4);
        f.write((char*)s.edges.data(), s.edges.size() * sizeof(Edge));
    }
    require(bool(f), "write cache");
}
std::vector<Snapshot>
load_cache(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    uint64_t n;
    f.read((char*)&n, 8);
    require(n < 1000000, "snapshot count");
    std::vector<Snapshot> ss(n);
    for (auto& s : ss) {
        uint64_t z;
        f.read((char*)&z, 8);
        f.read((char*)&s.k, 4);
        require(z <= 1024 && s.k <= 5, "snapshot bounds");
        s.members.resize(z);
        s.edges.resize(z * s.k);
        f.read((char*)s.members.data(), z * 4);
        f.read((char*)s.edges.data(), s.edges.size() * sizeof(Edge));
    }
    require(bool(f), "read cache");
    return ss;
}
using Adj = std::vector<std::vector<uint32_t>>;
Adj
adjacency(vsag::GraphInterfacePtr g, uint32_t n, vsag::Allocator* a) {
    Adj out(n);
    vsag::Vector<vsag::InnerIdType> row(a);
    for (uint32_t i = 0; i < n; ++i) {
        row.clear();
        g->GetNeighbors(i, row);
        out[i].assign(row.begin(), row.end());
    }
    return out;
}
Json
probe(const Adj& adj,
      const std::vector<float>& v,
      const std::vector<float>& q,
      const std::vector<uint8_t>& live) {
    uint32_t n = live.size();
    uint64_t hits = 0, evals = 0;
    Json per = Json::array();
    for (uint32_t qi = 0; qi < 64; ++qi) {
        auto* query = q.data() + qi * 128;
        std::vector<std::pair<float, uint32_t>> truth;
        for (uint32_t i = 0; i < n; ++i)
            if (live[i])
                truth.push_back(
                    {vsag::FP32ComputeL2Sqr(query, v.data() + uint64_t(i) * 128, 128), i});
        std::partial_sort(truth.begin(), truth.begin() + 10, truth.end());
        std::vector<uint8_t> seen(n);
        using P = std::pair<float, uint32_t>;
        std::priority_queue<P, std::vector<P>, std::greater<P>> frontier;
        std::priority_queue<P> best;
        uint64_t count = 0;
        auto add = [&](uint32_t id) {
            if (!live[id] || seen[id])
                return;
            seen[id] = 1;
            float d = vsag::FP32ComputeL2Sqr(query, v.data() + uint64_t(id) * 128, 128);
            ++count;
            if (best.size() < 256 || P(d, id) < best.top()) {
                frontier.push({d, id});
                best.push({d, id});
                if (best.size() > 256)
                    best.pop();
            }
        };
        for (uint32_t j = 0; j < 8; ++j) {
            uint32_t id = uint64_t(j) * n / 8;
            while (id < n && !live[id]) ++id;
            if (id < n)
                add(id);
        }
        while (!frontier.empty()) {
            auto cur = frontier.top();
            frontier.pop();
            if (best.size() == 256 && cur > best.top())
                break;
            for (auto id : adj[cur.second]) add(id);
        }
        std::vector<P> found;
        while (!best.empty()) {
            found.push_back(best.top());
            best.pop();
        }
        std::sort(found.begin(), found.end());
        uint32_t localhits = 0;
        for (uint32_t j = 0; j < std::min<uint64_t>(10, found.size()); ++j)
            for (uint32_t t = 0; t < 10; ++t)
                if (found[j].second == truth[t].second)
                    ++localhits;
        hits += localhits;
        evals += count;
        per.push_back({localhits, count});
    }
    return {{"recall10", hits / 640.0},
            {"distance_evaluations_mean", evals / 64.0},
            {"per_query_hits_distances", per},
            {"search",
             "custom bottom-graph best-first ef=256, 8 fixed entry points; not HGraph search"}};
}
int
main(int argc, char** argv) {
    try {
        require(argc == 4, "N DATA_DIR OUTPUT_DIR");
        uint32_t n = std::stoul(argv[1]);
        std::string data = argv[2], out = argv[3];
        auto v = read_fvecs(data + "/sift_base.fvecs", n),
             q = read_fvecs(data + "/sift_query.fvecs", 64);
        vsag::IndexCommonParam c;
        c.dim_ = 128;
        c.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
        c.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
        c.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();
        auto* a = c.allocator_.get();
        vsag::Vector<vsag::InnerIdType> ids(a);
        vsag::Vector<const float*> rows(a);
        for (uint32_t i = 0; i < n; ++i) {
            ids.push_back(i);
            rows.push_back(v.data() + uint64_t(i) * 128);
        }
        vsag::PiPNNGraphBuilderParameter param;
        param.alpha = 1.2f;
        auto t = Clock::now();
        auto oldgraph = make_graph(c, n);
        vsag::PiPNNPipeline old(param, 128, c.metric_, a, nullptr, 1, oldgraph, rows);
        old.prepare(ids);
        old.prepare_sketches();
        auto leaves = old.partition();
        double partition_time = elapsed(t);
        std::vector<Snapshot> cached;
        capture_sink = &cached;
        capture_only = true;
        t = Clock::now();
        for (auto& l : leaves)
            if (l.size() > 1)
                old.build_leaf(l);
        double snapshot_time = elapsed(t);
        capture_sink = nullptr;
        capture_only = false;
        // singleton leaves are deliberately omitted: they emit no candidates.
        uint64_t membership = 0, edges = 0, pairs = 0;
        for (auto& s : cached) {
            membership += s.members.size();
            edges += s.edges.size();
            pairs += s.members.size() * (s.members.size() - 1) / 2;
        }
        t = Clock::now();
        write_cache(out + "/leaf_cache.bin", cached);
        double export_time = elapsed(t);
        Json report = {{"n", n},
                       {"threads", 1},
                       {"scope",
                        "deletion-only; frozen real PiPNN leaves; stable internal IDs; no "
                        "partition restoration or full HGraph cache integration"},
                       {"partition_prepare_s", partition_time},
                       {"initial_leaf_snapshot_s", snapshot_time},
                       {"export_s", export_time},
                       {"leaf_count", cached.size()},
                       {"membership_count", membership},
                       {"edge_count", edges},
                       {"old_pair_count", pairs},
                       {"cache_payload_bytes", 8 + 12 * cached.size() + 4 * membership + 8 * edges},
                       {"runs", Json::array()}};
        std::cout << "snapshot n=" << n << " leaves=" << cached.size() << " members=" << membership
                  << " seconds=" << snapshot_time << std::endl;
        for (uint32_t seed : {20261104U, 20261105U, 20261106U})
            for (uint32_t keep : {90U, 95U, 99U}) {
                std::vector<uint32_t> order(n);
                std::iota(order.begin(), order.end(), 0);
                std::mt19937 random(seed);
                std::shuffle(order.begin(), order.end(), random);
                std::vector<uint8_t> live(n);
                for (uint32_t i = 0; i < uint64_t(n) * keep / 100; ++i) live[order[i]] = 1;
                Json r = {{"retention_percent", keep}, {"seed", seed}};
                std::vector<Snapshot> ref, actual;
                Adj refadj, newadj;
                Stats stats;
                // alternate execution order by seed; warm filesystem cache is explicitly reported.
                for (uint32_t pass = 0; pass < 2; ++pass) {
                    bool inc = ((seed + pass) % 2) == 1;
                    t = Clock::now();
                    std::vector<Snapshot> loaded;
                    if (inc)
                        loaded = load_cache(out + "/leaf_cache.bin");
                    double load_s = elapsed(t);
                    t = Clock::now();
                    auto g = make_graph(c, n);
                    vsag::PiPNNPipeline p(param, 128, c.metric_, a, nullptr, 1, g, rows);
                    p.prepare(ids);
                    p.prepare_sketches();
                    double prepare_s = elapsed(t);
                    auto b = Clock::now();
                    if (inc) {
                        for (auto& s : loaded) {
                            auto result = incremental(p, s, live, stats);
                            emit(p, result);
                            actual.push_back(std::move(result));
                        }
                    } else {
                        capture_sink = &ref;
                        for (auto& s : cached) {
                            vsag::Leaf leaf(a);
                            for (auto id : s.members)
                                if (live[id])
                                    leaf.push_back(id);
                            if (leaf.size() > 1)
                                p.build_leaf(leaf);
                            else {
                                Snapshot empty;
                                empty.k = 0;
                                empty.members.assign(leaf.begin(), leaf.end());
                                ref.push_back(empty);
                            }
                        }
                        capture_sink = nullptr;
                    }
                    double leaf_s = elapsed(b);
                    b = Clock::now();
                    p.write_graph();
                    double prune_s = elapsed(b);
                    double total = prepare_s + leaf_s + prune_s + (inc ? load_s : 0);
                    auto adj = adjacency(g, n, a);
                    for (uint32_t i = 0; i < n; ++i)
                        for (auto j : adj[i])
                            require(live[i] && live[j] && i != j, "invalid emitted graph edge");
                    r[inc ? "incremental" : "full"] = {{"cache_load_s", inc ? load_s : 0},
                                                       {"prepare_s", prepare_s},
                                                       {"leaf_and_merge_s", leaf_s},
                                                       {"prune_s", prune_s},
                                                       {"total_s", total}};
                    if (inc)
                        newadj = std::move(adj);
                    else
                        refadj = std::move(adj);
                }
                uint64_t mismatch = 0;
                require(ref.size() == actual.size(), "snapshot sizes");
                for (uint64_t l = 0; l < ref.size(); ++l) {
                    require(ref[l].members == actual[l].members && ref[l].k == actual[l].k,
                            "member/k mismatch");
                    require(ref[l].edges.size() == actual[l].edges.size(), "edge sizes");
                    for (uint64_t e = 0; e < ref[l].edges.size(); ++e)
                        if (ref[l].edges[e].target != actual[l].edges[e].target ||
                            ref[l].edges[e].distance != actual[l].edges[e].distance)
                            ++mismatch;
                }
                r["topk_edge_mismatches"] = mismatch;
                r["graphs_identical"] = refadj == newadj;
                r["clean_leaves"] = stats.clean;
                r["rows"] = stats.rows;
                r["reused_rows"] = stats.reused;
                r["refill_rows"] = stats.refill;
                r["full_unique_pairs"] = stats.fullpairs;
                r["incremental_rectangular_dot_products"] = stats.rectdots;
                r["k_change_fallback_leaves"] = stats.kfallback;
                if (seed == 20261104) {
                    r["full"]["query"] = probe(refadj, v, q, live);
                    r["incremental"]["query"] = probe(newadj, v, q, live);
                }
                report["runs"].push_back(r);
                std::ofstream(out + "/results.json") << report.dump(2) << '\n';
                std::cout << "keep=" << keep << " seed=" << seed << " full=" << r["full"]["total_s"]
                          << " cache=" << r["incremental"]["total_s"] << " mismatch=" << mismatch
                          << " graph=" << (refadj == newadj) << std::endl;
            }
        return 0;
    } catch (std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
