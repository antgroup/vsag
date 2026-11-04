// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
uint64_t prune_distance_calls = 0;
#include <sys/resource.h>

#include <filesystem>

#include "final_reuse_support.cpp"
uint64_t
pack(const vsag::ReservoirEntry& e) {
    return uint64_t(e.neighbor) | (uint64_t(e.hash) << 32) | (uint64_t(e.distance) << 48);
}
uint16_t
distance_key(uint64_t key) {
    return key >> 48;
}
struct OldState {
    uint32_t n;
    std::vector<uint16_t> sizes;
    std::vector<uint64_t> keys;
    Adj adjacency;
    std::vector<std::vector<uint16_t>> selected_distances;
};
OldState
save_state(vsag::PiPNNPipeline& p, uint32_t n, uint32_t cap) {
    OldState s;
    s.n = n;
    s.sizes.resize(n);
    s.keys.resize(uint64_t(n) * 64);
    s.adjacency = adjacency(p.graph_, cap, p.allocator_);
    s.selected_distances.resize(n);
    for (uint32_t id = 0; id < n; ++id) {
        auto size = p.reservoir_states_[id].size;
        s.sizes[id] = size;
        auto* row = p.reservoirs_.data() + uint64_t(id) * 64;
        auto* keys = s.keys.data() + uint64_t(id) * 64;
        for (uint32_t j = 0; j < size; ++j) keys[j] = pack(row[j]);
        std::sort(keys, keys + size);
        for (auto target : s.adjacency[id]) {
            bool found = false;
            for (uint32_t j = 0; j < size; ++j)
                if (uint32_t(keys[j]) == target) {
                    s.selected_distances[id].push_back(distance_key(keys[j]));
                    found = true;
                    break;
                }
            require(found, "old selected candidate absent");
        }
    }
    return s;
}
void
mark_changes(const Snapshot& old,
             const Snapshot& now,
             const std::vector<uint8_t>& live,
             std::vector<uint8_t>& dirty) {
    uint32_t ni = 0;
    auto mark_old = [&](uint32_t i) {
        auto source = old.members[i];
        dirty[source] = 1;
        for (uint32_t j = 0; j < old.k; ++j) dirty[old.edges[uint64_t(i) * old.k + j].target] = 1;
    };
    auto mark_new = [&](uint32_t i) {
        dirty[now.members[i]] = 1;
        for (uint32_t j = 0; j < now.k; ++j) dirty[now.edges[uint64_t(i) * now.k + j].target] = 1;
    };
    for (uint32_t oi = 0; oi < old.members.size(); ++oi) {
        auto source = old.members[oi];
        if (!live[source]) {
            mark_old(oi);
            continue;
        }
        require(ni < now.members.size() && now.members[ni] == source, "old/new row correspondence");
        bool same = old.k == now.k;
        if (same)
            for (uint32_t j = 0; j < old.k; ++j) {
                auto a = old.edges[uint64_t(oi) * old.k + j],
                     b = now.edges[uint64_t(ni) * now.k + j];
                if (a.target != b.target || a.distance != b.distance) {
                    same = false;
                    break;
                }
            }
        if (!same) {
            mark_old(oi);
            mark_new(ni);
        }
        ++ni;
    }
    for (; ni < now.members.size(); ++ni) mark_new(ni);
}
void
emit_dirty(vsag::PiPNNPipeline& p, const Snapshot& s, const std::vector<uint8_t>& dirty) {
    for (uint32_t i = 0; i < s.members.size(); ++i)
        for (uint32_t j = 0; j < s.k; ++j) {
            auto source = s.members[i];
            auto e = s.edges[uint64_t(i) * s.k + j];
            if (!dirty[source] && !dirty[e.target])
                continue;
            auto [forward, reverse] = p.relative_hash(source, e.target);
            auto key = vsag::distance_order_key(e.distance);
            if (dirty[source])
                p.insert_candidate(source, e.target, forward, key);
            if (dirty[e.target])
                p.insert_candidate(e.target, source, reverse, key);
        }
}
struct ReuseStats {
    uint64_t live = 0, stream_clean = 0, reservoir_equal = 0, certificate = 0, repruned = 0,
             certificate_distances = 0, certificate_attempts = 0;
};
bool
certificate(vsag::PiPNNPipeline& p,
            uint32_t source,
            const OldState& old,
            const uint64_t* keys,
            uint32_t size,
            ReuseStats& stats) {
    ++stats.certificate_attempts;
    const auto& selected = old.adjacency[source];
    const auto& ds = old.selected_distances[source];
    auto* oldkeys = old.keys.data() + uint64_t(source) * 64;
    auto oldsize = old.sizes[source];
    // Every previously selected candidate must still be present with exactly its old source distance.
    for (uint32_t i = 0; i < selected.size(); ++i) {
        bool found = false;
        for (uint32_t j = 0; j < size; ++j)
            if (uint32_t(keys[j]) == selected[i] && distance_key(keys[j]) == ds[i]) {
                found = true;
                break;
            }
        if (!found)
            return false;
    }
    auto less = [&](uint32_t id, uint16_t d, uint32_t other, uint16_t od) {
        if (d != od)
            return d < od;
        auto gap = vsag::id_gap(source, id), ogap = vsag::id_gap(source, other);
        return gap != ogap ? gap < ogap : id < other;
    };
    for (uint32_t j = 0; j < size; ++j) {
        auto key = keys[j];
        if (std::binary_search(oldkeys, oldkeys + oldsize, key))
            continue;
        uint32_t candidate = uint32_t(key);
        uint16_t d = distance_key(key);
        // A full old prefix terminates pruning before any candidate after its final retained neighbor.
        if (selected.size() == p.graph_->MaximumDegree() &&
            less(selected.back(), ds.back(), candidate, d))
            continue;
        bool selected_itself = false;
        for (auto id : selected)
            if (id == candidate) {
                selected_itself = true;
                break;
            }
        if (selected_itself)
            continue;
        bool blocked = false;
        for (uint32_t i = 0; i < selected.size(); ++i) {
            if (!less(selected[i], ds[i], candidate, d))
                continue;
            ++stats.certificate_distances;
            if (p.parameter_.alpha * p.pair_distance(selected[i], candidate) <
                vsag::distance_from_order_key(d)) {
                blocked = true;
                break;
            }
        }
        if (!blocked)
            return false;
    }
    return true;
}
ReuseStats
finish(vsag::PiPNNPipeline& p,
       const OldState& old,
       const std::vector<uint8_t>& live,
       const std::vector<uint8_t>* dirty,
       bool use_certificate) {
    ReuseStats stats;
    vsag::PruneScratch scratch(p.allocator_);
    std::array<uint64_t, 64> sorted;
    for (uint32_t source = 0; source < live.size(); ++source) {
        if (!live[source])
            continue;
        ++stats.live;
        bool reuse = false;
        if (source < old.n) {
            if (dirty && !(*dirty)[source]) {
                reuse = true;
                ++stats.stream_clean;
            } else {
                auto size = p.reservoir_states_[source].size;
                auto* row = p.reservoirs_.data() + uint64_t(source) * 64;
                for (uint32_t j = 0; j < size; ++j) sorted[j] = pack(row[j]);
                std::sort(sorted.begin(), sorted.begin() + size);
                auto* oldkeys = old.keys.data() + uint64_t(source) * 64;
                if (size == old.sizes[source] &&
                    std::equal(sorted.begin(), sorted.begin() + size, oldkeys)) {
                    reuse = true;
                    ++stats.reservoir_equal;
                } else if (use_certificate &&
                           certificate(p, source, old, sorted.data(), size, stats)) {
                    reuse = true;
                    ++stats.certificate;
                }
            }
        }
        if (reuse) {
            vsag::Vector<vsag::InnerIdType> row(p.allocator_);
            row.assign(old.adjacency[source].begin(), old.adjacency[source].end());
            p.graph_->InsertNeighborsById(source, row);
        } else {
            auto size = p.reservoir_states_[source].size;
            auto* row = p.reservoirs_.data() + uint64_t(source) * 64;
            p.robust_prune(source, row, size, scratch);
            p.graph_->InsertNeighborsById(source, scratch.neighbors);
            ++stats.repruned;
        }
    }
    return stats;
}
int
main(int argc, char** argv) {
    try {
        require(argc == 6, "N BASE_DIR LEARN OLD_CACHE OUTPUT");
        uint32_t n = std::stoul(argv[1]), extra = n / 10, cap = n + extra;
        std::string data = argv[2], out = argv[5];
        auto v = read_fvecs(data + "/sift_base.fvecs", n), ev = read_fvecs(argv[3], extra);
        v.insert(v.end(), ev.begin(), ev.end());
        ev.clear();
        ev.shrink_to_fit();
        auto queries = read_fvecs(data + "/sift_query.fvecs", 64);
        vsag::IndexCommonParam c;
        c.dim_ = 128;
        c.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
        c.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
        c.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();
        auto* a = c.allocator_.get();
        vsag::Vector<vsag::InnerIdType> ids(a);
        vsag::Vector<const float*> rows(a);
        for (uint32_t i = 0; i < cap; ++i) {
            ids.push_back(i);
            rows.push_back(v.data() + uint64_t(i) * 128);
        }
        auto loadstart = Clock::now();
        auto cache = load_cache(argv[4]);
        double load_s = elapsed(loadstart);
        vsag::PiPNNGraphBuilderParameter param;
        param.alpha = 1.2f;
        FrozenResult part;
        OldState old;
        double old_merge, old_prune;
        {
            auto g = make_graph(c, cap);
            vsag::PiPNNPipeline p(param, 128, c.metric_, a, nullptr, 1, g, rows);
            p.prepare(ids);
            p.prepare_sketches();
            part = derive_leaves(p, n, extra);
            require(part.leaves.size() == cache.size(), "cache partition count");
            for (uint32_t l = 0; l < cache.size(); ++l)
                require(part.leaves[l].old_members == cache[l].members, "cache partition members");
            auto t = Clock::now();
            for (auto& s : cache) emit(p, s);
            old_merge = elapsed(t);
            t = Clock::now();
            p.write_graph();
            old_prune = elapsed(t);
            old = save_state(p, n, cap);
        }
        std::cout << "old graph ready merge=" << old_merge << " prune=" << old_prune
                  << " cache_load=" << load_s << std::endl;
        Json report = {{"n", n},
                       {"threads", 1},
                       {"seed", 20261104},
                       {"initial_leaf_cache_read_s", load_s},
                       {"old_merge_s", old_merge},
                       {"old_prune_s", old_prune},
                       {"runs", Json::array()}};
        std::vector<uint32_t> permutation(n);
        std::iota(permutation.begin(), permutation.end(), 0);
        std::mt19937 rng(20261104);
        std::shuffle(permutation.begin(), permutation.end(), rng);
        for (uint32_t keep : {90U, 95U, 99U}) {
            uint32_t added_count = uint64_t(n) * (100 - keep) / 100;
            std::vector<uint8_t> live(cap);
            for (uint32_t i = 0; i < n - added_count; ++i) live[permutation[i]] = 1;
            for (uint32_t i = 0; i < added_count; ++i) live[n + i] = 1;
            Json r = {{"retention_percent", keep}};
            Adj reference;
            for (uint32_t pass = 0; pass < 2; ++pass) {
                bool selective = keep == 95 ? pass == 0 : pass == 1;
                auto start = Clock::now();
                auto g = make_graph(c, cap);
                vsag::PiPNNPipeline p(param, 128, c.metric_, a, nullptr, 1, g, rows);
                p.prepare(ids);
                p.prepare_sketches();
                double prep_s = elapsed(start);
                std::vector<Snapshot> current;
                MixStats mixstats;
                auto t = Clock::now();
                std::vector<uint8_t> dirty(cap);
                for (uint32_t i = 0; i < cap; ++i)
                    if (i >= n || !live[i])
                        dirty[i] = 1;
                for (uint32_t li = 0; li < cache.size(); ++li) {
                    std::vector<uint32_t> added;
                    for (auto id : part.leaves[li].new_members)
                        if (id < n + added_count)
                            added.push_back(id);
                    auto s = mixed(p, cache[li], added, live, mixstats);
                    if (selective)
                        mark_changes(cache[li], s, live, dirty);
                    current.push_back(std::move(s));
                }
                double candidates_s = elapsed(t);
                t = Clock::now();
                for (auto& s : current) {
                    if (selective)
                        emit_dirty(p, s, dirty);
                    else
                        emit(p, s);
                }
                double merge_s = elapsed(t);
                t = Clock::now();
                ReuseStats reuse;
                prune_distance_calls = 0;
                if (selective)
                    reuse = finish(p, old, live, &dirty, true);
                else
                    p.write_graph();
                double prune_s = elapsed(t);
                auto measured_prune_distances = prune_distance_calls;
                Json run = {{"prepare_s", prep_s},
                            {"candidates_and_dirty_detection_s", candidates_s},
                            {"merge_s", merge_s},
                            {"prune_or_reuse_s", prune_s},
                            {"prune_distance_evaluations", measured_prune_distances},
                            {"total_core_s", prep_s + candidates_s + merge_s + prune_s}};
                auto adj = adjacency(g, cap, a);
                for (uint32_t i = 0; i < cap; ++i)
                    for (auto j : adj[i])
                        require(live[i] && live[j] && i != j, "invalid final edge");
                if (selective) {
                    run["stream_clean"] = reuse.stream_clean;
                    run["reservoir_equal"] = reuse.reservoir_equal;
                    run["certificate_reuse"] = reuse.certificate;
                    run["repruned"] = reuse.repruned;
                    run["certificate_distance_evaluations"] = reuse.certificate_distances;
                    run["certificate_attempts"] = reuse.certificate_attempts;
                    run["live_nodes"] = reuse.live;
                }
                if (reference.empty()) {
                    reference = std::move(adj);
                } else {
                    run["graphs_identical"] = reference == adj;
                    if (reference != adj) {
                        uint64_t bad = 0;
                        for (uint32_t i = 0; i < cap; ++i) bad += reference[i] != adj[i];
                        run["different_adjacency_rows"] = bad;
                    }
                }
                r[selective ? "selective" : "full_prune"] = run;
                std::cout << "keep=" << keep << " path=" << (selective ? "selective" : "full")
                          << " total=" << run["total_core_s"] << " prune=" << prune_s;
                if (selective)
                    std::cout << " stream=" << reuse.stream_clean
                              << " reservoir=" << reuse.reservoir_equal
                              << " certificate=" << reuse.certificate
                              << " reprune=" << reuse.repruned;
                std::cout << std::endl;
            }
            bool equal = r["selective"].value("graphs_identical",
                                              r["full_prune"].value("graphs_identical", false));
            r["graphs_identical"] = equal;
            report["runs"].push_back(r);
            std::ofstream(out + "/results.json") << report.dump(2) << '\n';
            require(equal, "selective graph differs from full prune");
        }
    } catch (std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}
