// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#define main exact_final_reuse_main
#include "final_reuse.cpp"
#undef main
#include "quality512.cpp"
Snapshot
new_only(vsag::PiPNNPipeline& p,
         const Snapshot& old,
         const std::vector<uint32_t>& added,
         const std::vector<uint8_t>& live,
         uint64_t& dot_count) {
    Snapshot result;
    result.members = added;
    std::vector<uint32_t> members;
    for (auto id : old.members)
        if (live[id])
            members.push_back(id);
    uint32_t u = members.size();
    members.insert(members.end(), added.begin(), added.end());
    uint32_t n = members.size(), a = added.size();
    result.k = leaf_k(n);
    uint32_t k = result.k;
    if (!a || !k)
        return result;
    std::vector<float> matrix(uint64_t(n) * 128), dots(uint64_t(a) * n);
    for (uint32_t i = 0; i < n; ++i)
        std::copy_n(p.vector_by_local_id(members[i]), 128, matrix.data() + uint64_t(i) * 128);
    vsag::BlasFunction::Sgemm(vsag::BlasFunction::RowMajor,
                              vsag::BlasFunction::NoTrans,
                              vsag::BlasFunction::Trans,
                              a,
                              n,
                              128,
                              1,
                              matrix.data() + uint64_t(u) * 128,
                              128,
                              matrix.data(),
                              128,
                              0,
                              dots.data(),
                              n);
    dot_count += uint64_t(a) * n;
    result.edges.resize(uint64_t(a) * k);
    for (uint32_t ai = 0; ai < a; ++ai) {
        auto source = added[ai];
        uint64_t off = uint64_t(ai) * k;
        for (uint32_t j = 0; j < k; ++j)
            result.edges[off + j] = {std::numeric_limits<float>::infinity(), UINT32_MAX};
        for (uint32_t j = 0; j < n; ++j)
            if (j != u + ai) {
                auto target = members[j];
                float dot = dots[uint64_t(ai) * n + j];
                float distance = std::max(0.0f, p.norms_[source] + p.norms_[target] - dot - dot);
                offer(result.edges, off, k, source, {distance, target});
            }
    }
    return result;
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
        auto queries = read_fvecs(data + "/sift_query.fvecs", 512);
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
        auto cache = load_cache(argv[4]);
        vsag::PiPNNGraphBuilderParameter param;
        param.alpha = 1.2f;
        FrozenResult part;
        OldState old;
        {
            auto g = make_graph(c, cap);
            vsag::PiPNNPipeline p(param, 128, c.metric_, a, nullptr, 1, g, rows);
            p.prepare(ids);
            p.prepare_sketches();
            part = derive_leaves(p, n, extra);
            require(part.leaves.size() == cache.size(), "partition cache count");
            for (uint32_t li = 0; li < cache.size(); ++li)
                require(part.leaves[li].old_members == cache[li].members,
                        "partition cache members");
            for (auto& s : cache) emit(p, s);
            p.write_graph();
            old = save_state(p, n, cap);
        }
        Json report = {{"n", n},
                       {"threads", 1},
                       {"seed", 20261104},
                       {"scope",
                        "approximate warm start from old reservoirs and final adjacency; "
                        "new-node-only leaf TopK, no graph-search refinement; fixed leaders"},
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
            Adj baseline, warmed;
            for (uint32_t pass = 0; pass < 2; ++pass) {
                bool warm = keep == 95 ? pass == 0 : pass == 1;
                auto t = Clock::now();
                auto g = make_graph(c, cap);
                vsag::PiPNNPipeline p(param, 128, c.metric_, a, nullptr, 1, g, rows);
                p.prepare(ids);
                p.prepare_sketches();
                double prepare = elapsed(t);
                auto classify_start = Clock::now();
                std::vector<uint8_t> dirty(cap);
                for (uint32_t i = n; i < cap; ++i) dirty[i] = 1;
                if (warm)
                    for (uint32_t source = 0; source < n; ++source)
                        if (live[source])
                            for (auto target : old.adjacency[source])
                                if (!live[target]) {
                                    dirty[source] = 1;
                                    break;
                                }
                double classify_s = elapsed(classify_start);
                t = Clock::now();
                std::vector<Snapshot> candidates;
                MixStats mixstats;
                uint64_t dot_count = 0;
                for (uint32_t li = 0; li < cache.size(); ++li) {
                    std::vector<uint32_t> added;
                    for (auto id : part.leaves[li].new_members)
                        if (id < n + added_count)
                            added.push_back(id);
                    auto s = warm ? new_only(p, cache[li], added, live, dot_count)
                                  : mixed(p, cache[li], added, live, mixstats);
                    if (warm)
                        for (auto& e : s.edges) dirty[e.target] = 1;
                    candidates.push_back(std::move(s));
                }
                double candidate_s = elapsed(t);
                t = Clock::now();
                uint64_t seeded = 0;
                if (warm)
                    for (uint32_t source = 0; source < n; ++source)
                        if (live[source] && dirty[source]) {
                            auto* keys = old.keys.data() + uint64_t(source) * 64;
                            for (uint32_t j = 0; j < old.sizes[source]; ++j) {
                                auto key = keys[j];
                                auto target = uint32_t(key);
                                if (live[target]) {
                                    p.insert_candidate(
                                        source, target, (key >> 32) & 0xffff, distance_key(key));
                                    ++seeded;
                                }
                            }
                        }
                for (auto& s : candidates) emit(p, s);
                double merge_s = elapsed(t);
                t = Clock::now();
                prune_distance_calls = 0;
                ReuseStats stats;
                if (warm)
                    stats = finish(p, old, live, &dirty, true);
                else
                    p.write_graph();
                double prune_s = elapsed(t);
                auto paircalls = prune_distance_calls;
                Json timing = {
                    {"prepare_s", prepare},
                    {"classify_s", classify_s},
                    {"candidates_s", candidate_s},
                    {"merge_s", merge_s},
                    {"prune_s", prune_s},
                    {"total_core_s", prepare + classify_s + candidate_s + merge_s + prune_s},
                    {"prune_distance_evaluations", paircalls},
                    {"leaf_dot_products", warm ? dot_count : mixstats.dots},
                    {"seeded_old_candidates", seeded}};
                if (warm) {
                    timing["clean_old_adjacency_rows"] = stats.stream_clean;
                    timing["reservoir_equal"] = stats.reservoir_equal;
                    timing["certificate_reuse"] = stats.certificate;
                    timing["repruned"] = stats.repruned;
                }
                auto adj = adjacency(g, cap, a);
                for (uint32_t i = 0; i < cap; ++i)
                    for (auto j : adj[i])
                        require(live[i] && live[j] && i != j, "warm invalid edge");
                timing["query"] = probe512(adj, v, queries, live);
                std::cout << "keep=" << keep << " path=" << (warm ? "warm" : "reference")
                          << " core=" << timing["total_core_s"]
                          << " recall=" << timing["query"]["recall10"]
                          << " distances=" << timing["query"]["distance_evaluations_mean"]
                          << std::endl;
                r[warm ? "warm" : "reference"] = timing;
                if (warm)
                    warmed = std::move(adj);
                else
                    baseline = std::move(adj);
            }
            uint64_t changed = 0;
            for (uint32_t i = 0; i < cap; ++i) changed += baseline[i] != warmed[i];
            r["different_adjacency_rows"] = changed;
            report["runs"].push_back(r);
            std::ofstream(out + "/results.json") << report.dump(2) << '\n';
        }
    } catch (std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}
