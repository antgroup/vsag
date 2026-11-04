// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#include <sys/resource.h>

#include <filesystem>

#include "derived_support.cpp"
struct Twin {
    std::vector<uint32_t> old_members, new_members;
};
struct FrozenResult {
    std::vector<Twin> leaves;
    double route_s = 0;
    uint64_t fallback_points = 0;
};
FrozenResult
derive_leaves(vsag::PiPNNPipeline& p, uint32_t base, uint32_t extra) {
    struct Job {
        vsag::WorkItem work;
        vsag::Leaf added;
        Job(vsag::WorkItem&& w, vsag::Leaf&& a) : work(std::move(w)), added(std::move(a)) {};
    };
    std::vector<Job> work;
    vsag::Leaf root(p.allocator_), fresh(p.allocator_);
    for (uint32_t i = 0; i < base; ++i) root.push_back(i);
    for (uint32_t i = 0; i < extra; ++i) fresh.push_back(base + i);
    FrozenResult out;
    if (base <= p.parameter_.max_leaf_size) {
        out.leaves.push_back({{root.begin(), root.end()}, {fresh.begin(), fresh.end()}});
        return out;
    }
    work.emplace_back(vsag::WorkItem(std::move(root), 0, vsag::PARTITION_SEED), std::move(fresh));
    for (uint32_t iteration = 0; !work.empty(); ++iteration) {
        require(iteration < 30, "freeze depth");
        std::vector<Job> pending;
        for (auto& job : work) {
            auto& item = job.work;
            const uint64_t requested =
                item.level < p.parameter_.fanout.size() ? p.parameter_.fanout[item.level] : 1;
            auto leaders = p.sample_leaders(item);
            auto fanout = std::min<uint64_t>(requested, leaders.size());
            auto clusters = p.assign_to_leaders(item, leaders, fanout, false);
            auto rt = Clock::now();
            vsag::WorkItem added_item(std::move(job.added), item.level, item.seed);
            auto newclusters = p.assign_to_leaders(added_item, leaders, fanout, false);
            out.route_s += elapsed(rt);
            bool all = !clusters.empty();
            for (auto& cl : clusters) all = all && cl.size() == item.points.size();
            if (all && fanout > 1) {
                vsag::Leaf same(item.points.begin(), item.points.end(), p.allocator_);
                pending.emplace_back(vsag::WorkItem(std::move(same),
                                                    item.level + 1,
                                                    vsag::mix_seed(item.seed, item.points.size())),
                                     std::move(added_item.points));
                continue;
            }
            for (uint32_t ci = 0; ci < clusters.size(); ++ci) {
                auto& cl = clusters[ci];
                auto& added = newclusters[ci];
                if (cl.empty()) {
                    require(added.empty(), "new route enters an empty old cluster: unsupported");
                    continue;
                }
                if (cl.size() <= p.parameter_.max_leaf_size) {
                    out.leaves.push_back({{cl.begin(), cl.end()}, {added.begin(), added.end()}});
                    continue;
                }
                bool stuck = cl.size() == item.points.size() && fanout == 1;
                bool slow = item.level > vsag::CONCERNING_PARTITION_LEVEL &&
                            double(cl.size()) >
                                double(item.points.size()) * vsag::MIN_PARTITION_SHRINK_RATIO;
                if (stuck || slow) {
                    if (slow) {
                        std::mt19937_64 random(vsag::mix_seed(item.seed, cl.size()));
                        std::shuffle(cl.begin(), cl.end(), random);
                    }
                    uint64_t begin = 0;
                    std::vector<Twin> chunks;
                    while (begin < cl.size()) {
                        Twin chunk;
                        if (begin > 0)
                            chunk.old_members.push_back(cl[begin - 1]);
                        auto capacity = p.parameter_.max_leaf_size - chunk.old_members.size();
                        auto end = std::min<uint64_t>(begin + capacity, cl.size());
                        chunk.old_members.insert(
                            chunk.old_members.end(), cl.begin() + begin, cl.begin() + end);
                        chunks.push_back(std::move(chunk));
                        begin = end;
                    }
                    // A frozen fallback has no geometric sub-rule. Use the documented stable-ID hash extension.
                    for (auto id : added) {
                        chunks[vsag::mix_seed(id, item.seed) % chunks.size()].new_members.push_back(
                            id);
                        ++out.fallback_points;
                    }
                    for (auto& chunk : chunks) out.leaves.push_back(std::move(chunk));
                    continue;
                }
                pending.emplace_back(
                    vsag::WorkItem(std::move(cl),
                                   item.level + 1,
                                   vsag::mix_seed(item.seed, ci + item.points.size())),
                    std::move(added));
            }
        }
        work = std::move(pending);
    }
    // Preserve the original merge decisions based on old members only, carrying new members along.
    std::vector<Twin> merged, small;
    Twin accumulator;
    auto combine = [](const Twin& a, const Twin& b) {
        Twin c;
        std::set_union(a.old_members.begin(),
                       a.old_members.end(),
                       b.old_members.begin(),
                       b.old_members.end(),
                       std::back_inserter(c.old_members));
        std::set_union(a.new_members.begin(),
                       a.new_members.end(),
                       b.new_members.begin(),
                       b.new_members.end(),
                       std::back_inserter(c.new_members));
        return c;
    };
    for (auto& leaf : out.leaves) {
        if (leaf.old_members.size() >= p.parameter_.min_leaf_size) {
            merged.push_back(std::move(leaf));
            continue;
        }
        if (accumulator.old_members.empty()) {
            accumulator = std::move(leaf);
            continue;
        }
        auto candidate = combine(accumulator, leaf);
        if (candidate.old_members.size() > p.parameter_.max_leaf_size) {
            small.push_back(std::move(accumulator));
            accumulator = std::move(leaf);
        } else
            accumulator = std::move(candidate);
        if (accumulator.old_members.size() >= p.parameter_.min_leaf_size) {
            small.push_back(std::move(accumulator));
            accumulator = Twin{};
        }
    }
    if (!accumulator.old_members.empty()) {
        auto& preceding = small.empty() ? merged : small;
        if (!preceding.empty()) {
            auto candidate = combine(preceding.back(), accumulator);
            if (candidate.old_members.size() <= p.parameter_.max_leaf_size) {
                preceding.back() = std::move(candidate);
                accumulator = Twin{};
            }
        }
        if (!accumulator.old_members.empty())
            small.push_back(std::move(accumulator));
    }
    for (auto& leaf : small) merged.push_back(std::move(leaf));
    out.leaves = std::move(merged);
    return out;
}
struct MixStats {
    uint64_t old_rows = 0, reused_old_rows = 0, refilled_old_rows = 0, new_rows = 0, dots = 0,
             fullpairs = 0, fallback = 0, oversized = 0;
};
Snapshot
mixed(vsag::PiPNNPipeline& p,
      const Snapshot& old,
      const std::vector<uint32_t>& added,
      const std::vector<uint8_t>& live,
      MixStats& stats) {
    Snapshot s;
    std::vector<uint32_t> oldrow;
    for (uint32_t i = 0; i < old.members.size(); ++i)
        if (live[old.members[i]]) {
            s.members.push_back(old.members[i]);
            oldrow.push_back(i);
        }
    uint32_t u = s.members.size();
    s.members.insert(s.members.end(), added.begin(), added.end());
    uint32_t n = s.members.size(), a = added.size(), k = leaf_k(n);
    s.k = k;
    stats.old_rows += u;
    stats.new_rows += a;
    stats.fullpairs += uint64_t(n) * (n ? n - 1 : 0) / 2;
    if (n > 1024)
        ++stats.oversized;
    if (n < 2)
        return s;
    if (k != old.k) {
        ++stats.fallback;
        std::vector<Snapshot> ref;
        capture_sink = &ref;
        capture_only = true;
        vsag::Leaf leaf(p.allocator_);
        leaf.assign(s.members.begin(), s.members.end());
        p.build_leaf(leaf);
        capture_sink = nullptr;
        capture_only = false;
        stats.dots += uint64_t(n) * (n - 1) / 2;
        stats.refilled_old_rows += u;
        return std::move(ref[0]);
    }
    s.edges.resize(uint64_t(n) * k);
    std::vector<uint32_t> affected;
    for (uint32_t i = 0; i < u; ++i) {
        uint64_t off = uint64_t(oldrow[i]) * k;
        bool valid = true;
        for (uint32_t j = 0; j < k; ++j) valid = valid && live[old.edges[off + j].target];
        if (valid) {
            std::copy_n(old.edges.data() + off, k, s.edges.data() + uint64_t(i) * k);
            ++stats.reused_old_rows;
        } else
            affected.push_back(i);
    }
    stats.refilled_old_rows += affected.size();
    auto initialize = [&](uint32_t i) {
        for (uint32_t j = 0; j < k; ++j)
            s.edges[uint64_t(i) * k + j] = {std::numeric_limits<float>::infinity(), UINT32_MAX};
    };
    for (auto i : affected) initialize(i);
    for (uint32_t i = u; i < n; ++i) initialize(i);
    std::vector<float> matrix(uint64_t(n) * 128);
    for (uint32_t i = 0; i < n; ++i)
        std::copy_n(p.vector_by_local_id(s.members[i]), 128, matrix.data() + uint64_t(i) * 128);
    if (!affected.empty() && u) {
        std::vector<float> rows(uint64_t(affected.size()) * 128),
            dots(uint64_t(affected.size()) * u);
        for (uint32_t i = 0; i < affected.size(); ++i)
            std::copy_n(
                matrix.data() + uint64_t(affected[i]) * 128, 128, rows.data() + uint64_t(i) * 128);
        vsag::BlasFunction::Sgemm(vsag::BlasFunction::RowMajor,
                                  vsag::BlasFunction::NoTrans,
                                  vsag::BlasFunction::Trans,
                                  affected.size(),
                                  u,
                                  128,
                                  1,
                                  rows.data(),
                                  128,
                                  matrix.data(),
                                  128,
                                  0,
                                  dots.data(),
                                  u);
        stats.dots += uint64_t(affected.size()) * u;
        for (uint32_t ri = 0; ri < affected.size(); ++ri) {
            uint32_t i = affected[ri], source = s.members[i];
            for (uint32_t j = 0; j < u; ++j)
                if (i != j) {
                    float dot = dots[uint64_t(ri) * u + j];
                    float d = std::max(0.0f, p.norms_[source] + p.norms_[s.members[j]] - dot - dot);
                    offer(s.edges, uint64_t(i) * k, k, source, {d, s.members[j]});
                }
        }
    }
    if (a) {
        std::vector<float> dots(uint64_t(a) * n);
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
        stats.dots += uint64_t(a) * n;
        for (uint32_t ai = 0; ai < a; ++ai) {
            uint32_t i = u + ai, source = s.members[i];
            for (uint32_t j = 0; j < n; ++j)
                if (i != j) {
                    float dot = dots[uint64_t(ai) * n + j];
                    float d = std::max(0.0f, p.norms_[source] + p.norms_[s.members[j]] - dot - dot);
                    offer(s.edges, uint64_t(i) * k, k, source, {d, s.members[j]});
                    if (j < u)
                        offer(s.edges, uint64_t(j) * k, k, s.members[j], {d, source});
                }
        }
    }
    return s;
}
int
main(int argc, char** argv) {
    try {
        require(argc == 6, "N BASE_DIR LEARN OUTPUT QUERY_COUNT");
        uint32_t n = std::stoul(argv[1]), extra = n / 10;
        std::string data = argv[2], learn = argv[3], out = argv[4];
        bool runqueries = std::stoul(argv[5]) != 0;
        auto v = read_fvecs(data + "/sift_base.fvecs", n), extra_vectors = read_fvecs(learn, extra);
        v.insert(v.end(), extra_vectors.begin(), extra_vectors.end());
        extra_vectors.clear();
        extra_vectors.shrink_to_fit();
        auto queries = read_fvecs(data + "/sift_query.fvecs", 64);
        uint32_t cap = n + extra;
        vsag::IndexCommonParam c;
        c.dim_ = 128;
        c.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
        c.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
        c.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();
        auto* allocator = c.allocator_.get();
        vsag::Vector<vsag::InnerIdType> ids(allocator);
        vsag::Vector<const float*> rows(allocator);
        for (uint32_t i = 0; i < cap; ++i) {
            ids.push_back(i);
            rows.push_back(v.data() + uint64_t(i) * 128);
        }
        vsag::PiPNNGraphBuilderParameter param;
        param.alpha = 1.2f;
        FrozenResult partition;
        std::vector<Snapshot> cache;
        double prep, oldleaf, write_s;
        {
            auto start = Clock::now();
            auto g = make_graph(c, cap);
            vsag::PiPNNPipeline p(param, 128, c.metric_, allocator, nullptr, 1, g, rows);
            p.prepare(ids);
            p.prepare_sketches();
            partition = derive_leaves(p, n, extra);
            prep = elapsed(start);
            std::vector<uint32_t> coverage(extra);
            for (auto& leaf : partition.leaves)
                for (auto id : leaf.new_members) {
                    require(id >= n && id < cap, "new member range");
                    ++coverage[id - n];
                }
            for (auto count : coverage) require(count > 0, "new point has no routed leaf");
            if (n <= 10000) {
                vsag::Vector<vsag::InnerIdType> oldids(allocator);
                vsag::Vector<const float*> oldrows(allocator);
                for (uint32_t i = 0; i < n; ++i) {
                    oldids.push_back(i);
                    oldrows.push_back(rows[i]);
                }
                auto og = make_graph(c, n);
                vsag::PiPNNPipeline original(
                    param, 128, c.metric_, allocator, nullptr, 1, og, oldrows);
                original.prepare(oldids);
                original.prepare_sketches();
                auto check = original.partition();
                require(check.size() == partition.leaves.size(), "original partition leaf count");
                for (uint32_t i = 0; i < check.size(); ++i)
                    require(std::vector<uint32_t>(check[i].begin(), check[i].end()) ==
                                partition.leaves[i].old_members,
                            "original partition members");
            }
            start = Clock::now();
            capture_only = true;
            for (auto& twin : partition.leaves) {
                vsag::Leaf l(allocator);
                l.assign(twin.old_members.begin(), twin.old_members.end());
                if (l.size() > 1) {
                    capture_sink = &cache;
                    p.build_leaf(l);
                } else
                    cache.push_back({0, twin.old_members, {}});
            }
            capture_sink = nullptr;
            capture_only = false;
            oldleaf = elapsed(start);
        }
        auto start = Clock::now();
        write_cache(out + "/cache.bin", cache);
        write_s = elapsed(start);
        uint64_t membership = 0, edgecount = 0;
        for (auto& s : cache) {
            membership += s.members.size();
            edgecount += s.edges.size();
        }
        Json report = {{"base_count", n},
                       {"target_count", n},
                       {"union_slot_count", cap},
                       {"threads", 1},
                       {"seed", 20261104},
                       {"partition_prepare_s", prep},
                       {"added_routing_100k_or_n_over_10_s", partition.route_s},
                       {"fallback_added_memberships", partition.fallback_points},
                       {"old_leaf_snapshot_s", oldleaf},
                       {"cache_write_s", write_s},
                       {"cache_bytes", std::filesystem::file_size(out + "/cache.bin")},
                       {"leaf_count", cache.size()},
                       {"membership_count", membership},
                       {"edge_count", edgecount},
                       {"runs", Json::array()}};
        std::cout << "cache leaves=" << cache.size() << " memberships=" << membership
                  << " bytes=" << report["cache_bytes"] << " route_s=" << partition.route_s
                  << std::endl;
        std::vector<uint32_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::mt19937 rng(20261104);
        std::shuffle(order.begin(), order.end(), rng);
        for (uint32_t keep : {90U, 95U, 99U}) {
            uint32_t replaced = uint64_t(n) * (100 - keep) / 100;
            std::vector<uint8_t> live(cap, 0);
            for (uint32_t i = 0; i < uint64_t(n) * keep / 100; ++i) live[order[i]] = 1;
            for (uint32_t i = 0; i < replaced; ++i) live[n + i] = 1;
            std::ofstream recipe(out + "/target_" + std::to_string(keep) + "_source_ids.u32",
                                 std::ios::binary);
            uint32_t live_count = 0;
            for (uint32_t id = 0; id < cap; ++id)
                if (live[id]) {
                    recipe.write((char*)&id, 4);
                    ++live_count;
                }
            require(live_count == n && bool(recipe), "derived dataset count");
            recipe.close();
            Json r = {{"retention_percent", keep},
                      {"old_retained", n - replaced},
                      {"new_added", replaced},
                      {"target_count", n}};
            std::vector<Snapshot> ref, actual;
            Adj oldadj, newadj;
            MixStats stats;
            for (uint32_t pass = 0; pass < 2; ++pass) {
                bool inc = (keep == 95 ? pass == 0 : pass == 1);
                auto t = Clock::now();
                std::vector<Snapshot> loaded;
                if (inc)
                    loaded = load_cache(out + "/cache.bin");
                double read_s = elapsed(t);
                t = Clock::now();
                auto g = make_graph(c, cap);
                vsag::PiPNNPipeline p(param, 128, c.metric_, allocator, nullptr, 1, g, rows);
                p.prepare(ids);
                p.prepare_sketches();
                double setup_s = elapsed(t);
                auto b = Clock::now();
                for (uint32_t li = 0; li < cache.size(); ++li) {
                    std::vector<uint32_t> added;
                    for (auto id : partition.leaves[li].new_members)
                        if (id < n + replaced)
                            added.push_back(id);
                    if (inc) {
                        auto s = mixed(p, loaded[li], added, live, stats);
                        emit(p, s);
                        actual.push_back(std::move(s));
                    } else {
                        vsag::Leaf leaf(allocator);
                        for (auto id : cache[li].members)
                            if (live[id])
                                leaf.push_back(id);
                        leaf.insert(leaf.end(), added.begin(), added.end());
                        if (leaf.size() > 1) {
                            capture_sink = &ref;
                            p.build_leaf(leaf);
                            capture_sink = nullptr;
                        } else {
                            Snapshot s;
                            s.k = 0;
                            s.members.assign(leaf.begin(), leaf.end());
                            ref.push_back(std::move(s));
                        }
                    }
                }
                double leaf_s = elapsed(b);
                b = Clock::now();
                p.write_graph();
                double prune_s = elapsed(b);
                r[inc ? "incremental" : "full"] = {
                    {"cache_read_s", inc ? read_s : 0},
                    {"prepare_s", setup_s},
                    {"leaf_and_merge_s", leaf_s},
                    {"prune_s", prune_s},
                    {"core_total_s", setup_s + leaf_s + prune_s + (inc ? read_s : 0)}};
                auto adj = adjacency(g, cap, allocator);
                for (uint32_t i = 0; i < cap; ++i)
                    for (auto j : adj[i]) require(live[i] && live[j] && i != j, "invalid edge");
                if (inc)
                    newadj = std::move(adj);
                else
                    oldadj = std::move(adj);
                std::cout << "keep=" << keep << " version=" << (inc ? "cache" : "full")
                          << " core_s=" << r[inc ? "incremental" : "full"]["core_total_s"]
                          << std::endl;
            }
            uint64_t mismatch = 0;
            require(ref.size() == actual.size(), "leaf count compare");
            for (uint32_t l = 0; l < ref.size(); ++l) {
                require(ref[l].members == actual[l].members && ref[l].k == actual[l].k &&
                            ref[l].edges.size() == actual[l].edges.size(),
                        "leaf metadata compare");
                for (uint64_t i = 0; i < ref[l].edges.size(); ++i)
                    if (ref[l].edges[i].target != actual[l].edges[i].target ||
                        ref[l].edges[i].distance != actual[l].edges[i].distance)
                        ++mismatch;
            }
            r["topk_edge_mismatches"] = mismatch;
            r["graphs_identical"] = oldadj == newadj;
            r["old_rows"] = stats.old_rows;
            r["reused_old_rows_before_new_merge"] = stats.reused_old_rows;
            r["refilled_old_rows"] = stats.refilled_old_rows;
            r["new_rows"] = stats.new_rows;
            r["full_unique_pairs"] = stats.fullpairs;
            r["incremental_dot_products"] = stats.dots;
            r["k_change_fallback_leaves"] = stats.fallback;
            r["leaves_over_1024"] = stats.oversized;
            if (runqueries) {
                r["full"]["query"] = probe(oldadj, v, queries, live);
                if (oldadj == newadj)
                    r["incremental"]["query"] = r["full"]["query"];
                else
                    r["incremental"]["query"] = probe(newadj, v, queries, live);
            }
            rusage usage{};
            getrusage(RUSAGE_SELF, &usage);
            r["process_peak_rss_bytes"] = usage.ru_maxrss;
            report["runs"].push_back(r);
            std::ofstream(out + "/results.json") << report.dump(2) << '\n';
            std::cout << "validated keep=" << keep << " mismatches=" << mismatch
                      << " graph_equal=" << (oldadj == newadj)
                      << " peak_gib=" << usage.ru_maxrss / double(1ULL << 30) << std::endl;
        }
        return 0;
    } catch (std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
