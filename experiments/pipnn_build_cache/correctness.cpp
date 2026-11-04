// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#define main experiment_main
#include "bench.cpp"
#undef main
int
main() {
    try {
        uint64_t cases = 0;
        for (uint32_t n : {2U, 3U, 4U, 5U, 6U, 7U, 16U, 33U})
            for (uint32_t mode = 0; mode < 3; ++mode) {
                vsag::IndexCommonParam c;
                c.dim_ = 128;
                c.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
                c.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
                c.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();
                auto* a = c.allocator_.get();
                std::vector<float> v(uint64_t(n) * 128);
                for (uint32_t i = 0; i < n; ++i)
                    for (uint32_t j = 0; j < 128; ++j)
                        v[uint64_t(i) * 128 + j] = mode == 0   ? 0
                                                   : mode == 1 ? float(i % 3)
                                                               : float((i * 17 + j * 3) % 41);
                vsag::Vector<vsag::InnerIdType> ids(a);
                vsag::Vector<const float*> rows(a);
                for (uint32_t i = 0; i < n; ++i) {
                    ids.push_back(i);
                    rows.push_back(v.data() + uint64_t(i) * 128);
                }
                vsag::PiPNNGraphBuilderParameter param;
                auto g = make_graph(c, n);
                vsag::PiPNNPipeline old(param, 128, c.metric_, a, nullptr, 1, g, rows);
                old.prepare(ids);
                old.prepare_sketches();
                vsag::Leaf leaf(a);
                for (uint32_t i = 0; i < n; ++i) leaf.push_back(i);
                std::vector<Snapshot> ss;
                capture_sink = &ss;
                capture_only = true;
                old.build_leaf(leaf);
                capture_sink = nullptr;
                capture_only = false;
                require(ss.size() == 1, "old snapshot count");
                for (uint32_t deleted = 0; deleted < n; ++deleted) {
                    std::vector<uint8_t> live(n, 1);
                    for (uint32_t i = 0; i < deleted; ++i) live[(i * 5) % n] = 0;
                    // For n divisible by 5 the permutation is not bijective; still exercises valid masks.
                    vsag::Leaf kept(a);
                    for (uint32_t i = 0; i < n; ++i)
                        if (live[i])
                            kept.push_back(i);
                    auto ng = make_graph(c, n);
                    vsag::PiPNNPipeline p(param, 128, c.metric_, a, nullptr, 1, ng, rows);
                    p.prepare(ids);
                    p.prepare_sketches();
                    std::vector<Snapshot> ref;
                    if (kept.size() > 1) {
                        capture_sink = &ref;
                        capture_only = true;
                        p.build_leaf(kept);
                        capture_sink = nullptr;
                        capture_only = false;
                    } else {
                        Snapshot s;
                        s.k = 0;
                        s.members.assign(kept.begin(), kept.end());
                        ref.push_back(s);
                    }
                    Stats stats;
                    auto got = incremental(p, ss[0], live, stats);
                    require(ref[0].members == got.members && ref[0].k == got.k &&
                                ref[0].edges.size() == got.edges.size(),
                            "corner metadata");
                    for (uint64_t e = 0; e < got.edges.size(); ++e)
                        require(ref[0].edges[e].target == got.edges[e].target &&
                                    ref[0].edges[e].distance == got.edges[e].distance,
                                "corner top-k mismatch");
                    ++cases;
                }
            }
        std::cout << "PASS " << cases << " small deletion/tie/K-boundary cases\n";
    } catch (std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
