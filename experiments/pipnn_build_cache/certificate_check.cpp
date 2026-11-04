// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#define main final_reuse_benchmark_main
#include "final_reuse.cpp"
#undef main
int
main() {
    try {
        uint64_t checked = 0, accepted = 0;
        for (uint32_t mode = 0; mode < 3; ++mode)
            for (float alpha : {1.0f, 1.2f, 1.5f})
                for (uint32_t seed = 0; seed < 100; ++seed) {
                    constexpr uint32_t n = 64;
                    std::mt19937 random(seed + mode * 1000);
                    std::vector<float> v(n * 128);
                    for (uint32_t i = 0; i < n; ++i)
                        for (uint32_t j = 0; j < 128; ++j)
                            v[i * 128 + j] = mode == 0   ? 0.0f
                                             : mode == 1 ? float(i % 4)
                                                         : float(random() % 101);
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
                        rows.push_back(v.data() + i * 128);
                    }
                    vsag::PiPNNGraphBuilderParameter param;
                    param.alpha = alpha;
                    auto g = make_graph(c, n);
                    vsag::PiPNNPipeline p(param, 128, c.metric_, a, nullptr, 1, g, rows);
                    p.prepare(ids);
                    p.prepare_sketches();
                    std::vector<uint32_t> targets(n - 1);
                    std::iota(targets.begin(), targets.end(), 1);
                    std::shuffle(targets.begin(), targets.end(), random);
                    uint32_t os = 1 + random() % 63;
                    for (uint32_t i = 0; i < os; ++i) {
                        auto id = targets[i];
                        p.reservoirs_[i] = vsag::ReservoirEntry{
                            id, 0, vsag::distance_order_key(p.pair_distance(0, id))};
                    }
                    p.reservoir_states_[0].size = os;
                    p.write_graph();
                    auto old = save_state(p, n, n);
                    for (uint32_t iteration = 0; iteration < 10; ++iteration) {
                        std::shuffle(targets.begin(), targets.end(), random);
                        uint32_t size = 1 + random() % 63;
                        std::array<uint64_t, 64> keys;
                        for (uint32_t i = 0; i < size; ++i) {
                            auto id = targets[i];
                            p.reservoirs_[i] = vsag::ReservoirEntry{
                                id, 0, vsag::distance_order_key(p.pair_distance(0, id))};
                            keys[i] = pack(p.reservoirs_[i]);
                        }
                        p.reservoir_states_[0].size = size;
                        std::sort(keys.begin(), keys.begin() + size);
                        ReuseStats stats;
                        bool safe = certificate(p, 0, old, keys.data(), size, stats);
                        vsag::PruneScratch scratch(a);
                        p.robust_prune(0, p.reservoirs_.data(), size, scratch);
                        std::vector<uint32_t> got(scratch.neighbors.begin(),
                                                  scratch.neighbors.end());
                        if (safe) {
                            ++accepted;
                            require(got == old.adjacency[0], "certificate false positive");
                        }
                        ++checked;
                    }
                }
        std::cout << "PASS " << checked << " randomized reservoir mutations; " << accepted
                  << " accepted certificates; zero false positives\n";
    } catch (std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}
