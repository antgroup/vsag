// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

struct QueryResult {
    std::array<uint32_t, 10> ids;
    uint64_t distances;
};
QueryResult
search_graph(const Adj& adj,
             const std::vector<float>& v,
             const float* query,
             const std::vector<uint8_t>& live,
             uint32_t ef) {
    uint32_t n = live.size();
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
        if (best.size() < ef || P(d, id) < best.top()) {
            frontier.push({d, id});
            best.push({d, id});
            if (best.size() > ef)
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
        if (best.size() == ef && cur > best.top())
            break;
        for (auto id : adj[cur.second]) add(id);
    }
    std::vector<P> found;
    while (!best.empty()) {
        found.push_back(best.top());
        best.pop();
    }
    std::sort(found.begin(), found.end());
    QueryResult r;
    r.ids.fill(UINT32_MAX);
    r.distances = count;
    for (uint32_t j = 0; j < std::min<uint64_t>(10, found.size()); ++j) r.ids[j] = found[j].second;
    return r;
}
void
save_graph(const std::string& path, const Adj& adj) {
    std::ofstream f(path, std::ios::binary);
    uint32_t n = adj.size();
    f.write((char*)&n, 4);
    for (auto& row : adj) {
        uint32_t count = row.size();
        f.write((char*)&count, 4);
        f.write((char*)row.data(), uint64_t(count) * 4);
    }
    require(bool(f), "graph export");
}
Json
benchmark_pair(const Adj& reference,
               const Adj& warm,
               const std::vector<float>& v,
               const std::vector<float>& queries,
               const std::vector<uint8_t>& live,
               const std::string& out,
               uint32_t keep) {
    save_graph(out + "/reference_" + std::to_string(keep) + ".graph", reference);
    save_graph(out + "/warm_" + std::to_string(keep) + ".graph", warm);
    std::vector<std::array<uint32_t, 10>> truth(512);
    auto t = Clock::now();
    for (uint32_t qi = 0; qi < 512; ++qi) {
        std::vector<std::pair<float, uint32_t>> ds;
        ds.reserve(1000000);
        auto* q = queries.data() + qi * 128;
        for (uint32_t id = 0; id < live.size(); ++id)
            if (live[id])
                ds.push_back({vsag::FP32ComputeL2Sqr(q, v.data() + uint64_t(id) * 128, 128), id});
        std::partial_sort(ds.begin(), ds.begin() + 10, ds.end());
        for (uint32_t j = 0; j < 10; ++j) truth[qi][j] = ds[j].second;
    }
    double truth_s = elapsed(t);
    Json report = {
        {"truth_s_excluded", truth_s},
        {"query_count", 512},
        {"rounds", 4},
        {"search_scope",
         "custom bottom-graph best-first; 8 fixed entry points; single caller; not HGraph API"},
        {"ef", Json::array()}};
    for (uint32_t ef : {64U, 128U, 256U, 512U}) {
        std::array<std::vector<QueryResult>, 2> expected;
        Json quality = Json::array();
        for (uint32_t version = 0; version < 2; ++version) {
            auto& adj = version ? warm : reference;
            uint64_t hits = 0, distances = 0;
            Json per = Json::array();
            expected[version].reserve(512);
            for (uint32_t qi = 0; qi < 512; ++qi) {
                auto r = search_graph(adj, v, queries.data() + qi * 128, live, ef);
                uint32_t local = 0;
                for (auto id : r.ids)
                    for (auto target : truth[qi]) local += id == target;
                hits += local;
                distances += r.distances;
                per.push_back({local, r.distances});
                expected[version].push_back(r);
            }
            quality.push_back({{"recall10", hits / 5120.0},
                               {"distance_mean", distances / 512.0},
                               {"per_query_hits_distances", per}});
        }
        uint64_t sink = 0;
        for (uint32_t version = 0; version < 2; ++version)
            for (uint32_t qi = 0; qi < 64; ++qi) {
                auto r = search_graph(
                    version ? warm : reference, v, queries.data() + qi * 128, live, ef);
                sink += r.distances;
            }
        std::array<std::vector<double>, 2> seconds;
        std::vector<QueryResult> outputs(512);
        for (uint32_t round = 0; round < 4; ++round) {
            std::vector<uint32_t> order(512);
            std::iota(order.begin(), order.end(), 0);
            std::mt19937 random(20261104 + round);
            std::shuffle(order.begin(), order.end(), random);
            for (uint32_t pos = 0; pos < 2; ++pos) {
                bool swapped = round == 1 || round == 2;
                uint32_t version = swapped ? 1 - pos : pos;
                auto& adj = version ? warm : reference;
                auto start = Clock::now();
                for (uint32_t i = 0; i < 512; ++i)
                    outputs[i] = search_graph(adj, v, queries.data() + order[i] * 128, live, ef);
                double duration = elapsed(start);
                seconds[version].push_back(duration);
                for (uint32_t i = 0; i < 512; ++i) {
                    auto& want = expected[version][order[i]];
                    require(outputs[i].ids == want.ids && outputs[i].distances == want.distances,
                            "timed query changed result");
                    sink += outputs[i].distances + outputs[i].ids[0];
                }
            }
        }
        for (uint32_t version = 0; version < 2; ++version) {
            auto values = seconds[version];
            std::sort(values.begin(), values.end());
            double median = (values[1] + values[2]) / 2;
            quality[version]["qps_median_batch"] = 512 / median;
            quality[version]["qps_min"] = 512 / values.back();
            quality[version]["qps_max"] = 512 / values.front();
            quality[version]["batch_seconds"] = seconds[version];
        }
        Json row = {
            {"ef", ef}, {"reference", quality[0]}, {"warm", quality[1]}, {"checksum_sink", sink}};
        report["ef"].push_back(row);
        std::ofstream(out + "/search_" + std::to_string(keep) + ".json") << report.dump(2) << '\n';
        std::cout << "search keep=" << keep << " ef=" << ef
                  << " reference_recall=" << quality[0]["recall10"]
                  << " warm_recall=" << quality[1]["recall10"]
                  << " reference_qps=" << quality[0]["qps_median_batch"]
                  << " warm_qps=" << quality[1]["qps_median_batch"] << std::endl;
    }
    return report;
}
