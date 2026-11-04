// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

Json
probe512(const Adj& adj,
         const std::vector<float>& v,
         const std::vector<float>& q,
         const std::vector<uint8_t>& live) {
    uint32_t n = live.size();
    uint64_t hits = 0, evals = 0;
    Json per = Json::array();
    for (uint32_t qi = 0; qi < 512; ++qi) {
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
    return {{"recall10", hits / 5120.0},
            {"distance_evaluations_mean", evals / 512.0},
            {"per_query_hits_distances", per},
            {"search",
             "custom bottom-graph best-first ef=256, 8 fixed entry points; not HGraph search"}};
}
