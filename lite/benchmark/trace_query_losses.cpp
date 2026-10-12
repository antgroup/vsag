// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include <vsag/lite/index.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "lite/rabitq_snapshot.h"
namespace {
void
require(bool value, const char* message) {
    if (not value) {
        throw std::runtime_error(message);
    }
}
template <typename T>
std::vector<T>
record(std::istream& input, uint64_t dim) {
    int32_t width{};
    input.read(reinterpret_cast<char*>(&width), sizeof(width));
    require(static_cast<bool>(input) and width == static_cast<int32_t>(dim), "record dimension");
    std::vector<T> values(dim);
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(dim * sizeof(T)));
    require(static_cast<bool>(input), "truncated record");
    return values;
}
template <typename A, typename B>
void
equal(const A& a, const B& b) {
    require(a.size() == b.size(), "result count parity");
    for (uint64_t rank = 0; rank < a.size(); ++rank) {
        uint32_t left{}, right{};
        std::memcpy(&left, &a[rank].distance, sizeof(float));
        std::memcpy(&right, &b[rank].distance, sizeof(float));
        require(
            static_cast<int64_t>(a[rank].id) == static_cast<int64_t>(b[rank].id) and left == right,
            "ordered native result parity");
    }
}
template <typename A>
bool
contains(const A& candidates, int32_t id) {
    return std::any_of(candidates.begin(), candidates.end(), [id](const auto& c) {
        return static_cast<int64_t>(c.id) == id;
    });
}
int
run(int argc, char** argv) {
    require(argc == 6, "SNAPSHOT DATASET STORAGE OUTPUT_PREFIX EF required");
    namespace codec = vsag::lite::detail::rabitq;
    const std::string snapshot(argv[1]), dataset(argv[2]), storage(argv[3]), prefix(argv[4]);
    const uint64_t ef = std::stoull(argv[5]);
    require(ef == 2048, "frozen validation budget required");
    require(storage == "fp32" or storage == "fp16" or storage == "rabitq8", "invalid storage");
    std::ifstream input(snapshot, std::ios::binary);
    auto loaded = vsag::lite::Index::Load(input);
    require(static_cast<bool>(loaded), "native load failed");
    auto index = std::move(*loaded);
    require(index->Size() == 10000 and index->Dim() == 960, "dataset shape differs");
    codec::MutableGraphState* state_ptr = nullptr;
    std::unique_ptr<codec::MutableGraphState> state;
    if (storage == "rabitq8") {
        std::ifstream encoded(snapshot, std::ios::binary);
        state =
            std::make_unique<codec::MutableGraphState>(codec::load_mutable_snapshot(encoded, true));
        state_ptr = state.get();
        for (uint64_t slot = 0; slot < state->Size(); ++slot) {
            require(state->GetIds()[slot] == static_cast<int64_t>(slot), "ordinal IDs required");
        }
    }
    std::ifstream queries(dataset + "/queries.fvecs", std::ios::binary);
    std::ifstream truth(dataset + "/groundtruth.ivecs", std::ios::binary);
    std::ofstream summary(prefix + ".queries.csv"), evidence(prefix + ".truth.csv"),
        neighbors(prefix + ".neighbors.csv");
    require(static_cast<bool>(queries) and static_cast<bool>(truth) and
                static_cast<bool>(summary) and static_cast<bool>(evidence) and
                static_cast<bool>(neighbors),
            "stream open failed");
    summary << "query,formal_hits,visited_truth,visited_nodes,visited_rerank_hits,all_codes_hits\n";
    evidence << "query,id,visited,formal,visited_rerank,all_codes\n";
    neighbors << "query,rank,id,distance\n" << std::hexfloat;
    for (uint64_t query_id = 0; query_id < 1000; ++query_id) {
        const auto query = record<float>(queries, 960);
        const auto expected = record<int32_t>(truth, 10);
        const auto formal = index->SearchWithOptions(query.data(), 960, 10, {ef});
        require(static_cast<bool>(formal) and formal->size() == 10, "native query failed");
        std::vector<uint8_t> visited(10000);
        const auto traced = index->SearchWithOptions(query.data(), 960, 10, {ef}, [&](int64_t id) {
            require(id >= 0 and id < 10000, "visited ID bounds");
            require(visited[id] == 0, "duplicate visit callback");
            visited[id] = 1;
            return true;
        });
        require(static_cast<bool>(traced) and traced->size() == 10, "trace query failed");
        std::vector<codec::Candidate> complete;
        if (state_ptr != nullptr) {
            const auto internal = state->SearchWithOptions(query.data(), 10, ef);
            equal(internal.neighbors, *formal);
            require(internal.visited ==
                        static_cast<uint64_t>(std::count(visited.begin(), visited.end(), 1)),
                    "visit count parity");
            float norm{};
            const auto normalized = codec::normalize(state->GetModel(), query.data(), norm);
            const float sum = std::accumulate(normalized.begin(), normalized.end(), 0.0F);
            complete = codec::top_k(state->Size(), 10, [&](uint64_t slot) {
                const auto code = state->GetCodes().At(slot);
                const auto coarse = codec::filter_estimate(normalized, norm, code);
                return codec::full_distance(normalized, norm, code, coarse.centered_ip, sum);
            });
        } else {
            equal(*formal, *traced);
        }
        uint64_t a{}, b{}, c{}, d{};
        for (const auto id : expected) {
            require(id >= 0 and id < 10000, "truth ID bounds");
            const bool f = contains(*formal, id), r = contains(*traced, id);
            const bool g = state_ptr != nullptr and contains(complete, id);
            a += f;
            b += visited[id];
            c += r;
            d += g;
            evidence << query_id << ',' << id << ',' << static_cast<uint64_t>(visited[id]) << ','
                     << f << ',' << r << ',' << (state_ptr != nullptr ? static_cast<int>(g) : -1)
                     << '\n';
        }
        summary << query_id << ',' << a << ',' << b << ','
                << std::count(visited.begin(), visited.end(), 1) << ',' << c << ','
                << (state_ptr != nullptr ? static_cast<int64_t>(d) : -1) << '\n';
        for (uint64_t rank = 0; rank < formal->size(); ++rank) {
            neighbors << query_id << ',' << rank << ',' << (*formal)[rank].id << ','
                      << (*formal)[rank].distance << '\n';
        }
    }
    require(queries.peek() == std::char_traits<char>::eof() and
                truth.peek() == std::char_traits<char>::eof(),
            "extra input rows");
    summary.close();
    evidence.close();
    neighbors.close();
    require(
        static_cast<bool>(summary) and static_cast<bool>(evidence) and static_cast<bool>(neighbors),
        "output write failed");
    return 0;
}
}  // namespace
int
main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
