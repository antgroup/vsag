// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstring>
#include <istream>
#include <ostream>
#include <unordered_set>

#include "lite/rabitq_graph_state.h"

// Separate owned-memory encoded format: VSAGLQ01 version 1. Do not reinterpret
// existing VSAGLT01 or experimental VSLRBQ01 payloads as this public candidate.
namespace vsag::lite::detail::rabitq {

inline void
write_u64(std::ostream& output, uint64_t value, uint64_t bytes = 8) {
    for (uint64_t i = 0; i < bytes; ++i) {
        output.put(static_cast<char>((value >> (8U * i)) & 0xffU));
    }
    codec_require(static_cast<bool>(output), "snapshot write failed");
}

inline uint64_t
read_u64(std::istream& input, uint64_t bytes = 8) {
    uint64_t value = 0;
    for (uint64_t i = 0; i < bytes; ++i) {
        const int byte = input.get();
        codec_require(byte != std::char_traits<char>::eof(), "truncated snapshot");
        value |= static_cast<uint64_t>(static_cast<uint8_t>(byte)) << (8U * i);
    }
    return value;
}

inline void
write_i64(std::ostream& output, int64_t value) {
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    write_u64(output, bits);
}

inline int64_t
read_i64(std::istream& input) {
    const uint64_t bits = read_u64(input);
    int64_t value = 0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline void
write_float(std::ostream& output, float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    write_u64(output, bits, sizeof(bits));
}

inline float
read_float(std::istream& input) {
    const auto bits = static_cast<uint32_t>(read_u64(input, sizeof(uint32_t)));
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    codec_require(std::isfinite(value), "non-finite snapshot metadata");
    return value;
}

inline void
write_bytes(std::ostream& output, const uint8_t* bytes, uint64_t size) {
    if (size > 0) {
        output.write(reinterpret_cast<const char*>(bytes), static_cast<std::streamsize>(size));
    }
    codec_require(static_cast<bool>(output), "snapshot write failed");
}

inline void
write_bytes(std::ostream& output, const std::vector<uint8_t>& bytes) {
    write_bytes(output, bytes.data(), bytes.size());
}

inline void
read_bytes(std::istream& input, uint8_t* bytes, uint64_t size) {
    if (size > 0) {
        input.read(reinterpret_cast<char*>(bytes), static_cast<std::streamsize>(size));
    }
    codec_require(static_cast<bool>(input), "truncated snapshot");
}

inline void
read_bytes(std::istream& input, std::vector<uint8_t>& bytes) {
    read_bytes(input, bytes.data(), bytes.size());
}

inline void
write_snapshot_payload(std::ostream& output,
                       uint64_t version,
                       const Model& model,
                       const EncodedRecords& codes) {
    constexpr char magic[] = "VSAGLQ01";
    codec_require((version == 1 or version == 2 or version == 3) and model.dim == codes.dim,
                  "snapshot model and records disagree");
    output.write(magic, 8);
    write_u64(output, version);
    write_u64(output, model.dim);
    write_u64(output, codes.Size());
    write_u64(output, model.centroid.size());
    write_u64(output, model.flips.size());
    for (float value : model.centroid) {
        write_float(output, value);
    }
    write_bytes(output, model.flips);
    for (uint64_t id = 0; id < codes.Size(); ++id) {
        const auto code = codes.At(id);
        write_float(output, code.metadata.norm);
        write_float(output, code.metadata.code_norm);
        write_float(output, code.metadata.error);
        write_float(output, code.metadata.filter_norm);
        write_float(output, code.metadata.filter_error);
        write_float(output, code.metadata.lower_bound_error);
        write_bytes(output, code.filter, codes.FilterBytes());
        write_bytes(output, code.supplement, codes.SupplementBytes());
    }
}

struct EncodedSnapshot {
    Model model;
    EncodedRecords codes;
};

inline EncodedSnapshot
load_snapshot_payload(std::istream& input, uint64_t expected_version) {
    char magic[8]{};
    input.read(magic, sizeof(magic));
    codec_require(input and std::memcmp(magic, "VSAGLQ01", 8) == 0, "invalid snapshot magic");
    codec_require(read_u64(input) == expected_version, "unsupported snapshot version");
    const uint64_t dim = read_u64(input);
    const uint64_t count = read_u64(input);
    const uint64_t centroid_size = read_u64(input);
    const uint64_t flips_size = read_u64(input);
    codec_require(dim > 0 and dim <= (1U << 20U) and centroid_size == dim,
                  "invalid snapshot model");
    const uint64_t plane_bytes = (dim + 7) / 8;
    codec_require(flips_size == K_ROUNDS * plane_bytes and count <= 1000000,
                  "invalid snapshot layout");
    const auto position = input.tellg();
    input.seekg(0, std::ios::end);
    const auto end = input.tellg();
    codec_require(position != std::streampos(-1) and end >= position,
                  "seekable encoded input required");
    input.seekg(position);
    const uint64_t remaining = static_cast<uint64_t>(end - position);
    const uint64_t model_bytes = dim * sizeof(float) + flips_size;
    const uint64_t record_bytes = plane_bytes * K_TOTAL_BITS + 24 + 16;
    codec_require(
        remaining >= model_bytes + 48 and count <= (remaining - model_bytes - 48) / record_bytes,
        "encoded payload exceeds remaining bytes");
    Model model{dim, std::vector<float>(dim), std::vector<uint8_t>(flips_size)};
    for (float& value : model.centroid) {
        value = read_float(input);
    }
    read_bytes(input, model.flips);
    EncodedRecords codes(dim);
    codes.Resize(count);
    for (uint64_t id = 0; id < count; ++id) {
        auto& metadata = codes.metadata[id];
        metadata.norm = read_float(input);
        metadata.code_norm = read_float(input);
        metadata.error = read_float(input);
        metadata.filter_norm = read_float(input);
        metadata.filter_error = read_float(input);
        metadata.lower_bound_error = read_float(input);
        codec_require(metadata.norm > 0.0F and metadata.code_norm > 0.0F and
                          metadata.filter_norm > 0.0F and metadata.filter_error >= 1e-5F and
                          metadata.filter_error <= 1.0F and metadata.lower_bound_error >= 0.0F,
                      "invalid snapshot metadata");
        read_bytes(input, codes.filters.data() + id * codes.FilterBytes(), codes.FilterBytes());
        read_bytes(input,
                   codes.supplements.data() + id * codes.SupplementBytes(),
                   codes.SupplementBytes());
    }
    return {std::move(model), std::move(codes)};
}

inline void
validate_graph_topology(const GraphTopology& graph, uint64_t count) {
    codec_require(graph.offsets.size() == count + 1 and graph.offsets.front() == 0,
                  "invalid graph offset layout");
    codec_require(graph.neighbors.size() <= count * 64, "graph edge count exceeds limit");
    for (uint64_t slot = 0; slot < count; ++slot) {
        const uint64_t begin = graph.offsets[slot];
        const uint64_t end = graph.offsets[slot + 1];
        codec_require(begin <= end and end <= graph.neighbors.size() and end - begin <= 64,
                      "invalid graph adjacency bounds");
        for (uint64_t edge = begin; edge < end; ++edge) {
            const uint64_t neighbor = graph.neighbors[edge];
            codec_require(neighbor < count and neighbor != slot, "invalid graph neighbor");
            codec_require(
                std::find(graph.neighbors.begin() + static_cast<int64_t>(begin),
                          graph.neighbors.begin() + static_cast<int64_t>(edge),
                          neighbor) == graph.neighbors.begin() + static_cast<int64_t>(edge),
                "duplicate graph neighbor");
        }
    }
    codec_require(graph.offsets.back() == graph.neighbors.size(), "graph edge count mismatch");
}

inline void
save_mutable_snapshot(std::ostream& output, const MutableGraphState& state) {
    state.Validate();
    write_snapshot_payload(output, 1, state.GetModel(), state.GetCodes());
    write_u64(output, state.GetMaxDegree());
    write_u64(output, state.GetEfSearch());
    write_u64(output, state.GetIds().size());
    for (int64_t id : state.GetIds()) {
        write_i64(output, id);
    }
    const auto graph = state.GetGraph();
    write_u64(output, graph.offsets.size());
    write_u64(output, graph.neighbors.size());
    for (uint64_t offset : graph.offsets) {
        write_u64(output, offset);
    }
    for (uint64_t neighbor : graph.neighbors) {
        write_u64(output, neighbor);
    }
}

inline MutableGraphState
load_mutable_snapshot(std::istream& input) {
    auto loaded = load_snapshot_payload(input, 1);
    const uint64_t max_degree = read_u64(input);
    const uint64_t ef_search = read_u64(input);
    codec_require(max_degree >= 2 and max_degree <= 64 and ef_search >= max_degree,
                  "invalid encoded graph options");
    const uint64_t id_count = read_u64(input);
    codec_require(id_count == loaded.codes.Size(), "invalid mutable snapshot ID count");
    std::vector<int64_t> ids(id_count);
    std::unordered_set<int64_t> unique_ids;
    for (int64_t& id : ids) {
        id = read_i64(input);
        codec_require(unique_ids.insert(id).second, "duplicate mutable snapshot ID");
    }
    const uint64_t offset_count = read_u64(input);
    const uint64_t neighbor_count = read_u64(input);
    codec_require(
        offset_count == loaded.codes.Size() + 1 and neighbor_count <= loaded.codes.Size() * 64,
        "invalid mutable snapshot graph layout");
    const auto graph_position = input.tellg();
    input.seekg(0, std::ios::end);
    const auto graph_end = input.tellg();
    input.seekg(graph_position);
    codec_require(graph_position != std::streampos(-1) and graph_end >= graph_position and
                      static_cast<uint64_t>(graph_end - graph_position) ==
                          (offset_count + neighbor_count) * 8,
                  "encoded graph payload size mismatch");
    GraphTopology graph{std::vector<uint64_t>(offset_count), std::vector<uint64_t>(neighbor_count)};
    for (uint64_t& offset : graph.offsets) {
        offset = read_u64(input);
    }
    for (uint64_t& neighbor : graph.neighbors) {
        neighbor = read_u64(input);
    }
    validate_graph_topology(graph, loaded.codes.Size());
    codec_require(input.peek() == std::char_traits<char>::eof(), "snapshot trailing bytes");
    return {std::move(loaded.model),
            std::move(loaded.codes),
            graph,
            std::move(ids),
            max_degree,
            ef_search};
}

}  // namespace vsag::lite::detail::rabitq
