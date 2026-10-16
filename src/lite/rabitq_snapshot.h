// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstring>
#include <istream>
#include <limits>
#include <ostream>
#include <type_traits>

#include "lite/rabitq_graph_state.h"

// Separate owned-memory encoded format: VSAGLQ01 version 1. Do not reinterpret
// existing VSAGLT01 or experimental VSLRBQ01 payloads as this public candidate.
namespace vsag::lite::detail::rabitq {

inline void
write_u64(std::ostream& output, uint64_t value, uint64_t bytes = 8) {
    char encoded[8];
    for (uint64_t i = 0; i < bytes; ++i) {
        encoded[i] = static_cast<char>((value >> (8U * i)) & 0xffU);
    }
    output.write(encoded, static_cast<std::streamsize>(bytes));
    codec_require(static_cast<bool>(output), "snapshot write failed");
}

inline uint64_t
read_u64(std::istream& input, uint64_t bytes = 8) {
    char encoded[8];
    input.read(encoded, static_cast<std::streamsize>(bytes));
    codec_require(static_cast<bool>(input), "truncated snapshot");
    uint64_t value = 0;
    for (uint64_t i = 0; i < bytes; ++i) {
        value |= static_cast<uint64_t>(static_cast<uint8_t>(encoded[i])) << (8U * i);
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

inline bool
snapshot_little_endian() {
    const uint16_t value = 1;
    return *reinterpret_cast<const uint8_t*>(&value) == 1;
}

// The same block-I/O principle as native StreamReader::ReadVector, with this
// format's explicit little-endian conversion and capacity/failure contracts.
template <typename T>
inline void
write_integer_array(std::ostream& output,
                    const std::vector<T>& values,
                    bool bulk = snapshot_little_endian()) {
    static_assert(std::is_same_v<T, uint64_t> or std::is_same_v<T, int64_t>);
    codec_require(
        values.size() <=
            static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(T),
        "snapshot array exceeds stream capacity");
    if (values.empty()) {
        return;
    }
    if (bulk) {
        output.write(reinterpret_cast<const char*>(values.data()),
                     static_cast<std::streamsize>(values.size() * sizeof(T)));
        codec_require(static_cast<bool>(output), "snapshot write failed");
        return;
    }
    for (T value : values) {
        uint64_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        write_u64(output, bits);
    }
}

template <typename T>
inline void
read_integer_array(std::istream& input,
                   std::vector<T>& values,
                   bool bulk = snapshot_little_endian()) {
    static_assert(std::is_same_v<T, uint64_t> or std::is_same_v<T, int64_t>);
    codec_require(
        values.size() <=
            static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(T),
        "snapshot array exceeds stream capacity");
    if (values.empty()) {
        return;
    }
    if (bulk) {
        input.read(reinterpret_cast<char*>(values.data()),
                   static_cast<std::streamsize>(values.size() * sizeof(T)));
        codec_require(static_cast<bool>(input), "truncated snapshot");
        return;
    }
    for (T& value : values) {
        const uint64_t bits = read_u64(input);
        std::memcpy(&value, &bits, sizeof(bits));
    }
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
    const auto remaining = static_cast<uint64_t>(end - position);
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

// Preserve CSR wire order without materializing a second whole graph.
inline void
write_graph_adjacency(std::ostream& output,
                      const std::vector<std::vector<uint64_t>>& adjacency,
                      bool bulk = snapshot_little_endian()) {
    codec_require(adjacency.size() < std::numeric_limits<uint64_t>::max(),
                  "snapshot offset count overflow");
    uint64_t edges = 0;
    for (const auto& row : adjacency) {
        codec_require(row.size() <= std::numeric_limits<uint64_t>::max() - edges,
                      "snapshot edge count overflow");
        edges += row.size();
    }
    write_u64(output, adjacency.size() + 1);
    write_u64(output, edges);
    constexpr uint64_t k_buffer_entries = 8192;
    std::vector<uint64_t> buffer;
    buffer.reserve(
        std::min(k_buffer_entries, std::max(static_cast<uint64_t>(adjacency.size()) + 1, edges)));
    const auto flush = [&]() {
        write_integer_array(output, buffer, bulk);
        buffer.clear();
    };
    const auto append = [&](uint64_t value) {
        buffer.push_back(value);
        if (buffer.size() == k_buffer_entries) {
            flush();
        }
    };
    uint64_t offset = 0;
    append(offset);
    for (const auto& row : adjacency) {
        offset += row.size();
        append(offset);
    }
    flush();
    for (const auto& row : adjacency) {
        for (const uint64_t neighbor : row) {
            append(neighbor);
        }
    }
    flush();
}

inline void
save_mutable_snapshot(std::ostream& output, const MutableGraphState& state) {
    state.Validate();
    write_snapshot_payload(output, 1, state.GetModel(), state.GetCodes());
    write_u64(output, state.GetMaxDegree());
    write_u64(output, state.GetEfSearch());
    write_u64(output, state.GetIds().size());
    write_integer_array(output, state.GetIds());
    write_graph_adjacency(output, state.GetAdjacency());
}

inline MutableGraphState
load_mutable_snapshot(std::istream& input, bool use_incoming_adjacency = false) {
    auto loaded = load_snapshot_payload(input, 1);
    const uint64_t max_degree = read_u64(input);
    const uint64_t ef_search = read_u64(input);
    codec_require(max_degree >= 2 and max_degree <= 64 and ef_search >= max_degree,
                  "invalid encoded graph options");
    const uint64_t id_count = read_u64(input);
    codec_require(id_count == loaded.codes.Size(), "invalid mutable snapshot ID count");
    std::vector<int64_t> ids(id_count);
    read_integer_array(input, ids);
    // MutableGraphState rejects duplicate IDs while building its required slot map.
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
    read_integer_array(input, graph.offsets);
    read_integer_array(input, graph.neighbors);
    validate_graph_topology(graph, loaded.codes.Size());
    codec_require(input.peek() == std::char_traits<char>::eof(), "snapshot trailing bytes");
    return {std::move(loaded.model),
            std::move(loaded.codes),
            graph,
            std::move(ids),
            max_degree,
            ef_search,
            false,
            use_incoming_adjacency};
}

}  // namespace vsag::lite::detail::rabitq
