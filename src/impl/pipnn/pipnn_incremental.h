// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <bit>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

#include "storage/stream_reader.h"
#include "storage/stream_writer.h"

namespace vsag {

class AlignedBlockLayout;
class Allocator;

struct PiPNNRoute {
    std::vector<uint32_t> leaders;
    std::vector<PiPNNRoute> children;
    uint32_t fanout{1};
    uint32_t leaf{UINT32_MAX};
};

struct PiPNNLocalNeighbor {
    uint32_t target;
    float distance;
};

// Each entry retains its exact FP32 distance and a position in the leaf's member list.
// Packed byte loads also work on architectures that require aligned float access.
class PiPNNLeafTopK {
public:
    template <typename Byte>
    struct RowView {
        Byte* data;
        uint32_t stride;

        float
        Distance(uint32_t rank) const {
            float distance;
            std::memcpy(&distance, data + uint64_t{rank} * stride, sizeof(distance));
            return distance;
        }
        PiPNNLocalNeighbor
        Get(uint32_t rank) const {
            const auto* index = data + uint64_t{rank} * stride + sizeof(float);
            uint32_t target;
            if (stride == sizeof(float) + 1) {
                target = *index;
            } else if (stride == sizeof(float) + 2) {
                uint16_t small;
                std::memcpy(&small, index, sizeof(small));
                target = small;
            } else {
                std::memcpy(&target, index, sizeof(target));
            }
            return {target, Distance(rank)};
        }
        void
        Set(uint32_t rank, PiPNNLocalNeighbor neighbor) const {
            auto* entry = data + uint64_t{rank} * stride;
            std::memcpy(entry, &neighbor.distance, sizeof(float));
            auto* index = entry + sizeof(float);
            if (stride == sizeof(float) + 1) {
                *index = static_cast<uint8_t>(neighbor.target);
            } else if (stride == sizeof(float) + 2) {
                const auto small = static_cast<uint16_t>(neighbor.target);
                std::memcpy(index, &small, sizeof(small));
            } else {
                std::memcpy(index, &neighbor.target, sizeof(neighbor.target));
            }
        }
        void
        ReplaceLast(uint32_t rank, uint32_t last, PiPNNLocalNeighbor neighbor) const {
            auto* first = data + uint64_t{rank} * stride;
            std::memmove(first + stride, first, uint64_t{last - rank} * stride);
            Set(rank, neighbor);
        }
    };

    PiPNNLeafTopK() = default;
    PiPNNLeafTopK(uint32_t rows, uint32_t width);
    uint32_t
    Width() const {
        return width_;
    }
    RowView<const uint8_t>
    Row(uint32_t row) const {
        return {row_data(row), stride_};
    }
    RowView<uint8_t>
    Row(uint32_t row) {
        return {row_data(row), stride_};
    }
    void
    AppendRows(uint32_t count);
    uint64_t
    MemoryUsage() const;
    void
    Write(StreamWriter& writer, std::span<const uint32_t> points) const;

private:
    // New members use small row-aligned blocks. The packed base stays in place unless
    // the leaf grows beyond the range of the original target width.
    static constexpr uint32_t BLOCK_ROWS = 4;
    uint32_t base_rows_{0};
    uint32_t rows_{0};
    uint32_t width_{0};
    uint32_t stride_{0};
    std::unique_ptr<uint8_t[]> base_;
    std::vector<std::unique_ptr<uint8_t[]>> blocks_;

    static uint32_t
    entry_size(uint32_t rows) {
        return sizeof(float) + (rows <= 256 ? 1 : rows <= 65536 ? 2 : 4);
    }
    uint8_t*
    row_data(uint32_t row) const {
        if (width_ == 0)
            return nullptr;
        if (row < base_rows_)
            return base_.get() + uint64_t{row} * width_ * stride_;
        const auto offset = row - base_rows_;
        return blocks_[offset / BLOCK_ROWS].get() +
               uint64_t{offset % BLOCK_ROWS} * width_ * stride_;
    }
};

struct PiPNNCachedLeaf {
    std::vector<uint32_t> points;
    // Row-major, sorted local top-k.
    PiPNNLeafTopK nearest;
    uint64_t depth{UINT64_MAX};  // Minimum route depth across merged terminals.
    PiPNNRoute replacement;      // set when this leaf has been split

    PiPNNLocalNeighbor
    Neighbor(uint32_t row, uint32_t rank) const {
        auto neighbor = nearest.Row(row).Get(rank);
        neighbor.target = points[neighbor.target];
        return neighbor;
    }
};

// Fixed row blocks keep previous sketches at stable addresses during Add.
class PiPNNFloatRows {
public:
    void
    Resize(uint64_t rows, uint64_t width, Allocator* allocator);
    float*
    Row(uint32_t row) {
        return blocks_[row >> shift_].data() + (row & mask_) * width_;
    }
    const float*
    Row(uint32_t row) const {
        return blocks_[row >> shift_].data() + (row & mask_) * width_;
    }
    float&
    Scalar(uint32_t row) {
        return blocks_[row >> shift_][row & mask_];
    }
    uint64_t
    Size() const {
        return rows_;
    }
    uint64_t
    Width() const {
        return width_;
    }
    uint64_t
    MemoryUsage() const;
    void
    Write(StreamWriter& writer) const;
    void
    Read(StreamReader& reader, uint64_t rows, uint64_t width, bool optional, Allocator* allocator);

private:
    std::vector<Vector<float>> blocks_;
    uint64_t rows_{0};
    uint64_t width_{0};
    uint32_t shift_{0};
    uint32_t mask_{0};
};

struct PiPNNProjectionState {
    std::vector<float> planes;
    PiPNNFloatRows projections;
    PiPNNFloatRows norms;
    void
    WriteSketches(StreamWriter& writer) const;
    void
    ReadSketches(StreamReader& reader,
                 uint64_t count,
                 uint64_t dim,
                 uint64_t planes_count,
                 Allocator* allocator);
};

enum class PiPNNVectorPayload { Absent, Restore, Skip };

struct PiPNNIncrementalState : PiPNNProjectionState {
    PiPNNRoute root;
    std::vector<PiPNNCachedLeaf> leaves;
    std::vector<uint32_t> ids;

    // Derived on first use and omitted from snapshots.
    std::vector<float> root_leader_vectors;
    void
    UpdateLeafDepths(const PiPNNRoute& route, uint64_t depth);

    PiPNNIncrementalState();
    ~PiPNNIncrementalState();
    void
    ReserveVectors(uint64_t count, uint64_t dim, Allocator* allocator);
    const float*
    GetVector(uint32_t row) const;
    void
    SetVector(uint32_t row, const float* vector);
    std::span<const float* const>
    Rows() const {
        return {row_pointers_.data(), vectors_ ? ids.size() : 0};
    }
    bool
    ContainsId(uint32_t id) const {
        return (id < ids.size() && ids[id] == id) || sparse_ids_.count(id) != 0;
    }
    uint32_t
    LocalId(uint32_t id) const {
        return id < ids.size() && ids[id] == id ? id : sparse_ids_.at(id);
    }
    bool
    HasDenseIds() const {
        return sparse_ids_.empty();
    }
    void
    DropOwnedRows();
    void
    AppendIds(std::span<const uint32_t> incoming);
    uint64_t
    RequiredCapacity() const {
        return required_capacity_;
    }
    uint64_t
    MemoryUsage() const;
    void
    Serialize(StreamWriter& writer, bool with_vectors = true) const;
    void
    Deserialize(StreamReader& reader,
                uint64_t count,
                uint64_t dim,
                uint64_t planes_count,
                uint64_t graph_capacity,
                Allocator* allocator,
                PiPNNVectorPayload vectors = PiPNNVectorPayload::Restore,
                const std::function<void(StreamReader&)>& consume_vectors = {});

private:
    std::vector<const float*> row_pointers_;
    std::unordered_map<uint32_t, uint32_t> sparse_ids_;
    uint64_t required_capacity_{0};
    std::unique_ptr<AlignedBlockLayout> vectors_;
    uint64_t dimensions_{0};
};
}  // namespace vsag
