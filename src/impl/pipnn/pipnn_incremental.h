// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <bit>
#include <cstdint>
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

// Build and snapshot load supply one packed block. New members use small row-aligned
// blocks, so updating a leaf never relocates existing rows of the same width.
class PiPNNLeafTopK {
public:
    PiPNNLeafTopK() = default;
    PiPNNLeafTopK(uint32_t rows, std::vector<PiPNNLocalNeighbor>&& packed);

    uint32_t
    Width() const {
        return width_;
    }
    std::span<const PiPNNLocalNeighbor>
    Row(uint32_t row) const {
        return {row_data(*this, row), width_};
    }
    std::span<PiPNNLocalNeighbor>
    Row(uint32_t row) {
        return {row_data(*this, row), width_};
    }
    void
    AppendRows(uint32_t count);
    uint64_t
    MemoryUsage() const;
    void
    Write(StreamWriter& writer) const;

private:
    // A leaf often receives only a few new members. Bound unused tail capacity to three rows.
    static constexpr uint32_t BLOCK_ROWS = 4;
    uint32_t base_rows_{0};
    uint32_t rows_{0};
    uint32_t width_{0};
    std::vector<PiPNNLocalNeighbor> base_;
    std::vector<std::unique_ptr<PiPNNLocalNeighbor[]>> blocks_;

    template <typename Self>
    static auto
    row_data(Self& self, uint32_t row) -> decltype(self.base_.data()) {
        if (self.width_ == 0) {
            return nullptr;
        }
        if (row < self.base_rows_) {
            return self.base_.data() + uint64_t{row} * self.width_;
        }
        const auto offset = row - self.base_rows_;
        return self.blocks_[offset / BLOCK_ROWS].get() +
               uint64_t{offset % BLOCK_ROWS} * self.width_;
    }
};

struct PiPNNCachedLeaf {
    std::vector<uint32_t> points;
    // Row-major, sorted local top-k. Targets use pipeline-local point IDs.
    PiPNNLeafTopK nearest;
    uint64_t depth{UINT64_MAX};  // Minimum route depth across merged terminals.
    PiPNNRoute replacement;      // set when this leaf has been split
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

struct PiPNNIncrementalState : PiPNNProjectionState {
    PiPNNRoute root;
    std::vector<PiPNNCachedLeaf> leaves;
    // Derived from leaf membership, rebuilt on load.
    std::vector<std::vector<uint32_t>> memberships;
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
                bool with_vectors = true);

private:
    std::vector<const float*> row_pointers_;
    std::unordered_map<uint32_t, uint32_t> sparse_ids_;
    uint64_t required_capacity_{0};
    std::unique_ptr<AlignedBlockLayout> vectors_;
    uint64_t dimensions_{0};
};
}  // namespace vsag
