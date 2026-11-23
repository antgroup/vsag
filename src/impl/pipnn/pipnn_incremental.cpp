// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include "pipnn_incremental.h"

#include <cstring>

#include "common.h"
#include "layout/aligned_block_layout.h"

namespace vsag {
namespace {
void
write_route(StreamWriter& w, const PiPNNRoute& node) {
    StreamWriter::WriteObj(w, node.fanout);
    StreamWriter::WriteObj(w, node.leaf);
    StreamWriter::WriteVector(w, node.leaders);
    for (auto& child : node.children) write_route(w, child);
}
template <typename T>
void
read_array(StreamReader& r, std::vector<T>& dst, uint64_t limit) {
    uint64_t count;
    StreamReader::ReadObj(r, count);
    CHECK_ARGUMENT(count <= limit && count <= dst.max_size(), "bad PiPNN state array");
    dst.resize(count);
    if (count)
        r.Read(reinterpret_cast<char*>(dst.data()), count * sizeof(T));
}
void
read_route(StreamReader& r,
           PiPNNRoute& node,
           uint64_t count,
           uint64_t leaves,
           unsigned depth,
           uint64_t first_leaf = 0,
           bool allow_empty = false) {
    CHECK_ARGUMENT(depth < 64, "bad PiPNN routing depth");
    StreamReader::ReadObj(r, node.fanout);
    StreamReader::ReadObj(r, node.leaf);
    read_array(r, node.leaders, count);
    if (!node.leaders.empty()) {
        CHECK_ARGUMENT(node.fanout && node.fanout <= node.leaders.size(), "bad PiPNN fanout");
        for (auto id : node.leaders) CHECK_ARGUMENT(id < count, "bad PiPNN leader");
    } else {
        CHECK_ARGUMENT((node.leaf >= first_leaf && node.leaf < leaves) ||
                           (allow_empty && node.leaf == UINT32_MAX),
                       "bad PiPNN leaf");
    }
    node.children.resize(node.leaders.size());
    for (auto& child : node.children)
        read_route(r, child, count, leaves, depth + 1, first_leaf, false);
}
}  // namespace

void
PiPNNFloatRows::Resize(uint64_t rows, uint64_t width, Allocator* allocator) {
    CHECK_ARGUMENT(
        rows <= UINT32_MAX && width && width <= UINT32_MAX && (blocks_.empty() || width == width_),
        "bad PiPNN sketch shape");
    if (blocks_.empty()) {
        // Small Pyramid nodes start with small blocks. Larger indexes use 256-row blocks.
        shift_ = std::bit_width(std::clamp<uint64_t>(rows, 8, 256)) - 1;
        mask_ = (1U << shift_) - 1;
    }
    width_ = width;
    const uint64_t block_rows = mask_ + 1;
    const auto blocks = (rows + mask_) >> shift_;
    while (blocks_.size() < blocks) {
        blocks_.emplace_back(block_rows * width, allocator);
    }
    rows_ = rows;
}

uint64_t
PiPNNFloatRows::MemoryUsage() const {
    return blocks_.capacity() * sizeof(blocks_.front()) +
           blocks_.size() * (mask_ + 1) * width_ * sizeof(float);
}

void
PiPNNFloatRows::Write(StreamWriter& w) const {
    StreamWriter::WriteObj(w, rows_ * width_);
    for (uint64_t first = 0; first < rows_; first += mask_ + 1) {
        const auto n = std::min<uint64_t>(mask_ + 1, rows_ - first);
        w.Write(reinterpret_cast<const char*>(Row(first)), n * width_ * sizeof(float));
    }
}

void
PiPNNFloatRows::Read(
    StreamReader& r, uint64_t rows, uint64_t width, bool optional, Allocator* allocator) {
    uint64_t values;
    StreamReader::ReadObj(r, values);
    CHECK_ARGUMENT(width && rows <= UINT32_MAX && width <= UINT32_MAX &&
                       (values == rows * width || (optional && values == 0)),
                   "bad PiPNN sketch shape");
    Resize(values ? rows : 0, width, allocator);
    for (uint64_t first = 0; first < rows_; first += mask_ + 1) {
        const auto n = std::min<uint64_t>(mask_ + 1, rows_ - first);
        r.Read(reinterpret_cast<char*>(Row(first)), n * width_ * sizeof(float));
    }
}

PiPNNIncrementalState::PiPNNIncrementalState() = default;
PiPNNIncrementalState::~PiPNNIncrementalState() = default;

void
PiPNNIncrementalState::ReserveVectors(uint64_t count, uint64_t dim, Allocator* allocator) {
    CHECK_ARGUMENT(count <= UINT32_MAX && dim > 0 && dim <= UINT64_MAX / sizeof(float) &&
                       count <= UINT64_MAX / (dim * sizeof(float)),
                   "bad PiPNN vector shape");
    if (!vectors_) {
        // Keep small Pyramid nodes compact. A row always fits in one physical block.
        const auto limit =
            std::min<uint64_t>({2 * 1024 * 1024,
                                Options::Instance().block_size_limit(),
                                std::max<uint64_t>(4096, count * dim * sizeof(float))});
        vectors_ = std::make_unique<AlignedBlockLayout>(
            dim * sizeof(float), allocator, limit, alignof(float));
        dimensions_ = dim;
    }
    CHECK_ARGUMENT(dim == dimensions_, "PiPNN vector dimension changed");
    vectors_->Reserve(count);
    const auto first = row_pointers_.size();
    if (count > first) {
        row_pointers_.resize(count);
        for (uint64_t row = first; row < count; ++row)
            row_pointers_[row] = reinterpret_cast<const float*>(vectors_->GetRecord32(row));
    }
}

const float*
PiPNNIncrementalState::GetVector(uint32_t row) const {
    return row_pointers_[row];
}

void
PiPNNIncrementalState::SetVector(uint32_t row, const float* vector) {
    std::memcpy(vectors_->GetMutableRecord(row), vector, dimensions_ * sizeof(float));
}

void
PiPNNIncrementalState::AppendIds(std::span<const uint32_t> incoming) {
    const auto first = ids.size();
    ids.insert(ids.end(), incoming.begin(), incoming.end());
    auto capacity = required_capacity_;
    try {
        for (uint64_t i = first; i < ids.size(); ++i) {
            if (ids[i] != i)
                sparse_ids_.emplace(ids[i], static_cast<uint32_t>(i));
            capacity = std::max(capacity, uint64_t{ids[i]} + 1);
        }
    } catch (...) {
        for (auto id : incoming) sparse_ids_.erase(id);
        ids.resize(first);
        throw;
    }
    required_capacity_ = capacity;
}

uint64_t
PiPNNIncrementalState::MemoryUsage() const {
    const auto bytes = [](const auto& v) { return v.capacity() * sizeof(v.front()); };
    const auto tree = [&](const auto& self, const PiPNNRoute& r) -> uint64_t {
        uint64_t result = bytes(r.children) + bytes(r.leaders);
        for (const auto& child : r.children) result += self(self, child);
        return result;
    };
    uint64_t result =
        sizeof(*this) + bytes(ids) + (vectors_ ? vectors_->GetMemoryUsage() : 0) + bytes(planes) +
        projections.MemoryUsage() + norms.MemoryUsage() + bytes(row_pointers_) +
        sparse_ids_.bucket_count() * sizeof(void*) +
        sparse_ids_.size() * (sizeof(decltype(sparse_ids_)::value_type) + sizeof(void*)) +
        bytes(memberships) + bytes(leaves) + tree(tree, root);
    for (const auto& v : memberships) result += bytes(v);
    for (const auto& leaf : leaves)
        result += bytes(leaf.points) + bytes(leaf.nearest) + tree(tree, leaf.replacement);
    return result;
}
void
PiPNNIncrementalState::Serialize(StreamWriter& w) const {
    StreamWriter::WriteVector(w, ids);
    // Version 1 stores the flat FP32 element count followed by packed rows.
    StreamWriter::WriteObj(w, static_cast<uint64_t>(ids.size()) * dimensions_);
    if (!ids.empty())
        vectors_->WriteRecords(w, ids.size());
    StreamWriter::WriteVector(w, planes);
    projections.Write(w);
    norms.Write(w);
    StreamWriter::WriteObj(w, static_cast<uint64_t>(leaves.size()));
    write_route(w, root);
    for (auto& leaf : leaves) {
        StreamWriter::WriteVector(w, leaf.points);
        StreamWriter::WriteVector(w, leaf.nearest);
        write_route(w, leaf.replacement);
    }
}
void
PiPNNIncrementalState::Deserialize(StreamReader& r,
                                   uint64_t count,
                                   uint64_t dim,
                                   uint64_t h,
                                   uint64_t graph_capacity,
                                   Allocator* allocator) {
    std::vector<uint32_t> loaded_ids;
    read_array(r, loaded_ids, std::min<uint64_t>(count, UINT32_MAX));
    ids.clear();
    sparse_ids_.clear();
    required_capacity_ = 0;
    ids.reserve(loaded_ids.size());
    for (auto id : loaded_ids) {
        CHECK_ARGUMENT(id < graph_capacity && !ContainsId(id),
                       "invalid or duplicate PiPNN state ID");
        if (id != ids.size())
            sparse_ids_.emplace(id, static_cast<uint32_t>(ids.size()));
        ids.push_back(id);
        required_capacity_ = std::max(required_capacity_, uint64_t{id} + 1);
    }
    count = ids.size();
    uint64_t elements;
    StreamReader::ReadObj(r, elements);
    CHECK_ARGUMENT(dim && count <= UINT64_MAX / dim && elements == count * dim,
                   "bad PiPNN vector array");
    ReserveVectors(count, dim, allocator);
    vectors_->ReadRecords(r, count);
    CHECK_ARGUMENT(h && h <= UINT64_MAX / dim, "bad PiPNN plane dimensions");
    read_array(r, planes, h * dim);
    CHECK_ARGUMENT(planes.size() == h * dim, "bad PiPNN plane shape");
    projections.Read(r, count, h, false, allocator);
    norms.Read(r, count, 1, true, allocator);
    uint64_t n;
    StreamReader::ReadObj(r, n);
    CHECK_ARGUMENT(n && n < UINT32_MAX, "bad PiPNN leaf count");
    leaves.resize(n);
    read_route(r, root, count, n, 0);
    memberships.assign(count, {});
    for (auto& leaf : leaves) {
        read_array(r, leaf.points, count);
        const uint64_t m = leaf.points.size();
        read_array(r, leaf.nearest, m * m);
        CHECK_ARGUMENT(!m || leaf.nearest.size() % m == 0, "bad PiPNN top-k shape");
        for (auto p : leaf.points) {
            CHECK_ARGUMENT(p < count, "bad cached point");
            memberships[p].push_back(&leaf - leaves.data());
        }
        for (auto e : leaf.nearest) CHECK_ARGUMENT(e.target < count, "bad cached neighbor");
        read_route(r, leaf.replacement, count, n, 0, 1 + (&leaf - leaves.data()), true);
    }
}
}  // namespace vsag
