// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <iosfwd>
#include <memory>
#include <vector>

#include "vsag/errors.h"

namespace vsag::lite {

/** One owned search result, ordered by squared L2 then ascending external ID. */
struct Neighbor {
    int64_t id;
    float distance;
};

/**
 * Minimal FP32, squared-L2 index. No concurrent calls are supported.
 * Input vectors are borrowed for the duration of a call; stored data is owned.
 * This API and its versioned snapshot are separate from the Full Index ABI/format.
 */
class Index {
public:
    /** Create an empty fixed-dimension index. Zero dimensions are rejected. */
    static tl::expected<std::unique_ptr<Index>, Error>
    Create(uint64_t dim);

    ~Index();
    Index(const Index&) = delete;
    Index&
    operator=(const Index&) = delete;

    /** Insert one finite vector. Duplicate IDs fail without changing logical contents. */
    tl::expected<void, Error>
    Add(int64_t id, const float* vector, uint64_t dim);
    /** Update an existing ID; a missing ID or invalid vector is an error. */
    tl::expected<void, Error>
    Update(int64_t id, const float* vector, uint64_t dim);
    /** Physically remove a record. Returns false for a missing ID; capacity is retained. */
    bool
    Remove(int64_t id);
    /** k=0 or an empty index returns an empty result; k is capped at the record count. */
    tl::expected<std::vector<Neighbor>, Error>
    Search(const float* query, uint64_t dim, uint64_t k) const;

    /** Write a little-endian snapshot at the current stream position; no atomic file replace. */
    tl::expected<void, Error>
    Save(std::ostream& output) const;
    /** Load a new index from a seekable stream's remaining bytes; rejects trailing bytes. */
    static tl::expected<std::unique_ptr<Index>, Error>
    Load(std::istream& input);

    uint64_t
    Size() const;
    uint64_t
    Dim() const;

private:
    explicit Index(uint64_t dim);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vsag::lite
