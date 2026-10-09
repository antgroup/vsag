// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <functional>
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

/** Return true when an external ID is allowed in search results. */
using IdFilter = std::function<bool(int64_t)>;

/** Active Lite backend. Graph search is approximate; BruteForce is the default. */
enum class BackendKind { BRUTE_FORCE, GRAPH };

/** Vector storage used by the active backend. */
enum class VectorStorage { FP32, FP16 };

/** Per-call graph query budget; zero uses the configured default. */
struct SearchOptions {
    uint64_t ef_search = 0;
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
    /** Explicitly build a graph from the flat contents; failure leaves this index unchanged. */
    tl::expected<void, Error>
    BuildGraph(uint64_t max_degree = 16, uint64_t ef_search = 128);
    /** Explicitly build a graph with the selected vector storage. */
    tl::expected<void, Error>
    BuildGraph(VectorStorage storage, uint64_t max_degree = 16, uint64_t ef_search = 128);

    [[nodiscard]] BackendKind
    ActiveBackend() const;
    [[nodiscard]] VectorStorage
    ActiveVectorStorage() const;

    ~Index();
    Index(const Index&) = delete;
    Index&
    operator=(const Index&) = delete;

    /** Insert one finite vector. Duplicate IDs fail without changing logical contents. */
    tl::expected<void, Error>
    Add(int64_t id, const float* vector, uint64_t dim);
    /** Update an existing ID; a missing ID or invalid vector is an error.
     * Graph updates preserve topology when the validated stored representation is identical.
     */
    tl::expected<void, Error>
    Update(int64_t id, const float* vector, uint64_t dim);
    /** Physically remove a record. Returns false for a missing ID; capacity is retained. */
    bool
    Remove(int64_t id);
    /** k=0 or an empty index returns an empty result; k is capped at the record count. */
    tl::expected<std::vector<Neighbor>, Error>
    Search(const float* query, uint64_t dim, uint64_t k) const;
    /** Return only records whose external ID is accepted; an empty filter accepts all. */
    tl::expected<std::vector<Neighbor>, Error>
    Search(const float* query, uint64_t dim, uint64_t k, const IdFilter& filter) const;

    /**
     * Search with a per-call graph budget. BruteForce ignores the budget.
     * Graph ef is clamped to [min(k, Size()), Size()]. Zero uses the stored default.
     * Does not change construction/CRUD settings or persisted options.
     * The existing externally serialized-call requirement still applies.
     */
    tl::expected<std::vector<Neighbor>, Error>
    SearchWithOptions(const float* query,
                      uint64_t dim,
                      uint64_t k,
                      const SearchOptions& options,
                      const IdFilter& filter = {}) const;

    /** Write a little-endian snapshot at the current stream position; no atomic file replace. */
    tl::expected<void, Error>
    Save(std::ostream& output) const;
    /** Load a new index from a seekable stream's remaining bytes; rejects trailing bytes. */
    static tl::expected<std::unique_ptr<Index>, Error>
    Load(std::istream& input);

    [[nodiscard]] uint64_t
    Size() const;
    [[nodiscard]] uint64_t
    Dim() const;

private:
    struct Impl;
    explicit Index(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace vsag::lite
