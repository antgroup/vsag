// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>

#include "vsag/lite/index.h"

namespace vsag::lite {

/**
 * Opt-in Lite index with a checkpoint and an append-only log of successful additions.
 * A handle exclusively owns its local directory. Calls require external serialization.
 * The checkpoint may contain a prebuilt graph or an empty BruteForce index.
 */
class PersistentIndex {
public:
    /** Create a new directory from an existing index. Takes ownership of checkpoint. */
    static tl::expected<std::unique_ptr<PersistentIndex>, Error>
    Create(const std::string& directory, std::unique_ptr<Index> checkpoint);

    /** Open the checkpoint and replay complete additions in order. */
    static tl::expected<std::unique_ptr<PersistentIndex>, Error>
    Open(const std::string& directory);

    ~PersistentIndex();
    PersistentIndex(const PersistentIndex&) = delete;
    PersistentIndex&
    operator=(const PersistentIndex&) = delete;

    /**
     * Return success after syncing the addition to the log. An I/O failure
     * invalidates this handle. Reopen the directory before the next operation.
     */
    tl::expected<void, Error>
    Add(int64_t id, const float* vector, uint64_t dim);

    tl::expected<uint64_t, Error>
    Size() const;

    tl::expected<std::vector<Neighbor>, Error>
    Search(const float* query,
           uint64_t dim,
           uint64_t k,
           const SearchOptions& options = {},
           const IdFilter& filter = {}) const;

    /** Export the recovered state as an ordinary Lite snapshot. */
    tl::expected<void, Error>
    Save(std::ostream& output) const;

private:
    struct Impl;
    explicit PersistentIndex(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace vsag::lite
