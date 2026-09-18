// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "vsag/lite/index.h"

namespace vsag::lite::detail {

class Backend {
public:
    virtual ~Backend() = default;

    virtual tl::expected<void, Error>
    Add(int64_t id, const float* vector, uint64_t dim) = 0;
    virtual tl::expected<void, Error>
    Update(int64_t id, const float* vector, uint64_t dim) = 0;
    virtual bool
    Remove(int64_t id) = 0;
    virtual tl::expected<std::vector<Neighbor>, Error>
    Search(const float* query, uint64_t dim, uint64_t k) const = 0;
    virtual tl::expected<std::vector<Neighbor>, Error>
    Search(const float* query, uint64_t dim, uint64_t k, const IdFilter& filter) const = 0;

    [[nodiscard]] virtual uint64_t
    Size() const = 0;
    [[nodiscard]] virtual uint64_t
    Dim() const = 0;
    [[nodiscard]] virtual int64_t
    IdAt(uint64_t slot) const = 0;
    [[nodiscard]] virtual const float*
    VectorAt(uint64_t slot) const = 0;

    [[nodiscard]] virtual BackendKind
    Kind() const = 0;
    [[nodiscard]] virtual uint64_t
    MaxDegree() const {
        return 0;
    }
    [[nodiscard]] virtual uint64_t
    EfSearch() const {
        return 0;
    }
    [[nodiscard]] virtual uint64_t
    LinkCountAt([[maybe_unused]] uint64_t slot) const {
        return 0;
    }
    [[nodiscard]] virtual uint64_t
    LinkAt([[maybe_unused]] uint64_t slot, [[maybe_unused]] uint64_t edge) const {
        return 0;
    }
};

tl::expected<std::unique_ptr<Backend>, Error>
make_brute_force_backend(uint64_t dim);

tl::expected<std::unique_ptr<Backend>, Error>
restore_brute_force_backend(uint64_t dim, std::vector<int64_t> ids, std::vector<float> vectors);

tl::expected<std::unique_ptr<Backend>, Error>
make_graph_backend(const Backend& source, uint64_t max_degree, uint64_t ef_search);

tl::expected<std::unique_ptr<Backend>, Error>
restore_graph_backend(uint64_t dim,
                      uint64_t max_degree,
                      uint64_t ef_search,
                      std::vector<int64_t> ids,
                      std::vector<float> vectors,
                      std::vector<std::vector<uint64_t>> links);

}  // namespace vsag::lite::detail
