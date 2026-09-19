// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <queue>
#include <stdexcept>
#include <unordered_map>

#include "lite/backend.h"
#include "lite/fp32_distance.h"

namespace vsag::lite::detail {
namespace {

auto
failure(ErrorType type, const char* message) {
    return tl::unexpected(Error(type, message));
}

tl::expected<void, Error>
validate(const float* data, uint64_t actual, uint64_t expected) {
    if (actual != expected) {
        return failure(ErrorType::DIMENSION_NOT_EQUAL, "dimension mismatch");
    }
    if (data == nullptr) {
        return failure(ErrorType::INVALID_ARGUMENT, "null vector");
    }
    for (uint64_t i = 0; i < actual; ++i) {
        if (not std::isfinite(data[i])) {
            return failure(ErrorType::INVALID_ARGUMENT, "non-finite vector");
        }
    }
    return {};
}

bool
better(const Neighbor& a, const Neighbor& b) {
    return a.distance < b.distance or (a.distance == b.distance and a.id < b.id);
}

struct NeighborWorseFirst {
    bool
    operator()(const Neighbor& a, const Neighbor& b) const {
        return better(a, b);
    }
};

class BruteForceBackend final : public Backend {
public:
    explicit BruteForceBackend(uint64_t dimension) : dim_(dimension) {
    }

    tl::expected<void, Error>
    Add(int64_t id, const float* vector, uint64_t dim) override {
        auto valid = validate(vector, dim, Dim());
        if (not valid) {
            return tl::unexpected(valid.error());
        }
        if (slots_.count(id) != 0) {
            return failure(ErrorType::INVALID_ARGUMENT, "duplicate ID");
        }
        if (Size() >= vectors_.max_size() / dim or Size() >= ids_.max_size()) {
            return failure(ErrorType::NO_ENOUGH_MEMORY, "index capacity exceeded");
        }
        try {
            auto grow = [](auto& values, uint64_t required) {
                if (required > values.capacity()) {
                    const uint64_t maximum = values.max_size();
                    const uint64_t capacity = values.capacity();
                    const uint64_t doubled =
                        capacity == 0 ? 1 : (capacity > maximum / 2 ? maximum : capacity * 2);
                    values.reserve(std::max(required, doubled));
                }
            };
            grow(vectors_, (Size() + 1) * dim);
            grow(ids_, Size() + 1);
            slots_.emplace(id, Size());
            vectors_.insert(vectors_.end(), vector, vector + dim);
            ids_.push_back(id);
            return {};
        } catch (const std::bad_alloc&) {
            return failure(ErrorType::NO_ENOUGH_MEMORY, "add allocation failed");
        } catch (const std::length_error&) {
            return failure(ErrorType::NO_ENOUGH_MEMORY, "add capacity exceeded");
        }
    }

    tl::expected<void, Error>
    Update(int64_t id, const float* vector, uint64_t dim) override {
        auto valid = validate(vector, dim, Dim());
        if (not valid) {
            return tl::unexpected(valid.error());
        }
        auto slot = slots_.find(id);
        if (slot == slots_.end()) {
            return failure(ErrorType::INVALID_ARGUMENT, "missing ID");
        }
        std::copy_n(vector, dim, vectors_.data() + slot->second * dim);
        return {};
    }

    bool
    Remove(int64_t id) override {
        auto found = slots_.find(id);
        if (found == slots_.end()) {
            return false;
        }
        const uint64_t slot = found->second;
        const uint64_t last = Size() - 1;
        if (slot != last) {
            std::copy_n(vectors_.data() + last * Dim(), Dim(), vectors_.data() + slot * Dim());
            ids_[slot] = ids_[last];
            slots_.at(ids_[slot]) = slot;
        }
        slots_.erase(found);
        ids_.pop_back();
        vectors_.resize(last * Dim());
        return true;
    }

    tl::expected<std::vector<Neighbor>, Error>
    Search(const float* query, uint64_t dim, uint64_t k) const override {
        return SearchImpl(query, dim, k, nullptr);
    }

    tl::expected<std::vector<Neighbor>, Error>
    Search(const float* query, uint64_t dim, uint64_t k, const IdFilter& filter) const override {
        return SearchImpl(query, dim, k, filter ? &filter : nullptr);
    }

    tl::expected<std::vector<Neighbor>, Error>
    SearchImpl(const float* query, uint64_t dim, uint64_t k, const IdFilter* filter) const {
        auto valid = validate(query, dim, Dim());
        if (not valid) {
            return tl::unexpected(valid.error());
        }
        try {
            k = std::min(k, Size());
            if (k == 0) {
                return std::vector<Neighbor>{};
            }
            std::priority_queue<Neighbor, std::vector<Neighbor>, NeighborWorseFirst> heap;
            const auto distance_fn = dim < 16 ? generic_fp32_distance : select_fp32_distance();
            for (uint64_t slot = 0; slot < Size(); ++slot) {
                if (filter != nullptr and not(*filter)(IdAt(slot))) {
                    continue;
                }
                const auto distance = distance_fn(query, VectorAt(slot), dim);
                Neighbor next{IdAt(slot), distance};
                if (heap.size() < k) {
                    heap.push(next);
                } else if (better(next, heap.top())) {
                    heap.pop();
                    heap.push(next);
                }
            }
            std::vector<Neighbor> result(heap.size());
            for (uint64_t i = result.size(); i > 0; --i) {
                result[i - 1] = heap.top();
                heap.pop();
            }
            return result;
        } catch (const std::bad_alloc&) {
            return failure(ErrorType::NO_ENOUGH_MEMORY, "search allocation failed");
        } catch (const std::length_error&) {
            return failure(ErrorType::NO_ENOUGH_MEMORY, "search capacity exceeded");
        }
    }

    bool
    Restore(std::vector<int64_t> ids, std::vector<float> vectors) {
        slots_.reserve(ids.size());
        for (uint64_t slot = 0; slot < ids.size(); ++slot) {
            if (not slots_.emplace(ids[slot], slot).second) {
                return false;
            }
        }
        ids_ = std::move(ids);
        vectors_ = std::move(vectors);
        return true;
    }

    [[nodiscard]] BackendKind
    Kind() const override {
        return BackendKind::BRUTE_FORCE;
    }

    [[nodiscard]] uint64_t
    Size() const override {
        return ids_.size();
    }

    [[nodiscard]] uint64_t
    Dim() const override {
        return dim_;
    }

    [[nodiscard]] int64_t
    IdAt(uint64_t slot) const override {
        return ids_[slot];
    }

    [[nodiscard]] const float*
    VectorAt(uint64_t slot) const override {
        return vectors_.data() + slot * Dim();
    }

private:
    uint64_t dim_;
    std::vector<float> vectors_;
    std::vector<int64_t> ids_;
    std::unordered_map<int64_t, uint64_t> slots_;
};

}  // namespace

tl::expected<std::unique_ptr<Backend>, Error>
make_brute_force_backend(uint64_t dim) {
    if (dim == 0 or dim > std::vector<float>().max_size()) {
        return failure(ErrorType::INVALID_ARGUMENT, "invalid dimension");
    }
    try {
        return std::unique_ptr<Backend>(new BruteForceBackend(dim));
    } catch (const std::bad_alloc&) {
        return failure(ErrorType::NO_ENOUGH_MEMORY, "create allocation failed");
    }
}

tl::expected<std::unique_ptr<Backend>, Error>
restore_brute_force_backend(uint64_t dim, std::vector<int64_t> ids, std::vector<float> vectors) {
    if (dim == 0 or dim > vectors.max_size() or ids.size() > vectors.max_size() / dim or
        vectors.size() != ids.size() * dim) {
        return failure(ErrorType::INVALID_BINARY, "invalid snapshot layout");
    }
    try {
        auto backend = std::make_unique<BruteForceBackend>(dim);
        if (not backend->Restore(std::move(ids), std::move(vectors))) {
            return failure(ErrorType::INVALID_BINARY, "duplicate snapshot ID");
        }
        return std::unique_ptr<Backend>(std::move(backend));
    } catch (const std::bad_alloc&) {
        return failure(ErrorType::NO_ENOUGH_MEMORY, "load allocation failed");
    } catch (const std::length_error&) {
        return failure(ErrorType::INVALID_BINARY, "snapshot exceeds container capacity");
    }
}

}  // namespace vsag::lite::detail
