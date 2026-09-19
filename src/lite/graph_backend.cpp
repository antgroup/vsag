// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
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

struct Candidate {
    uint64_t slot;
    float distance;
};

bool
closer(const Candidate& left, const Candidate& right) {
    return left.distance < right.distance or
           (left.distance == right.distance and left.slot < right.slot);
}

bool
farther(const Candidate& left, const Candidate& right) {
    return left.distance > right.distance or
           (left.distance == right.distance and left.slot > right.slot);
}

class GraphBackend final : public Backend {
public:
    GraphBackend(uint64_t dimension, uint64_t max_degree, uint64_t ef_search)
        : dim_(dimension),
          max_degree_(max_degree),
          ef_search_(ef_search),
          distance_(select_fp32_distance()) {
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
        if (Size() >= vectors_.max_size() / dim or Size() >= ids_.max_size() or
            Size() >= extras_.max_size()) {
            return failure(ErrorType::NO_ENOUGH_MEMORY, "index capacity exceeded");
        }
        try {
            auto neighbors = nearest(vector, max_degree_);
            if (not neighbors) {
                return tl::unexpected(neighbors.error());
            }
            grow(vectors_, (Size() + 1) * dim);
            grow(ids_, Size() + 1);
            grow(extras_, Size() + 1);
            for (uint64_t neighbor : *neighbors) {
                extras_[neighbor].reserve(max_degree_ + 1);
            }
            slots_.emplace(id, Size());
            vectors_.insert(vectors_.end(), vector, vector + dim);
            ids_.push_back(id);
            extras_.push_back(std::move(*neighbors));
            for (uint64_t neighbor : extras_.back()) {
                link(neighbor, Size() - 1);
            }
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
        auto found = slots_.find(id);
        if (found == slots_.end()) {
            return failure(ErrorType::INVALID_ARGUMENT, "missing ID");
        }
        try {
            const uint64_t slot = found->second;
            auto neighbors = nearest(vector, max_degree_, slot);
            if (not neighbors) {
                return tl::unexpected(neighbors.error());
            }
            for (uint64_t neighbor : *neighbors) {
                extras_[neighbor].reserve(max_degree_ + 1);
            }
            for (uint64_t old_neighbor : extras_[slot]) {
                auto& reverse = extras_[old_neighbor];
                reverse.erase(std::remove(reverse.begin(), reverse.end(), slot), reverse.end());
            }
            std::copy_n(vector, dim, vectors_.data() + slot * dim);
            extras_[slot] = std::move(*neighbors);
            for (uint64_t neighbor : extras_[slot]) {
                link(neighbor, slot);
            }
            return {};
        } catch (const std::bad_alloc&) {
            return failure(ErrorType::NO_ENOUGH_MEMORY, "update allocation failed");
        } catch (const std::length_error&) {
            return failure(ErrorType::NO_ENOUGH_MEMORY, "update capacity exceeded");
        }
    }

    bool
    Remove(int64_t id) override {
        auto found = slots_.find(id);
        if (found == slots_.end()) {
            return false;
        }
        const uint64_t slot = found->second;
        const uint64_t last = Size() - 1;
        for (auto& neighbors : extras_) {
            neighbors.erase(std::remove(neighbors.begin(), neighbors.end(), slot), neighbors.end());
            for (auto& neighbor : neighbors) {
                if (neighbor == last) {
                    neighbor = slot;
                }
            }
        }
        if (slot != last) {
            std::copy_n(vectors_.data() + last * Dim(), Dim(), vectors_.data() + slot * Dim());
            ids_[slot] = ids_[last];
            extras_[slot] = std::move(extras_[last]);
            slots_.at(ids_[slot]) = slot;
        }
        slots_.erase(found);
        ids_.pop_back();
        vectors_.resize(last * Dim());
        extras_.pop_back();
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
            const uint64_t ef = std::min(Size(), std::max(k, ef_search_));
            // With closer as Compare, top() is the farthest candidate and pop() evicts it.
            std::priority_queue<Candidate, std::vector<Candidate>, decltype(&closer)> best(&closer);
            std::priority_queue<Candidate, std::vector<Candidate>, decltype(&closer)> accepted(
                &closer);
            std::priority_queue<Candidate, std::vector<Candidate>, decltype(&farther)> candidates(
                &farther);
            std::vector<uint8_t> visited(Size(), 0);

            auto visit = [&](uint64_t slot) {
                if (visited[slot] != 0) {
                    return;
                }
                visited[slot] = 1;
                Candidate next{slot, distance(query, VectorAt(slot))};
                candidates.push(next);
                best.push(next);
                if (best.size() > ef) {
                    best.pop();
                }
                if (filter != nullptr and (*filter)(IdAt(slot))) {
                    accepted.push(next);
                    if (accepted.size() > k) {
                        accepted.pop();
                    }
                }
            };
            visit(0);
            visit(Size() - 1);
            constexpr uint64_t k_extra_entry_points = 6;
            for (uint64_t i = 1; i <= k_extra_entry_points; ++i) {
                visit(i * (Size() - 1) / (k_extra_entry_points + 1));
            }

            while (not candidates.empty()) {
                const Candidate current = candidates.top();
                candidates.pop();
                if (best.size() == ef and closer(best.top(), current)) {
                    break;
                }
                if (Size() > 1) {
                    visit((current.slot + Size() - 1) % Size());
                    visit((current.slot + 1) % Size());
                }
                for (const auto neighbor : extras_[current.slot]) {
                    visit(neighbor);
                }
            }

            auto& output = filter == nullptr ? best : accepted;
            std::vector<Neighbor> result;
            result.reserve(output.size());
            while (not output.empty()) {
                const auto candidate = output.top();
                output.pop();
                result.push_back({IdAt(candidate.slot), candidate.distance});
            }
            std::sort(
                result.begin(), result.end(), [](const Neighbor& left, const Neighbor& right) {
                    return left.distance < right.distance or
                           (left.distance == right.distance and left.id < right.id);
                });
            result.resize(std::min(k, static_cast<uint64_t>(result.size())));
            return result;
        } catch (const std::bad_alloc&) {
            return failure(ErrorType::NO_ENOUGH_MEMORY, "search allocation failed");
        } catch (const std::length_error&) {
            return failure(ErrorType::NO_ENOUGH_MEMORY, "search capacity exceeded");
        }
    }

    bool
    Restore(std::vector<int64_t> ids,
            std::vector<float> vectors,
            std::vector<std::vector<uint64_t>> links) {
        slots_.reserve(ids.size());
        for (uint64_t slot = 0; slot < ids.size(); ++slot) {
            if (not slots_.emplace(ids[slot], slot).second) {
                return false;
            }
        }
        ids_ = std::move(ids);
        vectors_ = std::move(vectors);
        extras_ = std::move(links);
        return true;
    }

    [[nodiscard]] BackendKind
    Kind() const override {
        return BackendKind::GRAPH;
    }

    [[nodiscard]] uint64_t
    MaxDegree() const override {
        return max_degree_;
    }

    [[nodiscard]] uint64_t
    EfSearch() const override {
        return ef_search_;
    }

    [[nodiscard]] uint64_t
    LinkCountAt(uint64_t slot) const override {
        return extras_[slot].size();
    }

    [[nodiscard]] uint64_t
    LinkAt(uint64_t slot, uint64_t offset) const override {
        return extras_[slot][offset];
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
    template <typename T>
    static void
    grow(std::vector<T>& values, uint64_t required) {
        if (required > values.capacity()) {
            const uint64_t maximum = values.max_size();
            const uint64_t capacity = values.capacity();
            const uint64_t doubled =
                capacity == 0 ? 1 : (capacity > maximum / 2 ? maximum : capacity * 2);
            values.reserve(std::max(required, doubled));
        }
    }

    [[nodiscard]] float
    distance(const float* left, const float* right) const {
        return distance_(left, right, Dim());
    }

    tl::expected<std::vector<uint64_t>, Error>
    nearest(const float* vector,
            uint64_t count,
            uint64_t excluded = std::numeric_limits<uint64_t>::max()) const {
        if (Size() == 0) {
            return std::vector<uint64_t>{};
        }
        const auto found = Search(vector, Dim(), std::min(Size(), std::max(count * 4, ef_search_)));
        if (not found) {
            return tl::unexpected(found.error());
        }
        std::vector<uint64_t> result;
        result.reserve(count);
        for (const auto& neighbor : *found) {
            const uint64_t slot = slots_.at(neighbor.id);
            if (slot != excluded) {
                result.push_back(slot);
                if (result.size() == count) {
                    break;
                }
            }
        }
        return result;
    }

    void
    link(uint64_t source, uint64_t target) {
        auto& neighbors = extras_[source];
        if (std::find(neighbors.begin(), neighbors.end(), target) != neighbors.end()) {
            return;
        }
        neighbors.push_back(target);
        if (neighbors.size() > max_degree_) {
            // max_degree_ is at most 64; cache each distance once before sorting.
            std::array<Candidate, 65> ranked{};
            for (uint64_t i = 0; i < neighbors.size(); ++i) {
                ranked[i] = {neighbors[i], distance(VectorAt(source), VectorAt(neighbors[i]))};
            }
            std::sort(ranked.begin(),
                      ranked.begin() + static_cast<std::ptrdiff_t>(neighbors.size()),
                      closer);
            for (uint64_t i = 0; i < max_degree_; ++i) {
                neighbors[i] = ranked[i].slot;
            }
            neighbors.resize(max_degree_);
        }
    }

    uint64_t dim_;
    uint64_t max_degree_;
    uint64_t ef_search_;
    FP32Distance distance_;
    std::vector<float> vectors_;
    std::vector<int64_t> ids_;
    std::unordered_map<int64_t, uint64_t> slots_;
    std::vector<std::vector<uint64_t>> extras_;
};

}  // namespace

tl::expected<std::unique_ptr<Backend>, Error>
make_graph_backend(const Backend& source, uint64_t max_degree, uint64_t ef_search) {
    if (source.Kind() != BackendKind::BRUTE_FORCE or source.Dim() == 0 or max_degree < 2 or
        max_degree > 64 or ef_search < max_degree) {
        return failure(ErrorType::INVALID_ARGUMENT, "invalid graph options");
    }
    try {
        auto graph = std::make_unique<GraphBackend>(source.Dim(), max_degree, ef_search);
        for (uint64_t slot = 0; slot < source.Size(); ++slot) {
            auto added = graph->Add(source.IdAt(slot), source.VectorAt(slot), source.Dim());
            if (not added) {
                return tl::unexpected(added.error());
            }
        }
        return std::unique_ptr<Backend>(std::move(graph));
    } catch (const std::bad_alloc&) {
        return failure(ErrorType::NO_ENOUGH_MEMORY, "graph allocation failed");
    } catch (const std::length_error&) {
        return failure(ErrorType::NO_ENOUGH_MEMORY, "graph capacity exceeded");
    }
}

tl::expected<std::unique_ptr<Backend>, Error>
restore_graph_backend(uint64_t dim,
                      uint64_t max_degree,
                      uint64_t ef_search,
                      std::vector<int64_t> ids,
                      std::vector<float> vectors,
                      std::vector<std::vector<uint64_t>> links) {
    if (dim == 0 or dim > vectors.max_size() or max_degree < 2 or max_degree > 64 or
        ef_search < max_degree or ids.size() > vectors.max_size() / dim or
        vectors.size() != ids.size() * dim or links.size() != ids.size()) {
        return failure(ErrorType::INVALID_BINARY, "invalid graph snapshot layout");
    }
    for (uint64_t slot = 0; slot < links.size(); ++slot) {
        if (links[slot].size() > max_degree) {
            return failure(ErrorType::INVALID_BINARY, "graph degree exceeds limit");
        }
        for (uint64_t i = 0; i < links[slot].size(); ++i) {
            const auto neighbor = links[slot][i];
            if (neighbor >= ids.size() or neighbor == slot or
                std::find(links[slot].begin(),
                          links[slot].begin() + static_cast<std::ptrdiff_t>(i),
                          neighbor) != links[slot].begin() + static_cast<std::ptrdiff_t>(i)) {
                return failure(ErrorType::INVALID_BINARY, "invalid graph link");
            }
        }
    }
    try {
        auto graph = std::make_unique<GraphBackend>(dim, max_degree, ef_search);
        if (not graph->Restore(std::move(ids), std::move(vectors), std::move(links))) {
            return failure(ErrorType::INVALID_BINARY, "duplicate graph ID");
        }
        return std::unique_ptr<Backend>(std::move(graph));
    } catch (const std::bad_alloc&) {
        return failure(ErrorType::NO_ENOUGH_MEMORY, "graph load allocation failed");
    } catch (const std::length_error&) {
        return failure(ErrorType::INVALID_BINARY, "graph snapshot exceeds capacity");
    }
}

}  // namespace vsag::lite::detail
