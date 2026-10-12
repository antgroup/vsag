// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <new>
#include <queue>
#include <stdexcept>
#include <unordered_map>

#include "lite/backend.h"
#include "lite/backend_utils.h"
#include "lite/fp16_codec.h"
#include "lite/fp16_distance.h"
#include "lite/fp32_distance.h"
#include "lite/query_heap.h"

namespace vsag::lite::detail {
namespace {

constexpr uint64_t K_MAX_DEGREE = 64;
constexpr uint64_t K_MIN_INCOMING_COMPACT_INTERVAL = 1000;

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

// Concrete comparator types let heap operations inline the existing total ordering.
struct Closer {
    bool
    operator()(const Candidate& left, const Candidate& right) const {
        return closer(left, right);
    }
};

struct Farther {
    bool
    operator()(const Candidate& left, const Candidate& right) const {
        return farther(left, right);
    }
};

class GraphBackend final : public Backend {
public:
    GraphBackend(uint64_t dimension,
                 uint64_t max_degree,
                 uint64_t ef_search,
                 bool fp16 = false,
                 bool journal_mutations = true,
                 const Backend* build_source = nullptr)
        : dim_(dimension),
          max_degree_(max_degree),
          ef_search_(ef_search),
          fp16_(fp16),
          distance_(select_fp32_distance()),
          fp16_distance_(select_fp16_distance()),
          journal_mutations_(journal_mutations),
          build_source_(build_source) {
    }

    void
    ReserveBuild(uint64_t count) {
        if (count > ids_.max_size() or count > extras_.max_size() or count > incoming_.max_size() or
            (fp16_ ? count > fp16_vectors_.max_size() / Dim()
                   : count > vectors_.max_size() / Dim())) {
            throw std::length_error("graph build capacity exceeded");
        }
        if (fp16_) {
            fp16_vectors_.reserve(count * Dim());
        } else if (build_source_ == nullptr) {
            vectors_.reserve(count * Dim());
        }
        ids_.reserve(count);
        extras_.reserve(count);
        incoming_.reserve(count);
        slots_.reserve(count);
    }

    void
    EnableMutationJournal() {
        journal_mutations_ = true;
    }

    std::vector<std::vector<uint64_t>>
    TakeBuildLinks() {
        return std::move(extras_);
    }

    // Back up only rows touched by a mutation. Rollback reuses retained capacity
    // and swaps saved rows, so allocation failure cannot leave a partial commit.
    class MutationJournal {
    public:
        MutationJournal(GraphBackend& owner, uint64_t slot, int64_t added_id, bool adding)
            : owner_(owner),
              size_(owner.Size()),
              slot_(slot),
              added_id_(added_id),
              adding_(adding) {
            // Private construction is discarded on failure, then published once.
            if (not owner_.journal_mutations_) {
                committed_ = true;
                return;
            }
            if (slot_ < size_) {
                old_id_ = owner.ids_[slot_];
                last_id_ = owner.ids_.back();
                if (owner.fp16_) {
                    fp16_slot_.assign(owner.fp16_vectors_.data() + slot_ * owner.Dim(),
                                      owner.fp16_vectors_.data() + (slot_ + 1) * owner.Dim());
                    fp16_last_.assign(owner.fp16_vectors_.data() + (size_ - 1) * owner.Dim(),
                                      owner.fp16_vectors_.data() + size_ * owner.Dim());
                } else {
                    fp32_slot_.assign(owner.vectors_.data() + slot_ * owner.Dim(),
                                      owner.vectors_.data() + (slot_ + 1) * owner.Dim());
                    fp32_last_.assign(owner.vectors_.data() + (size_ - 1) * owner.Dim(),
                                      owner.vectors_.data() + size_ * owner.Dim());
                }
                Remember(slot_);
                Remember(size_ - 1);
            }
            owner_.active_journal_ = this;
        }

        ~MutationJournal() {
            owner_.active_journal_ = nullptr;
            if (committed_) {
                return;
            }
            if (adding_) {
                owner_.slots_.erase(added_id_);
            }
            owner_.ids_.resize(size_);
            owner_.extras_.resize(size_);
            owner_.incoming_.resize(size_);
            if (owner_.fp16_) {
                owner_.fp16_vectors_.resize(size_ * owner_.Dim());
            } else {
                owner_.vectors_.resize(size_ * owner_.Dim());
            }
            if (slot_ < size_) {
                owner_.ids_[slot_] = old_id_;
                owner_.ids_[size_ - 1] = last_id_;
                owner_.slots_.at(last_id_) = size_ - 1;
                if (owner_.fp16_) {
                    std::copy(fp16_slot_.begin(),
                              fp16_slot_.end(),
                              owner_.fp16_vectors_.data() + slot_ * owner_.Dim());
                    std::copy(fp16_last_.begin(),
                              fp16_last_.end(),
                              owner_.fp16_vectors_.data() + (size_ - 1) * owner_.Dim());
                } else {
                    std::copy(fp32_slot_.begin(),
                              fp32_slot_.end(),
                              owner_.vectors_.data() + slot_ * owner_.Dim());
                    std::copy(fp32_last_.begin(),
                              fp32_last_.end(),
                              owner_.vectors_.data() + (size_ - 1) * owner_.Dim());
                }
            }
            for (auto& entry : rows_) {
                if (entry.second.outgoing_saved) {
                    owner_.extras_[entry.first].swap(entry.second.outgoing);
                }
                if (entry.second.incoming_saved) {
                    owner_.incoming_[entry.first].swap(entry.second.incoming);
                }
            }
        }

        void
        Remember(uint64_t slot, bool outgoing = true, bool incoming = true) {
            if (slot >= size_) {
                return;
            }
            auto& saved = rows_.try_emplace(slot).first->second;
            if (outgoing and not saved.outgoing_saved) {
                saved.outgoing = owner_.extras_[slot];
                saved.outgoing_saved = true;
            }
            if (incoming and not saved.incoming_saved) {
                saved.incoming = owner_.incoming_[slot];
                saved.incoming_saved = true;
            }
        }

        void
        Commit() {
            committed_ = true;
            owner_.active_journal_ = nullptr;
        }

        MutationJournal(const MutationJournal&) = delete;
        MutationJournal&
        operator=(const MutationJournal&) = delete;

    private:
        struct Row {
            std::vector<uint64_t> outgoing;
            std::vector<uint64_t> incoming;
            bool outgoing_saved{};
            bool incoming_saved{};
        };
        GraphBackend& owner_;
        uint64_t size_;
        uint64_t slot_;
        int64_t added_id_;
        bool adding_;
        bool committed_{};
        int64_t old_id_{};
        int64_t last_id_{};
        std::vector<float> fp32_slot_;
        std::vector<float> fp32_last_;
        std::vector<uint16_t> fp16_slot_;
        std::vector<uint16_t> fp16_last_;
        std::unordered_map<uint64_t, Row> rows_;
    };

    tl::expected<void, Error>
    Add(int64_t id, const float* vector, uint64_t dim) override {
        auto valid = validate(vector, dim, Dim());
        if (not valid) {
            return tl::unexpected(valid.error());
        }
        if (slots_.count(id) != 0) {
            return failure(ErrorType::INVALID_ARGUMENT, "duplicate ID");
        }
        if (Size() >= ids_.max_size() or Size() >= extras_.max_size() or
            (fp16_ ? Size() >= fp16_vectors_.max_size() / dim
                   : Size() >= vectors_.max_size() / dim)) {
            return failure(ErrorType::NO_ENOUGH_MEMORY, "index capacity exceeded");
        }
        try {
            std::vector<uint16_t> encoded;
            if (fp16_) {
                encoded = encode(vector);
            }
            auto neighbors = nearest(vector, max_degree_);
            if (not neighbors) {
                return tl::unexpected(neighbors.error());
            }
            if (fp16_) {
                grow(fp16_vectors_, (Size() + 1) * dim);
            } else if (build_source_ == nullptr) {
                grow(vectors_, (Size() + 1) * dim);
            }
            grow(ids_, Size() + 1);
            grow(extras_, Size() + 1);
            grow(incoming_, Size() + 1);
            for (uint64_t neighbor : *neighbors) {
                extras_[neighbor].reserve(max_degree_ + 1);
            }
            MutationJournal journal(*this, Size(), id, true);
            slots_.emplace(id, Size());
            if (fp16_) {
                fp16_vectors_.insert(fp16_vectors_.end(), encoded.begin(), encoded.end());
            } else if (build_source_ == nullptr) {
                vectors_.insert(vectors_.end(), vector, vector + dim);
            }
            ids_.push_back(id);
            extras_.push_back(std::move(*neighbors));
            incoming_.emplace_back();
            sync_incoming(Size() - 1, {});
            for (uint64_t neighbor : extras_.back()) {
                link(neighbor, Size() - 1);
            }
            ensure_incoming(Size() - 1);
            journal.Commit();
            return {};
        } catch (const std::invalid_argument&) {
            return failure(ErrorType::INVALID_ARGUMENT, "vector exceeds FP16 range");
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
            std::vector<uint16_t> encoded;
            if (fp16_) {
                encoded = encode(vector);
            }
            const uint64_t slot = found->second;
            // Validation and FP16 encoding must precede the identity check. Preserve
            // the graph when the stored representation is unchanged.
            const bool unchanged =
                fp16_
                    ? std::equal(encoded.begin(), encoded.end(), fp16_vectors_.data() + slot * dim)
                    : std::memcmp(vector, vectors_.data() + slot * dim, dim * sizeof(float)) == 0;
            if (unchanged) {
                return {};
            }
            auto neighbors = nearest(vector, max_degree_, slot);
            if (not neighbors) {
                return tl::unexpected(neighbors.error());
            }
            for (uint64_t neighbor : *neighbors) {
                extras_[neighbor].reserve(max_degree_ + 1);
            }
            MutationJournal journal(*this, slot, 0, false);
            const auto old_neighbors = extras_[slot];
            const auto affected_nodes = incoming_[slot];
            for (const uint64_t source : affected_nodes) {
                remember_row(source, true, false);
                erase_value(extras_[source], slot);
            }
            incoming_[slot].clear();
            if (fp16_) {
                std::copy_n(encoded.data(), dim, fp16_vectors_.data() + slot * dim);
            } else {
                std::copy_n(vector, dim, vectors_.data() + slot * dim);
            }
            replace_neighbors(slot, *neighbors);
            for (uint64_t neighbor : extras_[slot]) {
                link(neighbor, slot);
            }
            ensure_incoming(slot);
            repair(affected_nodes, old_neighbors);
            // Replacing outgoing edges can orphan old targets, even when they did not
            // point back to this slot. Their outgoing lists do not need rebuilding.
            for (const uint64_t target : old_neighbors) {
                ensure_incoming(target);
            }
            journal.Commit();
            return {};
        } catch (const std::invalid_argument&) {
            return failure(ErrorType::INVALID_ARGUMENT, "vector exceeds FP16 range");
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
        try {
            const uint64_t slot = found->second;
            const uint64_t last = Size() - 1;
            MutationJournal journal(*this, slot, 0, false);
            const auto removed_neighbors = extras_[slot];
            const auto removed_incoming = incoming_[slot];
            std::vector<uint64_t> affected_nodes = removed_neighbors;
            affected_nodes.insert(
                affected_nodes.end(), removed_incoming.begin(), removed_incoming.end());
            for (const uint64_t target : removed_neighbors) {
                remember_row(target, false, true);
                erase_value(incoming_[target], slot);
            }
            for (const uint64_t source : removed_incoming) {
                remember_row(source, true, false);
                erase_value(extras_[source], slot);
            }
            incoming_[slot].clear();
            if (slot != last) {
                for (const uint64_t target : extras_[last]) {
                    remember_row(target, false, true);
                    replace_value(incoming_[target], last, slot);
                }
                for (const uint64_t source : incoming_[last]) {
                    remember_row(source, true, false);
                    replace_value(extras_[source], last, slot);
                }
            }
            if (slot != last) {
                if (fp16_) {
                    std::copy_n(fp16_vectors_.data() + last * Dim(),
                                Dim(),
                                fp16_vectors_.data() + slot * Dim());
                } else {
                    std::copy_n(
                        vectors_.data() + last * Dim(), Dim(), vectors_.data() + slot * Dim());
                }
                ids_[slot] = ids_[last];
                extras_[slot] = std::move(extras_[last]);
                incoming_[slot] = std::move(incoming_[last]);
                extras_[slot].erase(std::remove(extras_[slot].begin(), extras_[slot].end(), slot),
                                    extras_[slot].end());
                slots_.at(ids_[slot]) = slot;
            }
            ids_.pop_back();
            if (fp16_) {
                fp16_vectors_.resize(last * Dim());
            } else {
                vectors_.resize(last * Dim());
            }
            extras_.pop_back();
            incoming_.pop_back();
            auto remap = [slot, last](uint64_t node) { return node == last ? slot : node; };
            for (auto& node : affected_nodes) {
                node = remap(node);
            }
            std::vector<uint64_t> repair_candidates;
            repair_candidates.reserve(removed_neighbors.size());
            for (const uint64_t old_candidate : removed_neighbors) {
                repair_candidates.push_back(remap(old_candidate));
            }
            repair(std::move(affected_nodes), repair_candidates);
            slots_.erase(found);
            journal.Commit();
            maybe_compact_incoming();
            return true;
        } catch (const std::bad_alloc&) {
            return false;
        } catch (const std::length_error&) {
            return false;
        }
    }

    tl::expected<std::vector<Neighbor>, Error>
    Search(const float* query, uint64_t dim, uint64_t k) const override {
        return SearchImpl(query, dim, k, nullptr, ef_search_);
    }

    tl::expected<std::vector<Neighbor>, Error>
    Search(const float* query, uint64_t dim, uint64_t k, const IdFilter& filter) const override {
        return SearchImpl(query, dim, k, filter ? &filter : nullptr, ef_search_);
    }

    tl::expected<std::vector<Neighbor>, Error>
    SearchWithOptions(const float* query,
                      uint64_t dim,
                      uint64_t k,
                      const SearchOptions& options,
                      const IdFilter& filter) const override {
        const uint64_t budget = options.ef_search == 0 ? ef_search_ : options.ef_search;
        return SearchImpl(query, dim, k, filter ? &filter : nullptr, budget);
    }

    tl::expected<std::vector<Neighbor>, Error>
    SearchImpl(const float* query,
               uint64_t dim,
               uint64_t k,
               const IdFilter* filter,
               uint64_t budget) const {
        auto valid = validate(query, dim, Dim());
        if (not valid) {
            return tl::unexpected(valid.error());
        }
        try {
            k = std::min(k, Size());
            if (k == 0) {
                return std::vector<Neighbor>{};
            }
            const uint64_t ef = std::min(Size(), std::max(k, budget));
            // With closer as Compare, top() is the farthest candidate and pop() evicts it.
            QueryHeap<Candidate, Closer> best;
            std::priority_queue<Candidate, std::vector<Candidate>, Closer> accepted;
            std::priority_queue<Candidate, std::vector<Candidate>, Farther> candidates;
            std::vector<uint8_t> visited(Size(), 0);
            std::vector<uint16_t> encoded_query;
            if (fp16_) {
                encoded_query = encode(query);
            }

            auto visit = [&](uint64_t slot) {
                if (visited[slot] != 0) {
                    return;
                }
                visited[slot] = 1;
                Candidate next{slot, distance(query, encoded_query, slot)};
                // Filtered results include every allowed visited node, even if
                // its routing score does not improve the full best heap.
                if (filter != nullptr and (*filter)(IdAt(slot))) {
                    accepted.push(next);
                    if (accepted.size() > k) {
                        accepted.pop();
                    }
                }
                // Native BasicSearcher also bounds admission before heap writes.
                // The boundary only improves; rejected nodes cannot expand later.
                if (best.size() == ef and not closer(next, best.top())) {
                    return;
                }
                candidates.push(next);
                best.push(next);
                if (best.size() > ef) {
                    best.pop();
                }
            };
            visit(0);
            const uint64_t last = Size() - 1;
            if (last != 0) {
                visit(last);
            }
            constexpr uint64_t k_extra_entry_points = 6;
            for (uint64_t i = 1; i <= k_extra_entry_points; ++i) {
                const uint64_t entry = i * last / (k_extra_entry_points + 1);
                if (entry != 0 and entry != last) {
                    visit(entry);
                }
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

            std::vector<Neighbor> result;
            if (filter == nullptr) {
                auto values = best.TakeData();
                const auto rank = [this](const Candidate& left, const Candidate& right) {
                    return left.distance < right.distance or
                           (left.distance == right.distance and IdAt(left.slot) < IdAt(right.slot));
                };
                const uint64_t count = std::min(k, static_cast<uint64_t>(values.size()));
                const auto end = values.begin() + static_cast<std::ptrdiff_t>(count);
                if (count < values.size()) {
                    std::nth_element(values.begin(), end, values.end(), rank);
                }
                std::sort(values.begin(), end, rank);
                result.reserve(count);
                for (uint64_t i = 0; i < count; ++i) {
                    result.push_back({IdAt(values[i].slot), values[i].distance});
                }
            } else {
                result.reserve(accepted.size());
                while (not accepted.empty()) {
                    const auto candidate = accepted.top();
                    accepted.pop();
                    result.push_back({IdAt(candidate.slot), candidate.distance});
                }
                std::sort(
                    result.begin(), result.end(), [](const Neighbor& left, const Neighbor& right) {
                        return left.distance < right.distance or
                               (left.distance == right.distance and left.id < right.id);
                    });
            }
            result.resize(std::min(k, static_cast<uint64_t>(result.size())));
            return result;
        } catch (const std::invalid_argument&) {
            return failure(ErrorType::INVALID_ARGUMENT, "query exceeds FP16 range");
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
        if (fp16_) {
            return false;
        }
        slots_.reserve(ids.size());
        for (uint64_t slot = 0; slot < ids.size(); ++slot) {
            if (not slots_.emplace(ids[slot], slot).second) {
                return false;
            }
        }
        ids_ = std::move(ids);
        vectors_ = std::move(vectors);
        extras_ = std::move(links);
        incoming_ = build_incoming(extras_);
        return true;
    }

    bool
    RestoreFP16(std::vector<int64_t> ids,
                std::vector<uint16_t> vectors,
                std::vector<std::vector<uint64_t>> links) {
        if (not fp16_) {
            return false;
        }
        slots_.reserve(ids.size());
        for (uint64_t slot = 0; slot < ids.size(); ++slot) {
            if (not slots_.emplace(ids[slot], slot).second) {
                return false;
            }
        }
        ids_ = std::move(ids);
        fp16_vectors_ = std::move(vectors);
        extras_ = std::move(links);
        incoming_ = build_incoming(extras_);
        return true;
    }

    [[nodiscard]] BackendKind
    Kind() const override {
        return BackendKind::GRAPH;
    }

    [[nodiscard]] VectorStorage
    Storage() const override {
        return fp16_ ? VectorStorage::FP16 : VectorStorage::FP32;
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
    IncomingLinkCountAt(uint64_t slot) const override {
        return incoming_[slot].size();
    }

    [[nodiscard]] uint64_t
    IncomingLinkAt(uint64_t slot, uint64_t offset) const override {
        return incoming_[slot][offset];
    }

    [[nodiscard]] uint64_t
    IncomingLogicalBytes() const override {
        return incoming_bytes(false);
    }

    [[nodiscard]] uint64_t
    IncomingCapacityBytes() const override {
        return incoming_bytes(true);
    }

    [[nodiscard]] uint64_t
    IncomingCompactionCount() const override {
        return incoming_compactions_;
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
    VectorAt(uint64_t slot, std::vector<float>& scratch) const override {
        if (build_source_ != nullptr) {
            return build_source_->VectorAt(slot, scratch);
        }
        if (not fp16_) {
            return vectors_.data() + slot * Dim();
        }
        scratch.resize(Dim());
        for (uint64_t d = 0; d < Dim(); ++d) {
            scratch[d] = decode_fp16(fp16_vectors_[slot * Dim() + d]);
        }
        return scratch.data();
    }

    [[nodiscard]] const uint16_t*
    FP16VectorAt(uint64_t slot) const override {
        return fp16_ ? fp16_vectors_.data() + slot * Dim() : nullptr;
    }

private:
    static void
    erase_value(std::vector<uint64_t>& values, uint64_t value) {
        values.erase(std::remove(values.begin(), values.end(), value), values.end());
    }

    static void
    replace_value(std::vector<uint64_t>& values, uint64_t old_value, uint64_t new_value) {
        for (uint64_t& value : values) {
            if (value == old_value) {
                value = new_value;
            }
        }
    }

    static std::vector<std::vector<uint64_t>>
    build_incoming(const std::vector<std::vector<uint64_t>>& links) {
        std::vector<uint64_t> counts(links.size());
        for (const auto& neighbors : links) {
            for (const uint64_t target : neighbors) {
                ++counts[target];
            }
        }
        std::vector<std::vector<uint64_t>> incoming(links.size());
        for (uint64_t target = 0; target < links.size(); ++target) {
            incoming[target].reserve(counts[target]);
        }
        for (uint64_t source = 0; source < links.size(); ++source) {
            for (const uint64_t target : links[source]) {
                incoming[target].push_back(source);
            }
        }
        return incoming;
    }

    void
    remember_row(uint64_t slot, bool outgoing, bool incoming) {
        if (active_journal_ != nullptr) {
            active_journal_->Remember(slot, outgoing, incoming);
        }
    }

    void
    sync_incoming(uint64_t source, const std::vector<uint64_t>& old_neighbors) {
        const auto& neighbors = extras_[source];
        for (const uint64_t target : old_neighbors) {
            if (std::find(neighbors.begin(), neighbors.end(), target) == neighbors.end()) {
                remember_row(target, false, true);
                erase_value(incoming_[target], source);
            }
        }
        for (const uint64_t target : neighbors) {
            if (std::find(old_neighbors.begin(), old_neighbors.end(), target) ==
                old_neighbors.end()) {
                remember_row(target, false, true);
                incoming_[target].push_back(source);
            }
        }
    }

    void
    replace_neighbors(uint64_t source, const std::vector<uint64_t>& neighbors) {
        remember_row(source, true, false);
        const auto old_neighbors = extras_[source];
        extras_[source] = neighbors;
        sync_incoming(source, old_neighbors);
    }

    [[nodiscard]] uint64_t
    incoming_bytes(bool capacity) const {
        uint64_t bytes =
            (capacity ? incoming_.capacity() : incoming_.size()) * sizeof(std::vector<uint64_t>);
        for (const auto& sources : incoming_) {
            bytes += (capacity ? sources.capacity() : sources.size()) * sizeof(uint64_t);
        }
        return bytes;
    }

    void
    maybe_compact_incoming() {
        ++removes_since_incoming_check_;
        const uint64_t interval = std::max(K_MIN_INCOMING_COMPACT_INTERVAL, Size() / 100);
        if (removes_since_incoming_check_ < interval) {
            return;
        }
        removes_since_incoming_check_ = 0;
        const uint64_t logical = incoming_bytes(false);
        const uint64_t capacity = incoming_bytes(true);
        if (logical == 0 or capacity - logical <= logical / 4) {
            return;
        }
        try {
            for (auto& sources : incoming_) {
                sources.shrink_to_fit();
            }
            incoming_.shrink_to_fit();
            ++incoming_compactions_;
        } catch (const std::bad_alloc&) {
            // Capacity recovery is best-effort and must not make a successful Remove fail.
        }
    }

    void
    ensure_incoming(uint64_t target) {
        if (Size() <= 1 or not incoming_[target].empty()) {
            return;
        }
        uint64_t selected_source = Size();
        uint64_t selected_edge = 0;
        float farthest_distance = -1.0F;
        for (const uint64_t source : extras_[target]) {
            for (uint64_t edge = 0; edge < extras_[source].size(); ++edge) {
                const uint64_t displaced = extras_[source][edge];
                if (incoming_[displaced].size() <= 1) {
                    continue;
                }
                const float candidate_distance = distance(source, displaced);
                if (selected_source == Size() or candidate_distance > farthest_distance) {
                    selected_source = source;
                    selected_edge = edge;
                    farthest_distance = candidate_distance;
                }
            }
        }
        if (selected_source == Size()) {
            return;
        }
        remember_row(selected_source, true, false);
        const auto old_neighbors = extras_[selected_source];
        extras_[selected_source][selected_edge] = target;
        sync_incoming(selected_source, old_neighbors);
    }

    void
    repair(std::vector<uint64_t> affected_nodes,
           const std::vector<uint64_t>& additional_candidates) {
        std::sort(affected_nodes.begin(), affected_nodes.end());
        affected_nodes.erase(std::unique(affected_nodes.begin(), affected_nodes.end()),
                             affected_nodes.end());
        for (const uint64_t source : affected_nodes) {
            if (source >= Size()) {
                continue;
            }
            // Preserve valid links to limit topology drift; rank only candidates that can
            // refill the degree lost by the mutation.
            auto& repaired = extras_[source];
            if (repaired.size() >= max_degree_) {
                continue;
            }
            remember_row(source, true, false);
            const auto old_neighbors = repaired;
            std::vector<Candidate> ranked;
            ranked.reserve(additional_candidates.size());
            for (const uint64_t candidate : additional_candidates) {
                if (candidate < Size() and candidate != source and
                    std::find(repaired.begin(), repaired.end(), candidate) == repaired.end()) {
                    ranked.push_back({candidate, distance(source, candidate)});
                }
            }
            std::sort(ranked.begin(), ranked.end(), Closer{});
            const uint64_t needed = max_degree_ - repaired.size();
            const uint64_t retained = std::min(needed, ranked.size());
            repaired.reserve(repaired.size() + retained);
            for (uint64_t i = 0; i < retained; ++i) {
                repaired.push_back(ranked[i].slot);
            }
            sync_incoming(source, old_neighbors);
        }
        for (const uint64_t target : affected_nodes) {
            if (target < Size()) {
                ensure_incoming(target);
            }
        }
    }

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

    std::vector<uint16_t>
    encode(const float* vector) const {
        std::vector<uint16_t> result(Dim());
        for (uint64_t d = 0; d < Dim(); ++d) {
            result[d] = encode_fp16(vector[d]);
        }
        return result;
    }

    [[nodiscard]] float
    distance(const float* query, const std::vector<uint16_t>& encoded_query, uint64_t slot) const {
        if (fp16_) {
            return fp16_distance_(encoded_query.data(), fp16_vectors_.data() + slot * Dim(), Dim());
        }
        if (build_source_ == nullptr) {
            return distance_(query, vectors_.data() + slot * Dim(), Dim());
        }
        std::vector<float> scratch;
        return distance_(query, build_source_->VectorAt(slot, scratch), Dim());
    }

    [[nodiscard]] float
    distance(uint64_t left, uint64_t right) const {
        if (fp16_) {
            return fp16_distance_(
                fp16_vectors_.data() + left * Dim(), fp16_vectors_.data() + right * Dim(), Dim());
        }
        if (build_source_ == nullptr) {
            return distance_(
                vectors_.data() + left * Dim(), vectors_.data() + right * Dim(), Dim());
        }
        std::vector<float> left_scratch;
        std::vector<float> right_scratch;
        const auto* left_vector = VectorAt(left, left_scratch);
        const auto* right_vector = VectorAt(right, right_scratch);
        return distance_(left_vector, right_vector, Dim());
    }

    tl::expected<std::vector<uint64_t>, Error>
    nearest(const float* vector,
            uint64_t count,
            uint64_t excluded = std::numeric_limits<uint64_t>::max()) const {
        if (Size() == 0) {
            return std::vector<uint64_t>{};
        }
        // Explore four times the requested neighbor count during incremental construction to
        // improve graph quality before retaining the closest count candidates.
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
        if (source == target or
            std::find(neighbors.begin(), neighbors.end(), target) != neighbors.end()) {
            return;
        }
        remember_row(source, true, false);
        const auto old_neighbors = neighbors;
        neighbors.push_back(target);
        if (neighbors.size() > max_degree_) {
            // Cache each distance once before sorting. The shared limit keeps this capacity in
            // sync with graph option and snapshot validation.
            std::array<Candidate, K_MAX_DEGREE + 1> ranked{};
            for (uint64_t i = 0; i < neighbors.size(); ++i) {
                ranked[i] = {neighbors[i], distance(source, neighbors[i])};
            }
            std::sort(ranked.begin(),
                      ranked.begin() + static_cast<std::ptrdiff_t>(neighbors.size()),
                      Closer{});
            uint64_t dropped = max_degree_;
            for (uint64_t i = neighbors.size(); i > 0; --i) {
                const uint64_t candidate = ranked[i - 1].slot;
                const bool safe_to_drop = candidate == target ? not incoming_[candidate].empty()
                                                              : incoming_[candidate].size() > 1;
                if (safe_to_drop) {
                    dropped = i - 1;
                    break;
                }
            }
            for (uint64_t source = 0, destination = 0; source < neighbors.size(); ++source) {
                if (source != dropped) {
                    neighbors[destination++] = ranked[source].slot;
                }
            }
            neighbors.resize(max_degree_);
        }
        sync_incoming(source, old_neighbors);
    }

    uint64_t dim_;
    uint64_t max_degree_;
    uint64_t ef_search_;
    bool fp16_;
    FP32Distance distance_;
    FP16Distance fp16_distance_;
    std::vector<float> vectors_;
    std::vector<uint16_t> fp16_vectors_;
    std::vector<int64_t> ids_;
    std::unordered_map<int64_t, uint64_t> slots_;
    std::vector<std::vector<uint64_t>> extras_;
    std::vector<std::vector<uint64_t>> incoming_;
    uint64_t removes_since_incoming_check_{};
    uint64_t incoming_compactions_{};
    MutationJournal* active_journal_{};
    bool journal_mutations_;
    // Non-owning, synchronous build-only source. This graph never escapes its factory.
    const Backend* build_source_;
};

}  // namespace

tl::expected<std::unique_ptr<Backend>, Error>
make_graph_backend(const Backend& source, uint64_t max_degree, uint64_t ef_search) {
    if (source.Kind() != BackendKind::BRUTE_FORCE or source.Dim() == 0 or max_degree < 2 or
        max_degree > K_MAX_DEGREE or ef_search < max_degree) {
        return failure(ErrorType::INVALID_ARGUMENT, "invalid graph options");
    }
    try {
        auto graph =
            std::make_unique<GraphBackend>(source.Dim(), max_degree, ef_search, false, false);
        graph->ReserveBuild(source.Size());
        std::vector<float> scratch;
        for (uint64_t slot = 0; slot < source.Size(); ++slot) {
            auto added =
                graph->Add(source.IdAt(slot), source.VectorAt(slot, scratch), source.Dim());
            if (not added) {
                return tl::unexpected(added.error());
            }
        }
        graph->EnableMutationJournal();
        return std::unique_ptr<Backend>(std::move(graph));
    } catch (const std::bad_alloc&) {
        return failure(ErrorType::NO_ENOUGH_MEMORY, "graph allocation failed");
    } catch (const std::length_error&) {
        return failure(ErrorType::NO_ENOUGH_MEMORY, "graph capacity exceeded");
    }
}

tl::expected<std::vector<std::vector<uint64_t>>, Error>
build_graph_topology(const Backend& source, uint64_t max_degree, uint64_t ef_search) {
    if (source.Kind() != BackendKind::BRUTE_FORCE or source.Dim() == 0 or max_degree < 2 or
        max_degree > K_MAX_DEGREE or ef_search < max_degree) {
        return failure(ErrorType::INVALID_ARGUMENT, "invalid topology options");
    }
    try {
        GraphBackend graph(source.Dim(), max_degree, ef_search, false, false, &source);
        graph.ReserveBuild(source.Size());
        std::vector<float> scratch;
        for (uint64_t slot = 0; slot < source.Size(); ++slot) {
            auto added = graph.Add(source.IdAt(slot), source.VectorAt(slot, scratch), source.Dim());
            if (not added) {
                return tl::unexpected(added.error());
            }
        }
        // Only owned adjacency leaves this scope, never the borrowed source or row pointers.
        return graph.TakeBuildLinks();
    } catch (const std::bad_alloc&) {
        return failure(ErrorType::NO_ENOUGH_MEMORY, "topology allocation failed");
    } catch (const std::length_error&) {
        return failure(ErrorType::NO_ENOUGH_MEMORY, "topology capacity exceeded");
    }
}

tl::expected<std::unique_ptr<Backend>, Error>
make_fp16_graph_backend(const Backend& source, uint64_t max_degree, uint64_t ef_search) {
    if (source.Kind() != BackendKind::BRUTE_FORCE or source.Dim() == 0 or max_degree < 2 or
        max_degree > K_MAX_DEGREE or ef_search < max_degree) {
        return failure(ErrorType::INVALID_ARGUMENT, "invalid FP16 graph options");
    }
    try {
        auto graph =
            std::make_unique<GraphBackend>(source.Dim(), max_degree, ef_search, true, false);
        graph->ReserveBuild(source.Size());
        std::vector<float> scratch;
        for (uint64_t slot = 0; slot < source.Size(); ++slot) {
            auto added =
                graph->Add(source.IdAt(slot), source.VectorAt(slot, scratch), source.Dim());
            if (not added) {
                return tl::unexpected(added.error());
            }
        }
        graph->EnableMutationJournal();
        return std::unique_ptr<Backend>(std::move(graph));
    } catch (const std::bad_alloc&) {
        return failure(ErrorType::NO_ENOUGH_MEMORY, "FP16 graph allocation failed");
    } catch (const std::length_error&) {
        return failure(ErrorType::NO_ENOUGH_MEMORY, "FP16 graph capacity exceeded");
    }
}

tl::expected<std::unique_ptr<Backend>, Error>
restore_fp16_graph_backend(uint64_t dim,
                           uint64_t max_degree,
                           uint64_t ef_search,
                           std::vector<int64_t> ids,
                           std::vector<uint16_t> vectors,
                           std::vector<std::vector<uint64_t>> links) {
    if (dim == 0 or dim > vectors.max_size() or max_degree < 2 or max_degree > K_MAX_DEGREE or
        ef_search < max_degree or ids.size() > vectors.max_size() / dim or
        vectors.size() != ids.size() * dim or links.size() != ids.size()) {
        return failure(ErrorType::INVALID_BINARY, "invalid FP16 graph snapshot layout");
    }
    for (uint64_t slot = 0; slot < links.size(); ++slot) {
        if (links[slot].size() > max_degree) {
            return failure(ErrorType::INVALID_BINARY, "FP16 graph degree exceeds limit");
        }
        for (uint64_t i = 0; i < links[slot].size(); ++i) {
            const auto neighbor = links[slot][i];
            if (neighbor >= ids.size() or neighbor == slot or
                std::find(links[slot].begin(),
                          links[slot].begin() + static_cast<std::ptrdiff_t>(i),
                          neighbor) != links[slot].begin() + static_cast<std::ptrdiff_t>(i)) {
                return failure(ErrorType::INVALID_BINARY, "invalid FP16 graph link");
            }
        }
    }
    try {
        auto graph = std::make_unique<GraphBackend>(dim, max_degree, ef_search, true);
        if (not graph->RestoreFP16(std::move(ids), std::move(vectors), std::move(links))) {
            return failure(ErrorType::INVALID_BINARY, "invalid FP16 graph snapshot");
        }
        return std::unique_ptr<Backend>(std::move(graph));
    } catch (const std::bad_alloc&) {
        return failure(ErrorType::NO_ENOUGH_MEMORY, "FP16 graph load allocation failed");
    } catch (const std::length_error&) {
        return failure(ErrorType::INVALID_BINARY, "FP16 graph snapshot exceeds capacity");
    }
}

tl::expected<std::unique_ptr<Backend>, Error>
restore_graph_backend(uint64_t dim,
                      uint64_t max_degree,
                      uint64_t ef_search,
                      std::vector<int64_t> ids,
                      std::vector<float> vectors,
                      std::vector<std::vector<uint64_t>> links) {
    if (dim == 0 or dim > vectors.max_size() or max_degree < 2 or max_degree > K_MAX_DEGREE or
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
