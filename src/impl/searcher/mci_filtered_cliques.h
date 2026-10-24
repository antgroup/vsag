// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace vsag {

// Reused by one query worker. Only these three flat buffers allocate; no clique owns a container.
// Input IDs must be unique and increasing, and the immutable inverse CSR must match the base CSR.
// Members are emitted in increasing inner-ID order, which may differ from the base clique order.
template <typename Epoch = uint32_t>
class MCIFilteredCliques {
    static_assert(std::is_integral_v<Epoch> and std::is_unsigned_v<Epoch>);

public:
    struct Members {
        const uint32_t* data{nullptr};
        uint64_t count{0};
    };

    bool
    Build(const uint32_t* node_offsets,
          const uint32_t* node_cliques,
          uint64_t total,
          uint64_t clique_count,
          const uint32_t* valid_ids,
          uint64_t valid_count) {
        active_ = false;
        incidences_ = 0;
        kept_cliques_ = 0;
        member_count_ = 0;
        touched_.clear();
        if (clique_count > std::numeric_limits<uint32_t>::max() or valid_count > total or
            (valid_count != 0 and valid_ids == nullptr) or node_offsets == nullptr) {
            return false;
        }
        if (slots_.size() < clique_count) {
            slots_.resize(clique_count);
        }
        ++epoch_;
        if (epoch_ == 0) {
            std::fill(slots_.begin(), slots_.end(), Slot{});
            epoch_ = 1;
        }
        // Keep epoch, count/cursor and start together: avoid a dependent group-index lookup.
        for (uint64_t i = 0; i < valid_count; ++i) {
            const auto id = valid_ids[i];
            if (id >= total or (i != 0 and id <= valid_ids[i - 1])) {
                return false;
            }
            const auto begin = node_offsets[id];
            const auto end = node_offsets[id + 1];
            if (begin > end or end > node_offsets[total] or
                (begin != end and node_cliques == nullptr)) {
                return false;
            }
            incidences_ += end - begin;
            for (uint64_t offset = begin; offset < std::min<uint64_t>(end, begin + 32); ++offset) {
                PrefetchSlot(node_cliques[offset], clique_count);
            }
            for (uint64_t offset = begin; offset < end; ++offset) {
                if (offset + 32 < end) {
                    PrefetchSlot(node_cliques[offset + 32], clique_count);
                }
                const auto cid = node_cliques[offset];
                if (cid >= clique_count) {
                    return false;
                }
                auto& slot = slots_[cid];
                if (slot.epoch != epoch_) {
                    slot.epoch = epoch_;
                    slot.value = 1;
                    touched_.push_back(cid);
                } else {
                    ++slot.value;
                }
            }
        }
        // Prefix-sum only non-singleton groups: a singleton cannot discover another valid point.
        for (uint64_t i = 0; i < std::min<uint64_t>(32, touched_.size()); ++i) {
            PrefetchSlot(touched_[i], clique_count);
        }
        for (uint64_t i = 0; i < touched_.size(); ++i) {
            if (i + 32 < touched_.size()) {
                PrefetchSlot(touched_[i + 32], clique_count);
            }
            auto& slot = slots_[touched_[i]];
            const auto count = slot.value;
            slot.begin = std::numeric_limits<uint32_t>::max();
            if (count >= 2) {
                slot.begin = static_cast<uint32_t>(member_count_);
                slot.value = static_cast<uint32_t>(member_count_);
                member_count_ += count;
                ++kept_cliques_;
            }
        }
        // Keep the high-water size as well as capacity, avoiding zero-fill on every query.
        if (members_.size() < member_count_) {
            members_.resize(member_count_);
        }
        for (uint64_t i = 0; i < valid_count; ++i) {
            const auto id = valid_ids[i];
            const auto begin = node_offsets[id];
            const auto end = node_offsets[id + 1];
            for (uint64_t offset = begin; offset < std::min<uint64_t>(end, begin + 32); ++offset) {
                PrefetchSlot(node_cliques[offset], clique_count);
            }
            for (uint64_t offset = begin; offset < end; ++offset) {
                if (offset + 32 < end) {
                    PrefetchSlot(node_cliques[offset + 32], clique_count);
                }
                auto& slot = slots_[node_cliques[offset]];
                if (slot.begin != std::numeric_limits<uint32_t>::max()) {
                    members_[slot.value++] = id;
                }
            }
        }
        active_ = true;
        return true;
    }

    [[nodiscard]] Members
    Get(uint64_t cid) const {
        if (not active_ or cid >= slots_.size() or slots_[cid].epoch != epoch_) {
            return {};
        }
        const auto& slot = slots_[cid];
        return slot.begin == std::numeric_limits<uint32_t>::max()
                   ? Members{}
                   : Members{members_.data() + slot.begin, slot.value - slot.begin};
    }

    [[nodiscard]] uint64_t
    Incidences() const {
        return incidences_;
    }

    [[nodiscard]] uint64_t
    NonemptyCliques() const {
        return touched_.size();
    }

    [[nodiscard]] uint64_t
    KeptCliques() const {
        return kept_cliques_;
    }

    [[nodiscard]] uint64_t
    MemberCount() const {
        return member_count_;
    }

    [[nodiscard]] uint64_t
    CapacityBytes() const {
        return slots_.capacity() * sizeof(Slot) + touched_.capacity() * sizeof(uint32_t) +
               members_.capacity() * sizeof(uint32_t);
    }

    [[nodiscard]] const uint32_t*
    MemberStorage() const {
        return members_.data();
    }

private:
    struct Slot {
        Epoch epoch{0};
        uint32_t value{0};
        uint32_t begin{0};
    };

    void
    PrefetchSlot(uint32_t cid, uint64_t clique_count) const {
#if defined(__GNUC__) || defined(__clang__)
        if (cid < clique_count) {
            __builtin_prefetch(slots_.data() + cid, 1, 1);
        }
#endif
    }

    std::vector<Slot> slots_;
    std::vector<uint32_t> touched_;
    std::vector<uint32_t> members_;
    Epoch epoch_{0};
    uint64_t incidences_{0};
    uint64_t kept_cliques_{0};
    uint64_t member_count_{0};
    bool active_{false};
};

}  // namespace vsag
