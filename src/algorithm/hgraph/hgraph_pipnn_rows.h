// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cstring>
#include <span>
#include <vector>

#include "datacell/flatten_interface.h"
#include "impl/pipnn/pipnn_incremental.h"

namespace vsag {

// Logical-ID row pointers for one exact code store. A Reserve on contiguous IO can
// relocate its records. HGraph refreshes the entire view on its next capacity change.
class HGraphPiPNNRows {
public:
    explicit HGraphPiPNNRows(FlattenInterfacePtr codes) : codes_(std::move(codes)) {
    }

    std::span<const float* const>
    Prepare(const PiPNNIncrementalState& state,
            std::span<const InnerIdType> incoming,
            uint64_t capacity) {
        if (capacity_ != capacity) {
            rows_.clear();
            owned_.clear();
            Bind(state.ids, state.RequiredCapacity());
            capacity_ = capacity;
        }
        if (!incoming.empty()) {
            Bind(incoming, uint64_t{*std::max_element(incoming.begin(), incoming.end())} + 1);
        }
        return rows_;
    }

    const float*
    Get(InnerIdType id) const {
        return rows_.at(id);
    }

    const FlattenInterfacePtr&
    Store() const {
        return codes_;
    }

    // Older snapshots can have a separately tuned raw/query store. Borrow only
    // rows that are bit-identical to the historical inputs of the PiPNN builder.
    void
    RestoreLegacy(StreamReader& reader,
                  PiPNNIncrementalState& state,
                  uint64_t dim,
                  Allocator* allocator) const {
        const uint64_t batch = std::max<uint64_t>(1, (256 * 1024) / (dim * sizeof(float)));
        const auto n = state.ids.size();
        std::vector<float> buffer(std::min<uint64_t>(batch, n) * dim);
        bool owned = false;
        for (size_t first = 0; first < n; first += batch) {
            const auto count = std::min<uint64_t>(batch, n - first);
            reader.Read(reinterpret_cast<char*>(buffer.data()), count * dim * sizeof(float));
            for (size_t i = 0; i < count; ++i) {
                const auto position = first + i;
                const auto* saved = buffer.data() + i * dim;
                if (!owned) {
                    auto row = codes_->AcquireCodesById(state.ids[position]);
                    if (row && std::memcmp(row.Data(), saved, dim * sizeof(float)) == 0)
                        continue;
                    // The prefix was bit-identical. Recover it from the available store;
                    // the remaining input can also come from a forward-only compressed frame.
                    Materialize(state, dim, allocator, position);
                    owned = true;
                }
                state.SetVector(position, saved);
            }
        }
    }

    void
    Materialize(PiPNNIncrementalState& state,
                uint64_t dim,
                Allocator* allocator,
                uint64_t prefix = UINT64_MAX) const {
        state.ReserveVectors(state.ids.size(), dim, allocator);
        for (uint64_t i = 0; i < std::min<uint64_t>(state.ids.size(), prefix); ++i) {
            auto row = codes_->AcquireCodesById(state.ids[i]);
            CHECK_ARGUMENT(row, "missing exact PiPNN code");
            state.SetVector(i, reinterpret_cast<const float*>(row.Data()));
        }
    }

    uint64_t
    MemoryUsage() const {
        return sizeof(*this) + rows_.capacity() * sizeof(rows_.front()) +
               owned_.capacity() * sizeof(owned_.front()) +
               owned_.size() * codes_->GetQuantizerCodeSize();
    }

private:
    void
    Bind(std::span<const InnerIdType> ids, uint64_t required) {
        if (rows_.size() < required)
            rows_.resize(required, nullptr);
        for (auto id : ids) {
            if (rows_[id] != nullptr)
                continue;
            auto lease = codes_->AcquireCodesById(id);
            CHECK_ARGUMENT(lease, "missing exact PiPNN code");
            const auto* row = reinterpret_cast<const float*>(lease.Data());
            if (lease.NeedsRelease())
                owned_.emplace_back(std::move(lease));  // record straddles IO blocks
            rows_[id] = row;
        }
    }

    FlattenInterfacePtr codes_;
    std::vector<const float*> rows_;
    std::vector<FlattenCodesLease> owned_;
    uint64_t capacity_{UINT64_MAX};
};
}  // namespace vsag
