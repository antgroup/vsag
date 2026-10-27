
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <fmt/format.h>
#include <vsag/filter.h>

#include <atomic>
#include <functional>
#include <memory>
#include <shared_mutex>
#include <tuple>

#include "common.h"
#include "datacell/duplicate_interface.h"
#include "label_remap.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "typing.h"
#include "utils/pointer_define.h"
#include "utils/search_lock.h"
#include "vsag_exception.h"

namespace vsag {

DEFINE_POINTER(LabelTable);

using IdMapFunction = std::function<std::tuple<bool, int64_t>(int64_t)>;

class LabelTable {
public:
    explicit LabelTable(Allocator* allocator,
                        bool use_reverse_map = true,
                        bool compress_redundant_data = false,
                        LabelRemapType label_remap_type = LabelRemapType::PG);

    // Reserved sentinel, never a live slot ID; valid IDs must be strictly below this value.
    static constexpr InnerIdType INVALID_ID = std::numeric_limits<InnerIdType>::max();

    void
    Insert(InnerIdType id, LabelType label) {
        // Lock order is delete_ids_mutex_ (outer) then storage_mutex_ (inner). Everything below
        // touches label_table_/label_remap_ storage, so the whole body runs under the storage
        // lock: a concurrent Resize()/Reserve() may reallocate and free the buffers these writes
        // go to. delete_ids_mutex_ is taken first because a search traversal may pin it in
        // shared mode for its whole duration (RemoveListReadView) while still reading labels.
        std::scoped_lock wlock(delete_ids_mutex_);
        std::scoped_lock slock(storage_mutex_);
        if (use_reverse_map_) {
            label_remap_.InsertOrAssign(label, id);
        }
        const bool was_padding_label = id < label_table_.size() && label_table_[id] == -1;
        if (id + 1 > label_table_.size()) {
            label_table_.resize(id + 1);
        }
        label_table_[id] = label;
        MarkLabelsMutated();
        if (label == -1 && deleted_ids_.count(id) == 0) {
            active_padding_label_ids_.insert(id);
        } else if (was_padding_label) {
            active_padding_label_ids_.erase(id);
        }
        total_count_++;
    }

    void
    SetImmutable() {
        std::scoped_lock slock(storage_mutex_);
        this->use_reverse_map_ = false;
        this->label_remap_.Reset();
    }

    /**
     * Mark labels as removed.
     * @param labels The labels to mark as removed.
     * @return The number of labels marked as removed.
     */
    uint32_t
    MarkRemove(const std::vector<LabelType>& labels);

    /**
     * Mark a label as removed.
     * @param label The label to mark as removed.
     * @return The number of labels marked as removed.
     */
    uint32_t
    MarkRemove(const LabelType& label) {
        return MarkRemove(std::vector<LabelType>({label}));
    }

    /**
     * Check whether an id is removed.
     * @param id The id to check.
     * @return True if the id is removed, false otherwise.
     */
    bool
    IsRemoved(InnerIdType id) {
        auto rlock = make_search_read_lock(delete_ids_mutex_);
        return deleted_ids_.count(id) != 0;
    }

    void
    EraseFromDeletedIds(InnerIdType id) {
        // Reads label_table_[id] as well as deleted_ids_, so it needs both locks in the
        // class-wide order (delete first, then storage).
        std::scoped_lock wlock(delete_ids_mutex_);
        std::scoped_lock slock(storage_mutex_);
        if (deleted_ids_.erase(id) > 0 && id < label_table_.size() && label_table_[id] == -1) {
            active_padding_label_ids_.insert(id);
        }
    }

    /**
     * Get id by label.
     * @param label The label to query.
     * @param return_even_removed Whether to return even if the id is removed.
     * @return The id corresponding to the label.
     * @throws VsagException if the label does not exist or is removed.
     */
    InnerIdType
    GetIdByLabel(LabelType label, bool return_even_removed = false) const;

    /**
     * Try to get id by label without throwing exception.
     * @param label The label to query.
     * @param return_even_removed Whether to return even if the id is removed.
     * @return A pair where first indicates success and second is the inner_id.
     */
    std::pair<bool, InnerIdType>
    TryGetIdByLabel(LabelType label, bool return_even_removed = false) const noexcept;

    /**
     * Check whether a label exists and not been removed.
     * @param label The label to check.
     * @return True if the label exists and not been removed, false otherwise.
     */
    bool
    CheckLabel(LabelType label) const;

    bool
    HasActivePaddingLabel() const {
        auto rlock = make_search_read_lock(delete_ids_mutex_);
        return !active_padding_label_ids_.empty();
    }

    void
    RebuildActivePaddingLabelIds() {
        // Scans label_table_, so it must exclude reallocation while it reads. Both locks are
        // taken here (delete first, see storage_mutex_ ordering) and the body assumes both.
        std::scoped_lock wlock(delete_ids_mutex_);
        std::scoped_lock slock(storage_mutex_);
        RebuildActivePaddingLabelIdsLocked();
    }

    // HGraph's non-duplicate format serializes capacity separately from the
    // occupied label-to-id map. Unused -1 slots are not active external labels.
    void
    RebuildActivePaddingLabelIdsFromRemap() {
        // Rebuilds the padding set while reading label storage and deletion state.
        // Hold both mutexes exclusively, delete first, so storage-only lookups are safe.
        std::scoped_lock wlock(delete_ids_mutex_);
        std::scoped_lock slock(storage_mutex_);
        active_padding_label_ids_.clear();
        // Call label_remap_ directly instead of ForEachRemap(): the latter takes
        // storage_mutex_ in shared mode and would self-deadlock here.
        this->label_remap_.ForEach([this](LabelType label, InnerIdType id) {
            if (label == -1 and id < label_table_.size() and deleted_ids_.count(id) == 0) {
                active_padding_label_ids_.insert(id);
            }
        });
    }

    void
    UpdateLabel(LabelType old_label, LabelType new_label) {
        // A search can pin the deleted-ID view while invoking a user filter callback. For an
        // ordinary relabel into an absent destination, no deletion state changes or reads are
        // needed: acquire storage only so callback UpdateId does not self-deadlock on the view.
        if (old_label != -1 and new_label != -1) {
            std::scoped_lock slock(storage_mutex_);
            const bool destination_present =
                use_reverse_map_ ? get_id_by_label_with_reverse_map(new_label) != INVALID_ID
                                 : get_id_by_label_with_label_table(new_label) != INVALID_ID;
            if (not destination_present) {
                UpdateLabelStorageLocked(old_label, new_label);
                return;
            }
        }
        // Ordinary relabels only read deletion state. All deletion-state writers take
        // storage_mutex_ exclusively, so do not recursively take a pinned deletion view.
        // Padding relabels mutate deletion bookkeeping and still take delete -> storage.
        if (old_label != -1 and new_label != -1) {
            std::scoped_lock slock(storage_mutex_);
            if (CheckLabelLocked(new_label)) {
                throw VsagException(ErrorType::INTERNAL_ERROR,
                                    fmt::format("new label {} has been in Index", new_label));
            }
            UpdateLabelStorageLocked(old_label, new_label);
            return;
        }
        std::scoped_lock dlock(delete_ids_mutex_);
        std::scoped_lock slock(storage_mutex_);
        if (CheckLabelLocked(new_label)) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                fmt::format("new label {} has been in Index", new_label));
        }
        UpdateLabelStorageLocked(old_label, new_label);
    }

    // Both callers hold storage_mutex_. The caller has already checked that the new label
    // is not occupied; ordinary non-padding relabels do not mutate deletion state.
    void
    UpdateLabelStorageLocked(LabelType old_label, LabelType new_label) {
        // 2. update label_table_
        // Important: there may be multiple occurrences of old_label, so we need to update every one
        bool found = false;
        for (size_t i = 0; i < label_table_.size(); ++i) {
            if (label_table_[i] == old_label) {
                label_table_[i] = new_label;
                if (old_label == -1 || new_label == -1) {
                    // Only the padding call path can enter here; it holds delete_ids_mutex_
                    // exclusively. Ordinary relabels never touch deletion bookkeeping.
                    if (deleted_ids_.count(static_cast<InnerIdType>(i)) == 0) {
                        if (new_label == -1) {
                            active_padding_label_ids_.insert(static_cast<InnerIdType>(i));
                        } else {
                            active_padding_label_ids_.erase(static_cast<InnerIdType>(i));
                        }
                    }
                }
                found = true;
            }
        }
        MarkLabelsMutated();
        if (not found) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                fmt::format("old label {} does not exist", old_label));
        }

        // 3. update label_remap_
        if (use_reverse_map_) {
            // note that currently, old_label must exist
            InnerIdType internal_id = INVALID_ID;
            bool remap_found = label_remap_.Find(old_label, internal_id);
            CHECK_ARGUMENT(remap_found,
                           fmt::format("old label {} does not exist in remap", old_label));
            label_remap_.Erase(old_label);
            label_remap_.InsertOrAssign(new_label, internal_id);
        }
    }

    LabelType
    GetLabelById(InnerIdType inner_id) const {
        auto slock = make_search_read_lock(storage_mutex_);
        if (inner_id >= label_table_.size()) {
            throw VsagException(
                ErrorType::INTERNAL_ERROR,
                fmt::format("id is too large {} >= {}", inner_id, label_table_.size()));
        }
        return this->label_table_[inner_id];
    }

    // Copy the live labels into `out` while holding the storage lock in shared mode,
    // so the copy cannot race with a concurrent reallocation of label_table_.
    // `out` must have room for at least `count` entries. A stale logical count
    // must fail rather than leaving uninitialized ids in ExportIDs() output.
    void
    CopyAllLabels(LabelType* out, int64_t count) const {
        std::shared_lock slock(storage_mutex_);
        CHECK_ARGUMENT(count >= 0, "requested label count is negative");
        CHECK_ARGUMENT(static_cast<uint64_t>(count) <= label_table_.size(),
                       "requested label count exceeds label storage");
        if (count > 0) {
            std::memcpy(out, label_table_.data(), static_cast<size_t>(count) * sizeof(LabelType));
        }
    }

    void
    Serialize(StreamWriter& writer) const {
        std::shared_lock slock(storage_mutex_);
        StreamWriter::WriteVector(writer, label_table_);
    }

    void
    Deserialize(LvalueOrRvalue<StreamReader> reader) {
        std::scoped_lock wlock(delete_ids_mutex_);
        std::scoped_lock slock(storage_mutex_);
        StreamReader::ReadVector(reader, label_table_);
        MarkLabelsMutated();
        // Locked helper: both locks are already held here.
        RebuildActivePaddingLabelIdsLocked();
        if (use_reverse_map_) {
            this->label_remap_.Clear();
            this->label_remap_.Reserve(label_table_.size());
            for (InnerIdType id = 0; id < label_table_.size(); ++id) {
                this->label_remap_.InsertOrAssign(label_table_[id], id);
            }
        }

        this->total_count_.store(label_table_.size());
    }

    void
    Deserialize(StreamReader& reader, uint64_t max_labels = std::numeric_limits<uint64_t>::max());

    void
    Resize(uint64_t new_size) {
        // Grows (and may reallocate/free) the label storage. Must exclude concurrent
        // Insert()/Move()/ShrinkToFit(), which write into that same storage while holding
        // only delete_ids_mutex_ plus the owning index's label lock.
        std::scoped_lock slock(storage_mutex_);
        if (new_size < total_count_) {
            return;
        }
        label_table_.resize(new_size);
        MarkLabelsMutated();
    }

    // Dense append-only callers should reserve storage without creating label slots.
    // Resize remains available for callers that populate explicitly sized tables.
    void
    Reserve(uint64_t capacity) {
        // reserve() can also reallocate the buffer, so it needs the same exclusion as Resize().
        std::scoped_lock slock(storage_mutex_);
        label_table_.reserve(capacity);
    }

    // Legacy dense payloads may contain reserved slots. Their owning index must
    // supply the persisted vector count; label values cannot identify unused slots.
    void
    TrimUnusedSlots(uint64_t valid_count);

    int64_t
    GetTotalCount() {
        return total_count_;
    }

    void
    SetDuplicateTracker(DuplicateTrackerPtr tracker) {
        duplicate_tracker_ = std::move(tracker);
    }

    void
    MergeOther(const LabelTablePtr& other, const IdMapFunction& id_map = nullptr);

    /**
     * Get memory usage of the label table.
     * @return The memory usage in bytes.
     */
    uint64_t
    GetMemoryUsage() {
        // Reads label_table_ capacity and walks label_remap_ state, so it must not overlap a
        // reallocation or a remap mutation. deleted_ids_ is read under its own lock, taken
        // first to respect the class-wide order.
        std::shared_lock dlock(delete_ids_mutex_);
        std::shared_lock slock(storage_mutex_);
        return sizeof(LabelTable) + label_table_.capacity() * sizeof(LabelType) +
               label_remap_.GetMemoryUsage() + deleted_ids_.size() * sizeof(InnerIdType) * 2;
    }

    uint64_t
    GetRemapSize() const {
        std::shared_lock slock(storage_mutex_);
        return label_remap_.Size();
    }

    template <typename Visitor>
    void
    ForEachRemap(Visitor&& visitor) const {
        std::shared_lock slock(storage_mutex_);
        label_remap_.ForEach(visitor);
    }

    void
    ResetRemap(uint64_t size = 0) {
        std::scoped_lock slock(storage_mutex_);
        label_remap_.Clear();
        if (size > 0) {
            label_remap_.Reserve(size);
        }
    }

    void
    InsertRemap(LabelType label, InnerIdType inner_id) {
        std::scoped_lock slock(storage_mutex_);
        label_remap_.Emplace(label, inner_id);
    }

    /**
     * Get filter to filter out deleted ids.
     * @return The filter.
     */
    FilterPtr
    GetDeletedIdsFilter() {
        auto rlock = make_search_read_lock(delete_ids_mutex_);
        if (deleted_ids_.empty()) {
            return nullptr;
        }
        return deleted_ids_filter_;
    }

    // Pins the deletion set once; callers must release the filter before acquiring label locks.
    FilterPtr
    GetDeletedIdsReadView() const;

    // Restore a persisted physical-slot mask after loading labels, before publishing the index.
    void
    RestoreDeletedIds(const Vector<InnerIdType>& ids, uint64_t total);

    std::vector<InnerIdType>
    GetDeletedIds(InnerIdType max_count) {
        auto rlock = make_search_read_lock(delete_ids_mutex_);
        if (deleted_ids_.empty()) {
            return {};
        }
        auto size = std::min<uint64_t>(static_cast<uint64_t>(max_count), deleted_ids_.size());
        return std::vector<InnerIdType>(deleted_ids_.begin(),
                                        std::next(deleted_ids_.begin(), size));
    }

    std::vector<InnerIdType>
    GetAllDeletedIds() {
        auto rlock = make_search_read_lock(delete_ids_mutex_);
        return std::vector<InnerIdType>(deleted_ids_.begin(), deleted_ids_.end());
    }

    // Return an owned snapshot directly in the caller's allocator, without an intermediate copy.
    Vector<InnerIdType>
    GetAllDeletedIds(Allocator* allocator) const {
        auto rlock = make_search_read_lock(delete_ids_mutex_);
        return {deleted_ids_.begin(), deleted_ids_.end(), allocator};
    }

private:
    InnerIdType
    get_id_by_label_with_reverse_map(LabelType label) const noexcept;

    InnerIdType
    get_id_by_label_with_label_table(LabelType label) const noexcept;

    // Body of CheckLabel() without taking storage_mutex_, for callers that already hold it
    // (std::shared_mutex is not recursive, so re-acquiring would self-deadlock).
    bool
    CheckLabelLocked(LabelType label) const;

    // Body of RebuildActivePaddingLabelIds(). Callers hold delete_ids_mutex_ and
    // storage_mutex_ exclusively: the padding set is also read under storage alone.
    void
    RebuildActivePaddingLabelIdsLocked() {
        active_padding_label_ids_.clear();
        for (InnerIdType id = 0; id < label_table_.size(); ++id) {
            if (label_table_[id] == -1 && deleted_ids_.count(id) == 0) {
                active_padding_label_ids_.insert(id);
            }
        }
    }

    // Must be called after every write to label_table_ so that cached label-space properties are
    // never reused for the new content.
    void
    MarkLabelsMutated() {
        this->labels_version_.fetch_add(1, std::memory_order_release);
    }

public:
    /**
     * @brief Whether this table maps every live inner id to itself, i.e. the label space and the
     * inner-id space coincide.
     *
     * Components that reuse an id-space-indexed structure as an inner-id-indexed one (for example
     * the validity bitmap that the MCI searcher inlines) must check this first, because the label
     * a filter observes is not necessarily the inner id it is probed with. Only the live ids
     * `[0, GetTotalCount())` take part: the table pre-allocates capacity, and those spare slots are
     * never probed. The answer is cached and recomputed after any label mutation.
     *
     * Concurrency: scanning the table requires concurrent label mutations to be excluded, because a
     * mutation can reallocate the storage being read. `labels_version_` only protects the cached
     * answer, not the scan. Search paths therefore call this while holding the label lock in shared
     * mode, the same way they read labels anywhere else.
     */
    [[nodiscard]] bool
    IsIdentityMapping() const {
        // Exclude reallocation while scanning. HGraph's MCI filter can call this while
        // holding label_lookup_mutex_ in shared mode; no LabelTable method calls it
        // while already holding storage_mutex_. Resize takes global_mutex_ before
        // storage_mutex_ and does not subsequently acquire label_lookup_mutex_.
        // search_no_lock callers promise exclusive access to the index for the whole query.
        auto slock = make_search_read_lock(storage_mutex_);
        const auto version = this->labels_version_.load(std::memory_order_acquire);
        if (version == this->identity_version_.load(std::memory_order_acquire)) {
            return this->identity_mapping_.load(std::memory_order_relaxed);
        }
        const auto live_count =
            static_cast<uint64_t>(this->total_count_.load(std::memory_order_relaxed));
        bool identity = true;
        for (uint64_t id = 0; id < live_count and id < label_table_.size(); ++id) {
            if (label_table_[id] != static_cast<LabelType>(id)) {
                identity = false;
                break;
            }
        }
        this->identity_mapping_.store(identity, std::memory_order_relaxed);
        this->identity_version_.store(version, std::memory_order_release);
        return identity;
    }

    // Label table, map from id to label.
    //
    // NOTE: this is public for historical reasons. Serialization paths (e.g.
    // hgraph_serialize.cpp) bypass storage_mutex_ while the index is exclusively held.
    // HGraph::resize() also accesses it, but through LabelTable::Resize(), which
    // takes storage_mutex_. New code should use the locked accessors above.
    Vector<LabelType> label_table_;

    // Temporary compatibility switch for legacy duplicate payload layout.
    bool is_legacy_duplicate_format_{false};

    // Whether to use reverse map to speed up GetIdByLabel.
    bool use_reverse_map_{true};
    // Reverse map from label to id.
    LabelRemap label_remap_;

    DuplicateTrackerPtr duplicate_tracker_{nullptr};

    Allocator* allocator_{nullptr};
    std::atomic<int64_t> total_count_{0L};

    // Cached answer of IsIdentityMapping() and the label version it was computed from. Every label
    // mutation bumps labels_version_ *after* the write, so a cached answer can never be reused for
    // different table content.
    mutable std::atomic<uint64_t> labels_version_{1};
    mutable std::atomic<uint64_t> identity_version_{0};
    mutable std::atomic<bool> identity_mapping_{false};

    bool
    InsertSourceId(InnerIdType inner_id, const std::string& source_id) {
        std::scoped_lock slock(storage_mutex_);
        if (inner_id >= source_id_table_.size()) {
            source_id_table_.resize(inner_id + 1);
        }
        source_id_table_[inner_id] = source_id;
        return true;
    }

    std::string
    GetSourceId(InnerIdType inner_id) const {
        std::shared_lock slock(storage_mutex_);
        if (inner_id >= source_id_table_.size()) {
            return {};
        }
        return source_id_table_[inner_id];
    }

    // Returns a reference into source_id_table_'s storage. No lock is held after this call
    // returns; callers must exclude concurrent mutations themselves (serialization paths do).
    const Vector<std::string>&
    GetSourceIdTableRef() const {
        return source_id_table_;
    }

    // Replace the entire source_id_table_ with a pre-built table.
    // Used during single-threaded deserialization to move in the persisted
    // data without exposing a mutable reference to callers.
    void
    ReplaceSourceIdTable(Vector<std::string>&& table) {
        std::scoped_lock slock(storage_mutex_);
        source_id_table_ = std::move(table);
    }

    void
    Move(InnerIdType from, InnerIdType to) {
        if (from == to) {
            return;
        }
        // Writes label_table_, label_remap_ and source_id_table_, and reads label_table_
        // while updating the delete set, so take both locks for the whole move
        // (delete first, per the class-wide order).
        std::scoped_lock wlock(delete_ids_mutex_);
        std::scoped_lock slock(storage_mutex_);

        bool from_removed = false;
        {
            from_removed = deleted_ids_.erase(from) > 0;
            deleted_ids_.erase(to);
            active_padding_label_ids_.erase(from);
            active_padding_label_ids_.erase(to);
            if (from_removed) {
                deleted_ids_.insert(to);
            } else if (label_table_[from] == -1) {
                active_padding_label_ids_.insert(to);
            }
        }

        bool remap_from = false;
        if (use_reverse_map_) {
            InnerIdType mapped = INVALID_ID;
            remap_from = label_remap_.Find(label_table_[from], mapped) and mapped == from;
            if (label_remap_.Find(label_table_[to], mapped) and mapped == to) {
                label_remap_.Erase(label_table_[to]);
            }
        }
        label_table_[to] = label_table_[from];
        MarkLabelsMutated();
        // The caller retires the old tail; ShrinkToFit removes its stale label/source-ID row.
        // An old tombstone may share its label with a later Add. Moving it must not hide
        // the active incarnation by replacing that label's current reverse mapping.
        if (remap_from) {
            label_remap_.InsertOrAssign(label_table_[to], to);
        }
        if (from < source_id_table_.size()) {
            auto source_id = std::move(source_id_table_[from]);
            if (to >= source_id_table_.size()) {
                source_id_table_.resize(static_cast<uint64_t>(to) + 1);
            }
            source_id_table_[to] = std::move(source_id);
        } else if (to < source_id_table_.size()) {
            source_id_table_[to].clear();
        }
    }

    void
    ShrinkToFit(InnerIdType capacity) {
        // Compacts/truncates label_table_ and source_id_table_ and prunes the padding set;
        // must exclude readers and concurrent writers of that storage. Delete lock first.
        std::scoped_lock wlock(delete_ids_mutex_);
        std::scoped_lock slock(storage_mutex_);
        if (source_id_table_.size() > capacity) {
            source_id_table_.resize(capacity);
        }
        {
            for (auto it = active_padding_label_ids_.begin();
                 it != active_padding_label_ids_.end();) {
                if (*it >= capacity) {
                    it = active_padding_label_ids_.erase(it);
                } else {
                    ++it;
                }
            }
        }
        // Avoid a full-table copy for small removals; vector storage is still compacted by BruteForce.
        if (capacity <= label_table_.capacity() / 2) {
            try {
                Vector<LabelType> compacted_labels(label_table_.begin(),
                                                   label_table_.begin() + capacity,
                                                   label_table_.get_allocator());
                label_table_.swap(compacted_labels);
            } catch (const std::bad_alloc&) {
                // Deletion has already completed; retain capacity but remove stale labels.
                label_table_.resize(capacity);
            }
        } else {
            label_table_.resize(capacity);
        }
        if (duplicate_tracker_ != nullptr) {
            duplicate_tracker_->Resize(capacity);
        }
        MarkLabelsMutated();
    }

    void
    ForceRemove(LabelType label, InnerIdType inner_id) {
        std::scoped_lock wlock(delete_ids_mutex_);
        std::scoped_lock slock(storage_mutex_);
        if (use_reverse_map_) {
            label_remap_.Erase(label);
        }
        {
            active_padding_label_ids_.erase(inner_id);
            deleted_ids_.erase(inner_id);
        }
        total_count_.fetch_sub(1);
    }

private:
    UnorderedSet<InnerIdType> deleted_ids_;  // Record deleted ids.
    UnorderedSet<InnerIdType> active_padding_label_ids_;
    FilterPtr deleted_ids_filter_{nullptr};       // Filter to filter out deleted ids.
    mutable std::shared_mutex delete_ids_mutex_;  // Pins deletion views across traversal.

    // Guards label_table_, label_remap_, and deletion state on lookup. Every writer of
    // deleted_ids_ / active_padding_label_ids_ also holds this mutex exclusively, so
    // callback lookups under a pinned deletion view need only one shared storage lock.
    // Guards the *storage* of label_table_ and label_remap_: their buffers may be
    // reallocated (and the old ones freed) by Resize()/Reserve()/ShrinkToFit()/Deserialize()
    // while other threads write elements into them. delete_ids_mutex_ does not cover this:
    // it alone cannot protect label storage, and before this mutex existed
    // the reallocating side (HGraph::resize under global_mutex_) and the writing side
    // (LabelTable::Insert under label_lookup_mutex_) held two disjoint lock sets, so a write
    // could land in a freed buffer (heap-use-after-free, the reported production crash).
    //
    // Lock ordering (must be preserved to avoid deadlock):
    //   HGraph label_lookup_mutex_ (when held) -> delete_ids_mutex_ -> storage_mutex_
    //   HGraph label_lookup_mutex_ (when held) -> storage_mutex_
    //   HGraph global_mutex_ (resize) -> storage_mutex_
    // No path may acquire label_lookup_mutex_ or global_mutex_ while holding storage_mutex_.
    // delete_ids_mutex_ comes first because a search traversal pins it in shared mode for its
    // whole duration (RemoveListReadView) while still reading labels through GetLabelById(),
    // so delete-then-storage is the order a live search already imposes. Never acquire
    // delete_ids_mutex_ while holding storage_mutex_. Methods that need both take the delete
    // lock first, and helpers called with both already held must not re-acquire either one
    // (see the *_Locked() helpers below).
    mutable std::shared_mutex storage_mutex_;

    Vector<std::string> source_id_table_;  // Map from id to source id, used for cache.
};

}  // namespace vsag
