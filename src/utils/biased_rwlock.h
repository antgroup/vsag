
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

#include <atomic>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <thread>

namespace vsag {

// Reader-biased read/write indicator built from two primitives:
//
//   fast path (readers): one seq_cst fetch_add on a per-thread padded slot,
//   then one seq_cst load of writer_pending_. Readers hold no shared cache
//   line, so N concurrent readers never invalidate each other -- unlike a
//   pthread rwlock whose reader count sits on one hot line.
//
//   slow path: when a writer is pending, readers fall back to an underlying
//   std::shared_mutex, preserving classic exclusion semantics.
//
//   writers: serialize on the same std::shared_mutex (unique), set
//   writer_pending_, drain every reader slot, run the critical section, clear
//   the flag. In-flight fast-path readers finish without touching any lock,
//   so the drain is bounded by one search duration and cannot deadlock.
//
// Correctness of the handoff uses the seq_cst total order: a reader that
// observed writer_pending_ == false ordered its slot increment before the
// writer's store in that order, so the writer's subsequent scan must observe
// the nonzero slot and wait for it.
class BiasedRwLock {
public:
    static constexpr uint32_t kReaderSlots = 128;

    enum class SharedLockKind { kFast, kSlow };

    BiasedRwLock() = default;

    BiasedRwLock(const BiasedRwLock&) = delete;
    BiasedRwLock&
    operator=(const BiasedRwLock&) = delete;

    [[nodiscard]] SharedLockKind
    LockShared(uint32_t slot_index) {
        auto& slot = reader_slots_[slot_index].count;
        if (!writer_pending_.load(std::memory_order_relaxed)) {
            slot.fetch_add(1, std::memory_order_seq_cst);
            if (!writer_pending_.load(std::memory_order_seq_cst)) {
                return SharedLockKind::kFast;
            }
            slot.fetch_sub(1, std::memory_order_release);
        }
        mutex_.lock_shared();
        return SharedLockKind::kSlow;
    }

    void
    FastUnlockShared(uint32_t slot_index) {
        reader_slots_[slot_index].count.fetch_sub(1, std::memory_order_release);
    }

    void
    UnlockShared() {
        mutex_.unlock_shared();
    }

    // Exclusive access. Caller must hold NO fast-path slot on this lock
    // (release it first), otherwise the drain below would spin forever.
    template <typename CriticalSection>
    void
    WithWriterCriticalSection(CriticalSection&& critical) {
        std::unique_lock<std::shared_mutex> exclusive(mutex_);
        writer_pending_.store(true, std::memory_order_seq_cst);
        for (auto& slot : reader_slots_) {
            while (slot.count.load(std::memory_order_acquire) != 0) {
                std::this_thread::yield();
            }
        }
        critical();
        writer_pending_.store(false, std::memory_order_release);
    }

private:
    struct alignas(64) Slot {
        std::atomic<uint32_t> count{0};
    };

    std::shared_mutex mutex_;
    Slot reader_slots_[kReaderSlots];
    std::atomic<bool> writer_pending_{false};
};

}  // namespace vsag
