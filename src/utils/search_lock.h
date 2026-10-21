// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mutex>
#include <shared_mutex>

namespace vsag {

// A request-local policy. Index mutex objects are never swapped or globally disabled.
// The caller must exclude all index mutation for the duration of an unlocked search.
class SearchLockScope {
public:
    explicit SearchLockScope(bool no_lock) : previous_(no_lock_) {
        no_lock_ = no_lock;
    }

    ~SearchLockScope() {
        no_lock_ = previous_;
    }

    SearchLockScope(const SearchLockScope&) = delete;
    SearchLockScope&
    operator=(const SearchLockScope&) = delete;

    [[nodiscard]] static bool
    IsNoLock() {
        return no_lock_;
    }

private:
    bool previous_;
    inline static thread_local bool no_lock_{false};
};

template <typename Mutex>
[[nodiscard]] std::shared_lock<Mutex>
MakeSearchReadLock(Mutex& mutex) {
    if (SearchLockScope::IsNoLock()) {
        return {};
    }
    return std::shared_lock<Mutex>(mutex);
}

// For immutable filter reads and query-private diagnostic state guarded by std::mutex.
template <typename Mutex>
[[nodiscard]] std::unique_lock<Mutex>
MakeSearchMutexLock(Mutex& mutex) {
    if (SearchLockScope::IsNoLock()) {
        return {};
    }
    return std::unique_lock<Mutex>(mutex);
}

}  // namespace vsag
