// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mutex>
#include <shared_mutex>

namespace vsag {

// A request-local policy. Index mutex objects are never swapped or globally disabled.
// The caller must exclude all index mutation for the duration of an unlocked search,
// including callbacks. Each nested public search installs its own explicit policy and
// restores this one on return; lower-level helpers inherit the active request policy.
class SearchLockScope {
public:
    explicit SearchLockScope(bool no_lock) : previous_(search_no_lock) {
        search_no_lock = no_lock;
    }

    ~SearchLockScope() {
        search_no_lock = previous_;
    }

    SearchLockScope(const SearchLockScope&) = delete;
    SearchLockScope&
    operator=(const SearchLockScope&) = delete;
    SearchLockScope(SearchLockScope&&) = delete;
    SearchLockScope&
    operator=(SearchLockScope&&) = delete;

    [[nodiscard]] static bool
    IsNoLock() {
        return search_no_lock;
    }

private:
    bool previous_;
    inline static thread_local bool search_no_lock{false};
};

template <typename Mutex>
[[nodiscard]] std::shared_lock<Mutex>
make_search_read_lock(Mutex& mutex) {
    if (SearchLockScope::IsNoLock()) {
        return {};
    }
    return std::shared_lock<Mutex>(mutex);
}

}  // namespace vsag
