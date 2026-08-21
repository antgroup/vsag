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

#include "io/read_cache/page_cache.h"

#include <algorithm>
#include <condition_variable>
#include <exception>
#include <vector>

namespace vsag {

struct PageCache::LoadingPage {
    std::condition_variable cv;
    PagePtr page;
    bool done{false};
    bool stale{false};
};

PageCache::PageCache(uint64_t max_pages) : max_pages_(max_pages) {
}

PagePtr
PageCache::Get(uint64_t page_id) {
    std::scoped_lock<std::mutex> lock(mutex_);
    auto it = pages_.find(page_id);
    if (it == pages_.end()) {
        return nullptr;
    }
    OnAccess(page_id);
    return it->second;
}

PageCache::LoadResult
PageCache::Acquire(uint64_t page_id) {
    std::scoped_lock<std::mutex> lock(mutex_);
    if (const auto iter = pages_.find(page_id); iter != pages_.end()) {
        OnAccess(page_id);
        LoadResult result;
        result.page = iter->second;
        return result;
    }
    if (const auto iter = loading_pages_.find(page_id); iter != loading_pages_.end()) {
        LoadResult result;
        result.handle.state_ = iter->second;
        return result;
    }
    auto state = std::make_shared<LoadingPage>();
    loading_pages_.emplace(page_id, state);
    LoadResult result;
    result.handle.state_ = std::move(state);
    result.should_load = true;
    return result;
}

PagePtr
PageCache::Wait(const LoadHandle& handle) {
    std::unique_lock<std::mutex> lock(mutex_);
    handle.state_->cv.wait(lock, [&handle] { return handle.state_->done; });
    return handle.state_->page;
}

bool
PageCache::IsStale(const LoadHandle& handle) const {
    std::scoped_lock<std::mutex> lock(mutex_);
    return handle.state_->stale;
}

PagePtr
PageCache::Complete(uint64_t page_id, const LoadHandle& handle, PagePtr page, bool success) {
    PagePtr result;
    std::exception_ptr error;
    {
        std::scoped_lock<std::mutex> lock(mutex_);
        const auto state = handle.state_;
        try {
            if (success and not state->stale) {
                result = InsertLocked(page_id, std::move(page));
            }
        } catch (...) {
            error = std::current_exception();
        }
        state->page = result;
        state->done = true;
        if (const auto iter = loading_pages_.find(page_id);
            iter != loading_pages_.end() and iter->second == state) {
            loading_pages_.erase(iter);
        }
    }
    handle.state_->cv.notify_all();
    if (error != nullptr) {
        std::rethrow_exception(error);
    }
    return result;
}

PagePtr
PageCache::Insert(uint64_t page_id, PagePtr page) {
    std::scoped_lock<std::mutex> lock(mutex_);
    return InsertLocked(page_id, std::move(page));
}

PagePtr
PageCache::InsertLocked(uint64_t page_id, PagePtr page) {
    auto existing = pages_.find(page_id);
    if (existing != pages_.end()) {
        OnAccess(page_id);
        return existing->second;
    }
    if (max_pages_ == 0) {
        return page;
    }
    while (pages_.size() >= max_pages_) {
        uint64_t victim = PickVictim();
        if (victim == UINT64_MAX or pages_.find(victim) == pages_.end()) {
            victim = pages_.begin()->first;
        }
        OnRemove(victim);
        pages_.erase(victim);
    }
    pages_[page_id] = std::move(page);
    OnInsert(page_id);
    return pages_[page_id];
}

void
PageCache::Remove(uint64_t page_id) {
    std::scoped_lock<std::mutex> lock(mutex_);
    auto it = pages_.find(page_id);
    if (it != pages_.end()) {
        OnRemove(page_id);
        pages_.erase(it);
    }
    if (const auto loading = loading_pages_.find(page_id); loading != loading_pages_.end()) {
        loading->second->stale = true;
    }
}

void
PageCache::Clear() {
    std::vector<std::shared_ptr<LoadingPage>> states_to_notify;
    {
        std::scoped_lock<std::mutex> lock(mutex_);
        for (const auto& page_pair : pages_) {
            OnRemove(page_pair.first);
        }
        pages_.clear();
        for (const auto& [page_id, state] : loading_pages_) {
            state->stale = true;
            state->done = true;
            states_to_notify.push_back(state);
        }
        loading_pages_.clear();
    }
    for (const auto& state : states_to_notify) {
        state->cv.notify_all();
    }
}

uint64_t
PageCache::Size() const {
    std::scoped_lock<std::mutex> lock(mutex_);
    return pages_.size();
}

}  // namespace vsag
