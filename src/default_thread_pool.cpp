
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

#include "default_thread_pool.h"

namespace vsag {

DefaultThreadPool::DefaultThreadPool(std::size_t threads) {
    pool_ = std::make_unique<progschj::ThreadPool>(threads);
}

std::future<void>
DefaultThreadPool::Enqueue(std::function<void(void)> func) {
    return pool_->enqueue(func);
}

void
DefaultThreadPool::WaitUntilEmpty() {
    pool_->wait_until_empty();
}

void
DefaultThreadPool::SetPoolSize(std::size_t limit) {
    pool_->set_pool_size(limit);
}

void
DefaultThreadPool::SetQueueSizeLimit(std::size_t limit) {
    pool_->set_queue_size_limit(limit);
}

}  // namespace vsag