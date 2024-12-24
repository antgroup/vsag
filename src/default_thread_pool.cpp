//
// Created by root on 12/22/24.
//

#include "default_thread_pool.h"

namespace vsag {

DefaultThreadPool::DefaultThreadPool(std::size_t threads) {
    pool_ = std::make_unique<progschj::ThreadPool>(threads);
}

std::future<void>
DefaultThreadPool::enqueue(std::function<void(void)> func) {
    return pool_->enqueue(func);
}

void
DefaultThreadPool::wait_until_empty() {
    pool_->wait_until_empty();
}

void
DefaultThreadPool::set_pool_size(std::size_t limit) {
    pool_->set_pool_size(limit);
}

void
DefaultThreadPool::set_queue_size_limit(std::size_t limit) {
    pool_->set_queue_size_limit(limit);
}

}  // namespace vsag