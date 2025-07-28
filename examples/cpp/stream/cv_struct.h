
#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>

struct CvStruct {
    std::condition_variable cv_;
    std::atomic<bool> pause_signal_{true};
    std::mutex mu_;
};
