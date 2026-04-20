#pragma once

#include <mutex>

class windows_exclusive_slim_lock {
public:
    void lock() { _mutex.lock(); }
    void unlock() { _mutex.unlock(); }
private:
    std::mutex _mutex;
};

class windows_exclusive_slim_lock_guard {
public:
    explicit windows_exclusive_slim_lock_guard(windows_exclusive_slim_lock& lock)
        : _lock(lock) {
        _lock.lock();
    }
    ~windows_exclusive_slim_lock_guard() {
        _lock.unlock();
    }
    windows_exclusive_slim_lock_guard(const windows_exclusive_slim_lock_guard&) = delete;
    windows_exclusive_slim_lock_guard& operator=(const windows_exclusive_slim_lock_guard&) = delete;
private:
    windows_exclusive_slim_lock& _lock;
};
