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

#include <vsag/engine.h>
#include <vsag/factory.h>
#include <vsag/filter.h>
#include <vsag/index.h>
#include <vsag/logger.h>
#include <vsag/options.h>
#include <vsag/resource.h>
#include <vsag/thread_pool.h>
#include <vsag/vsag_c_api.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <functional>
#include <future>
#include <istream>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <utility>
#include <vector>

struct vsag_error {
    int32_t code;
    std::string message;
};

struct vsag_dataset {
    int64_t num_elements{0};
    int64_t dim{0};
    const int64_t* ids{nullptr};
    const float* float32_vectors{nullptr};
    const uint16_t* float16_vectors{nullptr};
    const int8_t* int8_vectors{nullptr};
    const uint32_t* index_pointers{nullptr};
    const uint32_t* indices{nullptr};
    const float* values{nullptr};
};

struct vsag_filter {
    vsag::FilterPtr filter;
};

struct vsag_engine_options {
    std::string allocator_name;
    vsag_allocate_fn allocate_fn{nullptr};
    vsag_deallocate_fn deallocate_fn{nullptr};
    vsag_reallocate_fn reallocate_fn{nullptr};
    void* allocator_user_data{nullptr};
    vsag_submit_fn submit_fn{nullptr};
    void* pool_user_data{nullptr};
};

struct vsag_engine {
    std::shared_ptr<void> state;  // EngineState
};

struct vsag_index {
    std::shared_ptr<vsag::Index> index;
    // Keeps the engine's host resources alive for as long as the index exists.
    std::shared_ptr<void> owner;
};

struct vsag_search_result {
    vsag::DatasetPtr dataset;
    int64_t count{0};
    // Rendered on first request: most callers never read the statistics.
    std::once_flag statistics_once;
    std::string statistics;
};

namespace {

// Returned when allocating an error object fails; vsag_error_destroy() never frees it.
vsag_error*
out_of_memory_error() {
    static vsag_error error{VSAG_NO_ENOUGH_MEMORY, "out of memory"};
    return &error;
}

vsag_error_t*
make_error(int32_t code, const std::string& message) noexcept {
    try {
        return new vsag_error{code, message};
    } catch (...) {
        return out_of_memory_error();
    }
}

vsag_error_t*
make_error(const vsag::Error& error) noexcept {
    return make_error(-static_cast<int32_t>(error.type), error.message);
}

vsag_error_t*
invalid_argument(const std::string& message) noexcept {
    return make_error(VSAG_INVALID_ARGUMENT, message);
}

template <typename Fn>
vsag_error_t*
guarded(Fn&& fn) noexcept {
    try {
        return fn();
    } catch (const std::bad_alloc&) {
        return out_of_memory_error();
    } catch (const std::invalid_argument& e) {
        return invalid_argument(e.what());
    } catch (const std::exception& e) {
        return make_error(VSAG_INTERNAL_ERROR, e.what());
    } catch (...) {
        return make_error(VSAG_UNKNOWN_ERROR, "unknown error");
    }
}

template <typename T>
vsag_error_t*
to_error(const tl::expected<T, vsag::Error>& result) noexcept {
    return result.has_value() ? nullptr : make_error(result.error());
}

vsag_error_t*
require(bool condition, const char* message) noexcept {
    return condition ? nullptr : invalid_argument(message);
}

#define VSAG_REQUIRE(condition, message)                               \
    do {                                                               \
        if (vsag_error_t* _vsag_error = require(condition, message)) { \
            return _vsag_error;                                        \
        }                                                              \
    } while (0)

// A vsag::Dataset built from a handle, plus the sparse rows it points at.
struct DatasetView {
    vsag::DatasetPtr dataset;
    std::vector<vsag::SparseVector> sparse_vectors;
};

DatasetView
make_view(const vsag_dataset_t* handle, bool single_query) {
    if (handle == nullptr) {
        throw std::invalid_argument("dataset is NULL");
    }
    if (handle->num_elements <= 0) {
        throw std::invalid_argument("dataset num_elements must be positive");
    }
    if (single_query and handle->num_elements != 1) {
        throw std::invalid_argument("dataset must hold exactly one vector");
    }
    const bool is_sparse = handle->index_pointers != nullptr;
    const bool is_dense = handle->float32_vectors != nullptr or
                          handle->float16_vectors != nullptr or handle->int8_vectors != nullptr;
    if (not is_sparse and not is_dense) {
        throw std::invalid_argument("dataset has no vectors");
    }
    if (is_dense and handle->dim <= 0) {
        throw std::invalid_argument("dataset dim must be positive");
    }

    DatasetView view;
    view.dataset = vsag::Dataset::Make();
    view.dataset->Owner(false)->NumElements(handle->num_elements);
    if (handle->dim > 0) {
        view.dataset->Dim(handle->dim);
    }
    if (handle->ids != nullptr) {
        view.dataset->Ids(handle->ids);
    }
    if (handle->float32_vectors != nullptr) {
        view.dataset->Float32Vectors(handle->float32_vectors);
    }
    if (handle->float16_vectors != nullptr) {
        view.dataset->Float16Vectors(handle->float16_vectors);
    }
    if (handle->int8_vectors != nullptr) {
        view.dataset->Int8Vectors(handle->int8_vectors);
    }
    if (is_sparse) {
        const auto* pointers = handle->index_pointers;
        if (pointers[0] != 0) {
            throw std::invalid_argument("index_pointers[0] must be 0");
        }
        view.sparse_vectors.resize(handle->num_elements);
        for (int64_t i = 0; i < handle->num_elements; ++i) {
            const uint32_t begin = pointers[i];
            const uint32_t end = pointers[i + 1];
            if (end < begin) {
                throw std::invalid_argument("index_pointers must be non-decreasing");
            }
            if (end > begin and (handle->indices == nullptr or handle->values == nullptr)) {
                throw std::invalid_argument("sparse indices and values must not be NULL");
            }
            auto& row = view.sparse_vectors[i];
            row.len_ = end - begin;
            row.ids_ = const_cast<uint32_t*>(handle->indices) + begin;
            row.vals_ = const_cast<float*>(handle->values) + begin;
        }
        view.dataset->SparseVectors(view.sparse_vectors.data());
    }
    return view;
}

class IdListFilter : public vsag::Filter {
public:
    IdListFilter(const int64_t* ids, uint64_t count, bool allow)
        : ids_(ids, ids + count), allow_(allow) {
        std::sort(ids_.begin(), ids_.end());
        ids_.erase(std::unique(ids_.begin(), ids_.end()), ids_.end());
    }

    [[nodiscard]] bool
    CheckValid(int64_t id) const override {
        return std::binary_search(ids_.begin(), ids_.end(), id) == allow_;
    }

private:
    std::vector<int64_t> ids_;
    bool allow_;
};

class BitmapFilter : public vsag::Filter {
public:
    BitmapFilter(const uint8_t* bitmap, uint64_t num_ids) : bitmap_(bitmap), num_ids_(num_ids) {
    }

    [[nodiscard]] bool
    CheckValid(int64_t id) const override {
        if (id < 0 or static_cast<uint64_t>(id) >= num_ids_) {
            return false;
        }
        return ((bitmap_[id >> 3] >> (id & 7)) & 1) != 0;
    }

private:
    const uint8_t* bitmap_;
    uint64_t num_ids_;
};

class IdRangeFilter : public vsag::Filter {
public:
    IdRangeFilter(int64_t min_id, int64_t max_id) : min_id_(min_id), max_id_(max_id) {
    }

    [[nodiscard]] bool
    CheckValid(int64_t id) const override {
        return id >= min_id_ and id <= max_id_;
    }

private:
    int64_t min_id_;
    int64_t max_id_;
};

vsag_error_t*
wrap_filter(vsag::FilterPtr filter, vsag_filter_t** out_filter) {
    auto* handle = new vsag_filter{std::move(filter)};
    *out_filter = handle;
    return nullptr;
}

class CallbackFilter : public vsag::Filter {
public:
    CallbackFilter(vsag_filter_fn filter_fn, void* user_data)
        : filter_fn_(filter_fn), user_data_(user_data) {
    }

    [[nodiscard]] bool
    CheckValid(int64_t id) const override {
        return filter_fn_(user_data_, id) != 0;
    }

private:
    vsag_filter_fn filter_fn_;
    void* user_data_;
};

class CallbackAllocator : public vsag::Allocator {
public:
    CallbackAllocator(std::string name,
                      vsag_allocate_fn allocate_fn,
                      vsag_deallocate_fn deallocate_fn,
                      vsag_reallocate_fn reallocate_fn,
                      void* user_data)
        : name_(std::move(name)),
          allocate_fn_(allocate_fn),
          deallocate_fn_(deallocate_fn),
          reallocate_fn_(reallocate_fn),
          user_data_(user_data) {
    }

    std::string
    Name() override {
        return name_;
    }

    void*
    Allocate(uint64_t size) override {
        return allocate_fn_(user_data_, size);
    }

    void
    Deallocate(void* p) override {
        deallocate_fn_(user_data_, p);
    }

    void*
    Reallocate(void* p, uint64_t size) override {
        return reallocate_fn_(user_data_, p, size);
    }

private:
    std::string name_;
    vsag_allocate_fn allocate_fn_;
    vsag_deallocate_fn deallocate_fn_;
    vsag_reallocate_fn reallocate_fn_;
    void* user_data_;
};

// Hands tasks to the host pool. The host owns pool sizing, so the size and queue limits are
// ignored; pending tasks are counted so WaitUntilEmpty() and destruction can wait for them.
class CallbackThreadPool : public vsag::ThreadPool {
public:
    CallbackThreadPool(vsag_submit_fn submit_fn, void* user_data)
        : submit_fn_(submit_fn), user_data_(user_data) {
    }

    ~CallbackThreadPool() override {
        wait_idle();
    }

    void
    WaitUntilEmpty() override {
        wait_idle();
    }

    void
    SetQueueSizeLimit(std::uint64_t /*limit*/) override {
    }

    void
    SetPoolSize(std::uint64_t /*limit*/) override {
    }

    std::future<void>
    Enqueue(std::function<void(void)> task) override {
        auto* job = new Job{this, std::packaged_task<void()>(std::move(task))};
        auto future = job->task.get_future();
        {
            std::lock_guard lock(mutex_);
            ++pending_;
        }
        if (submit_fn_(user_data_, &CallbackThreadPool::run, job) != 0) {
            run(job);
        }
        return future;
    }

private:
    void
    wait_idle() {
        std::unique_lock lock(mutex_);
        idle_.wait(lock, [this]() { return pending_ == 0; });
    }

    struct Job {
        CallbackThreadPool* pool;
        std::packaged_task<void()> task;
    };

    static void
    run(void* task_data) {
        std::unique_ptr<Job> job(static_cast<Job*>(task_data));
        job->task();
        auto* pool = job->pool;
        std::lock_guard lock(pool->mutex_);
        if (--pool->pending_ == 0) {
            pool->idle_.notify_all();
        }
    }

    vsag_submit_fn submit_fn_;
    void* user_data_;
    std::mutex mutex_;
    std::condition_variable idle_;
    uint64_t pending_{0};
};

// Host resources and the engine built on them; shared with every index the engine creates.
struct EngineState {
    std::shared_ptr<vsag::Allocator> allocator;
    std::shared_ptr<vsag::ThreadPool> thread_pool;
    std::unique_ptr<vsag::Resource> resource;
    std::unique_ptr<vsag::Engine> engine;
};

struct LogSink {
    vsag_log_fn log_fn;
    void* user_data;
};

// Forwards records to the host callback, or to the logger that was active before it was
// installed when no callback is set. It is installed once and never removed, so replacing the
// callback only swaps `sink_` under a mutex and never touches Options while other threads log.
class CallbackLogger : public vsag::Logger {
public:
    explicit CallbackLogger(vsag::Logger* fallback) : fallback_(fallback) {
    }

    void
    SetSink(vsag_log_fn log_fn, void* user_data) {
        auto sink = std::make_shared<const LogSink>(LogSink{log_fn, user_data});
        std::lock_guard lock(sink_mutex_);
        sink_ = std::move(sink);
    }

    void
    SetLevel(Level log_level) override {
        level_.store(log_level);
        fallback_->SetLevel(log_level);
    }

    void
    Trace(const std::string& msg) override {
        emit(kTRACE, msg);
    }

    void
    Debug(const std::string& msg) override {
        emit(kDEBUG, msg);
    }

    void
    Info(const std::string& msg) override {
        emit(kINFO, msg);
    }

    void
    Warn(const std::string& msg) override {
        emit(kWARN, msg);
    }

    void
    Error(const std::string& msg) override {
        emit(kERR, msg);
    }

    void
    Critical(const std::string& msg) override {
        emit(kCRITICAL, msg);
    }

private:
    void
    emit(int level, const std::string& msg) {
        std::shared_ptr<const LogSink> sink;
        {
            std::lock_guard lock(sink_mutex_);
            sink = sink_;
        }
        if (sink == nullptr or sink->log_fn == nullptr) {
            forward_to_fallback(level, msg);
        } else if (level >= level_.load()) {
            sink->log_fn(sink->user_data, level, msg.c_str());
        }
    }

    void
    forward_to_fallback(int level, const std::string& msg) {
        switch (level) {
            case kTRACE:
                fallback_->Trace(msg);
                break;
            case kDEBUG:
                fallback_->Debug(msg);
                break;
            case kINFO:
                fallback_->Info(msg);
                break;
            case kWARN:
                fallback_->Warn(msg);
                break;
            case kERR:
                fallback_->Error(msg);
                break;
            default:
                fallback_->Critical(msg);
                break;
        }
    }

    vsag::Logger* fallback_;
    std::mutex sink_mutex_;
    std::shared_ptr<const LogSink> sink_;
    std::atomic<int> level_{kINFO};
};

std::mutex logger_mutex;
int32_t log_level = -1;                     // -1: never set through this API
CallbackLogger* callback_logger = nullptr;  // intentionally leaked; see CallbackLogger

// Presents a host reader as a seekable std::istream. A failing callback ends the stream, which
// the index reports as a read error.
class ReaderStreamBuf : public std::streambuf {
public:
    ReaderStreamBuf(vsag_read_fn read_fn, vsag_size_fn size_fn, void* user_data)
        : read_fn_(read_fn), size_fn_(size_fn), user_data_(user_data) {
    }

protected:
    std::streamsize
    xsgetn(char* dest, std::streamsize count) override {
        if (count <= 0) {
            return 0;
        }
        const auto size = static_cast<uint64_t>(count);
        if (read_fn_(user_data_, position_, size, dest) != 0) {
            return 0;
        }
        position_ += size;
        return count;
    }

    pos_type
    seekoff(off_type off, std::ios_base::seekdir way, std::ios_base::openmode which) override {
        if (which != std::ios_base::in) {
            return {-1};
        }
        int64_t base = 0;
        if (way == std::ios_base::cur) {
            base = static_cast<int64_t>(position_);
        } else if (way == std::ios_base::end) {
            if (size_fn_ == nullptr) {
                return {-1};
            }
            base = static_cast<int64_t>(size_fn_(user_data_));
        }
        const int64_t target = base + off;
        if (target < 0) {
            return {-1};
        }
        position_ = static_cast<uint64_t>(target);
        return {static_cast<off_type>(target)};
    }

    pos_type
    seekpos(pos_type pos, std::ios_base::openmode which) override {
        return seekoff(off_type(pos), std::ios_base::beg, which);
    }

private:
    vsag_read_fn read_fn_;
    vsag_size_fn size_fn_;
    void* user_data_;
    uint64_t position_{0};
};

vsag_error_t*
make_result(const tl::expected<vsag::DatasetPtr, vsag::Error>& searched,
            vsag_search_result_t** out_result) {
    if (not searched.has_value()) {
        return make_error(searched.error());
    }
    auto result = std::make_unique<vsag_search_result>();
    result->dataset = searched.value();
    result->count = result->dataset->GetDim();
    *out_result = result.release();
    return nullptr;
}

}  // namespace

extern "C" {

uint32_t
vsag_c_api_version(void) {
    return VSAG_C_API_VERSION;
}

int32_t
vsag_error_code(const vsag_error_t* error) {
    return error == nullptr ? VSAG_SUCCESS : error->code;
}

const char*
vsag_error_message(const vsag_error_t* error) {
    return error == nullptr ? "" : error->message.c_str();
}

void
vsag_error_destroy(vsag_error_t* error) {
    if (error != out_of_memory_error()) {
        delete error;
    }
}

vsag_error_t*
vsag_set_log_level(int32_t level) {
    return guarded([&]() -> vsag_error_t* {
        VSAG_REQUIRE(static_cast<uint32_t>(level) <= VSAG_LOG_OFF, "unsupported log level");
        std::lock_guard lock(logger_mutex);
        log_level = level;
        vsag::Options::Instance().logger()->SetLevel(static_cast<vsag::Logger::Level>(level));
        return nullptr;
    });
}

vsag_error_t*
vsag_set_logger(vsag_log_fn log_fn, void* user_data) {
    return guarded([&]() -> vsag_error_t* {
        std::lock_guard lock(logger_mutex);
        if (callback_logger == nullptr) {
            if (log_fn == nullptr) {
                return nullptr;  // still using the built-in logger
            }
            auto& options = vsag::Options::Instance();
            callback_logger = new CallbackLogger(options.logger());
            if (log_level >= 0) {
                callback_logger->SetLevel(static_cast<vsag::Logger::Level>(log_level));
            }
            callback_logger->SetSink(log_fn, user_data);
            options.set_logger(callback_logger);
            return nullptr;
        }
        callback_logger->SetSink(log_fn, user_data);
        return nullptr;
    });
}

vsag_error_t*
vsag_dataset_create(vsag_dataset_t** out_dataset) {
    return guarded([&]() -> vsag_error_t* {
        VSAG_REQUIRE(out_dataset != nullptr, "out_dataset is NULL");
        *out_dataset = new vsag_dataset();
        return nullptr;
    });
}

void
vsag_dataset_destroy(vsag_dataset_t* dataset) {
    delete dataset;
}

vsag_error_t*
vsag_dataset_set_num_elements(vsag_dataset_t* dataset, int64_t num_elements) {
    VSAG_REQUIRE(dataset != nullptr, "dataset is NULL");
    dataset->num_elements = num_elements;
    return nullptr;
}

vsag_error_t*
vsag_dataset_set_dim(vsag_dataset_t* dataset, int64_t dim) {
    VSAG_REQUIRE(dataset != nullptr, "dataset is NULL");
    dataset->dim = dim;
    return nullptr;
}

vsag_error_t*
vsag_dataset_set_ids(vsag_dataset_t* dataset, const int64_t* ids) {
    VSAG_REQUIRE(dataset != nullptr, "dataset is NULL");
    dataset->ids = ids;
    return nullptr;
}

vsag_error_t*
vsag_dataset_set_float32_vectors(vsag_dataset_t* dataset, const float* vectors) {
    VSAG_REQUIRE(dataset != nullptr, "dataset is NULL");
    dataset->float32_vectors = vectors;
    return nullptr;
}

vsag_error_t*
vsag_dataset_set_float16_vectors(vsag_dataset_t* dataset, const uint16_t* vectors) {
    VSAG_REQUIRE(dataset != nullptr, "dataset is NULL");
    dataset->float16_vectors = vectors;
    return nullptr;
}

vsag_error_t*
vsag_dataset_set_int8_vectors(vsag_dataset_t* dataset, const int8_t* vectors) {
    VSAG_REQUIRE(dataset != nullptr, "dataset is NULL");
    dataset->int8_vectors = vectors;
    return nullptr;
}

vsag_error_t*
vsag_dataset_set_sparse_vectors(vsag_dataset_t* dataset,
                                const uint32_t* index_pointers,
                                const uint32_t* indices,
                                const float* values) {
    VSAG_REQUIRE(dataset != nullptr, "dataset is NULL");
    dataset->index_pointers = index_pointers;
    dataset->indices = indices;
    dataset->values = values;
    return nullptr;
}

vsag_error_t*
vsag_filter_create_id_list(const int64_t* ids,
                           uint64_t count,
                           int32_t mode,
                           vsag_filter_t** out_filter) {
    return guarded([&]() -> vsag_error_t* {
        VSAG_REQUIRE(out_filter != nullptr, "out_filter is NULL");
        VSAG_REQUIRE(ids != nullptr or count == 0, "ids is NULL");
        VSAG_REQUIRE(static_cast<uint32_t>(mode) <= VSAG_FILTER_DENY, "unsupported filter mode");
        return wrap_filter(std::make_shared<IdListFilter>(ids, count, mode == VSAG_FILTER_ALLOW),
                           out_filter);
    });
}

vsag_error_t*
vsag_filter_create_bitmap(const uint8_t* bitmap, uint64_t num_ids, vsag_filter_t** out_filter) {
    return guarded([&]() -> vsag_error_t* {
        VSAG_REQUIRE(out_filter != nullptr, "out_filter is NULL");
        VSAG_REQUIRE(bitmap != nullptr or num_ids == 0, "bitmap is NULL");
        return wrap_filter(std::make_shared<BitmapFilter>(bitmap, num_ids), out_filter);
    });
}

vsag_error_t*
vsag_filter_create_id_range(int64_t min_id, int64_t max_id, vsag_filter_t** out_filter) {
    return guarded([&]() -> vsag_error_t* {
        VSAG_REQUIRE(out_filter != nullptr, "out_filter is NULL");
        return wrap_filter(std::make_shared<IdRangeFilter>(min_id, max_id), out_filter);
    });
}

vsag_error_t*
vsag_filter_create_callback(vsag_filter_fn filter_fn, void* user_data, vsag_filter_t** out_filter) {
    return guarded([&]() -> vsag_error_t* {
        VSAG_REQUIRE(out_filter != nullptr, "out_filter is NULL");
        VSAG_REQUIRE(filter_fn != nullptr, "filter_fn is NULL");
        return wrap_filter(std::make_shared<CallbackFilter>(filter_fn, user_data), out_filter);
    });
}

void
vsag_filter_destroy(vsag_filter_t* filter) {
    delete filter;
}

int64_t
vsag_search_result_count(const vsag_search_result_t* result) {
    return result == nullptr ? 0 : result->count;
}

const int64_t*
vsag_search_result_ids(const vsag_search_result_t* result) {
    return result == nullptr ? nullptr : result->dataset->GetIds();
}

const float*
vsag_search_result_distances(const vsag_search_result_t* result) {
    return result == nullptr ? nullptr : result->dataset->GetDistances();
}

const char*
vsag_search_result_statistics(const vsag_search_result_t* result) {
    if (result == nullptr) {
        return "";
    }
    try {
        auto* mutable_result = const_cast<vsag_search_result_t*>(result);
        std::call_once(mutable_result->statistics_once, [&]() {
            mutable_result->statistics = mutable_result->dataset->GetStatistics();
        });
    } catch (...) {  // NOLINT(bugprone-empty-catch)
        // Statistics are optional; report none rather than throw across the C boundary.
    }
    return result->statistics.c_str();
}

void
vsag_search_result_destroy(vsag_search_result_t* result) {
    delete result;
}

vsag_error_t*
vsag_index_create(const char* index_name, const char* index_param, vsag_index_t** out_index) {
    return guarded([&]() -> vsag_error_t* {
        VSAG_REQUIRE(out_index != nullptr, "out_index is NULL");
        *out_index = nullptr;
        VSAG_REQUIRE(index_name != nullptr and index_param != nullptr,
                     "index name and parameters must not be NULL");
        auto created = vsag::Factory::CreateIndex(index_name, index_param);
        if (not created.has_value()) {
            return make_error(created.error());
        }
        *out_index = new vsag_index{created.value()};
        return nullptr;
    });
}

vsag_error_t*
vsag_engine_options_create(vsag_engine_options_t** out_options) {
    return guarded([&]() -> vsag_error_t* {
        VSAG_REQUIRE(out_options != nullptr, "out_options is NULL");
        *out_options = new vsag_engine_options();
        return nullptr;
    });
}

void
vsag_engine_options_destroy(vsag_engine_options_t* options) {
    delete options;
}

vsag_error_t*
vsag_engine_options_set_allocator(vsag_engine_options_t* options,
                                  const char* name,
                                  vsag_allocate_fn allocate_fn,
                                  vsag_deallocate_fn deallocate_fn,
                                  vsag_reallocate_fn reallocate_fn,
                                  void* user_data) {
    return guarded([&]() -> vsag_error_t* {
        VSAG_REQUIRE(options != nullptr, "options is NULL");
        VSAG_REQUIRE(
            allocate_fn != nullptr and deallocate_fn != nullptr and reallocate_fn != nullptr,
            "allocator callbacks must not be NULL");
        options->allocator_name = name == nullptr ? "host_allocator" : name;
        options->allocate_fn = allocate_fn;
        options->deallocate_fn = deallocate_fn;
        options->reallocate_fn = reallocate_fn;
        options->allocator_user_data = user_data;
        return nullptr;
    });
}

vsag_error_t*
vsag_engine_options_set_thread_pool(vsag_engine_options_t* options,
                                    vsag_submit_fn submit_fn,
                                    void* user_data) {
    return guarded([&]() -> vsag_error_t* {
        VSAG_REQUIRE(options != nullptr, "options is NULL");
        VSAG_REQUIRE(submit_fn != nullptr, "submit_fn is NULL");
        options->submit_fn = submit_fn;
        options->pool_user_data = user_data;
        return nullptr;
    });
}

vsag_error_t*
vsag_engine_create(const vsag_engine_options_t* options, vsag_engine_t** out_engine) {
    return guarded([&]() -> vsag_error_t* {
        VSAG_REQUIRE(out_engine != nullptr, "out_engine is NULL");
        *out_engine = nullptr;
        auto state = std::make_shared<EngineState>();
        if (options != nullptr and options->allocate_fn != nullptr) {
            state->allocator = std::make_shared<CallbackAllocator>(options->allocator_name,
                                                                   options->allocate_fn,
                                                                   options->deallocate_fn,
                                                                   options->reallocate_fn,
                                                                   options->allocator_user_data);
        } else {
            state->allocator = vsag::Engine::CreateDefaultAllocator();
        }
        if (options != nullptr and options->submit_fn != nullptr) {
            state->thread_pool =
                std::make_shared<CallbackThreadPool>(options->submit_fn, options->pool_user_data);
        } else {
            auto pool = vsag::Engine::CreateThreadPool(
                static_cast<uint32_t>(vsag::Options::Instance().num_threads_building()));
            if (not pool.has_value()) {
                return make_error(pool.error());
            }
            state->thread_pool = pool.value();
        }
        state->resource = std::make_unique<vsag::Resource>(state->allocator, state->thread_pool);
        state->engine = std::make_unique<vsag::Engine>(state->resource.get());
        *out_engine = new vsag_engine{std::move(state)};
        return nullptr;
    });
}

void
vsag_engine_destroy(vsag_engine_t* engine) {
    delete engine;
}

vsag_error_t*
vsag_engine_create_index(vsag_engine_t* engine,
                         const char* index_name,
                         const char* index_param,
                         vsag_index_t** out_index) {
    return guarded([&]() -> vsag_error_t* {
        VSAG_REQUIRE(out_index != nullptr, "out_index is NULL");
        *out_index = nullptr;
        VSAG_REQUIRE(engine != nullptr, "engine is NULL");
        VSAG_REQUIRE(index_name != nullptr and index_param != nullptr,
                     "index name and parameters must not be NULL");
        auto* state = static_cast<EngineState*>(engine->state.get());
        auto created = state->engine->CreateIndex(index_name, index_param);
        if (not created.has_value()) {
            return make_error(created.error());
        }
        *out_index = new vsag_index{created.value(), engine->state};
        return nullptr;
    });
}

void
vsag_index_destroy(vsag_index_t* index) {
    delete index;
}

vsag_error_t*
vsag_index_build(vsag_index_t* index, const vsag_dataset_t* base) {
    return guarded([&]() -> vsag_error_t* {
        VSAG_REQUIRE(index != nullptr, "index is NULL");
        auto view = make_view(base, false);
        return to_error(index->index->Build(view.dataset));
    });
}

vsag_error_t*
vsag_index_train(vsag_index_t* index, const vsag_dataset_t* data) {
    return guarded([&]() -> vsag_error_t* {
        VSAG_REQUIRE(index != nullptr, "index is NULL");
        auto view = make_view(data, false);
        return to_error(index->index->Train(view.dataset));
    });
}

vsag_error_t*
vsag_index_add(vsag_index_t* index, const vsag_dataset_t* base, uint64_t* out_failed_count) {
    return guarded([&]() -> vsag_error_t* {
        VSAG_REQUIRE(index != nullptr, "index is NULL");
        auto view = make_view(base, false);
        auto added = index->index->Add(view.dataset);
        if (not added.has_value()) {
            return make_error(added.error());
        }
        if (out_failed_count != nullptr) {
            *out_failed_count = added.value().size();
        }
        return nullptr;
    });
}

vsag_error_t*
vsag_index_remove(vsag_index_t* index,
                  const int64_t* ids,
                  uint64_t count,
                  uint64_t* out_removed_count) {
    return guarded([&]() -> vsag_error_t* {
        VSAG_REQUIRE(index != nullptr, "index is NULL");
        VSAG_REQUIRE(ids != nullptr or count == 0, "ids is NULL");
        auto removed = index->index->Remove(std::vector<int64_t>(ids, ids + count));
        if (not removed.has_value()) {
            return make_error(removed.error());
        }
        if (out_removed_count != nullptr) {
            *out_removed_count = removed.value();
        }
        return nullptr;
    });
}

vsag_error_t*
vsag_index_update_ids(vsag_index_t* index,
                      const int64_t* old_ids,
                      const int64_t* new_ids,
                      uint64_t count) {
    return guarded([&]() -> vsag_error_t* {
        VSAG_REQUIRE(index != nullptr, "index is NULL");
        VSAG_REQUIRE((old_ids != nullptr and new_ids != nullptr) or count == 0, "ids is NULL");
        for (uint64_t i = 0; i < count; ++i) {
            auto updated = index->index->UpdateId(old_ids[i], new_ids[i]);
            if (not updated.has_value()) {
                return make_error(updated.error());
            }
            if (not updated.value()) {
                return make_error(VSAG_INTERNAL_ERROR, "failed to update id");
            }
        }
        return nullptr;
    });
}

vsag_error_t*
vsag_index_update_vector(vsag_index_t* index,
                         int64_t id,
                         const vsag_dataset_t* vector,
                         int32_t force_update) {
    return guarded([&]() -> vsag_error_t* {
        VSAG_REQUIRE(index != nullptr, "index is NULL");
        auto view = make_view(vector, true);
        auto updated = index->index->UpdateVector(id, view.dataset, force_update != 0);
        if (not updated.has_value()) {
            return make_error(updated.error());
        }
        if (not updated.value()) {
            return make_error(VSAG_INTERNAL_ERROR, "failed to update vector");
        }
        return nullptr;
    });
}

vsag_error_t*
vsag_index_knn_search(const vsag_index_t* index,
                      const vsag_dataset_t* query,
                      int64_t k,
                      const char* parameters,
                      const vsag_filter_t* filter,
                      vsag_search_result_t** out_result) {
    return guarded([&]() -> vsag_error_t* {
        VSAG_REQUIRE(out_result != nullptr, "out_result is NULL");
        *out_result = nullptr;
        VSAG_REQUIRE(index != nullptr, "index is NULL");
        VSAG_REQUIRE(parameters != nullptr, "parameters is NULL");
        VSAG_REQUIRE(k > 0, "knn_search requires k >= 1");
        auto view = make_view(query, true);
        const auto& searched =
            filter == nullptr
                ? index->index->KnnSearch(view.dataset, k, parameters)
                : index->index->KnnSearch(view.dataset, k, parameters, filter->filter);
        return make_result(searched, out_result);
    });
}

vsag_error_t*
vsag_index_range_search(const vsag_index_t* index,
                        const vsag_dataset_t* query,
                        float radius,
                        int64_t limit,
                        const char* parameters,
                        const vsag_filter_t* filter,
                        vsag_search_result_t** out_result) {
    return guarded([&]() -> vsag_error_t* {
        VSAG_REQUIRE(out_result != nullptr, "out_result is NULL");
        *out_result = nullptr;
        VSAG_REQUIRE(index != nullptr, "index is NULL");
        VSAG_REQUIRE(parameters != nullptr, "parameters is NULL");
        VSAG_REQUIRE(limit == -1 or limit > 0, "range_search limit must be -1 or positive");
        auto view = make_view(query, true);
        const auto& searched =
            filter == nullptr ? index->index->RangeSearch(view.dataset, radius, parameters, limit)
                              : index->index->RangeSearch(
                                    view.dataset, radius, parameters, filter->filter, limit);
        return make_result(searched, out_result);
    });
}

vsag_error_t*
vsag_index_knn_search_float32(const vsag_index_t* index,
                              const float* queries,
                              uint64_t num_queries,
                              uint64_t dim,
                              int64_t k,
                              const char* parameters,
                              const vsag_filter_t* filter,
                              int64_t* out_ids,
                              float* out_distances,
                              int64_t* out_counts) {
    return guarded([&]() -> vsag_error_t* {
        VSAG_REQUIRE(index != nullptr, "index is NULL");
        VSAG_REQUIRE(queries != nullptr and parameters != nullptr, "query or parameters is NULL");
        VSAG_REQUIRE(out_ids != nullptr and out_distances != nullptr, "output buffer is NULL");
        VSAG_REQUIRE(k > 0, "knn_search requires k >= 1");
        VSAG_REQUIRE(dim > 0, "dim must be positive");
        const std::string search_parameters(parameters);
        for (uint64_t i = 0; i < num_queries; ++i) {
            auto query = vsag::Dataset::Make();
            query->Owner(false)
                ->NumElements(1)
                ->Dim(static_cast<int64_t>(dim))
                ->Float32Vectors(queries + i * dim);
            const auto& searched =
                filter == nullptr
                    ? index->index->KnnSearch(query, k, search_parameters)
                    : index->index->KnnSearch(query, k, search_parameters, filter->filter);
            if (not searched.has_value()) {
                return make_error(searched.error());
            }
            const auto count = std::min<int64_t>(searched.value()->GetDim(), k);
            std::memcpy(out_ids + i * k, searched.value()->GetIds(), count * sizeof(int64_t));
            std::memcpy(
                out_distances + i * k, searched.value()->GetDistances(), count * sizeof(float));
            if (out_counts != nullptr) {
                out_counts[i] = count;
            }
        }
        return nullptr;
    });
}

vsag_error_t*
vsag_index_calc_distances_by_ids(const vsag_index_t* index,
                                 const vsag_dataset_t* query,
                                 const int64_t* ids,
                                 uint64_t count,
                                 float* out_distances) {
    return guarded([&]() -> vsag_error_t* {
        VSAG_REQUIRE(index != nullptr, "index is NULL");
        VSAG_REQUIRE((ids != nullptr and out_distances != nullptr) or count == 0,
                     "ids or out_distances is NULL");
        auto view = make_view(query, true);
        auto calculated =
            index->index->CalcDistancesById(view.dataset, ids, static_cast<int64_t>(count));
        if (not calculated.has_value()) {
            return make_error(calculated.error());
        }
        if (count > 0) {
            std::memcpy(out_distances, calculated.value()->GetDistances(), count * sizeof(float));
        }
        return nullptr;
    });
}

vsag_error_t*
vsag_index_get_vectors_by_ids(const vsag_index_t* index,
                              const int64_t* ids,
                              uint64_t count,
                              float* out_vectors) {
    return guarded([&]() -> vsag_error_t* {
        VSAG_REQUIRE(index != nullptr, "index is NULL");
        VSAG_REQUIRE(ids != nullptr and out_vectors != nullptr and count > 0,
                     "ids or out_vectors is NULL, or count is 0");
        auto fetched = index->index->GetRawVectorByIds(ids, static_cast<int64_t>(count));
        if (not fetched.has_value()) {
            return make_error(fetched.error());
        }
        const auto* vectors = fetched.value()->GetFloat32Vectors();
        if (vectors == nullptr) {
            return make_error(VSAG_INTERNAL_ERROR, "index returned no float32 vectors");
        }
        std::memcpy(out_vectors, vectors, count * fetched.value()->GetDim() * sizeof(float));
        return nullptr;
    });
}

vsag_error_t*
vsag_index_clone(const vsag_index_t* index, vsag_index_t** out_index) {
    return guarded([&]() -> vsag_error_t* {
        VSAG_REQUIRE(out_index != nullptr, "out_index is NULL");
        *out_index = nullptr;
        VSAG_REQUIRE(index != nullptr, "index is NULL");
        auto cloned = index->index->Clone();
        if (not cloned.has_value()) {
            return make_error(cloned.error());
        }
        *out_index = new vsag_index{cloned.value(), index->owner};
        return nullptr;
    });
}

vsag_error_t*
vsag_index_export_model(const vsag_index_t* index, vsag_index_t** out_index) {
    return guarded([&]() -> vsag_error_t* {
        VSAG_REQUIRE(out_index != nullptr, "out_index is NULL");
        *out_index = nullptr;
        VSAG_REQUIRE(index != nullptr, "index is NULL");
        auto exported = index->index->ExportModel();
        if (not exported.has_value()) {
            return make_error(exported.error());
        }
        *out_index = new vsag_index{exported.value(), index->owner};
        return nullptr;
    });
}

vsag_error_t*
vsag_index_serialize_file(const vsag_index_t* index, const char* file_path) {
    return guarded([&]() -> vsag_error_t* {
        VSAG_REQUIRE(index != nullptr, "index is NULL");
        VSAG_REQUIRE(file_path != nullptr, "file_path is NULL");
        std::ofstream file(file_path, std::ios::binary);
        if (not file.is_open()) {
            return make_error(VSAG_INTERNAL_ERROR,
                              std::string("failed to open file for writing: ") + file_path);
        }
        return to_error(index->index->Serialize(file));
    });
}

vsag_error_t*
vsag_index_deserialize_file(vsag_index_t* index, const char* file_path) {
    return guarded([&]() -> vsag_error_t* {
        VSAG_REQUIRE(index != nullptr, "index is NULL");
        VSAG_REQUIRE(file_path != nullptr, "file_path is NULL");
        std::ifstream file(file_path, std::ios::binary);
        if (not file.is_open()) {
            return make_error(VSAG_MISSING_FILE,
                              std::string("failed to open file for reading: ") + file_path);
        }
        return to_error(index->index->Deserialize(file));
    });
}

vsag_error_t*
vsag_index_serialize_writer(const vsag_index_t* index, vsag_write_fn write_fn, void* user_data) {
    return guarded([&]() -> vsag_error_t* {
        VSAG_REQUIRE(index != nullptr, "index is NULL");
        VSAG_REQUIRE(write_fn != nullptr, "write_fn is NULL");
        auto failed = std::make_shared<bool>(false);
        auto serialized =
            index->index->Serialize([=](uint64_t offset, uint64_t size, const void* data) {
                if (*failed) {
                    return;
                }
                if (write_fn(user_data, offset, size, data) != 0) {
                    *failed = true;
                    throw std::runtime_error("write callback failed");
                }
            });
        if (*failed) {
            return make_error(VSAG_INTERNAL_ERROR, "write callback failed");
        }
        return to_error(serialized);
    });
}

vsag_error_t*
vsag_index_deserialize_reader(vsag_index_t* index,
                              vsag_read_fn read_fn,
                              vsag_size_fn size_fn,
                              void* user_data) {
    return guarded([&]() -> vsag_error_t* {
        VSAG_REQUIRE(index != nullptr, "index is NULL");
        VSAG_REQUIRE(read_fn != nullptr and size_fn != nullptr, "read_fn or size_fn is NULL");
        ReaderStreamBuf buffer(read_fn, size_fn, user_data);
        std::istream stream(&buffer);
        return to_error(index->index->Deserialize(stream));
    });
}

vsag_error_t*
vsag_index_get_num_elements(const vsag_index_t* index, int64_t* out_num_elements) {
    return guarded([&]() -> vsag_error_t* {
        VSAG_REQUIRE(index != nullptr, "index is NULL");
        VSAG_REQUIRE(out_num_elements != nullptr, "out_num_elements is NULL");
        *out_num_elements = index->index->GetNumElements();
        return nullptr;
    });
}

vsag_error_t*
vsag_index_get_memory_usage(const vsag_index_t* index, int64_t* out_memory_usage) {
    return guarded([&]() -> vsag_error_t* {
        VSAG_REQUIRE(index != nullptr, "index is NULL");
        VSAG_REQUIRE(out_memory_usage != nullptr, "out_memory_usage is NULL");
        *out_memory_usage = static_cast<int64_t>(index->index->GetMemoryUsage());
        return nullptr;
    });
}

vsag_error_t*
vsag_index_check_id_exist(const vsag_index_t* index, int64_t id, int32_t* out_exist) {
    return guarded([&]() -> vsag_error_t* {
        VSAG_REQUIRE(index != nullptr, "index is NULL");
        VSAG_REQUIRE(out_exist != nullptr, "out_exist is NULL");
        *out_exist = index->index->CheckIdExist(id) ? 1 : 0;
        return nullptr;
    });
}

vsag_error_t*
vsag_index_get_min_max_id(const vsag_index_t* index, int64_t* out_min_id, int64_t* out_max_id) {
    return guarded([&]() -> vsag_error_t* {
        VSAG_REQUIRE(index != nullptr, "index is NULL");
        VSAG_REQUIRE(out_min_id != nullptr and out_max_id != nullptr, "out id is NULL");
        auto range = index->index->GetMinAndMaxId();
        if (not range.has_value()) {
            return make_error(range.error());
        }
        *out_min_id = range.value().first;
        *out_max_id = range.value().second;
        return nullptr;
    });
}

}  // extern "C"
