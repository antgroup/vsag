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

#include <algorithm>
#include <cstdint>
#include <exception>
#include <functional>
#include <future>
#include <istream>
#include <mutex>
#include <vector>

#include "impl/logger/logger.h"
#include "vsag/deserialize_reader.h"
#include "vsag/thread_pool.h"
#include "vsag_exception.h"

namespace vsag {

/// Building blocks for restoring a chunked index body concurrently: moving
/// frame bytes into pre-allocated io extents and propagating the first task
/// failure out of the pool. They carry no index-format knowledge, which is
/// why they live here in src/storage/ rather than beside an algorithm;
/// HGraph's parallel restore (hgraph_parallel_deserialize.cpp) is the only
/// consumer today.
///
/// Allocation contract: consume_chunk_stream and fill_extent_from_reader each
/// allocate one staging buffer per call, bounded by
/// PARALLEL_DESERIALIZE_SCRATCH_SIZE. That suits the intended usage of one call
/// per pool task, where the allocation is amortized against the frame I/O. A
/// caller that needs them in a tighter loop should add an overload taking a
/// caller-owned buffer rather than pay the repeated allocation.

/// Staging buffer size for moving io data bytes into the pre-allocated extents.
///
/// This is the request granularity, not just an allocation size: a task issues
/// ceil(bytes / this) Read calls and the same number of WriteRaw calls, and one
/// buffer of min(bytes, this) is allocated per call. So the value trades three
/// things off.
///
/// Round-trip count wants it large. A DeserializeReader may be network backed,
/// where every Read is a round trip; at 4 MiB a default 128 MiB chunk costs 32
/// of them, at 64 KiB it would cost 2048.
///
/// Concurrent footprint wants it small. Peak staging memory is (threads running
/// a fill task) * min(bytes, this), which is what rules out simply using the
/// chunk size: 32 threads * 128 MiB would stage 4 GiB.
///
/// Cache residency also wants it small, and loses. The buffer is written once
/// by Read and read once by WriteRaw with no reuse, so keeping it inside L2
/// would make that hop cheaper; 4 MiB does not, it goes through L3 or memory.
/// That is a deliberate choice: for the components this path exists for the
/// round-trip count dominates a per-byte cache hop.
constexpr uint64_t PARALLEL_DESERIALIZE_SCRATCH_SIZE = 4ULL * 1024 * 1024;

/// writes raw io bytes as (data, size, offset), type-erased over the component
using WriteRawFunc = std::function<void(const uint8_t*, uint64_t, uint64_t)>;

/// Consume one decompressed chunk stream into [io_offset, io_offset + logical).
/// The byte count must match exactly: with the skip-zeroing reserve a short
/// stream would leave uninitialized bytes and an over-long stream would write
/// past the pre-allocated extent.
inline void
consume_chunk_stream(std::istream& is,
                     const WriteRawFunc& write_raw,
                     uint64_t io_offset,
                     uint64_t logical) {
    std::vector<char> scratch(std::min(logical, PARALLEL_DESERIALIZE_SCRATCH_SIZE));
    uint64_t written = 0;
    while (written < logical) {
        const auto want = std::min<uint64_t>(scratch.size(), logical - written);
        is.read(scratch.data(), static_cast<std::streamsize>(want));
        const auto got = static_cast<uint64_t>(is.gcount());
        if (got == 0) {
            throw VsagException(
                ErrorType::INVALID_BINARY,
                fmt::format("decompressed chunk ends early: {} of {} bytes", written, logical));
        }
        write_raw(reinterpret_cast<const uint8_t*>(scratch.data()), got, io_offset + written);
        written += got;
    }
    char probe = 0;
    is.read(&probe, 1);
    if (is.gcount() != 0) {
        throw VsagException(ErrorType::INVALID_BINARY,
                            fmt::format("decompressed chunk longer than {} bytes", logical));
    }
}

/// Fill [io_offset, io_offset + size) with plain bytes read from
/// [file_offset, file_offset + size).
inline void
fill_extent_from_reader(DeserializeReader& reader,
                        const WriteRawFunc& write_raw,
                        uint64_t file_offset,
                        uint64_t io_offset,
                        uint64_t size) {
    std::vector<char> scratch(std::min(size, PARALLEL_DESERIALIZE_SCRATCH_SIZE));
    uint64_t pos = 0;
    while (pos < size) {
        const auto step = std::min<uint64_t>(scratch.size(), size - pos);
        reader.Read(file_offset + pos, step, scratch.data());
        write_raw(reinterpret_cast<const uint8_t*>(scratch.data()), step, io_offset + pos);
        pos += step;
    }
}

/// Invoke reader.ReadDecompressed while guaranteeing that the parallel
/// pipeline only surfaces VsagException. The public DeserializeReader
/// interface defaults ReadDecompressed to throw std::runtime_error (the
/// public header cannot depend on internal exception types), so a plain
/// reader handed a compressed body would otherwise leak that type through
/// the task error propagation and force callers of ParallelDeserialize to
/// catch two exception families for one failure path.
inline void
read_decompressed_checked(DeserializeReader& reader,
                          uint64_t offset,
                          uint64_t compressed_size,
                          const std::function<void(std::istream&)>& consume) {
    try {
        reader.ReadDecompressed(offset, compressed_size, consume);
    } catch (const VsagException&) {
        throw;
    } catch (const std::exception& e) {
        throw VsagException(
            ErrorType::INTERNAL_ERROR,
            fmt::format("ReadDecompressed failed at offset {} ({} compressed bytes): {}",
                        offset,
                        compressed_size,
                        e.what()));
    }
}

/// Wait for all tasks; the first exception wins, the rest are drained so no
/// task outlives the stack frames it references.
inline void
join_all(std::vector<std::future<void>>& futures) {
    std::exception_ptr first = nullptr;
    for (auto& future : futures) {
        try {
            future.get();
        } catch (...) {
            if (first == nullptr) {
                first = std::current_exception();
            }
        }
    }
    if (first != nullptr) {
        std::rethrow_exception(first);
    }
}

/// Thread pools are not required to propagate task exceptions through the
/// returned futures (SafeThreadPool logs and swallows them), so every task
/// records its failure here and the orchestrator rethrows after the join.
/// TaskBatch owns one of these and does the recording, so callers normally use
/// TaskBatch rather than this class directly.
class FirstTaskError {
public:
    void
    Capture() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (error_ == nullptr) {
            error_ = std::current_exception();
        }
    }

    void
    RethrowIfAny() {
        // uncontended after the join; keeps the read correct even for
        // ThreadPool implementations whose futures synchronize loosely
        std::lock_guard<std::mutex> lock(mutex_);
        if (error_ != nullptr) {
            std::rethrow_exception(error_);
        }
    }

    [[nodiscard]] bool
    HasError() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return error_ != nullptr;
    }

private:
    mutable std::mutex mutex_;
    std::exception_ptr error_{nullptr};
};

/// One round of "fan tasks out, wait, surface the first failure".
///
/// Submit wraps every task body, so a task cannot lose its exception in a pool
/// that swallows them; that contract is structural here rather than something
/// each call site has to remember. Tasks capture references into the caller's
/// stack frame, so the batch drains in its destructor too: if building a task
/// throws midway, the already running ones are still joined before the frame
/// they point at unwinds.
///
/// Run() is the normal exit and must be called before destruction. If it was
/// not (a bug at the call site, or an early return between Submit and Run),
/// the destructor still joins the tasks and then rethrows the first recorded
/// failure, so a task exception is never silently dropped; the rethrow is
/// suppressed while the stack is already unwinding, where a second in-flight
/// exception would terminate.
class TaskBatch {
public:
    /// expected_tasks must be the exact task count: the reservation keeps
    /// Submit from reallocating, which would otherwise be able to throw while
    /// tasks are already running.
    TaskBatch(ThreadPool& pool, size_t expected_tasks) : pool_(pool) {
        futures_.reserve(expected_tasks);
    }

    ~TaskBatch() noexcept(false) {
        try {
            join_all(futures_);
        } catch (...) {
            // a failure here is already recorded in error_, and the only
            // remaining job is to not leave tasks running
        }
        if (!ran_ && std::uncaught_exceptions() == 0) {
            error_.RethrowIfAny();
        } else if (!ran_ && error_.HasError()) {
            // an exception is already in flight, so rethrowing here would
            // call std::terminate; make the suppressed failure at least
            // visible instead of losing it without a trace
            logger::warn(
                "TaskBatch destroyed during stack unwinding with a recorded task failure; "
                "the failure is suppressed in favor of the in-flight exception");
        }
    }

    TaskBatch(const TaskBatch&) = delete;
    TaskBatch&
    operator=(const TaskBatch&) = delete;

    template <class Fn>
    void
    Submit(Fn&& task) {
        futures_.push_back(pool_.Enqueue([this, task = std::forward<Fn>(task)]() {
            try {
                task();
            } catch (...) {
                error_.Capture();
            }
        }));
    }

    /// Join every task, then rethrow the first failure any of them hit.
    void
    Run() {
        ran_ = true;
        // join_all consumes the futures even when it rethrows, so move them
        // out first: the destructor must not wait on them a second time
        std::vector<std::future<void>> pending;
        pending.swap(futures_);
        join_all(pending);
        error_.RethrowIfAny();
    }

private:
    ThreadPool& pool_;
    // std::vector rather than the allocator-aware Vector on purpose, same as
    // the scratch buffers above: futures are move-only task machinery, not
    // index data, the count is bounded by the task fan-out, and TaskBatch has
    // no handle to an index allocator.
    std::vector<std::future<void>> futures_;
    FirstTaskError error_;
    bool ran_{false};
};

}  // namespace vsag
