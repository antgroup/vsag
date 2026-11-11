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

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <utility>

/// Owning handles for the CUDA resources the backend allocates. Every entry point
/// allocates several buffers and can fail at any of them; these let it return
/// early, with no cleanup path to keep in step with the acquisitions.
///
/// Only .cu translation units include this header.
namespace vsag::gpu {

namespace detail {

/// Common shape for the handles below: move-only, released on destruction.
template <typename T, typename Deleter>
class UniqueResource {
public:
    UniqueResource() = default;

    ~UniqueResource() {
        Reset();
    }

    UniqueResource(const UniqueResource&) = delete;
    UniqueResource&
    operator=(const UniqueResource&) = delete;

    UniqueResource(UniqueResource&& other) noexcept : value_(other.Release()) {
    }

    UniqueResource&
    operator=(UniqueResource&& other) noexcept {
        if (this != &other) {
            Reset(other.Release());
        }
        return *this;
    }

    [[nodiscard]] T
    Get() const {
        return value_;
    }

    void
    Reset(T value = T{}) {
        if (value_ != T{}) {
            Deleter{}(value_);
        }
        value_ = value;
    }

    T
    Release() {
        return std::exchange(value_, T{});
    }

private:
    T value_{};
};

struct DeviceFree {
    void
    operator()(void* p) const {
        cudaFree(p);
    }
};

struct HostFree {
    void
    operator()(void* p) const {
        cudaFreeHost(p);
    }
};

struct StreamDestroy {
    void
    operator()(cudaStream_t s) const {
        cudaStreamDestroy(s);
    }
};

struct BlasDestroy {
    void
    operator()(cublasHandle_t h) const {
        cublasDestroy(h);
    }
};

}  // namespace detail

/// A device allocation of `T`, freed when it goes out of scope.
template <typename T>
class DeviceArray {
public:
    /// Replaces any previous allocation. False leaves the object empty.
    bool
    Alloc(uint64_t count) {
        void* raw = nullptr;
        if (cudaMalloc(&raw, count * sizeof(T)) != cudaSuccess) {
            storage_.Reset();
            return false;
        }
        storage_.Reset(raw);
        return true;
    }

    [[nodiscard]] T*
    Get() const {
        return static_cast<T*>(storage_.Get());
    }

private:
    detail::UniqueResource<void*, detail::DeviceFree> storage_;
};

/// A page-locked host allocation of `T`, freed when it goes out of scope, so a
/// copy on one stream can overlap compute on another. Worth its cost here, unlike
/// in the accumulation pass: on the shape kAssignStreams names, pinning the rows
/// took 0.098 s against 0.103 s reading them out of ordinary pages.
template <typename T>
class PinnedArray {
public:
    bool
    Alloc(uint64_t count) {
        void* raw = nullptr;
        if (cudaHostAlloc(&raw, count * sizeof(T), cudaHostAllocDefault) != cudaSuccess) {
            storage_.Reset();
            return false;
        }
        storage_.Reset(raw);
        return true;
    }

    [[nodiscard]] T*
    Get() const {
        return static_cast<T*>(storage_.Get());
    }

private:
    detail::UniqueResource<void*, detail::HostFree> storage_;
};

class Stream {
public:
    bool
    Create() {
        cudaStream_t s = nullptr;
        if (cudaStreamCreate(&s) != cudaSuccess) {
            return false;
        }
        handle_.Reset(s);
        return true;
    }

    [[nodiscard]] cudaStream_t
    Get() const {
        return handle_.Get();
    }

private:
    detail::UniqueResource<cudaStream_t, detail::StreamDestroy> handle_;
};

/// A cuBLAS handle bound to one stream, one per stream.
///
/// Not for throughput: on that same shape a single handle rebound before each
/// product measured 0.095 s against 0.098 s for two, so sharing one is if anything
/// a shade faster. What two buy is that each stays bound for the whole pass, so the
/// binding the argmin kernel depends on is established once and checked once,
/// rather than before every product inside the chunk loop.
class BlasHandle {
public:
    /// Both settings are checked, because the caller's correctness rests on them:
    /// the stream orders the product before the kernel that reads it, and pedantic
    /// math keeps the product in FP32, where TF32 would change which centroid wins
    /// a near tie. A handle that cannot promise either is no handle.
    bool
    Create(cudaStream_t stream) {
        cublasHandle_t h = nullptr;
        if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) {
            return false;
        }
        handle_.Reset(h);
        return cublasSetMathMode(h, CUBLAS_PEDANTIC_MATH) == CUBLAS_STATUS_SUCCESS and
               cublasSetStream(h, stream) == CUBLAS_STATUS_SUCCESS;
    }

    [[nodiscard]] cublasHandle_t
    Get() const {
        return handle_.Get();
    }

private:
    detail::UniqueResource<cublasHandle_t, detail::BlasDestroy> handle_;
};

/// Clears any error the runtime has latched for this thread. A failed probe leaves
/// its error in place and later successful calls do not clear it, so without this
/// a post-launch cudaGetLastError() could read a failure from before the call.
///
/// Discarded rather than asserted on, because refusing is this backend's ordinary
/// answer and a refusal latches. Measured: a cudaMalloc of 400 GiB reports out of
/// memory, a 1 MiB one straight after it succeeds, and the pending error is still
/// out of memory until a cudaGetLastError takes it. So an allocation the budget
/// was too optimistic about leaves one behind on its way to the host path, and
/// asserting here would turn that documented fallback into a crash.
inline void
ClearPendingError() {
    cudaGetLastError();
}

}  // namespace vsag::gpu
