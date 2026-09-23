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

#include <cuda_runtime.h>

#include "cuda_backend.h"
#include "gpu_plan.h"

namespace vsag::gpu {

bool
CudaAvailable() {
    int count = 0;
    // A machine with no driver reports an error rather than a count of zero.
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        return false;
    }
    return count > 0;
}

bool
CudaSelectDevice(int32_t device_id) {
    // No machine has a device at a negative ordinal, so refuse it before the
    // runtime is initialised on its behalf.
    if (device_id < 0) {
        return false;
    }
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || device_id >= count) {
        return false;
    }
    return cudaSetDevice(device_id) == cudaSuccess;
}

CudaDeviceScope::CudaDeviceScope(int32_t device_id) {
    if (device_id < 0) {
        return;
    }
    // cudaGetDevice creates no context of its own, so asking first costs
    // nothing on a thread that has none.
    if (cudaGetDevice(&previous_) != cudaSuccess) {
        // Only to leave the member determinate. bound_ stays false on this path
        // and the destructor restores nothing unless it is true, so this 0 is
        // never the one the header's ambiguity is about.
        previous_ = 0;
        return;
    }
    bound_ = CudaSelectDevice(device_id);
}

CudaDeviceScope::~CudaDeviceScope() {
    // See the header: 0 is ambiguous and is left alone.
    if (bound_ and previous_ != 0) {
        cudaSetDevice(previous_);
    }
}

uint64_t
CudaSuggestedBudget(uint64_t cap_bytes) {
    // cudaMemGetInfo takes size_t*; everything downstream is uint64_t.
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        return 0;
    }
    return CapBudget(static_cast<uint64_t>(free_bytes), cap_bytes);
}

}  // namespace vsag::gpu
