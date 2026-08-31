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
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || device_id < 0 || device_id >= count) {
        return false;
    }
    return cudaSetDevice(device_id) == cudaSuccess;
}

}  // namespace vsag::gpu
