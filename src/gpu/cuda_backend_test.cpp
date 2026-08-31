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

#include "cuda_backend.h"

#include "unittest.h"

#ifdef VSAG_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

// Holds in every build: no machine has a device at a negative ordinal, so the
// backend must refuse one before it consults the runtime at all.
TEST_CASE("A negative ordinal is refused", "[ut][gpu_backend]") {
    REQUIRE_FALSE(vsag::gpu::CudaSelectDevice(-1));
}

#ifndef VSAG_ENABLE_CUDA

// Compiled out, the stub reports no device, so callers stay on the CPU path.

TEST_CASE("Compiled out: the backend reports no device", "[ut][gpu_backend]") {
    REQUIRE_FALSE(vsag::gpu::CudaAvailable());
}

TEST_CASE("Compiled out: no ordinal can be selected", "[ut][gpu_backend]") {
    REQUIRE_FALSE(vsag::gpu::CudaSelectDevice(0));
    // An ordinal the machine does not have must be refused rather than rounded
    // to one that exists.
    REQUIRE_FALSE(vsag::gpu::CudaSelectDevice(99));
}

#else

// Compiled in, what the backend reports must agree with the runtime. This
// needs no device: with none the count is 0, and 0 is then the first ordinal
// that does not exist.
TEST_CASE("Compiled in: availability agrees with the runtime", "[ut][gpu_backend]") {
    int count = 0;
    const bool runtime_has_device = cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
    if (not runtime_has_device) {
        count = 0;
    }

    REQUIRE(vsag::gpu::CudaAvailable() == runtime_has_device);
    // `count` is the first ordinal past the end, whether that is 0 or N.
    REQUIRE_FALSE(vsag::gpu::CudaSelectDevice(count));
}

// Selecting a device binds the calling thread to it, and a refusal leaves the
// thread where it was. Both need a device to observe, so skip without one.
TEST_CASE("Compiled in: selection binds the thread to the device", "[ut][gpu_backend]") {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        SKIP("no CUDA device on this machine");
    }
    const auto current_device = [] {
        int device = -1;
        return cudaGetDevice(&device) == cudaSuccess ? device : -1;
    };

    // The runtime defaults to device 0, so selecting 0 alone would pass even if
    // the backend never called cudaSetDevice. With more than one device, going
    // to the last ordinal first is a switch the runtime can confirm.
    const int last = count - 1;
    REQUIRE(vsag::gpu::CudaSelectDevice(last));
    REQUIRE(current_device() == last);
    REQUIRE(vsag::gpu::CudaSelectDevice(0));
    REQUIRE(current_device() == 0);

    REQUIRE_FALSE(vsag::gpu::CudaSelectDevice(count));
    REQUIRE(current_device() == 0);
}

#endif  // VSAG_ENABLE_CUDA
