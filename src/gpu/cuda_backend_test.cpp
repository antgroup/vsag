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

TEST_CASE("Compiled out: the device scope binds nothing", "[ut][gpu_backend]") {
    vsag::gpu::CudaDeviceScope scope(0);
    REQUIRE_FALSE(scope.Bound());
}

TEST_CASE("Compiled out: no budget is ever suggested", "[ut][gpu_backend]") {
    // A zero budget is what every caller then passes on, and each entry point
    // reads it as "do not offload" rather than sizing a plan around it. That the
    // passes leave the caller's buffers untouched is covered in
    // cuda_pass_test.cpp, which checks it in both configurations.
    REQUIRE(vsag::gpu::CudaSuggestedBudget(0) == 0);
    REQUIRE(vsag::gpu::CudaSuggestedBudget(2ULL << 30) == 0);
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

// The scope is what callers use, so that a build does not leave the caller's
// thread on a device the caller never chose.
TEST_CASE("Compiled in: the device scope puts the thread back", "[ut][gpu_backend]") {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess or count < 2) {
        SKIP("needs at least two CUDA devices to observe a switch and a restore");
    }
    const auto current_device = [] {
        int device = -1;
        return cudaGetDevice(&device) == cudaSuccess ? device : -1;
    };

    // From a non-zero device, which only a thread that already holds a context
    // can report, the scope has to put the thread back.
    REQUIRE(vsag::gpu::CudaSelectDevice(1));
    REQUIRE(current_device() == 1);
    {
        vsag::gpu::CudaDeviceScope scope(0);
        REQUIRE(scope.Bound());
        REQUIRE(current_device() == 0);
    }
    REQUIRE(current_device() == 1);

    // A scope that binds nothing leaves the thread alone, and says so.
    {
        vsag::gpu::CudaDeviceScope none(-1);
        REQUIRE_FALSE(none.Bound());
        REQUIRE(current_device() == 1);
    }
    REQUIRE(current_device() == 1);

    {
        vsag::gpu::CudaDeviceScope absent(count);
        REQUIRE_FALSE(absent.Bound());
        REQUIRE(current_device() == 1);
    }
    REQUIRE(current_device() == 1);

    // From device 0 the restore is skipped, because the runtime reports 0 for a
    // thread that has never bound one too and putting that thread "back" would
    // create a context on device 0 that nothing asked for.
    REQUIRE(vsag::gpu::CudaSelectDevice(0));
    {
        vsag::gpu::CudaDeviceScope scope(1);
        REQUIRE(scope.Bound());
        REQUIRE(current_device() == 1);
    }
    REQUIRE(current_device() == 1);

    // Which is the one case that leaves the binding somewhere the runtime would
    // not have put it, so this case puts it back. Every other case here sets the
    // device it needs, but the passes in cuda_pass_test.cpp run on whichever one
    // the thread already holds, and size their working set from that device's free
    // memory. On a machine whose cards are not equally busy, inheriting device 1
    // from here would make those cases depend on the order they ran in.
    REQUIRE(vsag::gpu::CudaSelectDevice(0));
    REQUIRE(current_device() == 0);
}

#endif  // VSAG_ENABLE_CUDA
