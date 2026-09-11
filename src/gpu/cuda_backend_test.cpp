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

// The contract callers rely on when the backend is compiled out: no device is
// reported, so they stay on the CPU path. These cases need no device, which is
// what lets them run in CI.
#ifndef VSAG_ENABLE_CUDA

TEST_CASE("Compiled out, the backend reports no device", "[ut][gpu_backend]") {
    REQUIRE_FALSE(vsag::gpu::CudaAvailable());
}

TEST_CASE("Compiled out, no ordinal can be selected", "[ut][gpu_backend]") {
    REQUIRE_FALSE(vsag::gpu::CudaSelectDevice(0));
    // An ordinal the machine does not have must be refused rather than rounded
    // to one that exists.
    REQUIRE_FALSE(vsag::gpu::CudaSelectDevice(99));
    REQUIRE_FALSE(vsag::gpu::CudaSelectDevice(-1));
}

#endif  // VSAG_ENABLE_CUDA
