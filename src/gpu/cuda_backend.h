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

#include <cstdint>

/// Device discovery for the optional CUDA backend.
///
/// With ENABLE_CUDA off, a stub exporting the same symbols is compiled instead
/// and reports no device, so callers link unconditionally and decide at
/// runtime. No CUDA type appears here, so includers do not need the toolkit.
namespace vsag::gpu {

/// True when this build has the CUDA backend and the machine has at least one
/// usable device.
bool
CudaAvailable();

/// Binds the calling thread to a device.
///
/// Returns false for an ordinal the machine does not have; the caller keeps its
/// CPU path rather than being moved to a device it did not ask for. Returns
/// false too for an ordinal that exists but cannot be used (a device in a
/// prohibited or exclusive compute mode); the runtime does not promise where
/// the thread is left in that case, so select again before issuing more work.
bool
CudaSelectDevice(int32_t device_id);

}  // namespace vsag::gpu
