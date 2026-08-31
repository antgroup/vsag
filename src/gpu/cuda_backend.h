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
/// Everything here has a stub compiled in its place when ENABLE_CUDA is off, so
/// callers link against it unconditionally and decide at runtime rather than
/// behind a preprocessor branch. The stub reports no device, which sends every
/// caller down the CPU path it already has.
///
/// No CUDA type appears in this header, so translation units that include it do
/// not need the toolkit.
namespace vsag::gpu {

/// True when this build has the CUDA backend and the machine has at least one
/// usable device.
bool
CudaAvailable();

/// Binds the calling thread to a device.
///
/// Returns false for an ordinal the machine does not have, so a caller that
/// named a device it cannot get keeps its CPU path instead of running on a
/// different one. On a machine whose devices are shared, substituting a card
/// consumes a resource the caller did not ask for, and does so silently.
bool
CudaSelectDevice(int32_t device_id);

}  // namespace vsag::gpu
