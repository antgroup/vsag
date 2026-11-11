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

namespace vsag::pipnn_half {

enum class Format { FP16, BF16 };

using DecodeRowType = void (*)(const uint16_t*, float*, uint64_t);
using PreparationType = float (*)(const uint16_t*, const float*, uint64_t, Format);

// Explicit inputs allow testing dispatch without relying on the host CPU.
struct Capabilities {
    bool compiled_avx2;
    bool compiled_f16c;
    bool cpuinfo_initialized;
    bool cpu_avx;
    bool cpu_avx2;
    bool cpu_f16c;
};

struct ImplementationSelection {
    bool decode_fp16_avx2_f16c;
    bool decode_bf16_avx2;
    bool preparation_avx2_f16c;
};

constexpr ImplementationSelection
SelectImplementations(Capabilities caps) {
    const bool avx2 =
        caps.compiled_avx2 && caps.cpuinfo_initialized && caps.cpu_avx && caps.cpu_avx2;
    const bool f16c = avx2 && caps.compiled_f16c && caps.cpu_f16c;
    return {f16c, avx2, f16c};
}

struct Dispatch {
    DecodeRowType decode_fp16;
    DecodeRowType decode_bf16;
    PreparationType preparation;
};

// Resolve a selection using only implementations present in this build. This does
// not execute the selected functions; callers must ensure the CPU supports them.
Dispatch
ResolveDispatch(ImplementationSelection selection);

namespace detail {
#if defined(ENABLE_AVX2)
void
DecodeRowBF16(const uint16_t*, float*, uint64_t);
#if defined(ENABLE_PIPNN_F16C)
void
DecodeRowFP16(const uint16_t*, float*, uint64_t);
float
PreparationAVX2(const uint16_t*, const float*, uint64_t, Format);
#endif
#endif
}  // namespace detail

// Initialized once on first use, including from another TU's initializer.
const Dispatch&
GetDispatch();

}  // namespace vsag::pipnn_half
