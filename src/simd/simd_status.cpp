
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

#include "simd_status.h"

#if defined(ENABLE_AMX) && defined(__linux__) && (defined(__x86_64__) || defined(_M_X64))
#include <sys/syscall.h>
#include <unistd.h>

// Linux-specific values for arch_prctl. We avoid pulling in <asm/prctl.h>
// because not all distributions install it; the constants are stable ABI.
#ifndef ARCH_GET_XCOMP_PERM
#define ARCH_GET_XCOMP_PERM 0x1022
#endif
#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif
#endif  // ENABLE_AMX && Linux x86_64

namespace vsag {
bool SimdStatus::is_inited = false;

bool
SimdStatus::SupportAMX() {
#if defined(ENABLE_AMX) && defined(__linux__) && (defined(__x86_64__) || defined(_M_X64))
    // The body is evaluated exactly once per process by the C++11
    // thread-safe static-local initializer (a.k.a. "Magic Statics"),
    // which is simpler and safer than a hand-rolled DCLP.
    static const bool supported = []() -> bool {
        Init();

        // cpuinfo currently lacks a dedicated AMX query in some versions, so
        // we fall back to a direct CPUID check (leaf 7, sub-leaf 0, EDX bits
        // 24/25 = AMX_TILE / AMX_INT8). This is cheap and self-contained.
        unsigned int eax = 7, ebx = 0, ecx = 0, edx = 0;
        __asm__ volatile("cpuid" : "+a"(eax), "=b"(ebx), "+c"(ecx), "=d"(edx));
        bool has_tile = (edx >> 24) & 1u;
        bool has_int8 = (edx >> 25) & 1u;
        if (!has_tile || !has_int8) {
            return false;
        }

        // Ask the kernel for permission to use XFEATURE_XTILEDATA in this
        // process. Without this, executing any tile* instruction raises #UD.
        long rc = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
        return rc == 0;
    }();
    return supported;
#else
    return false;
#endif
}

bool
SimdStatus::SupportAMXBF16() {
#if defined(ENABLE_AMX) && defined(__linux__) && (defined(__x86_64__) || defined(_M_X64))
    static const bool supported = []() -> bool {
        // Require AMX_TILE + INT8 + the kernel XFEATURE_XTILEDATA grant
        // first; AMX_BF16 is meaningless without those.
        if (!SimdStatus::SupportAMX()) {
            return false;
        }
        // CPUID leaf 7, sub-leaf 0, EDX bit 22 == AMX_BF16.
        unsigned int eax = 7, ebx = 0, ecx = 0, edx = 0;
        __asm__ volatile("cpuid" : "+a"(eax), "=b"(ebx), "+c"(ecx), "=d"(edx));
        return ((edx >> 22) & 1u) != 0;
    }();
    return supported;
#else
    return false;
#endif
}

}  // namespace vsag
