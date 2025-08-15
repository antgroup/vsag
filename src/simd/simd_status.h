
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

#include <cpuinfo.h>

#include <iostream>
#include <string>

namespace vsag {

class SimdStatus {
public:
    bool dist_support_sse = false;
    bool dist_support_avx = false;
    bool dist_support_avx2 = false;
    bool dist_support_avx512f = false;
    bool dist_support_avx512dq = false;
    bool dist_support_avx512bw = false;
    bool dist_support_avx512vl = false;
    bool dist_support_neon = false;
    bool dist_support_sve = false;
    bool dist_support_avx512vpopcntdq = false;
    bool runtime_has_sse = false;
    bool runtime_has_avx = false;
    bool runtime_has_avx2 = false;
    bool runtime_has_avx512f = false;
    bool runtime_has_avx512dq = false;
    bool runtime_has_avx512bw = false;
    bool runtime_has_avx512vl = false;
    bool runtime_has_neon = false;
    bool runtime_has_sve = false;
    bool runtime_has_avx512vpopcntdq = false;

    static bool is_inited;

    static inline void
    Init() {
        if (is_inited) {
            return;
        }
        is_inited = cpuinfo_initialize();
    }

    static inline bool
    SupportAVX512() {
        Init();
        bool ret = false;
#if defined(ENABLE_AVX512)
        ret = true;
#endif
        ret &= cpuinfo_has_x86_avx512f() & cpuinfo_has_x86_avx512dq() & cpuinfo_has_x86_avx512bw() &
               cpuinfo_has_x86_avx512vl();
        return ret;
    }

    static inline bool
    SupportAVX512VPOPCNTDQ() {
        Init();
#if defined(ENABLE_AVX512VPOPCNTDQ)
        return cpuinfo_has_x86_avx512vpopcntdq();
#else
        return false;
#endif
    }

    static inline bool
    SupportAVX2() {
        Init();
        bool ret = false;
#if defined(ENABLE_AVX2)
        ret = true;
#endif
        ret &= cpuinfo_has_x86_avx2();
        return ret;
    }

    static inline bool
    SupportAVX() {
        Init();
        bool ret = false;
#if defined(ENABLE_AVX)
        ret = true;
#endif
        ret &= cpuinfo_has_x86_avx();
        return ret;
    }

    static inline bool
    SupportSSE() {
        Init();
        bool ret = false;
#if defined(ENABLE_SSE)
        ret = true;
#endif
        ret &= cpuinfo_has_x86_sse();
        return ret;
    }

    static inline bool
    SupportNEON() {
        bool ret = false;
#if defined(ENABLE_NEON)
        ret = true;
#endif
        ret &= cpuinfo_has_arm_neon();
        return ret;
    }

    static inline bool
    SupportSVE() {
        Init();
        bool ret = false;
#if defined(ENABLE_SVE)
        ret = true;
#endif
        ret &= cpuinfo_has_arm_sve();
        return ret;
    }

    [[nodiscard]] std::string
    sse() const {
        return status_to_string(dist_support_sse, runtime_has_sse);
    }

    [[nodiscard]] std::string
    avx() const {
        return status_to_string(dist_support_avx, runtime_has_avx);
    }

    [[nodiscard]] std::string
    avx2() const {
        return status_to_string(dist_support_avx2, runtime_has_avx2);
    }

    [[nodiscard]] std::string
    avx512f() const {
        return status_to_string(dist_support_avx512f, runtime_has_avx512f);
    }

    [[nodiscard]] std::string
    avx512dq() const {
        return status_to_string(dist_support_avx512dq, runtime_has_avx512dq);
    }

    [[nodiscard]] std::string
    avx512bw() const {
        return status_to_string(dist_support_avx512bw, runtime_has_avx512bw);
    }

    [[nodiscard]] std::string
    avx512vl() const {
        return status_to_string(dist_support_avx512vl, runtime_has_avx512vl);
    }

    [[nodiscard]] std::string
    neon() const {
        return status_to_string(dist_support_neon, runtime_has_neon);
    }

    [[nodiscard]] std::string
    sve() const {
        return status_to_string(dist_support_sve, runtime_has_sve);
    }

    [[nodiscard]] std::string
    avx512vpopcntdq() const {
        return status_to_string(dist_support_avx512vpopcntdq, runtime_has_avx512vpopcntdq);
    }

    static std::string
    boolean_to_string(bool value) {
        if (value) {
            return "Y";
        } else {
            return "N";
        }
    }

    static std::string
    status_to_string(bool dist, bool runtime) {
        return "dist_support:" + boolean_to_string(dist) +
               " + platform:" + boolean_to_string(runtime) +
               " = using:" + boolean_to_string(dist & runtime);
    }
};

}  // namespace vsag
