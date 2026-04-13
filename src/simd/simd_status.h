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

/**
 * @file simd_status.h
 * @brief SIMD capability detection and status reporting.
 *
 * This header provides the SimdStatus class for detecting and reporting
 * available SIMD instruction sets on the current platform. It supports
 * both x86 (SSE, AVX, AVX2, AVX512 variants) and ARM (NEON, SVE) architectures.
 */

#pragma once

#include <cpuinfo.h>

#include <iostream>
#include <string>

namespace vsag {

/**
 * @brief Class for detecting and reporting SIMD instruction set support.
 *
 * SimdStatus provides runtime detection of CPU SIMD capabilities and reports
 * which instruction sets are available. It distinguishes between:
 * - Distance support: Whether the distance computation code was compiled with
 *   support for a particular instruction set
 * - Runtime support: Whether the CPU actually supports that instruction set
 *
 * The actual SIMD function selection requires both distance support and
 * runtime support to be enabled.
 */
class SimdStatus {
public:
    /** @brief Whether distance computation supports SSE instructions */
    bool dist_support_sse = false;
    /** @brief Whether distance computation supports AVX instructions */
    bool dist_support_avx = false;
    /** @brief Whether distance computation supports AVX2 instructions */
    bool dist_support_avx2 = false;
    /** @brief Whether distance computation supports AVX512F instructions */
    bool dist_support_avx512f = false;
    /** @brief Whether distance computation supports AVX512DQ instructions */
    bool dist_support_avx512dq = false;
    /** @brief Whether distance computation supports AVX512BW instructions */
    bool dist_support_avx512bw = false;
    /** @brief Whether distance computation supports AVX512VL instructions */
    bool dist_support_avx512vl = false;
    /** @brief Whether distance computation supports ARM NEON instructions */
    bool dist_support_neon = false;
    /** @brief Whether distance computation supports ARM SVE instructions */
    bool dist_support_sve = false;
    /** @brief Whether distance computation supports AVX512VPOPCNTDQ instructions */
    bool dist_support_avx512vpopcntdq = false;

    /** @brief Whether the CPU supports SSE at runtime */
    bool runtime_has_sse = false;
    /** @brief Whether the CPU supports AVX at runtime */
    bool runtime_has_avx = false;
    /** @brief Whether the CPU supports AVX2 at runtime */
    bool runtime_has_avx2 = false;
    /** @brief Whether the CPU supports AVX512F at runtime */
    bool runtime_has_avx512f = false;
    /** @brief Whether the CPU supports AVX512DQ at runtime */
    bool runtime_has_avx512dq = false;
    /** @brief Whether the CPU supports AVX512BW at runtime */
    bool runtime_has_avx512bw = false;
    /** @brief Whether the CPU supports AVX512VL at runtime */
    bool runtime_has_avx512vl = false;
    /** @brief Whether the CPU supports NEON at runtime */
    bool runtime_has_neon = false;
    /** @brief Whether the CPU supports SVE at runtime */
    bool runtime_has_sve = false;
    /** @brief Whether the CPU supports AVX512VPOPCNTDQ at runtime */
    bool runtime_has_avx512vpopcntdq = false;

    /** @brief Flag indicating whether cpuinfo has been initialized */
    static bool is_inited;

    /**
     * @brief Initialize the cpuinfo library for CPU feature detection.
     *
     * This method is called automatically before any CPU feature check.
     * It ensures cpuinfo_initialize() is called only once.
     */
    static inline void
    Init() {
        if (is_inited) {
            return;
        }
        is_inited = cpuinfo_initialize();
    }

    /**
     * @brief Check if AVX512 (F, DQ, BW, VL) is supported.
     *
     * @return true if AVX512 is compiled in and available on CPU,
     *         false otherwise.
     */
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

    /**
     * @brief Check if AVX512VPOPCNTDQ (population count) is supported.
     *
     * @return true if AVX512VPOPCNTDQ is compiled in and available on CPU,
     *         false otherwise.
     */
    static inline bool
    SupportAVX512VPOPCNTDQ() {
        Init();
#if defined(ENABLE_AVX512VPOPCNTDQ)
        return cpuinfo_has_x86_avx512vpopcntdq();
#else
        return false;
#endif
    }

    /**
     * @brief Check if AVX2 is supported.
     *
     * @return true if AVX2 is compiled in and available on CPU,
     *         false otherwise.
     */
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

    /**
     * @brief Check if AVX is supported.
     *
     * @return true if AVX is compiled in and available on CPU,
     *         false otherwise.
     */
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

    /**
     * @brief Check if SSE is supported.
     *
     * @return true if SSE is compiled in and available on CPU,
     *         false otherwise.
     */
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

    /**
     * @brief Check if ARM NEON is supported.
     *
     * @return true if NEON is compiled in and available on CPU,
     *         false otherwise.
     */
    static inline bool
    SupportNEON() {
        bool ret = false;
#if defined(ENABLE_NEON)
        ret = true;
#endif
        ret &= cpuinfo_has_arm_neon();
        return ret;
    }

    /**
     * @brief Check if ARM SVE is supported.
     *
     * @return true if SVE is compiled in and available on CPU,
     *         false otherwise.
     */
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

    /**
     * @brief Get SSE support status as a formatted string.
     * @return Formatted string showing distance support, platform support,
     *         and usage status.
     */
    [[nodiscard]] std::string
    sse() const {
        return status_to_string(dist_support_sse, runtime_has_sse);
    }

    /**
     * @brief Get AVX support status as a formatted string.
     * @return Formatted string showing distance support, platform support,
     *         and usage status.
     */
    [[nodiscard]] std::string
    avx() const {
        return status_to_string(dist_support_avx, runtime_has_avx);
    }

    /**
     * @brief Get AVX2 support status as a formatted string.
     * @return Formatted string showing distance support, platform support,
     *         and usage status.
     */
    [[nodiscard]] std::string
    avx2() const {
        return status_to_string(dist_support_avx2, runtime_has_avx2);
    }

    /**
     * @brief Get AVX512F support status as a formatted string.
     * @return Formatted string showing distance support, platform support,
     *         and usage status.
     */
    [[nodiscard]] std::string
    avx512f() const {
        return status_to_string(dist_support_avx512f, runtime_has_avx512f);
    }

    /**
     * @brief Get AVX512DQ support status as a formatted string.
     * @return Formatted string showing distance support, platform support,
     *         and usage status.
     */
    [[nodiscard]] std::string
    avx512dq() const {
        return status_to_string(dist_support_avx512dq, runtime_has_avx512dq);
    }

    /**
     * @brief Get AVX512BW support status as a formatted string.
     * @return Formatted string showing distance support, platform support,
     *         and usage status.
     */
    [[nodiscard]] std::string
    avx512bw() const {
        return status_to_string(dist_support_avx512bw, runtime_has_avx512bw);
    }

    /**
     * @brief Get AVX512VL support status as a formatted string.
     * @return Formatted string showing distance support, platform support,
     *         and usage status.
     */
    [[nodiscard]] std::string
    avx512vl() const {
        return status_to_string(dist_support_avx512vl, runtime_has_avx512vl);
    }

    /**
     * @brief Get NEON support status as a formatted string.
     * @return Formatted string showing distance support, platform support,
     *         and usage status.
     */
    [[nodiscard]] std::string
    neon() const {
        return status_to_string(dist_support_neon, runtime_has_neon);
    }

    /**
     * @brief Get SVE support status as a formatted string.
     * @return Formatted string showing distance support, platform support,
     *         and usage status.
     */
    [[nodiscard]] std::string
    sve() const {
        return status_to_string(dist_support_sve, runtime_has_sve);
    }

    /**
     * @brief Get AVX512VPOPCNTDQ support status as a formatted string.
     * @return Formatted string showing distance support, platform support,
     *         and usage status.
     */
    [[nodiscard]] std::string
    avx512vpopcntdq() const {
        return status_to_string(dist_support_avx512vpopcntdq, runtime_has_avx512vpopcntdq);
    }

    /**
     * @brief Convert a boolean value to "Y" or "N" string.
     * @param value Boolean value to convert.
     * @return "Y" if true, "N" if false.
     */
    static std::string
    boolean_to_string(bool value) {
        if (value) {
            return "Y";
        } else {
            return "N";
        }
    }

    /**
     * @brief Format SIMD status as a human-readable string.
     *
     * Creates a string in the format:
     * "dist_support:Y/N + platform:Y/N = using:Y/N"
     *
     * @param dist Whether the distance computation supports this instruction set.
     * @param runtime Whether the CPU supports this instruction set at runtime.
     * @return Formatted status string.
     */
    static std::string
    status_to_string(bool dist, bool runtime) {
        return "dist_support:" + boolean_to_string(dist) +
               " + platform:" + boolean_to_string(runtime) +
               " = using:" + boolean_to_string(dist & runtime);
    }
};

}  // namespace vsag