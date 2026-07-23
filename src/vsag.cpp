
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

#include "vsag/vsag.h"

#include <../extern/diskann/DiskANN/include/diskann_logger.h>
#include <cpuinfo.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string_view>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#endif

#include "impl/logger/logger.h"
#include "simd/simd.h"
#include "version.h"

namespace vsag {
namespace {

[[nodiscard]] char
ascii_lower(char value) {
    if (value >= 'A' and value <= 'Z') {
        return static_cast<char>(value - 'A' + 'a');
    }
    return value;
}

[[nodiscard]] bool
equals_ignore_case(std::string_view lhs, std::string_view rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }

    for (uint64_t i = 0; i < lhs.size(); ++i) {
        if (ascii_lower(lhs[i]) != ascii_lower(rhs[i])) {
            return false;
        }
    }

    return true;
}

[[nodiscard]] bool
is_truthy_env_value(std::string_view value) {
    return value == "1" or equals_ignore_case(value, "on") or equals_ignore_case(value, "true");
}

[[nodiscard]] bool
should_suppress_init_banner() {
    const auto* suppress = std::getenv("VSAG_SUPPRESS_INIT_BANNER");
    return suppress != nullptr and is_truthy_env_value(suppress);
}

[[nodiscard]] bool
get_total_physical_memory(uint64_t& memory_bytes) {
#if defined(_WIN32)
    MEMORYSTATUSEX memory_status{};
    memory_status.dwLength = sizeof(memory_status);
    if (GlobalMemoryStatusEx(&memory_status) == 0) {
        return false;
    }
    memory_bytes = memory_status.ullTotalPhys;
    return true;
#elif defined(__APPLE__)
    uint64_t value = 0;
    size_t value_size = sizeof(value);
    if (sysctlbyname("hw.memsize", &value, &value_size, nullptr, 0) != 0 or value == 0) {
        return false;
    }
    memory_bytes = value;
    return true;
#elif defined(__linux__)
    struct sysinfo memory_info {};
    if (sysinfo(&memory_info) != 0 or memory_info.totalram == 0 or memory_info.mem_unit == 0) {
        return false;
    }
    if (static_cast<uint64_t>(memory_info.totalram) >
        std::numeric_limits<uint64_t>::max() / memory_info.mem_unit) {
        return false;
    }
    memory_bytes = static_cast<uint64_t>(memory_info.totalram) * memory_info.mem_unit;
    return memory_bytes != 0;
#else
    (void)memory_bytes;
    return false;
#endif
}

[[nodiscard]] std::string
instance_spec(uint64_t core_count) {
    constexpr uint64_t bytes_per_gib = 1024ULL * 1024ULL * 1024ULL;
    uint64_t memory_bytes = 0;
    std::stringstream spec;
    spec << core_count << "C";
    if (get_total_physical_memory(memory_bytes)) {
        // Report whole GiB by flooring the byte count for deterministic output.
        spec << memory_bytes / bytes_per_gib;
    } else {
        spec << "?";
    }
    spec << "G";
    return spec.str();
}

}  // namespace

std::string
version() {
    return VSAG_VERSION;
}

bool
init() {
#ifndef NDEBUG
    // set debug level by default in debug version of VSAG
    logger::set_level(logger::level::debug);
#endif

    cpuinfo_initialize();
    auto simd_status = setup_simd();
    if (not should_suppress_init_banner()) {
        std::stringstream ss;

        ss << std::boolalpha;
        // Keep this leading newline to separate the banner from the preceding shell command.
        ss << "\n====vsag start init====";
        ss << "\nrunning on " << cpuinfo_get_package(0)->name;
        ss << "\ninstance spec: " << instance_spec(cpuinfo_get_cores_count());
        ss << "\ncpu sse >> " << simd_status.sse();
        ss << "\ncpu avx >> " << simd_status.avx();
        ss << "\ncpu avx2 >> " << simd_status.avx2();
        ss << "\ncpu avx512f >> " << simd_status.avx512f();
        ss << "\ncpu avx512dq >> " << simd_status.avx512dq();
        ss << "\ncpu avx512bw >> " << simd_status.avx512bw();
        ss << "\ncpu avx512vl >> " << simd_status.avx512vl();
        ss << "\ncpu avx512vpopcntdq >> " << simd_status.avx512vpopcntdq();
        ss << "\ncpu neon >> " << simd_status.neon();
        ss << "\ncpu sve >> " << simd_status.sve();
        ss << "\n====vsag init done====";
        logger::info(ss.str());
    }

    return true;
}

// to trigger initial
static bool init_status = init();

}  // namespace vsag
