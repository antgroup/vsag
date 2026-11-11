// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <string>

namespace vsag {

// Resolve the current process's visible CPU hierarchy, including ancestor quotas.
// An unresolved cgroup namespace root conservatively limits execution to one slot.
uint32_t
CgroupCpuSlots(const std::string& membership,
               const std::string& mountinfo,
               const std::function<std::string(const std::string&)>& read,
               uint32_t affinity);

}  // namespace vsag
