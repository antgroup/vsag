// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include "cpu_resource.h"

#include <string>
#include <unordered_map>

#include "unittest.h"
using namespace vsag;

TEST_CASE("CPU budget resolves nested and namespaced quotas", "[ut][CPUResource]") {
    std::unordered_map<std::string, std::string> files;
    const auto read = [&](const std::string& path) {
        const auto found = files.find(path);
        return found == files.end() ? std::string{} : found->second;
    };
    const std::string v2 = "29 23 0:26 / /cg rw - cgroup2 cgroup rw\n";
    files["/cg/pod/task/cpu.max"] = "800000 100000";
    files["/cg/pod/cpu.max"] = "400000 100000";
    files["/cg/cpu.max"] = "max 100000";
    REQUIRE(CgroupCpuSlots("0::/pod/task\n", v2, read, 16) == 4);
    REQUIRE(CgroupCpuSlots("0::/pod/task\n", v2, read, 2) == 2);
    files["/cg/pod/cpu.max"] = "50000 100000";
    REQUIRE(CgroupCpuSlots("0::/pod/task\n", v2, read, 16) == 1);

    const std::string v1 = "30 23 0:27 / /cpu rw - cgroup cgroup rw,cpu,cpuacct\n";
    files["/cpu/pod/cpu.cfs_quota_us"] = "200000";
    files["/cpu/pod/cpu.cfs_period_us"] = "100000";
    REQUIRE(CgroupCpuSlots("3:cpu,cpuacct:/pod\n", v1, read, 16) == 2);
    REQUIRE(CgroupCpuSlots("3:cpuacct:/pod\n", v1, read, 16) == 16);

    const std::string namespaced = "29 23 0:26 /pod /bound rw - cgroup2 cgroup rw\n";
    files["/bound/cpu.max"] = "300000 100000";
    REQUIRE(CgroupCpuSlots("0::/\n", namespaced, read, 16) == 3);
    REQUIRE(CgroupCpuSlots("0::/pod\n", namespaced, read, 16) == 3);
    REQUIRE(CgroupCpuSlots("0::/../../pod\n", namespaced, read, 16) == 1);

    const std::string sibling_mount =
        "29 23 0:26 / /cg rw - cgroup2 cgroup rw\n"
        "30 23 0:26 /podB /other rw - cgroup2 cgroup rw\n";
    files["/cg/podA/task/cpu.max"] = "800000 100000";
    files["/cg/podA/cpu.max"] = "800000 100000";
    files["/cg/cpu.max"] = "max 100000";
    files["/other/podA/task/cpu.max"] = "100000 100000";
    REQUIRE(CgroupCpuSlots("0::/podA/task\n", sibling_mount, read, 16) == 8);
    REQUIRE(CgroupCpuSlots(
                "0::/podA/task\n", "30 23 0:26 /podB /other rw - cgroup2 cgroup rw\n", read, 16) ==
            16);

    REQUIRE(CgroupCpuSlots("invalid", v2, read, 2) == 2);
    REQUIRE(CgroupCpuSlots("", "", read, 64) == 64);  // No index-specific cap.
}
