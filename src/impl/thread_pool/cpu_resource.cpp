// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#include "cpu_resource.h"

#include <algorithm>
#include <filesystem>
#include <sstream>
#include <vector>

namespace vsag {
namespace {

bool
has_controller(const std::string& controllers, const std::string& name) {
    return ("," + controllers + ",").find("," + name + ",") != std::string::npos;
}

std::string
unescape_mount(const std::string& path) {
    std::string result;
    for (uint64_t i = 0; i < path.size(); ++i) {
        if (path[i] == '\\' and i + 3 < path.size() and path[i + 1] >= '0' and
            path[i + 1] <= '7' and path[i + 2] >= '0' and path[i + 2] <= '7' and
            path[i + 3] >= '0' and path[i + 3] <= '7') {
            result += char((path[i + 1] - '0') * 64 + (path[i + 2] - '0') * 8 + path[i + 3] - '0');
            i += 3;
        } else {
            result += path[i];
        }
    }
    return result;
}

}  // namespace

uint32_t
CgroupCpuSlots(const std::string& membership,
               const std::string& mountinfo,
               const std::function<std::string(const std::string&)>& read,
               uint32_t affinity) {
    uint64_t slots = std::max(1U, affinity);
    std::istringstream members(membership);
    std::string member;
    while (std::getline(members, member)) {
        const auto first = member.find(':');
        const auto second = member.find(':', first == std::string::npos ? 0 : first + 1);
        if (first == std::string::npos or second == std::string::npos) {
            continue;
        }
        const auto controllers = member.substr(first + 1, second - first - 1);
        const bool unified = member.substr(0, first) == "0" and controllers.empty();
        if (not unified and not has_controller(controllers, "cpu")) {
            continue;
        }
        const std::filesystem::path group(member.substr(second + 1));
        if (not group.is_absolute() or
            std::any_of(
                group.begin(), group.end(), [](const auto& part) { return part == ".."; })) {
            return 1;
        }
        std::istringstream mounts(mountinfo);
        std::string mount;
        while (std::getline(mounts, mount)) {
            std::istringstream fields(mount);
            std::vector<std::string> words;
            for (std::string word; fields >> word;) {
                words.push_back(std::move(word));
            }
            const auto separator = std::find(words.begin(), words.end(), "-");
            if (words.size() < 10 or separator == words.end() or words.end() - separator < 4 or
                separator - words.begin() < 6) {
                continue;
            }
            if (unified ? separator[1] != "cgroup2"
                        : separator[1] != "cgroup" or not has_controller(separator[3], "cpu")) {
                continue;
            }
            const auto root = unescape_mount(words[3]);
            const std::filesystem::path point(unescape_mount(words[4]));
            auto relative = group.generic_string();
            if (relative == root) {
                relative.clear();
            } else if (root != "/" and relative.compare(0, root.size() + 1, root + "/") == 0) {
                relative.erase(0, root.size());
            }
            // A cgroup namespace reports membership relative to its own root.
            auto current =
                (point / std::filesystem::path(relative).relative_path()).lexically_normal();
            for (;;) {
                std::string quota;
                uint64_t period = 0;
                if (unified) {
                    std::istringstream value(read((current / "cpu.max").string()));
                    value >> quota >> period;
                } else {
                    quota = read((current / "cpu.cfs_quota_us").string());
                    std::istringstream value(read((current / "cpu.cfs_period_us").string()));
                    value >> period;
                }
                int64_t limit = 0;
                std::istringstream value(quota);
                if (value >> limit and limit > 0 and period > 0) {
                    slots = std::min(slots, std::max<uint64_t>(1, uint64_t(limit) / period));
                }
                if (current == point or current == current.parent_path()) {
                    break;
                }
                current = current.parent_path();
            }
        }
    }
    return static_cast<uint32_t>(slots);
}

}  // namespace vsag
