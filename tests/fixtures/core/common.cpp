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

#include "common.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>

#include "algorithm/pyramid.h"
#include "data/vector_generator.h"
#include "fmt/format.h"

namespace fixtures {

const int RABITQ_MIN_RACALL_DIM = 960;

static std::vector<int>
select_dims(const std::vector<int>& dims, uint64_t count, int seed, int limited_dim) {
    if (count == -1 || count >= dims.size()) {
        return dims;
    }
    if (limited_dim > 0) {
        auto it = std::upper_bound(dims.begin(), dims.end(), limited_dim);
        if (it != dims.begin()) {
            std::vector<int> result(dims.begin(), it);
            if (result.size() < count) {
                return result;
            } else {
                std::shuffle(result.begin(), result.end(), std::mt19937(seed));
                result.resize(count);
                return result;
            }
        }
    }
    std::vector<int> result(dims.begin(), dims.end());
    std::shuffle(result.begin(), result.end(), std::mt19937(seed));
    result.resize(count);
    return result;
}

std::vector<int>
get_common_used_dims(uint64_t count, int seed, int limited_dim) {
    const std::vector<int> dims = {7,   8,   9,   32,  33,  48,  64,   65,  70,
                                   96,  97,  109, 128, 129, 160, 161,  192, 193,
                                   224, 225, 256, 512, 784, 960, 1024, 1536};
    return select_dims(dims, count, seed, limited_dim);
}

std::vector<int>
get_index_test_dims(uint64_t count, int seed, int limited_dim) {
    const std::vector<int> dims = {32, 57, 128, 256, 768, 1536};
    return select_dims(dims, count, seed, limited_dim);
}

bool
is_path_belong_to(const std::string& a, const std::string& b) {
    auto paths = vsag::split(a, '|');
    for (const auto& path : paths) {
        if (b.compare(0, path.size(), path) == 0) {
            return true;
        }
    }
    return false;
}

std::string
create_random_string(bool is_full) {
    const std::vector<std::string> level1 = {"a", "b", "c"};
    const std::vector<std::string> level2 = {"d", "e"};
    const std::vector<std::string> level3 = {"f", "g", "h"};

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<> distr;

    std::vector<std::string> selected_levels;

    if (is_full) {
        selected_levels.push_back(level1[distr(mt) % level1.size()]);
        selected_levels.push_back(level2[distr(mt) % level2.size()]);
        selected_levels.push_back(level3[distr(mt) % level3.size()]);
        return fmt::to_string(fmt::join(selected_levels, "/"));
    } else {
        std::uniform_int_distribution<> dist(1, 3);
        int num_path = dist(mt);
        std::vector<std::string> paths;
        for (int i = 0; i < num_path; ++i) {
            selected_levels.clear();
            int num_levels = dist(mt);
            if (num_levels >= 1) {
                selected_levels.push_back(level1[distr(mt) % level1.size()]);
            }
            if (num_levels >= 2) {
                selected_levels.push_back(level2[distr(mt) % level2.size()]);
            }
            if (num_levels == 3) {
                selected_levels.push_back(level3[distr(mt) % level3.size()]);
            }
            bool is_same = false;
            auto new_path = fmt::to_string(fmt::join(selected_levels, "/"));
            for (const auto& path : paths) {
                if (path == new_path || is_path_belong_to(new_path, path) ||
                    is_path_belong_to(path, new_path)) {
                    is_same = true;
                    break;
                }
            }
            if (not is_same) {
                paths.push_back(fmt::to_string(fmt::join(selected_levels, "/")));
            }
        }
        return fmt::to_string(fmt::join(paths, "|"));
    }
}

uint64_t
GetFileSize(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    return static_cast<uint64_t>(file.tellg());
}

std::vector<std::string>
SplitString(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::stringstream ss(s);

    while (std::getline(ss, token, delimiter)) {
        tokens.emplace_back(token);
    }

    return tokens;
}

std::vector<IOItem>
GenTestItems(uint64_t count, uint64_t max_length, uint64_t max_index) {
    std::vector<IOItem> result(count);
    std::unordered_set<uint64_t> maps;
    for (auto& item : result) {
        while (true) {
            item.start_ = (random() % max_index) * max_length;
            if (not maps.count(item.start_)) {
                maps.insert(item.start_);
                break;
            }
        };
        item.length_ = random() % max_length + 1;
        item.data_ = new uint8_t[item.length_];
        auto vec = GenerateVectors<float>(1, max_length, random(), false);
        memcpy(item.data_, vec.data(), item.length_);
    }
    return result;
}

}  // namespace fixtures
