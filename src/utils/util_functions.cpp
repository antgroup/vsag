
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

#include "util_functions.h"

#include <random>

#include "vsag_exception.h"

namespace vsag {

std::string
format_map(const std::string& str, const std::unordered_map<std::string, std::string>& mappings) {
    std::string result = str;

    for (const auto& [key, value] : mappings) {
        size_t pos = result.find("{" + key + "}");
        while (pos != std::string::npos) {
            result.replace(pos, key.length() + 2, value);
            pos = result.find("{" + key + "}");
        }
    }
    return result;
}

void
mapping_external_param_to_inner(const JsonType& external_json,
                                ConstParamMap& param_map,
                                JsonType& inner_json) {
    for (const auto& [key, value] : external_json.items()) {
        const auto& iter = param_map.find(key);

        if (iter != param_map.end()) {
            const auto& vec = iter->second;
            auto* json = &inner_json;
            for (const auto& str : vec) {
                json = &(json->operator[](str));
            }
            *json = value;
        } else {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                fmt::format("invalid config param: {}", key));
        }
    }
}

std::tuple<DatasetPtr, float*, int64_t*>
create_fast_dataset(int64_t dim, Allocator* allocator) {
    auto dataset = Dataset::Make();
    dataset->Dim(static_cast<int64_t>(dim))->NumElements(1)->Owner(true, allocator);
    auto* ids = reinterpret_cast<int64_t*>(allocator->Allocate(sizeof(int64_t) * dim));
    dataset->Ids(ids);
    auto* dists = reinterpret_cast<float*>(allocator->Allocate(sizeof(float) * dim));
    dataset->Distances(dists);
    return {dataset, dists, ids};
}

std::vector<int>
select_k_numbers(int64_t n, int k) {
    if (k > n || k <= 0) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "Invalid values for N or K");
    }

    std::vector<int> numbers(n);
    std::iota(numbers.begin(), numbers.end(), 0);

    std::random_device rd;
    std::mt19937 gen(rd());
    for (int i = 0; i < k; ++i) {
        std::uniform_int_distribution<> dist(i, static_cast<int>(n - 1));
        std::swap(numbers[i], numbers[dist(gen)]);
    }
    numbers.resize(k);
    return numbers;
}

uint64_t
next_multiple_of_power_of_two(uint64_t x, uint64_t n) {
    if (n > 63) {
        throw std::runtime_error(fmt::format("n is larger than 63, n is {}", n));
    }
    uint64_t y = 1 << n;
    auto result = (x + y - 1) & ~(y - 1);
    return result;
}

bool
check_equal_on_string_stream(std::stringstream& s1, std::stringstream& s2) {
    if (!s1.good() || !s2.good()) {
        return false;
    }

    auto get_length = [](std::stringstream& s) -> std::streamoff {
        s.seekg(0, std::ios::end);
        std::streamoff len = s.tellg();
        s.seekg(0, std::ios::beg);
        return len;
    };

    std::streamoff len1 = get_length(s1);
    std::streamoff len2 = get_length(s2);

    if (len1 != len2) {
        return false;
    }

    if (len1 == 0) {
        return true;
    }

    constexpr int64_t chunk_size = 1024L * 2L;
    char buffer1[chunk_size];
    char buffer2[chunk_size];

    while (len1 > 0) {
        int64_t chunk = std::min(static_cast<int64_t>(len1), chunk_size);

        s1.read(buffer1, chunk);
        s2.read(buffer2, chunk);

        if (!s1 || !s2 || std::memcmp(buffer1, buffer2, chunk) != 0) {
            return false;
        }

        len1 -= chunk;
    }
    return true;
}

std::vector<std::string>
split_string(const std::string& str, const char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
        tokens.emplace_back(token);
    }
    return tokens;
}

std::string
get_current_time() {
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    std::tm* now_tm = std::localtime(&now_c);
    std::ostringstream oss;
    oss << std::put_time(now_tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

static const std::string BASE64_CHARS =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

std::string
base64_encode(const std::string& in) {
    std::string out;
    int32_t val = 0;
    int32_t valb = -6;
    for (unsigned char c : in) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            out.push_back(BASE64_CHARS[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }
    if (valb > -6) {
        out.push_back(BASE64_CHARS[((val << 8) >> (valb + 8)) & 0x3F]);
    }
    while (out.size() % 4 != 0) {
        out.push_back('=');
    }
    return out;
}

std::string
base64_decode(const std::string& in) {
    std::string out;
    std::vector<int> t(256, -1);
    for (int i = 0; i < 64; i++) {
        t[BASE64_CHARS[i]] = i;
    }
    int32_t val = 0;
    int32_t valb = -8;
    for (unsigned char c : in) {
        if (t[c] == -1) {
            break;
        }
        val = (val << 6) + t[c];
        valb += 6;
        if (valb >= 0) {
            out.push_back(char((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

}  // namespace vsag
