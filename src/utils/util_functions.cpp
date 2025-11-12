
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

#include <iomanip>
#include <nlohmann/json.hpp>
#include <random>

#include "datacell/bucket_datacell.h"
#include "dataset_impl.h"
#include "impl/allocator/safe_allocator.h"
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
    auto* external_raw_json = external_json.GetInnerJson();
    auto* inner_raw_json = inner_json.GetInnerJson();
    for (const auto& [key, value] : external_raw_json->items()) {
        auto ranges = param_map.equal_range(key);
        if (ranges.first == ranges.second) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                fmt::format("invalid config param: {}", key));
        }
        for (auto iter = ranges.first; iter != ranges.second; ++iter) {
            const auto& vec = iter->second;
            auto* json = inner_raw_json;
            for (const auto& str : vec) {
                json = &(json->operator[](str));
            }
            *json = value;
        }
    }
}

std::tuple<DatasetPtr, float*, int64_t*>
create_fast_dataset(int64_t dim, Allocator* allocator) {
    auto dataset = Dataset::Make();
    dataset->Dim(static_cast<int64_t>(dim))->NumElements(1)->Owner(true, allocator);
    if (dim == 0) {
        return {dataset, nullptr, nullptr};
    }
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
    std::tm now_tm{};
    if (localtime_r(&now_c, &now_tm) == nullptr) {
        return "invalid time";
    }
    std::ostringstream oss;
    oss << std::put_time(&now_tm, "%Y-%m-%d %H:%M:%S");
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

void
get_vectors(DataTypes type,
            int64_t dim,
            const vsag::DatasetPtr& base,
            void** vectors_ptr,
            size_t* data_size_ptr) {
    if (type == DataTypes::DATA_TYPE_FLOAT) {
        *vectors_ptr = (void*)base->GetFloat32Vectors();
        *data_size_ptr = dim * sizeof(float);
    } else if (type == DataTypes::DATA_TYPE_INT8) {
        *vectors_ptr = (void*)base->GetInt8Vectors();
        *data_size_ptr = dim * sizeof(int8_t);
    } else {
        throw std::invalid_argument(fmt::format("no support for this metric: {}", (int)type));
    }
}

void
set_dataset(DataTypes type,
            int64_t dim,
            const DatasetPtr& base,
            const void* vectors_ptr,
            uint32_t num_element) {
    if (type == DataTypes::DATA_TYPE_FLOAT) {
        base->Float32Vectors((float*)vectors_ptr)->Dim(dim)->Owner(false)->NumElements(num_element);
    } else if (type == DataTypes::DATA_TYPE_INT8) {
        base->Int8Vectors((int8_t*)vectors_ptr)->Dim(dim)->Owner(false)->NumElements(num_element);
    } else {
        throw std::invalid_argument(fmt::format("no support for this type: {}", (int)type));
    }
}

DatasetPtr
sample_train_data(const vsag::DatasetPtr& data,
                  int64_t total_elements,
                  int64_t dim,
                  int64_t train_sample_count,
                  Allocator* allocator) {
    const int64_t min_train_size = 512;
    const int64_t max_train_size = 65536;

    int64_t sample_count =
        std::min({max_train_size, total_elements, std::max(min_train_size, train_sample_count)});

    // If no sampling is needed, return the original dataset
    if (sample_count >= total_elements) {
        return data;
    }

    // If the allocator is null, use the default allocator
    std::shared_ptr<Allocator> owned_allocator;
    Allocator* safe_allocator;
    if (allocator != nullptr) {
        safe_allocator = allocator;
    } else {
        owned_allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
        safe_allocator = owned_allocator.get();
    }

    vsag::Vector<float> sampled_data_buffer(safe_allocator);
    vsag::Vector<int64_t> sampled_ids(safe_allocator);
    vsag::Vector<int64_t> sampled_indices(safe_allocator);
    sampled_indices.reserve(sample_count);

    int64_t actual_size = std::min(sample_count, total_elements);
    sampled_indices.resize(actual_size);
    std::iota(sampled_indices.begin(), sampled_indices.end(), 0);
    std::random_device rd;
    std::mt19937_64 gen(rd());
    for (int64_t i = sample_count; i < total_elements; ++i) {
        std::uniform_int_distribution<int64_t> dist(0, i);
        int64_t j = dist(gen);
        if (j < sample_count) {
            sampled_indices[j] = i;
        }
    }
    sampled_data_buffer.resize(sample_count * dim);
    const auto* original_data = data->GetFloat32Vectors();
    for (int64_t i = 0; i < sample_count; ++i) {
        std::copy(original_data + sampled_indices[i] * dim,
                  original_data + (sampled_indices[i] + 1) * dim,
                  sampled_data_buffer.data() + i * dim);
    }
    auto* new_data_buffer =
        static_cast<float*>(safe_allocator->Allocate(sample_count * dim * sizeof(float)));
    std::copy(sampled_data_buffer.begin(), sampled_data_buffer.end(), new_data_buffer);

    auto sampled_dataset = std::make_shared<DatasetImpl>();
    sampled_dataset->NumElements(sample_count)
        ->Dim(dim)
        ->Float32Vectors(new_data_buffer)
        ->Owner(true, safe_allocator);
    if (data->GetIds() != nullptr) {
        sampled_ids.reserve(sample_count);
        const auto* original_ids = data->GetIds();
        for (int64_t i = 0; i < sample_count; ++i) {
            sampled_ids.push_back(original_ids[sampled_indices[i]]);
        }

        auto* new_ids_buffer =
            static_cast<int64_t*>(safe_allocator->Allocate(sample_count * sizeof(int64_t)));
        std::copy(sampled_ids.begin(), sampled_ids.end(), new_ids_buffer);

        sampled_dataset->Ids(new_ids_buffer)->Owner(true, safe_allocator);
    }
    return sampled_dataset;
}

}  // namespace vsag
