
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

#include <cstring>
#include <iomanip>
#include <limits>
#include <random>

#include "common.h"
#include "container_types.h"
#include "dataset_impl.h"
#include "impl/allocator/safe_allocator.h"
#include "vsag_exception.h"
namespace vsag {

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
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            fmt::format("n is larger than 63, n is {}", n));
    }
    uint64_t y = uint64_t{1} << n;
    CHECK_ARGUMENT(x <= std::numeric_limits<uint64_t>::max() - (y - 1),
                   "next multiple of power of two exceeds uint64_t range");
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
    uint32_t val = 0;
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
    uint32_t val = 0;
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
            uint64_t* data_size_ptr) {
    CHECK_ARGUMENT(dim > 0, "vector dimension must be positive");
    uint64_t element_bytes;
    switch (type) {
        case DataTypes::DATA_TYPE_FLOAT:
            element_bytes = sizeof(float);
            *vectors_ptr = const_cast<float*>(base->GetFloat32Vectors());
            break;
        case DataTypes::DATA_TYPE_INT8:
            element_bytes = sizeof(int8_t);
            *vectors_ptr = const_cast<int8_t*>(base->GetInt8Vectors());
            break;
        case DataTypes::DATA_TYPE_FP16:
        case DataTypes::DATA_TYPE_BF16:
            element_bytes = sizeof(uint16_t);
            *vectors_ptr = const_cast<uint16_t*>(base->GetFloat16Vectors());
            break;
        case DataTypes::DATA_TYPE_BINARY:
            *vectors_ptr = const_cast<uint8_t*>(base->GetBinaryVectors());
            *data_size_ptr = (static_cast<uint64_t>(dim) + 7U) / 8U;
            return;
        default:
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                fmt::format("no support for this type: {}", (int)type));
    }
    CHECK_ARGUMENT(static_cast<uint64_t>(dim) <= std::numeric_limits<size_t>::max() / element_bytes,
                   "vector row size overflows");
    *data_size_ptr = static_cast<uint64_t>(dim) * element_bytes;
}

void
set_dataset(DataTypes type,
            int64_t dim,
            const DatasetPtr& base,
            const void* vectors_ptr,
            int64_t num_element) {
    if (type == DataTypes::DATA_TYPE_FLOAT) {
        base->Float32Vectors((float*)vectors_ptr)->Dim(dim)->Owner(false)->NumElements(num_element);
    } else if (type == DataTypes::DATA_TYPE_INT8) {
        base->Int8Vectors((int8_t*)vectors_ptr)->Dim(dim)->Owner(false)->NumElements(num_element);
    } else if (type == DataTypes::DATA_TYPE_FP16 || type == DataTypes::DATA_TYPE_BF16) {
        base->Float16Vectors((uint16_t*)vectors_ptr)
            ->Dim(dim)
            ->Owner(false)
            ->NumElements(num_element);
    } else if (type == DataTypes::DATA_TYPE_BINARY) {
        base->BinaryVectors((uint8_t*)vectors_ptr)
            ->Dim(dim)
            ->Owner(false)
            ->NumElements(num_element);
    } else {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("no support for this type: {}", (int)type));
    }
}

DatasetPtr
sample_train_data(const vsag::DatasetPtr& data,
                  int64_t total_elements,
                  int64_t dim,
                  int64_t train_sample_count,
                  DataTypes data_type,
                  Allocator* allocator,
                  std::optional<uint64_t> random_seed) {
    const int64_t min_train_size = 512;

    int64_t sample_count = std::min(total_elements, std::max(min_train_size, train_sample_count));

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

    void* original_data = nullptr;
    uint64_t row_bytes = 0;
    get_vectors(data_type, dim, data, &original_data, &row_bytes);
    CHECK_ARGUMENT(original_data != nullptr, "training vectors must not be null");
    CHECK_ARGUMENT(
        static_cast<uint64_t>(total_elements) <= std::numeric_limits<size_t>::max() / row_bytes,
        "training input size overflows");
    CHECK_ARGUMENT(
        static_cast<uint64_t>(sample_count) <= std::numeric_limits<size_t>::max() / row_bytes,
        "training sample size overflows");
    CHECK_ARGUMENT(
        static_cast<uint64_t>(sample_count) <= std::numeric_limits<size_t>::max() / sizeof(int64_t),
        "training sample index size overflows");
    if (data->GetIds() != nullptr and static_cast<uint64_t>(total_elements) >
                                          std::numeric_limits<size_t>::max() / sizeof(int64_t)) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "training input ID size overflows");
    }
    const auto sample_bytes = static_cast<uint64_t>(sample_count) * row_bytes;
    vsag::Vector<int64_t> sampled_indices(safe_allocator);
    sampled_indices.reserve(sample_count);

    int64_t actual_size = std::min(sample_count, total_elements);
    sampled_indices.resize(actual_size);
    std::iota(sampled_indices.begin(), sampled_indices.end(), 0);
    std::random_device rd;
    std::mt19937_64 gen(random_seed.has_value() ? *random_seed : rd());
    for (int64_t i = sample_count; i < total_elements; ++i) {
        std::uniform_int_distribution<int64_t> dist(0, i);
        int64_t j = dist(gen);
        if (j < sample_count) {
            sampled_indices[j] = i;
        }
    }
    // DatasetImpl stores only a raw allocator pointer. Retain a default allocator
    // until after the sampled dataset has released its owned buffers.
    auto sampled_dataset = std::shared_ptr<DatasetImpl>(
        new DatasetImpl(), [owned_allocator](DatasetImpl* dataset) { delete dataset; });
    // Metadata setters may allocate after registering the row pointer. Until
    // publication succeeds, only data_guard may release that storage.
    sampled_dataset->Owner(false, nullptr);
    auto release_buffer = [safe_allocator](void* buffer) { safe_allocator->Deallocate(buffer); };
    std::unique_ptr<void, decltype(release_buffer)> data_guard(
        safe_allocator->Allocate(sample_bytes), release_buffer);
    CHECK_ARGUMENT(data_guard != nullptr, "training sample allocation failed");
    auto* new_data_buffer = static_cast<uint8_t*>(data_guard.get());
    const auto* source = static_cast<const uint8_t*>(original_data);
    for (int64_t i = 0; i < sample_count; ++i) {
        std::memcpy(
            new_data_buffer + i * row_bytes, source + sampled_indices[i] * row_bytes, row_bytes);
    }
    set_dataset(data_type, dim, sampled_dataset, new_data_buffer, sample_count);
    sampled_dataset->Owner(true, safe_allocator);
    [[maybe_unused]] auto* released_data = data_guard.release();
    if (data->GetIds() != nullptr) {
        std::unique_ptr<void, decltype(release_buffer)> ids_guard(
            safe_allocator->Allocate(static_cast<uint64_t>(sample_count) * sizeof(int64_t)),
            release_buffer);
        CHECK_ARGUMENT(ids_guard != nullptr, "training sample ID allocation failed");
        auto* new_ids_buffer = static_cast<int64_t*>(ids_guard.get());
        const auto* original_ids = data->GetIds();
        for (int64_t i = 0; i < sample_count; ++i) {
            new_ids_buffer[i] = original_ids[sampled_indices[i]];
        }
        sampled_dataset->Ids(new_ids_buffer);
        [[maybe_unused]] auto* released_ids = ids_guard.release();
    }
    return sampled_dataset;
}

}  // namespace vsag
