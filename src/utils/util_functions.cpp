
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
    if (type == DataTypes::DATA_TYPE_FLOAT) {
        *vectors_ptr = (void*)base->GetFloat32Vectors();
        *data_size_ptr = dim * sizeof(float);
    } else if (type == DataTypes::DATA_TYPE_INT8) {
        *vectors_ptr = (void*)base->GetInt8Vectors();
        *data_size_ptr = dim * sizeof(int8_t);
    } else if (type == DataTypes::DATA_TYPE_FP16 || type == DataTypes::DATA_TYPE_BF16) {
        *vectors_ptr = (void*)base->GetFloat16Vectors();
        *data_size_ptr = dim * sizeof(uint16_t);
    } else if (type == DataTypes::DATA_TYPE_BINARY) {
        *vectors_ptr = (void*)base->GetBinaryVectors();
        *data_size_ptr = BinaryVectorBytes(dim);
    } else {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("no support for this type: {}", (int)type));
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
    } else if (type == DataTypes::DATA_TYPE_FP16 || type == DataTypes::DATA_TYPE_BF16) {
        base->Float16Vectors((uint16_t*)vectors_ptr)
            ->Dim(dim)
            ->Owner(false)
            ->NumElements(num_element);
    } else if (type == DataTypes::DATA_TYPE_BINARY) {
        base->BinaryVectors(static_cast<const uint8_t*>(vectors_ptr))
            ->Dim(dim)
            ->Owner(false)
            ->NumElements(num_element);
    } else {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("no support for this type: {}", (int)type));
    }
}

namespace {

// Dense rows of a dataset in their own element type: the first vector field that is set, in the
// order float32, binary, int8, float16/bfloat16. Training samples keep the input type, so the
// quantizers that read them see the same layout as the full dataset.
struct DenseRows {
    const uint8_t* data{nullptr};
    uint64_t row_bytes{0};
    DataTypes type{DataTypes::DATA_TYPE_FLOAT};
};

DenseRows
dense_rows(const DatasetPtr& data, int64_t dim) {
    const auto elements = static_cast<uint64_t>(dim);
    if (const auto* vectors = data->GetFloat32Vectors()) {
        return {reinterpret_cast<const uint8_t*>(vectors),
                elements * sizeof(float),
                DataTypes::DATA_TYPE_FLOAT};
    }
    if (const auto* vectors = data->GetBinaryVectors()) {
        return {vectors, BinaryVectorBytes(elements), DataTypes::DATA_TYPE_BINARY};
    }
    if (const auto* vectors = data->GetInt8Vectors()) {
        return {reinterpret_cast<const uint8_t*>(vectors),
                elements * sizeof(int8_t),
                DataTypes::DATA_TYPE_INT8};
    }
    if (const auto* vectors = data->GetFloat16Vectors()) {
        return {reinterpret_cast<const uint8_t*>(vectors),
                elements * sizeof(uint16_t),
                DataTypes::DATA_TYPE_FP16};
    }
    throw VsagException(ErrorType::INVALID_ARGUMENT,
                        "sample_train_data requires float32, binary, int8 or float16 vectors");
}

}  // namespace

DatasetPtr
sample_train_data(const vsag::DatasetPtr& data,
                  int64_t total_elements,
                  int64_t dim,
                  int64_t train_sample_count,
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

    vsag::Vector<int64_t> sampled_ids(safe_allocator);
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

    // Copy whole rows of the input element type (binary rows are ceil(dim / 8) bytes).
    const auto rows = dense_rows(data, dim);
    auto* new_data_buffer = static_cast<uint8_t*>(
        safe_allocator->Allocate(static_cast<uint64_t>(sample_count) * rows.row_bytes));
    for (int64_t i = 0; i < sample_count; ++i) {
        std::memcpy(new_data_buffer + static_cast<uint64_t>(i) * rows.row_bytes,
                    rows.data + static_cast<uint64_t>(sampled_indices[i]) * rows.row_bytes,
                    rows.row_bytes);
    }

    auto sampled_dataset = std::make_shared<DatasetImpl>();
    sampled_dataset->NumElements(sample_count)->Dim(dim)->Owner(true, safe_allocator);
    switch (rows.type) {
        case DataTypes::DATA_TYPE_BINARY:
            sampled_dataset->BinaryVectors(new_data_buffer);
            break;
        case DataTypes::DATA_TYPE_INT8:
            sampled_dataset->Int8Vectors(reinterpret_cast<const int8_t*>(new_data_buffer));
            break;
        case DataTypes::DATA_TYPE_FP16:
        case DataTypes::DATA_TYPE_BF16:
            sampled_dataset->Float16Vectors(reinterpret_cast<const uint16_t*>(new_data_buffer));
            break;
        default:
            sampled_dataset->Float32Vectors(reinterpret_cast<const float*>(new_data_buffer));
            break;
    }
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
