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

#pragma once

// Handle definitions and error helpers shared by the C ABI implementations
// (vsag_c_api.cpp and the evaluation C ABI in tools/eval/vsag_eval_c_api.cpp).

#include <vsag/dataset.h>
#include <vsag/errors.h>
#include <vsag/index.h>
#include <vsag/vsag_c_api.h>

#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

struct vsag_error {
    int32_t code;
    std::string message;
};

struct vsag_dataset {
    int64_t num_elements{0};
    int64_t dim{0};
    const int64_t* ids{nullptr};
    const float* float32_vectors{nullptr};
    const uint16_t* float16_vectors{nullptr};
    const int8_t* int8_vectors{nullptr};
    const uint32_t* index_pointers{nullptr};
    const uint32_t* indices{nullptr};
    const float* values{nullptr};
};

struct vsag_index {
    std::shared_ptr<vsag::Index> index;
    // Keeps the engine's host resources alive for as long as the index exists.
    std::shared_ptr<void> owner;
};

namespace vsag::c_api {

// Returned when allocating an error object fails; vsag_error_destroy() never frees it.
inline vsag_error*
out_of_memory_error() {
    static vsag_error error{VSAG_NO_ENOUGH_MEMORY, "out of memory"};
    return &error;
}

inline vsag_error_t*
make_error(int32_t code, const std::string& message) noexcept {
    try {
        return new vsag_error{code, message};
    } catch (...) {
        return out_of_memory_error();
    }
}

inline vsag_error_t*
make_error(const vsag::Error& error) noexcept {
    return make_error(-static_cast<int32_t>(error.type), error.message);
}

inline vsag_error_t*
invalid_argument(const std::string& message) noexcept {
    return make_error(VSAG_INVALID_ARGUMENT, message);
}

template <typename Fn>
vsag_error_t*
guarded(Fn&& fn) noexcept {
    try {
        return fn();
    } catch (const std::bad_alloc&) {
        return out_of_memory_error();
    } catch (const std::invalid_argument& e) {
        return invalid_argument(e.what());
    } catch (const std::exception& e) {
        return make_error(VSAG_INTERNAL_ERROR, e.what());
    } catch (...) {
        return make_error(VSAG_UNKNOWN_ERROR, "unknown error");
    }
}

template <typename T>
vsag_error_t*
to_error(const tl::expected<T, vsag::Error>& result) noexcept {
    return result.has_value() ? nullptr : make_error(result.error());
}

inline vsag_error_t*
require(bool condition, const char* message) noexcept {
    return condition ? nullptr : invalid_argument(message);
}

#define VSAG_REQUIRE(condition, message)                                            \
    do {                                                                            \
        if (vsag_error_t* _vsag_error = vsag::c_api::require(condition, message)) { \
            return _vsag_error;                                                     \
        }                                                                           \
    } while (0)

// A vsag::Dataset built from a handle, plus the sparse rows it points at.
struct DatasetView {
    vsag::DatasetPtr dataset;
    std::vector<vsag::SparseVector> sparse_vectors;
};

inline DatasetView
make_view(const vsag_dataset_t* handle, bool single_query) {
    if (handle == nullptr) {
        throw std::invalid_argument("dataset is NULL");
    }
    if (handle->num_elements <= 0) {
        throw std::invalid_argument("dataset num_elements must be positive");
    }
    if (single_query and handle->num_elements != 1) {
        throw std::invalid_argument("dataset must hold exactly one vector");
    }
    const bool is_sparse = handle->index_pointers != nullptr;
    const bool is_dense = handle->float32_vectors != nullptr or
                          handle->float16_vectors != nullptr or handle->int8_vectors != nullptr;
    if (not is_sparse and not is_dense) {
        throw std::invalid_argument("dataset has no vectors");
    }
    if (is_dense and handle->dim <= 0) {
        throw std::invalid_argument("dataset dim must be positive");
    }

    DatasetView view;
    view.dataset = vsag::Dataset::Make();
    view.dataset->Owner(false)->NumElements(handle->num_elements);
    if (handle->dim > 0) {
        view.dataset->Dim(handle->dim);
    }
    if (handle->ids != nullptr) {
        view.dataset->Ids(handle->ids);
    }
    if (handle->float32_vectors != nullptr) {
        view.dataset->Float32Vectors(handle->float32_vectors);
    }
    if (handle->float16_vectors != nullptr) {
        view.dataset->Float16Vectors(handle->float16_vectors);
    }
    if (handle->int8_vectors != nullptr) {
        view.dataset->Int8Vectors(handle->int8_vectors);
    }
    if (is_sparse) {
        const auto* pointers = handle->index_pointers;
        if (pointers[0] != 0) {
            throw std::invalid_argument("index_pointers[0] must be 0");
        }
        view.sparse_vectors.resize(handle->num_elements);
        for (int64_t i = 0; i < handle->num_elements; ++i) {
            const uint32_t begin = pointers[i];
            const uint32_t end = pointers[i + 1];
            if (end < begin) {
                throw std::invalid_argument("index_pointers must be non-decreasing");
            }
            if (end > begin and (handle->indices == nullptr or handle->values == nullptr)) {
                throw std::invalid_argument("sparse indices and values must not be NULL");
            }
            auto& row = view.sparse_vectors[i];
            row.len_ = end - begin;
            row.ids_ = const_cast<uint32_t*>(handle->indices) + begin;
            row.vals_ = const_cast<float*>(handle->values) + begin;
        }
        view.dataset->SparseVectors(view.sparse_vectors.data());
    }
    return view;
}

}  // namespace vsag::c_api
