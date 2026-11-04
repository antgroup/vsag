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

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <fstream>
#include <nlohmann/json.hpp>
#include <vector>

#include "binding.h"
#include "fmt/format.h"
#include "vsag/constants.h"
#include "vsag/dataset.h"
#include "vsag/vsag.h"

namespace {

int64_t
to_int64(uint64_t value) {
    return static_cast<int64_t>(value);
}

enum class DenseVectorKind { FLOAT32, FLOAT16, BFLOAT16, BINARY, UNSUPPORTED };

DenseVectorKind
parse_dense_vector_kind(const std::string& parameters) {
    const auto json = nlohmann::json::parse(parameters);
    const auto dtype = json.find(vsag::PARAMETER_DTYPE);
    if (dtype == json.end() || !dtype->is_string()) {
        return DenseVectorKind::FLOAT32;
    }

    const auto dtype_name = dtype->get<std::string>();
    if (dtype_name == vsag::DATATYPE_FLOAT32) {
        return DenseVectorKind::FLOAT32;
    }
    if (dtype_name == vsag::DATATYPE_FLOAT16) {
        return DenseVectorKind::FLOAT16;
    }
    if (dtype_name == vsag::DATATYPE_BFLOAT16) {
        return DenseVectorKind::BFLOAT16;
    }
    if (dtype_name == vsag::DATATYPE_BINARY) {
        return DenseVectorKind::BINARY;
    }
    return DenseVectorKind::UNSUPPORTED;
}

// Logical dimension declared in the index parameters, or 0 when absent. For binary indexes it
// counts bits and fixes the number of bytes of every packed vector.
uint64_t
parse_index_dim(const std::string& parameters) {
    const auto json = nlohmann::json::parse(parameters);
    const auto dim = json.find(vsag::PARAMETER_DIM);
    if (dim == json.end() || !dim->is_number_integer() || dim->get<int64_t>() <= 0) {
        return 0;
    }
    return dim->get<uint64_t>();
}

const char*
expected_numpy_dtype(DenseVectorKind kind) {
    switch (kind) {
        case DenseVectorKind::FLOAT32:
            return "numpy.float32";
        case DenseVectorKind::FLOAT16:
            return "numpy.float16";
        case DenseVectorKind::BFLOAT16:
            return "numpy.uint16 (raw bfloat16 bits)";
        case DenseVectorKind::BINARY:
            return "numpy.uint8 (bit-packed binary vectors)";
        case DenseVectorKind::UNSUPPORTED:
            return "unsupported";
    }
    return "unsupported";
}

const char*
expected_numpy_dtype_name(DenseVectorKind kind) {
    switch (kind) {
        case DenseVectorKind::FLOAT32:
            return "float32";
        case DenseVectorKind::FLOAT16:
            return "float16";
        case DenseVectorKind::BFLOAT16:
            return "uint16";
        case DenseVectorKind::BINARY:
            return "uint8";
        case DenseVectorKind::UNSUPPORTED:
            return "";
    }
    return "";
}

void
validate_dense_index_kind(DenseVectorKind kind, const char* operation_name) {
    if (kind == DenseVectorKind::UNSUPPORTED) {
        throw std::invalid_argument(
            fmt::format("{} only supports indexes with dtype in [{}, {}, {}, {}]",
                        operation_name,
                        vsag::DATATYPE_FLOAT32,
                        vsag::DATATYPE_FLOAT16,
                        vsag::DATATYPE_BFLOAT16,
                        vsag::DATATYPE_BINARY));
    }
}

py::buffer_info
validate_dense_array(const py::array& array,
                     DenseVectorKind kind,
                     const char* arg_name,
                     uint64_t expected_size = 0,
                     uint64_t expected_dim = 0) {
    auto buf = array.request();
    const auto dtype_name = py::str(array.attr("dtype")).cast<std::string>();
    if (dtype_name != expected_numpy_dtype_name(kind)) {
        throw std::invalid_argument(
            fmt::format("{} must be {} array", arg_name, expected_numpy_dtype(kind)));
    }
    if (expected_size == 0) {
        if (buf.ndim != 1) {
            throw std::invalid_argument(fmt::format("{} must be 1-dimensional", arg_name));
        }
        if (buf.strides[0] != buf.itemsize) {
            throw std::invalid_argument(fmt::format("{} must be contiguous", arg_name));
        }
        return buf;
    }

    uint64_t actual_size = 0;
    if (buf.ndim == 1) {
        if (buf.strides[0] != buf.itemsize) {
            throw std::invalid_argument(fmt::format("{} must be contiguous", arg_name));
        }
        actual_size = static_cast<uint64_t>(buf.shape[0]);
    } else if (buf.ndim == 2) {
        if (buf.strides[1] != buf.itemsize || buf.strides[0] != buf.itemsize * buf.shape[1]) {
            throw std::invalid_argument(
                fmt::format("{} must be contiguous in row-major order", arg_name));
        }
        if (expected_dim != 0 && static_cast<uint64_t>(buf.shape[1]) != expected_dim) {
            throw std::invalid_argument(fmt::format("{} second dimension ({}) must equal dim ({})",
                                                    arg_name,
                                                    buf.shape[1],
                                                    expected_dim));
        }
        actual_size = static_cast<uint64_t>(buf.shape[0]) * static_cast<uint64_t>(buf.shape[1]);
    } else {
        throw std::invalid_argument(fmt::format(
            "{} must be a 1-dimensional array or a 2-dimensional contiguous matrix", arg_name));
    }

    if (actual_size != expected_size) {
        throw std::invalid_argument(
            fmt::format("{} size ({}) must equal {}", arg_name, actual_size, expected_size));
    }
    return buf;
}

// Validates bit-packed binary vectors: a C-contiguous numpy.uint8 array holding `rows` vectors of
// ceil(dim / 8) bytes, either as a (rows, ceil(dim / 8)) matrix or as a flat buffer. Arrays are
// never converted implicitly, so a wrong dtype, shape or memory layout is reported instead of
// silently reinterpreted.
py::buffer_info
validate_binary_array(const py::array& array, const char* arg_name, uint64_t rows, uint64_t dim) {
    const uint64_t row_bytes = (dim + 7) / 8;
    const auto dtype_name = py::str(array.attr("dtype")).cast<std::string>();
    if (dtype_name != "uint8") {
        throw std::invalid_argument(fmt::format(
            "{} must be a numpy.uint8 array of bit-packed binary vectors, got dtype {}; "
            "pack booleans with numpy.packbits(bits, axis=-1)",
            arg_name,
            dtype_name));
    }
    auto buf = array.request();
    const bool contiguous_1d = buf.ndim == 1 && (buf.shape[0] <= 1 || buf.strides[0] == 1);
    const bool contiguous_2d = buf.ndim == 2 && (buf.shape[1] <= 1 || buf.strides[1] == 1) &&
                               (buf.shape[0] <= 1 || buf.strides[0] == buf.shape[1]);
    if (buf.ndim != 1 && buf.ndim != 2) {
        throw std::invalid_argument(fmt::format(
            "{} must be a 1-dimensional buffer or a 2-dimensional matrix of packed bytes",
            arg_name));
    }
    if (!contiguous_1d && !contiguous_2d) {
        throw std::invalid_argument(
            fmt::format("{} must be C-contiguous; pass numpy.ascontiguousarray({}) to copy it",
                        arg_name,
                        arg_name));
    }
    if (buf.ndim == 2) {
        if (static_cast<uint64_t>(buf.shape[0]) != rows ||
            static_cast<uint64_t>(buf.shape[1]) != row_bytes) {
            throw std::invalid_argument(
                fmt::format("{} shape ({}, {}) must be ({}, {}): {} vectors of ceil(dim / 8) "
                            "bytes for dim={} bits",
                            arg_name,
                            buf.shape[0],
                            buf.shape[1],
                            rows,
                            row_bytes,
                            rows,
                            dim));
        }
    } else if (static_cast<uint64_t>(buf.shape[0]) != rows * row_bytes) {
        throw std::invalid_argument(
            fmt::format("{} holds {} bytes but must hold {} = {} vectors x ceil(dim / 8) bytes "
                        "for dim={} bits",
                        arg_name,
                        buf.shape[0],
                        rows * row_bytes,
                        rows,
                        dim,
                        dim));
    }
    return buf;
}

void
validate_ids(const py::array_t<int64_t>& ids, uint64_t expected_size) {
    auto buf = ids.request();
    if (buf.ndim != 1) {
        throw std::invalid_argument("ids must be 1-dimensional");
    }
    if (buf.strides[0] != buf.itemsize) {
        throw std::invalid_argument("ids must be contiguous");
    }
    if (static_cast<uint64_t>(buf.shape[0]) != expected_size) {
        throw std::invalid_argument(fmt::format(
            "ids length ({}) must equal num_elements ({})", buf.shape[0], expected_size));
    }
}

void
set_dense_vectors(const vsag::DatasetPtr& dataset,
                  const py::buffer_info& buf,
                  DenseVectorKind kind) {
    if (kind == DenseVectorKind::FLOAT32) {
        dataset->Float32Vectors(static_cast<const float*>(buf.ptr));
        return;
    }
    if (kind == DenseVectorKind::BINARY) {
        dataset->BinaryVectors(static_cast<const uint8_t*>(buf.ptr));
        return;
    }
    dataset->Float16Vectors(static_cast<const uint16_t*>(buf.ptr));
}

struct SparseVectors {  // NOLINT(readability-identifier-naming)
    std::vector<vsag::SparseVector> sparse_vectors;
    uint32_t num_elements;
    uint32_t num_non_zeros = 0;

    explicit SparseVectors(uint32_t count) : sparse_vectors(count), num_elements(count) {
    }
};

SparseVectors
build_sparse_vectors_from_csr(const py::array_t<uint32_t>& index_pointers,
                              const py::array_t<uint32_t>& indices,
                              const py::array_t<float>& values) {
    auto buf_ptr = index_pointers.request();
    auto buf_idx = indices.request();
    auto buf_val = values.request();

    if (buf_ptr.ndim != 1 || buf_idx.ndim != 1 || buf_val.ndim != 1) {
        throw std::invalid_argument("all inputs must be 1-dimensional");
    }

    if (buf_ptr.shape[0] < 2) {
        throw std::invalid_argument("index_pointers length must be at least 2");
    }
    const auto num_elements = static_cast<uint32_t>(buf_ptr.shape[0] - 1);

    const uint32_t* ptr_data = index_pointers.data();
    const uint32_t* idx_data = indices.data();
    const float* val_data = values.data();

    const uint32_t num_non_zeros = ptr_data[num_elements];

    if (static_cast<uint64_t>(num_non_zeros) != static_cast<uint64_t>(buf_idx.shape[0])) {
        throw std::invalid_argument(
            fmt::format("Size of 'indices'({}) must equal index_pointers[last]",
                        buf_idx.shape[0],
                        num_non_zeros));
    }
    if (static_cast<uint64_t>(num_non_zeros) != static_cast<uint64_t>(buf_val.shape[0])) {
        throw std::invalid_argument(
            fmt::format("Size of 'values'({}) must equal index_pointers[last]({})",
                        buf_val.shape[0],
                        num_non_zeros));
    }

    if (ptr_data[0] != 0) {
        throw std::invalid_argument("index_pointers[0] must be 0");
    }
    for (uint32_t i = 1; i <= num_elements; ++i) {
        if (ptr_data[i] < ptr_data[i - 1]) {
            throw std::invalid_argument(
                fmt::format("index_pointers[{}]({}) > index_pointers[{}]({})",
                            i - 1,
                            ptr_data[i - 1],
                            i,
                            ptr_data[i]));
        }
    }

    SparseVectors sparse_vectors(num_elements);
    sparse_vectors.num_non_zeros = num_non_zeros;

    for (uint32_t i = 0; i < num_elements; ++i) {
        const uint32_t start = ptr_data[i];
        const uint32_t end = ptr_data[i + 1];
        const uint32_t len = end - start;

        sparse_vectors.sparse_vectors[i].len_ = len;
        sparse_vectors.sparse_vectors[i].ids_ = const_cast<uint32_t*>(idx_data + start);
        sparse_vectors.sparse_vectors[i].vals_ = const_cast<float*>(val_data + start);
    }

    return sparse_vectors;
}

class Index {
public:
    Index(const std::string& name, const std::string& parameters) {
        if (auto index = vsag::Factory::CreateIndex(name, parameters)) {
            index_ = index.value();
        } else {
            const vsag::Error error_code = index.error();
            switch (error_code.type) {
                case vsag::ErrorType::UNSUPPORTED_INDEX:
                    throw std::runtime_error("error type: UNSUPPORTED_INDEX");
                case vsag::ErrorType::INVALID_ARGUMENT:
                    throw std::runtime_error("error type: invalid_parameter");
                default:
                    throw std::runtime_error("error type: unexpectedError");
            }
        }
        dense_vector_kind_ = parse_dense_vector_kind(parameters);
        index_dim_ = parse_index_dim(parameters);
    }

    void
    Build(py::array vectors, py::array_t<int64_t> ids, uint64_t num_elements, uint64_t dim) {
        auto dataset = make_base_dataset(vectors, ids, num_elements, dim, "build");
        auto build_result = index_->Build(dataset);
        if (!build_result.has_value()) {
            throw std::runtime_error(fmt::format("build failed: {}", build_result.error().message));
        }
    }

    void
    SparseBuild(const py::array_t<uint32_t>& index_pointers,
                const py::array_t<uint32_t>& indices,
                const py::array_t<float>& values,
                const py::array_t<int64_t>& ids) {
        auto batch = build_sparse_vectors_from_csr(index_pointers, indices, values);

        auto buf_id = ids.request();
        if (buf_id.ndim != 1) {
            throw std::invalid_argument("all inputs must be 1-dimensional");
        }
        if (batch.num_elements != static_cast<uint32_t>(buf_id.shape[0])) {
            throw std::invalid_argument(
                fmt::format("Length of 'ids'({}) must match number of vectors({})",
                            buf_id.shape[0],
                            batch.num_elements));
        }

        auto dataset = vsag::Dataset::Make();
        dataset->Owner(false)
            ->NumElements(batch.num_elements)
            ->Ids(ids.data())
            ->SparseVectors(batch.sparse_vectors.data());

        index_->Build(dataset);
    }

    py::object
    KnnSearch(py::array vector, uint64_t k, std::string& parameters) {
        auto query = make_query_dataset(vector, "vector", "knn_search");

        uint64_t ids_shape[1]{k};
        uint64_t ids_strides[1]{sizeof(int64_t)};
        uint64_t dists_shape[1]{k};
        uint64_t dists_strides[1]{sizeof(float)};

        auto ids = py::array_t<int64_t>(ids_shape, ids_strides);
        auto dists = py::array_t<float>(dists_shape, dists_strides);
        auto ids_view = ids.mutable_unchecked<1>();
        auto dists_view = dists.mutable_unchecked<1>();
        for (uint64_t i = 0; i < k; ++i) {
            ids_view(i) = -1;
            dists_view(i) = -1.0f;
        }

        if (auto result = index_->KnnSearch(query, to_int64(k), parameters); result.has_value()) {
            const auto* vsag_ids = result.value()->GetIds();
            const auto* vsag_distances = result.value()->GetDistances();
            const auto count = static_cast<uint64_t>(result.value()->GetDim());
            for (uint64_t i = 0; i < k && i < count; ++i) {
                ids_view(i) = vsag_ids[i];
                dists_view(i) = vsag_distances[i];
            }
        }

        return py::make_tuple(ids, dists);
    }

    py::tuple
    KnnSearchWithStatistics(const py::array& vector, uint64_t k, const std::string& parameters) {
        auto query = make_query_dataset(vector, "vector", "knn_search_with_statistics");
        uint64_t ids_shape[1]{k};
        uint64_t ids_strides[1]{sizeof(int64_t)};
        uint64_t dists_shape[1]{k};
        uint64_t dists_strides[1]{sizeof(float)};
        py::array_t<int64_t> ids(ids_shape, ids_strides);
        py::array_t<float> dists(dists_shape, dists_strides);
        auto ids_view = ids.mutable_unchecked<1>();
        auto dists_view = dists.mutable_unchecked<1>();
        for (uint64_t i = 0; i < k; ++i) {
            ids_view(i) = -1;
            dists_view(i) = -1.0F;
        }
        auto result = index_->KnnSearch(query, to_int64(k), parameters);
        if (not result.has_value()) {
            throw std::runtime_error(fmt::format("knn search failed: {}", result.error().message));
        }
        const auto statistics = result.value()->GetStatistics();
        const auto count = static_cast<uint64_t>(result.value()->GetDim());
        for (uint64_t i = 0; i < k && i < count; ++i) {
            ids_view(i) = result.value()->GetIds()[i];
            dists_view(i) = result.value()->GetDistances()[i];
        }
        return py::make_tuple(ids, dists, py::str(statistics));
    }

    py::tuple
    SparseKnnSearch(const py::array_t<uint32_t>& index_pointers,
                    const py::array_t<uint32_t>& indices,
                    const py::array_t<float>& values,
                    uint32_t k,
                    const std::string& parameters) {
        auto batch = build_sparse_vectors_from_csr(index_pointers, indices, values);

        std::vector<uint32_t> shape{batch.num_elements, k};
        auto res_ids = py::array_t<int64_t>(shape);
        auto res_dists = py::array_t<float>(shape);

        auto ids_view = res_ids.mutable_unchecked<2>();
        auto dists_view = res_dists.mutable_unchecked<2>();

        for (uint32_t i = 0; i < batch.num_elements; ++i) {
            auto query = vsag::Dataset::Make();
            query->Owner(false)->NumElements(1)->SparseVectors(batch.sparse_vectors.data() + i);

            auto result = index_->KnnSearch(query, k, parameters);
            if (result.has_value()) {
                for (uint32_t j = 0; j < k; ++j) {
                    if (j < result.value()->GetDim()) {
                        ids_view(i, j) = result.value()->GetIds()[j];
                        dists_view(i, j) = result.value()->GetDistances()[j];
                    }
                }
            }
        }

        return py::make_tuple(res_ids, res_dists);
    }

    py::tuple
    SparseKnnSearchWithStatistics(const py::array_t<uint32_t>& index_pointers,
                                  const py::array_t<uint32_t>& indices,
                                  const py::array_t<float>& values,
                                  uint32_t k,
                                  const std::string& parameters) {
        auto batch = build_sparse_vectors_from_csr(index_pointers, indices, values);
        std::vector<uint32_t> shape{batch.num_elements, k};
        py::array_t<int64_t> res_ids(shape);
        py::array_t<float> res_dists(shape);
        auto ids_view = res_ids.mutable_unchecked<2>();
        auto dists_view = res_dists.mutable_unchecked<2>();
        py::list statistics;

        for (uint32_t i = 0; i < batch.num_elements; ++i) {
            for (uint32_t j = 0; j < k; ++j) {
                ids_view(i, j) = -1;
                dists_view(i, j) = -1.0F;
            }
            auto query = vsag::Dataset::Make();
            query->Owner(false)->NumElements(1)->SparseVectors(batch.sparse_vectors.data() + i);
            auto result = index_->KnnSearch(query, k, parameters);
            if (not result.has_value()) {
                throw std::runtime_error(
                    fmt::format("sparse knn search failed: {}", result.error().message));
            }
            const auto count = static_cast<uint64_t>(result.value()->GetDim());
            for (uint32_t j = 0; j < k and j < count; ++j) {
                ids_view(i, j) = result.value()->GetIds()[j];
                dists_view(i, j) = result.value()->GetDistances()[j];
            }
            statistics.append(py::str(result.value()->GetStatistics()));
        }

        return py::make_tuple(res_ids, res_dists, statistics);
    }

    py::object
    RangeSearch(py::array point, float threshold, std::string& parameters) {
        auto query = make_query_dataset(point, "point", "range_search");

        py::array_t<int64_t> labels;
        py::array_t<float> dists;
        if (auto result = index_->RangeSearch(query, threshold, parameters); result.has_value()) {
            const auto* ids = result.value()->GetIds();
            const auto* distances = result.value()->GetDistances();
            const auto count = static_cast<uint64_t>(result.value()->GetDim());
            labels.resize({count});
            dists.resize({count});
            auto* labels_data = labels.mutable_data();
            auto* dists_data = dists.mutable_data();
            for (uint64_t i = 0; i < count; ++i) {
                labels_data[i] = ids[i];
                dists_data[i] = distances[i];
            }
        }

        return py::make_tuple(labels, dists);
    }

    py::tuple
    RangeSearchWithStatistics(const py::array& point,
                              float threshold,
                              const std::string& parameters) {
        auto query = make_query_dataset(point, "point", "range_search_with_statistics");

        auto result = index_->RangeSearch(query, threshold, parameters);
        if (not result.has_value()) {
            throw std::runtime_error(
                fmt::format("range search failed: {}", result.error().message));
        }
        const auto count = static_cast<py::ssize_t>(result.value()->GetDim());
        py::array_t<int64_t> labels(count);
        py::array_t<float> dists(count);
        auto* labels_data = labels.mutable_data();
        auto* dists_data = dists.mutable_data();
        for (py::ssize_t i = 0; i < count; ++i) {
            labels_data[i] = result.value()->GetIds()[i];
            dists_data[i] = result.value()->GetDistances()[i];
        }
        return py::make_tuple(labels, dists, py::str(result.value()->GetStatistics()));
    }

    [[nodiscard]] int64_t
    GetNumElements() const {
        return index_->GetNumElements();
    }

    [[nodiscard]] int64_t
    GetMemoryUsage() const {
        return index_->GetMemoryUsage();
    }

    [[nodiscard]] bool
    CheckIdExist(int64_t id) const {
        return index_->CheckIdExist(id);
    }

    [[nodiscard]] py::tuple
    GetMinAndMaxId() const {
        auto result = index_->GetMinAndMaxId();
        if (result.has_value()) {
            auto [min_id, max_id] = result.value();
            return py::make_tuple(min_id, max_id);
        }
        return py::make_tuple(-1, -1);
    }

    void
    Add(py::array vectors, py::array_t<int64_t> ids, uint64_t num_elements, uint64_t dim) {
        auto dataset = make_base_dataset(vectors, ids, num_elements, dim, "add");
        auto result = index_->Add(dataset);
        if (!result.has_value()) {
            throw std::runtime_error(fmt::format("add failed: {}", result.error().message));
        }
    }

    uint32_t
    Remove(const py::array_t<int64_t>& ids) {
        auto buf = ids.request();
        if (buf.ndim != 1) {
            throw std::invalid_argument("ids must be 1-dimensional");
        }

        std::vector<int64_t> ids_vec(ids.data(), ids.data() + buf.shape[0]);
        auto result = index_->Remove(ids_vec);
        if (result.has_value()) {
            return result.value();
        }
        return 0;
    }

    py::array_t<float>
    CalcDistancesById(const py::object& query_object,
                      const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& ids) {
        auto buf_ids = ids.request();
        if (buf_ids.ndim != 1) {
            throw std::invalid_argument("query and ids must be 1-dimensional");
        }
        const int64_t count = buf_ids.shape[0];

        vsag::DatasetPtr dataset;
        // Keeps a converted float32 query alive until the distances are computed.
        py::array_t<float, py::array::c_style | py::array::forcecast> float_query;
        if (dense_vector_kind_ == DenseVectorKind::BINARY) {
            if (!py::isinstance<py::array>(query_object)) {
                throw std::invalid_argument(
                    "query must be a numpy.uint8 array of bit-packed binary vectors");
            }
            dataset = make_query_dataset(
                py::reinterpret_borrow<py::array>(query_object), "query", "calc_distances_by_id");
        } else {
            float_query =
                py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(query_object);
            if (!float_query) {
                throw std::invalid_argument("query must be convertible to a float32 array");
            }
            auto buf_query = float_query.request();
            if (buf_query.ndim != 1) {
                throw std::invalid_argument("query and ids must be 1-dimensional");
            }
            if (buf_query.shape[0] == 0) {
                throw std::runtime_error("calc_distances_by_id failed: query must not be empty");
            }
            dataset = vsag::Dataset::Make();
            dataset->Owner(false)
                ->NumElements(1)
                ->Dim(buf_query.shape[0])
                ->Float32Vectors(float_query.data());
        }
        // Specify the byte stride explicitly: pybind11 2.11 infers a zero itemsize
        // for NumPy 2 dtypes, otherwise all output elements alias the same slot.
        auto distances = py::array_t<float>({count}, {sizeof(float)});
        auto dist_view = distances.mutable_unchecked<1>();

        auto result = index_->CalcDistancesById(dataset, ids.data(), count);
        if (!result.has_value()) {
            throw std::runtime_error(
                fmt::format("calc_distances_by_id failed: {}", result.error().message));
        }

        const auto* dist_data = result.value()->GetDistances();
        for (int64_t i = 0; i < count; ++i) {
            dist_view(i) = dist_data[i];
        }

        return distances;
    }

    py::array_t<float>
    CalDistanceById(const py::object& query,
                    const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& ids) {
        return this->CalcDistancesById(query, ids);
    }

    void
    Save(const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        index_->Serialize(file);
        file.close();
    }

    void
    Load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        index_->Deserialize(file);
        file.close();
    }

private:
    // Validates a base batch and wraps it into a non-owning dataset of the index dtype.
    vsag::DatasetPtr
    make_base_dataset(const py::array& vectors,
                      const py::array_t<int64_t>& ids,
                      uint64_t num_elements,
                      uint64_t dim,
                      const char* operation_name) const {
        validate_dense_index_kind(dense_vector_kind_, operation_name);
        py::buffer_info buf;
        if (dense_vector_kind_ == DenseVectorKind::BINARY) {
            buf = validate_binary_array(vectors, "vectors", num_elements, dim);
        } else {
            buf = validate_dense_array(
                vectors, dense_vector_kind_, "vectors", num_elements * dim, dim);
        }
        validate_ids(ids, num_elements);

        auto dataset = vsag::Dataset::Make();
        dataset->Owner(false)
            ->Dim(to_int64(dim))
            ->NumElements(to_int64(num_elements))
            ->Ids(ids.data());
        set_dense_vectors(dataset, buf, dense_vector_kind_);
        return dataset;
    }

    // Validates one query vector and wraps it into a non-owning single-row dataset. Binary
    // queries must hold exactly ceil(dim / 8) bytes of the index dimension.
    vsag::DatasetPtr
    make_query_dataset(const py::array& vector,
                       const char* arg_name,
                       const char* operation_name) const {
        validate_dense_index_kind(dense_vector_kind_, operation_name);
        auto query = vsag::Dataset::Make();
        query->NumElements(1)->Owner(false);
        if (dense_vector_kind_ == DenseVectorKind::BINARY) {
            auto buf = validate_binary_array(vector, arg_name, 1, index_dim_);
            query->Dim(to_int64(index_dim_));
            set_dense_vectors(query, buf, dense_vector_kind_);
            return query;
        }
        auto buf = validate_dense_array(vector, dense_vector_kind_, arg_name);
        query->Dim(to_int64(static_cast<uint64_t>(buf.shape[0])));
        set_dense_vectors(query, buf, dense_vector_kind_);
        return query;
    }

    DenseVectorKind dense_vector_kind_{DenseVectorKind::FLOAT32};
    uint64_t index_dim_{0};
    std::shared_ptr<vsag::Index> index_;
};

}  // namespace

void
bind_index(py::module_& module) {
    py::class_<Index>(module, "Index", R"pbdoc(
        Vector search index class for efficient similarity search.

        This class supports both dense and sparse vector indexing and searching.
        It provides methods for building indexes, performing k-NN and range searches,
        and saving/loading indexes to/from disk.

        Supported dense input types:
        - float32 indexes accept numpy.float32 vectors
        - float16 indexes accept numpy.float16 vectors
        - bfloat16 indexes accept numpy.uint16 arrays containing raw bfloat16 bits
        - binary indexes ("dtype": "binary", "metric_type": "hamming") accept C-contiguous
          numpy.uint8 arrays of bit-packed vectors: dim counts bits, each vector holds
          ceil(dim / 8) bytes packed MSB-first (numpy.packbits default), and padding bits of
          the last byte are ignored
    )pbdoc")
        .def(py::init<const std::string&, const std::string&>(),
             py::arg("name"),
             py::arg("parameters"),
             R"pbdoc(
         Initialize a new Index instance.

         Args:
             name (str): Name of the index type (e.g., "hgraph", "ivf", "sindi")
             parameters (str): JSON string containing index configuration parameters

         )pbdoc")
        .def("build",
             &Index::Build,
             py::arg("vectors"),
             py::arg("ids"),
             py::arg("num_elements"),
             py::arg("dim"),
             R"pbdoc(
         Build index from dense vectors.

         Args:
              vectors (numpy.ndarray): Dense vector buffer with shape (num_elements * dim,) or
                  a row-major contiguous matrix with shape (num_elements, dim)
             ids (numpy.ndarray): 1D array of int64 values with shape (num_elements,)
             num_elements (int): Number of vectors in the dataset
             dim (int): Dimensionality of each vector

         Note:
              - The index parameters determine the VSAG dtype
              - Accepted dense inputs are numpy.float32 for float32 indexes, numpy.float16 for float16 indexes,
                and numpy.uint16 raw-bit buffers for bfloat16 indexes
              - Binary indexes take numpy.uint8 bit-packed vectors with shape (num_elements, ceil(dim / 8))
                or (num_elements * ceil(dim / 8),); dim is the number of bits
              - Dense build inputs may be passed as either a flat array or a contiguous 2D matrix
         )pbdoc")
        .def("build",
             &Index::SparseBuild,
             py::arg("index_pointers"),
             py::arg("indices"),
             py::arg("values"),
             py::arg("ids"),
             R"pbdoc(
         Build index from sparse vectors in CSR (Compressed Sparse Row) format.

         Args:
             index_pointers (numpy.ndarray): 1D array of uint32 values with shape (num_elements + 1,)
             indices (numpy.ndarray): 1D array of uint32 values containing column indices
             values (numpy.ndarray): 1D array of float32 values containing non-zero element values
             ids (numpy.ndarray): 1D array of int64 values with shape (num_elements,)

         CSR Format Requirements:
             - index_pointers[0] must be 0
             - index_pointers should be monotonically non-decreasing
             - len(indices) == len(values) == index_pointers[-1]
             - All arrays use 0-based indexing
         )pbdoc")
        .def("knn_search",
             &Index::KnnSearch,
             py::arg("vector"),
             py::arg("k"),
             py::arg("parameters"),
             R"pbdoc(
         Perform k-nearest neighbors search on a single dense query vector.

         Args:
             vector (numpy.ndarray): 1D dense query vector buffer
             k (int): Number of nearest neighbors to retrieve
             parameters (str): JSON-formatted string containing search-specific parameters

         Returns:
             tuple: (ids, distances) where:
                 - ids: numpy.ndarray of int64 with shape (k,) containing neighbor IDs
                 - distances: numpy.ndarray of float32 with shape (k,) containing distances to neighbors

         Note:
             - The query dtype must match the index dtype declared in the index parameters
             - Use numpy.uint16 raw-bit buffers for bfloat16 queries
             - Binary queries are numpy.uint8 buffers of exactly ceil(dim / 8) bytes; distances
               are exact Hamming distances (numbers of differing bits)
         )pbdoc")
        .def("knn_search_with_statistics",
             &Index::KnnSearchWithStatistics,
             py::arg("vector"),
             py::arg("k"),
             py::arg("parameters"),
             "Dense k-nearest-neighbor search returning (ids, distances, statistics_json).")
        .def("knn_search_with_statistics",
             &Index::SparseKnnSearchWithStatistics,
             py::arg("index_pointers"),
             py::arg("indices"),
             py::arg("values"),
             py::arg("k"),
             py::arg("parameters"),
             "Sparse k-nearest-neighbor search returning "
             "(ids, distances, statistics_json_per_query).")
        .def("knn_search",
             &Index::SparseKnnSearch,
             py::arg("index_pointers"),
             py::arg("indices"),
             py::arg("values"),
             py::arg("k"),
             py::arg("parameters"),
             R"pbdoc(
         Perform k-nearest neighbors search on a single sparse query vector in CSR format.

         Args:
             index_pointers (numpy.ndarray): 1D array of uint32 with shape (2,) = [0, nnz]
             indices (numpy.ndarray): 1D array of uint32 containing column indices of non-zero elements
             values (numpy.ndarray): 1D array of float32 containing non-zero values
             k (int): Number of nearest neighbors to retrieve
             parameters (str): JSON-formatted string containing search-specific parameters

         Returns:
             tuple: (ids, distances) with the same format as dense knn_search

         )pbdoc")
        .def("range_search",
             &Index::RangeSearch,
             py::arg("point"),
             py::arg("threshold"),
             py::arg("parameters"),
             R"pbdoc(
         Perform range search to find all vectors within a specified distance threshold.

         Args:
             point (numpy.ndarray): 1D dense query vector buffer
             threshold (float): Maximum distance threshold for inclusion in results
             parameters (str): JSON-formatted string containing search-specific parameters

         Returns:
             tuple: (ids, distances) where:
                 - ids: numpy.ndarray of int64 containing corresponding IDs
                 - distances: numpy.ndarray of float32 containing distances to all qualifying vectors

         Note:
             - The query dtype must match the index dtype declared in the index parameters
             - Use numpy.uint16 raw-bit buffers for bfloat16 queries
             - For binary indexes a vector is returned when its Hamming distance is <= threshold
         )pbdoc")
        .def("range_search_with_statistics",
             &Index::RangeSearchWithStatistics,
             py::arg("point"),
             py::arg("threshold"),
             py::arg("parameters"),
             "Dense range search returning (ids, distances, statistics_json).")
        .def("save",
             &Index::Save,
             py::arg("filename"),
             R"pbdoc(
         Save the built index to a binary file.

         Args:
             filename (str): Path to the output file where the index will be saved

         Note:
             The saved file can be loaded later using the load() method.
         )pbdoc")
        .def("load",
             &Index::Load,
             py::arg("filename"),
             R"pbdoc(
         Load a previously saved index from a binary file.

         Args:
             filename (str): Path to the input file containing the saved index

         Note:
             The Index object must be constructed with the same parameters
             that were used when the index was originally built and saved.
         )pbdoc")
        .def("get_num_elements",
             &Index::GetNumElements,
             R"pbdoc(
         Get the number of elements in the index.

         Returns:
             int: Number of vectors currently stored in the index.
         )pbdoc")
        .def("get_memory_usage",
             &Index::GetMemoryUsage,
             R"pbdoc(
         Get the memory usage of the index in bytes.

         Returns:
             int: Memory occupied by the index in bytes.
         )pbdoc")
        .def("check_id_exist",
             &Index::CheckIdExist,
             py::arg("id"),
             R"pbdoc(
         Check if a specific ID exists in the index.

         Args:
             id (int): The ID to check for existence.

         Returns:
             bool: True if the ID exists, False otherwise.
         )pbdoc")
        .def("get_min_max_id",
             &Index::GetMinAndMaxId,
             R"pbdoc(
         Get the minimum and maximum IDs in the index.

         Returns:
             tuple: (min_id, max_id) where:
                 - min_id (int): Minimum ID in the index
                 - max_id (int): Maximum ID in the index

         Note:
             Returns (-1, -1) if the operation fails.
         )pbdoc")
        .def("add",
             &Index::Add,
             py::arg("vectors"),
             py::arg("ids"),
             py::arg("num_elements"),
             py::arg("dim"),
             R"pbdoc(
         Add new dense vectors to the index dynamically.

         Args:
              vectors (numpy.ndarray): Dense vector buffer with shape (num_elements * dim,) or
                  a row-major contiguous matrix with shape (num_elements, dim)
             ids (numpy.ndarray): 1D array of int64 values with shape (num_elements,)
             num_elements (int): Number of vectors to add
             dim (int): Dimensionality of each vector

         Raises:
             RuntimeError: If the add operation fails.

         Note:
              - The vector dtype must match the index dtype declared in the index parameters
              - Use numpy.uint16 raw-bit buffers for bfloat16 additions
              - Binary indexes take numpy.uint8 bit-packed vectors of ceil(dim / 8) bytes each
              - Dense add inputs may be passed as either a flat array or a contiguous 2D matrix
          )pbdoc")
        .def("remove",
             &Index::Remove,
             py::arg("ids"),
             R"pbdoc(
         Remove vectors from the index by their IDs.

         Args:
             ids (numpy.ndarray): 1D array of int64 IDs to remove

         Returns:
             int: Number of vectors successfully removed from the index.
         )pbdoc")
        .def("calc_distances_by_id",
             &Index::CalcDistancesById,
             py::arg("query"),
             py::arg("ids"),
             R"pbdoc(
         Calculate distances between a query vector and vectors specified by IDs.

         Args:
             query (numpy.ndarray): 1D query vector; float32 values, or for binary indexes a
                 numpy.uint8 buffer of ceil(dim / 8) bit-packed bytes
             ids (numpy.ndarray): 1D array of int64 IDs to calculate distances for

         Returns:
             numpy.ndarray: 1D float32 distances in ID order; missing IDs produce -1.

         Notes:
             Query length must match the index dimension. Strided inputs are copied
             to contiguous arrays. C++ operation failures raise RuntimeError rather
             than returning missing-ID sentinels. Only one dense query is accepted.
         )pbdoc")
        .def("cal_distance_by_id",
             &Index::CalDistanceById,
             py::arg("query"),
             py::arg("ids"),
             R"pbdoc(
         Deprecated alias for calc_distances_by_id.
         )pbdoc");
}
