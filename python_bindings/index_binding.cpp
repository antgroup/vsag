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

#include <algorithm>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <vector>

#include "binding.h"
#include "fmt/format.h"
#include "vsag/constants.h"
#include "vsag/dataset.h"
#include "vsag/vsag.h"

namespace {

template <typename T>
using ContiguousArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

[[noreturn]] void
throw_vsag_error(const char* operation, const vsag::Error& error) {
    auto type = py::module_::import("_pyvsag").attr("VsagError");
    auto instance = type(fmt::format(
        "{} failed (code {}): {}", operation, static_cast<int>(error.type), error.message));
    instance.attr("code") = static_cast<int>(error.type);
    instance.attr("operation") = operation;
    PyErr_SetObject(type.ptr(), instance.ptr());
    throw py::error_already_set();
}

[[noreturn]] void
throw_file_error(const std::string& message) {
    PyErr_SetString(PyExc_OSError, message.c_str());
    throw py::error_already_set();
}

int64_t
to_int64(uint64_t value) {
    if (value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::invalid_argument("value exceeds int64 range");
    }
    return static_cast<int64_t>(value);
}

py::ssize_t
to_ssize(uint64_t value) {
    if (value > static_cast<uint64_t>(std::numeric_limits<py::ssize_t>::max())) {
        throw std::invalid_argument("array size exceeds Python ssize range");
    }
    return static_cast<py::ssize_t>(value);
}

uint64_t
checked_product(uint64_t count, uint64_t width) {
    if (width != 0 &&
        count > static_cast<uint64_t>(std::numeric_limits<py::ssize_t>::max()) / width) {
        throw std::invalid_argument("array size product exceeds Python ssize range");
    }
    return count * width;
}

template <typename T>
py::array_t<T>
make_array(uint64_t count) {
    checked_product(count, sizeof(T));
    return py::array_t<T>({to_ssize(count)}, {static_cast<py::ssize_t>(sizeof(T))});
}

py::tuple
search_arrays(const vsag::DatasetPtr& result, const char* operation) {
    const auto dim = result->GetDim();
    if (dim < 0) {
        throw_vsag_error(
            operation,
            {vsag::ErrorType::INTERNAL_ERROR, "core returned a negative search result size"});
    }
    auto ids = make_array<int64_t>(dim);
    auto distances = make_array<float>(dim);
    if (dim != 0) {
        std::copy_n(result->GetIds(), dim, ids.mutable_data());
        std::copy_n(result->GetDistances(), dim, distances.mutable_data());
    }
    return py::make_tuple(ids, distances);
}

enum class DenseVectorKind { FLOAT32, FLOAT16, BFLOAT16, UNSUPPORTED };

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
    return DenseVectorKind::UNSUPPORTED;
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
        case DenseVectorKind::UNSUPPORTED:
            return "";
    }
    return "";
}

void
validate_dense_index_kind(DenseVectorKind kind, const char* operation_name) {
    if (kind == DenseVectorKind::UNSUPPORTED) {
        throw std::invalid_argument(
            fmt::format("{} only supports indexes with dtype in [{}, {}, {}]",
                        operation_name,
                        vsag::DATATYPE_FLOAT32,
                        vsag::DATATYPE_FLOAT16,
                        vsag::DATATYPE_BFLOAT16));
    }
}

py::buffer_info
validate_dense_array(const py::array& array,
                     DenseVectorKind kind,
                     const char* arg_name,
                     int64_t expected_size = -1,
                     uint64_t expected_dim = 0,
                     uint64_t expected_rows = 0) {
    auto buf = array.request();
    checked_product(static_cast<uint64_t>(buf.size), static_cast<uint64_t>(buf.itemsize));
    const auto dtype_name = py::str(array.attr("dtype")).cast<std::string>();
    if (dtype_name != expected_numpy_dtype_name(kind)) {
        throw std::invalid_argument(
            fmt::format("{} must be {} array", arg_name, expected_numpy_dtype(kind)));
    }
    if (expected_size < 0) {
        if (buf.ndim != 1) {
            throw std::invalid_argument(fmt::format("{} must be 1-dimensional", arg_name));
        }
        if (!(array.flags() & py::array::c_style)) {
            throw std::invalid_argument(fmt::format("{} must be contiguous", arg_name));
        }
        return buf;
    }

    uint64_t actual_size = 0;
    if (buf.ndim == 1) {
        if (!(array.flags() & py::array::c_style)) {
            throw std::invalid_argument(fmt::format("{} must be contiguous", arg_name));
        }
        actual_size = static_cast<uint64_t>(buf.shape[0]);
    } else if (buf.ndim == 2) {
        if (!(array.flags() & py::array::c_style)) {
            throw std::invalid_argument(
                fmt::format("{} must be contiguous in row-major order", arg_name));
        }
        if (static_cast<uint64_t>(buf.shape[0]) != expected_rows) {
            throw std::invalid_argument(
                fmt::format("{} first dimension ({}) must equal num_elements ({})",
                            arg_name,
                            buf.shape[0],
                            expected_rows));
        }
        if (static_cast<uint64_t>(buf.shape[1]) != expected_dim) {
            throw std::invalid_argument(fmt::format("{} second dimension ({}) must equal dim ({})",
                                                    arg_name,
                                                    buf.shape[1],
                                                    expected_dim));
        }
        actual_size = checked_product(buf.shape[0], buf.shape[1]);
    } else {
        throw std::invalid_argument(fmt::format(
            "{} must be a 1-dimensional array or a 2-dimensional contiguous matrix", arg_name));
    }

    if (actual_size != static_cast<uint64_t>(expected_size)) {
        throw std::invalid_argument(
            fmt::format("{} size ({}) must equal {}", arg_name, actual_size, expected_size));
    }
    return buf;
}

void
validate_ids(const ContiguousArray<int64_t>& ids, uint64_t expected_size) {
    auto buf = ids.request();
    if (buf.ndim != 1) {
        throw std::invalid_argument("ids must be 1-dimensional");
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
build_sparse_vectors_from_csr(const ContiguousArray<uint32_t>& index_pointers,
                              const ContiguousArray<uint32_t>& indices,
                              const ContiguousArray<float>& values) {
    auto buf_ptr = index_pointers.request();
    auto buf_idx = indices.request();
    auto buf_val = values.request();

    if (buf_ptr.ndim != 1 || buf_idx.ndim != 1 || buf_val.ndim != 1) {
        throw std::invalid_argument("all inputs must be 1-dimensional");
    }

    if (buf_ptr.shape[0] < 1) {
        throw std::invalid_argument("index_pointers length must be at least 1");
    }
    if (static_cast<uint64_t>(buf_ptr.shape[0] - 1) > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("CSR row count exceeds uint32 range");
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
    for (uint64_t i = 1; i <= num_elements; ++i) {
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
            throw_vsag_error("create_index", index.error());
        }
        dense_vector_kind_ = parse_dense_vector_kind(parameters);
    }

    std::vector<int64_t>
    Build(py::array vectors, ContiguousArray<int64_t> ids, uint64_t num_elements, uint64_t dim) {
        to_int64(num_elements);
        to_int64(dim);
        validate_dense_index_kind(dense_vector_kind_, "build");
        auto buf = validate_dense_array(vectors,
                                        dense_vector_kind_,
                                        "vectors",
                                        to_int64(checked_product(num_elements, dim)),
                                        dim,
                                        num_elements);
        validate_ids(ids, num_elements);

        auto dataset = vsag::Dataset::Make();
        dataset->Owner(false)
            ->Dim(to_int64(dim))
            ->NumElements(to_int64(num_elements))
            ->Ids(ids.data());
        set_dense_vectors(dataset, buf, dense_vector_kind_);

        auto build_result = index_->Build(dataset);
        if (!build_result.has_value()) {
            throw_vsag_error("build", build_result.error());
        }
        return build_result.value();
    }

    std::vector<int64_t>
    SparseBuild(const ContiguousArray<uint32_t>& index_pointers,
                const ContiguousArray<uint32_t>& indices,
                const ContiguousArray<float>& values,
                const ContiguousArray<int64_t>& ids) {
        auto batch = build_sparse_vectors_from_csr(index_pointers, indices, values);

        validate_ids(ids, batch.num_elements);

        auto dataset = vsag::Dataset::Make();
        dataset->Owner(false)
            ->NumElements(batch.num_elements)
            ->Ids(ids.data())
            ->SparseVectors(batch.sparse_vectors.data());

        auto result = index_->Build(dataset);
        if (!result.has_value()) {
            throw_vsag_error("build", result.error());
        }
        return result.value();
    }

    py::object
    KnnSearch(py::array vector, uint64_t k, std::string& parameters) {
        validate_dense_index_kind(dense_vector_kind_, "knn_search");
        auto buf = validate_dense_array(vector, dense_vector_kind_, "vector");

        auto query = vsag::Dataset::Make();
        query->NumElements(1)->Dim(to_int64(static_cast<uint64_t>(buf.shape[0])))->Owner(false);
        set_dense_vectors(query, buf, dense_vector_kind_);

        auto result = index_->KnnSearch(query, to_int64(k), parameters);
        if (!result.has_value()) {
            throw_vsag_error("knn_search", result.error());
        }
        return search_arrays(result.value(), "knn_search");
    }

    py::tuple
    KnnSearchWithStatistics(const py::array& vector, uint64_t k, const std::string& parameters) {
        validate_dense_index_kind(dense_vector_kind_, "knn_search_with_statistics");
        auto buf = validate_dense_array(vector, dense_vector_kind_, "vector");
        auto query = vsag::Dataset::Make();
        query->NumElements(1)->Dim(to_int64(buf.shape[0]))->Owner(false);
        set_dense_vectors(query, buf, dense_vector_kind_);
        auto result = index_->KnnSearch(query, to_int64(k), parameters);
        if (!result.has_value()) {
            throw_vsag_error("knn_search_with_statistics", result.error());
        }
        auto arrays = search_arrays(result.value(), "knn_search_with_statistics");
        return py::make_tuple(arrays[0], arrays[1], py::str(result.value()->GetStatistics()));
    }

    py::tuple
    SparseKnnSearch(const ContiguousArray<uint32_t>& index_pointers,
                    const ContiguousArray<uint32_t>& indices,
                    const ContiguousArray<float>& values,
                    uint32_t k,
                    const std::string& parameters) {
        auto batch = build_sparse_vectors_from_csr(index_pointers, indices, values);

        return SparseSearch(batch, k, parameters, false);
    }

    py::tuple
    SparseKnnSearchWithStatistics(const ContiguousArray<uint32_t>& index_pointers,
                                  const ContiguousArray<uint32_t>& indices,
                                  const ContiguousArray<float>& values,
                                  uint32_t k,
                                  const std::string& parameters) {
        auto batch = build_sparse_vectors_from_csr(index_pointers, indices, values);
        return SparseSearch(batch, k, parameters, true);
    }

    py::tuple
    SparseSearch(const SparseVectors& batch,
                 uint32_t k,
                 const std::string& parameters,
                 bool with_statistics) {
        const auto count = checked_product(batch.num_elements, k);
        checked_product(count, sizeof(int64_t));
        const std::vector<py::ssize_t> shape{to_ssize(batch.num_elements), to_ssize(k)};
        const std::vector<py::ssize_t> ids_strides{to_ssize(checked_product(k, sizeof(int64_t))),
                                                   sizeof(int64_t)};
        const std::vector<py::ssize_t> dist_strides{to_ssize(checked_product(k, sizeof(float))),
                                                    sizeof(float)};
        py::array_t<int64_t> ids(shape, ids_strides);
        py::array_t<float> distances(shape, dist_strides);
        if (count != 0) {
            std::fill_n(ids.mutable_data(), count, int64_t{-1});
            std::fill_n(distances.mutable_data(), count, std::numeric_limits<float>::infinity());
        }
        auto ids_view = ids.mutable_unchecked<2>();
        auto dist_view = distances.mutable_unchecked<2>();
        py::list statistics;
        for (uint64_t i = 0; i < batch.num_elements; ++i) {
            auto query = vsag::Dataset::Make();
            query->Owner(false)->NumElements(1)->SparseVectors(batch.sparse_vectors.data() + i);
            auto result = index_->KnnSearch(query, k, parameters);
            if (!result.has_value()) {
                throw_vsag_error(with_statistics ? "knn_search_with_statistics" : "knn_search",
                                 result.error());
            }
            const auto result_count = result.value()->GetDim();
            if (result_count < 0 || static_cast<uint64_t>(result_count) > k) {
                throw_vsag_error(with_statistics ? "knn_search_with_statistics" : "knn_search",
                                 {vsag::ErrorType::INTERNAL_ERROR,
                                  "core returned invalid sparse search result size"});
            }
            for (int64_t j = 0; j < result_count; ++j) {
                ids_view(i, j) = result.value()->GetIds()[j];
                dist_view(i, j) = result.value()->GetDistances()[j];
            }
            if (with_statistics) {
                statistics.append(py::str(result.value()->GetStatistics()));
            }
        }
        if (with_statistics) {
            return py::make_tuple(ids, distances, statistics);
        }
        return py::make_tuple(ids, distances);
    }

    py::object
    RangeSearch(py::array point, float threshold, std::string& parameters) {
        validate_dense_index_kind(dense_vector_kind_, "range_search");
        auto buf = validate_dense_array(point, dense_vector_kind_, "point");

        auto query = vsag::Dataset::Make();
        query->NumElements(1)->Dim(to_int64(static_cast<uint64_t>(buf.shape[0])))->Owner(false);
        set_dense_vectors(query, buf, dense_vector_kind_);

        auto result = index_->RangeSearch(query, threshold, parameters);
        if (!result.has_value()) {
            throw_vsag_error("range_search", result.error());
        }
        return search_arrays(result.value(), "range_search");
    }

    py::tuple
    RangeSearchWithStatistics(const py::array& point,
                              float threshold,
                              const std::string& parameters) {
        validate_dense_index_kind(dense_vector_kind_, "range_search_with_statistics");
        auto buf = validate_dense_array(point, dense_vector_kind_, "point");
        auto query = vsag::Dataset::Make();
        query->NumElements(1)->Dim(to_int64(buf.shape[0]))->Owner(false);
        set_dense_vectors(query, buf, dense_vector_kind_);
        auto result = index_->RangeSearch(query, threshold, parameters);
        if (!result.has_value()) {
            throw_vsag_error("range_search_with_statistics", result.error());
        }
        auto arrays = search_arrays(result.value(), "range_search_with_statistics");
        return py::make_tuple(arrays[0], arrays[1], py::str(result.value()->GetStatistics()));
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
        throw_vsag_error("get_min_max_id", result.error());
    }

    std::vector<int64_t>
    Add(py::array vectors, ContiguousArray<int64_t> ids, uint64_t num_elements, uint64_t dim) {
        to_int64(num_elements);
        to_int64(dim);
        validate_dense_index_kind(dense_vector_kind_, "add");
        auto buf = validate_dense_array(vectors,
                                        dense_vector_kind_,
                                        "vectors",
                                        to_int64(checked_product(num_elements, dim)),
                                        dim,
                                        num_elements);
        validate_ids(ids, num_elements);

        auto dataset = vsag::Dataset::Make();
        dataset->Owner(false)
            ->Dim(to_int64(dim))
            ->NumElements(to_int64(num_elements))
            ->Ids(ids.data());
        set_dense_vectors(dataset, buf, dense_vector_kind_);

        auto result = index_->Add(dataset);
        if (!result.has_value()) {
            throw_vsag_error("add", result.error());
        }
        return result.value();
    }

    uint32_t
    Remove(const ContiguousArray<int64_t>& ids) {
        auto buf = ids.request();
        if (buf.ndim != 1) {
            throw std::invalid_argument("ids must be 1-dimensional");
        }

        std::vector<int64_t> ids_vec(ids.data(), ids.data() + buf.shape[0]);
        auto result = index_->Remove(ids_vec);
        if (result.has_value()) {
            return result.value();
        }
        throw_vsag_error("remove", result.error());
    }

    py::array_t<float>
    CalcDistancesById(const py::array_t<float, py::array::c_style | py::array::forcecast>& query,
                      const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& ids) {
        auto buf_query = query.request();
        auto buf_ids = ids.request();

        if (buf_query.ndim != 1 || buf_ids.ndim != 1) {
            throw std::invalid_argument("query and ids must be 1-dimensional");
        }

        const int64_t count = buf_ids.shape[0];
        // Specify the byte stride explicitly: pybind11 2.11 infers a zero itemsize
        // for NumPy 2 dtypes, otherwise all output elements alias the same slot.
        auto distances = make_array<float>(count);
        auto dist_view = distances.mutable_unchecked<1>();

        auto dataset = vsag::Dataset::Make();
        dataset->Owner(false)
            ->NumElements(1)
            ->Dim(buf_query.shape[0])
            ->Float32Vectors(query.data());
        auto result = index_->CalcDistancesById(dataset, ids.data(), count);
        if (!result.has_value()) {
            throw_vsag_error("calc_distances_by_id", result.error());
        }

        const auto* dist_data = result.value()->GetDistances();
        for (int64_t i = 0; i < count; ++i) {
            dist_view(i) = dist_data[i];
        }

        return distances;
    }

    py::array_t<float>
    CalDistanceById(const py::array_t<float, py::array::c_style | py::array::forcecast>& query,
                    const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& ids) {
        return this->CalcDistancesById(query, ids);
    }

    void
    Save(const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw_file_error(fmt::format("save: cannot open file '{}'", filename));
        }
        auto result = index_->Serialize(file);
        if (!result.has_value()) {
            throw_vsag_error("save", result.error());
        }
        file.flush();
        if (!file.good()) {
            throw_file_error(fmt::format("save: failed writing file '{}'", filename));
        }
        file.close();
        if (file.fail()) {
            throw_file_error(fmt::format("save: failed closing file '{}'", filename));
        }
    }

    void
    Load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw_file_error(fmt::format("load: cannot open file '{}'", filename));
        }
        auto result = index_->Deserialize(file);
        if (!result.has_value()) {
            throw_vsag_error("load", result.error());
        }
        if (file.bad() || file.fail()) {
            throw_file_error(fmt::format("load: failed reading file '{}'", filename));
        }
        file.close();
        if (file.fail()) {
            throw_file_error(fmt::format("load: failed closing file '{}'", filename));
        }
    }

private:
    DenseVectorKind dense_vector_kind_{DenseVectorKind::FLOAT32};
    std::shared_ptr<vsag::Index> index_;
};

}  // namespace

void
bind_index(py::module_& module) {
    module.attr("VsagError") = py::reinterpret_steal<py::object>(
        PyErr_NewException("_pyvsag.VsagError", PyExc_RuntimeError, nullptr));
    py::class_<Index>(module, "Index", R"pbdoc(
        Vector search index class for efficient similarity search.

        This class supports both dense and sparse vector indexing and searching.
        It provides methods for building indexes, performing k-NN and range searches,
        and saving/loading indexes to/from disk.

        Supported dense input types:
        - float32 indexes accept numpy.float32 vectors
        - float16 indexes accept numpy.float16 vectors
        - bfloat16 indexes accept numpy.uint16 arrays containing raw bfloat16 bits
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
         Build index from dense vectors. Return the list of IDs that failed to insert.
         Overall C++ failures raise VsagError (RuntimeError) with code and operation attributes.

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
         Return the list of IDs that failed to insert; overall failures raise VsagError.

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
             - [0] represents a zero-row batch
             - CSR and ID arrays are converted to contiguous buffers when necessary
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
                 - ids: numpy.ndarray of int64 with shape (count,) containing neighbor IDs
                 - distances: numpy.ndarray of float32 with shape (count,) containing distances to neighbors

         Note:
             - The query dtype must match the index dtype declared in the index parameters
             - Use numpy.uint16 raw-bit buffers for bfloat16 queries
             - count is the actual number of results, not necessarily k; no padding is added
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
             "(ids, distances, statistics_json_per_query). Arrays have shape (num_queries, k); "
             "valid results are row prefixes, followed by (-1, +inf) padding.")
        .def("knn_search",
             &Index::SparseKnnSearch,
             py::arg("index_pointers"),
             py::arg("indices"),
             py::arg("values"),
             py::arg("k"),
             py::arg("parameters"),
             R"pbdoc(
         Perform k-nearest neighbors search on sparse query rows in CSR format.

         Args:
             index_pointers (numpy.ndarray): 1D array of uint32 with shape (num_queries + 1,)
             indices (numpy.ndarray): 1D array of uint32 containing column indices of non-zero elements
             values (numpy.ndarray): 1D array of float32 containing non-zero values
             k (int): Number of nearest neighbors to retrieve
             parameters (str): JSON-formatted string containing search-specific parameters

         Returns:
             tuple: (ids, distances), rectangular arrays of shape (num_queries, k).
             Valid results form a prefix of each row; padding is (-1, +inf).

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
             Raises VsagError if the core operation fails.
         )pbdoc")
        .def("add",
             &Index::Add,
             py::arg("vectors"),
             py::arg("ids"),
             py::arg("num_elements"),
             py::arg("dim"),
             R"pbdoc(
         Add new dense vectors to the index dynamically.
         Return the list of IDs that failed to insert; overall failures raise VsagError.

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
             query (numpy.ndarray): 1D array of float32 values representing the query vector
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
