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
#include <vector>

#include "binding.h"
#include "fmt/format.h"
#include "vsag/dataset.h"
#include "vsag/vsag.h"

namespace {

int64_t
to_int64(uint64_t value) {
    return static_cast<int64_t>(value);
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
    }

    void
    Build(py::array_t<float> vectors,
          py::array_t<int64_t> ids,
          uint64_t num_elements,
          uint64_t dim) {
        auto dataset = vsag::Dataset::Make();
        dataset->Owner(false)
            ->Dim(to_int64(dim))
            ->NumElements(to_int64(num_elements))
            ->Ids(ids.mutable_data())
            ->Float32Vectors(vectors.mutable_data());
        index_->Build(dataset);
    }

    void
    BuildFloat16(py::array vectors, py::array_t<int64_t> ids, uint64_t num_elements, uint64_t dim) {
        auto buf = vectors.request();
        if (buf.format != "e") {
            throw std::invalid_argument("vectors must be float16 (numpy.float16) array");
        }
        if (buf.ndim != 1) {
            throw std::invalid_argument("vectors must be 1-dimensional");
        }
        if (static_cast<uint64_t>(buf.shape[0]) != num_elements * dim) {
            throw std::invalid_argument(
                fmt::format("vectors length ({}) must equal num_elements * dim ({})",
                            buf.shape[0],
                            num_elements * dim));
        }
        auto buf_ids = ids.request();
        if (buf_ids.ndim != 1 || static_cast<uint64_t>(buf_ids.shape[0]) != num_elements) {
            throw std::invalid_argument(fmt::format(
                "ids length ({}) must equal num_elements ({})", buf_ids.shape[0], num_elements));
        }
        auto dataset = vsag::Dataset::Make();
        dataset->Owner(false)
            ->Dim(to_int64(dim))
            ->NumElements(to_int64(num_elements))
            ->Ids(ids.data())
            ->Float16Vectors(static_cast<const uint16_t*>(buf.ptr));
        auto build_result = index_->Build(dataset);
        if (!build_result.has_value()) {
            throw std::runtime_error(fmt::format("BuildFloat16 failed: {}", build_result.error()));
        }
    }

    void
    BuildBfloat16(py::array vectors,
                  py::array_t<int64_t> ids,
                  uint64_t num_elements,
                  uint64_t dim) {
        auto buf = vectors.request();
        if (buf.format != "H") {
            throw std::invalid_argument("vectors must be uint16 (raw bfloat16 bits) array");
        }
        if (buf.ndim != 1) {
            throw std::invalid_argument("vectors must be 1-dimensional");
        }
        if (static_cast<uint64_t>(buf.shape[0]) != num_elements * dim) {
            throw std::invalid_argument(
                fmt::format("vectors length ({}) must equal num_elements * dim ({})",
                            buf.shape[0],
                            num_elements * dim));
        }
        auto buf_ids = ids.request();
        if (buf_ids.ndim != 1 || static_cast<uint64_t>(buf_ids.shape[0]) != num_elements) {
            throw std::invalid_argument(fmt::format(
                "ids length ({}) must equal num_elements ({})", buf_ids.shape[0], num_elements));
        }
        auto dataset = vsag::Dataset::Make();
        dataset->Owner(false)
            ->Dim(to_int64(dim))
            ->NumElements(to_int64(num_elements))
            ->Ids(ids.data())
            ->BFloat16Vectors(static_cast<const uint16_t*>(buf.ptr));
        auto build_result = index_->Build(dataset);
        if (!build_result.has_value()) {
            throw std::runtime_error(fmt::format("BuildBfloat16 failed: {}", build_result.error()));
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
    KnnSearch(py::array_t<float> vector, uint64_t k, std::string& parameters) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(to_int64(vector.size()))
            ->Float32Vectors(vector.mutable_data())
            ->Owner(false);

        uint64_t ids_shape[1]{k};
        uint64_t ids_strides[1]{sizeof(int64_t)};
        uint64_t dists_shape[1]{k};
        uint64_t dists_strides[1]{sizeof(float)};

        auto ids = py::array_t<int64_t>(ids_shape, ids_strides);
        auto dists = py::array_t<float>(dists_shape, dists_strides);
        if (auto result = index_->KnnSearch(query, to_int64(k), parameters); result.has_value()) {
            auto ids_view = ids.mutable_unchecked<1>();
            auto dists_view = dists.mutable_unchecked<1>();

            const auto* vsag_ids = result.value()->GetIds();
            const auto* vsag_distances = result.value()->GetDistances();
            for (uint32_t i = 0; i < k; ++i) {
                ids_view(i) = vsag_ids[i];
                dists_view(i) = vsag_distances[i];
            }
        }

        return py::make_tuple(ids, dists);
    }

    py::object
    KnnSearchFloat16(py::array vector, uint64_t k, std::string& parameters) {
        auto buf = vector.request();
        if (buf.format != "e") {
            throw std::invalid_argument("vector must be float16 (numpy.float16) array");
        }
        if (buf.ndim != 1) {
            throw std::invalid_argument("vector must be 1-dimensional");
        }

        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(to_int64(buf.shape[0]))
            ->Float16Vectors(static_cast<const uint16_t*>(buf.ptr))
            ->Owner(false);

        uint64_t ids_shape[1]{k};
        uint64_t ids_strides[1]{sizeof(int64_t)};
        uint64_t dists_shape[1]{k};
        uint64_t dists_strides[1]{sizeof(float)};

        auto ids = py::array_t<int64_t>(ids_shape, ids_strides);
        auto dists = py::array_t<float>(dists_shape, dists_strides);
        auto ids_view = ids.mutable_unchecked<1>();
        auto dists_view = dists.mutable_unchecked<1>();
        for (uint32_t i = 0; i < k; ++i) {
            ids_view(i) = -1;
            dists_view(i) = -1.0f;
        }

        if (auto result = index_->KnnSearch(query, to_int64(k), parameters); result.has_value()) {
            const auto* vsag_ids = result.value()->GetIds();
            const auto* vsag_distances = result.value()->GetDistances();
            const auto count = static_cast<uint32_t>(result.value()->GetDim());
            for (uint32_t i = 0; i < k && i < count; ++i) {
                ids_view(i) = vsag_ids[i];
                dists_view(i) = vsag_distances[i];
            }
        }

        return py::make_tuple(ids, dists);
    }

    py::object
    KnnSearchBfloat16(py::array vector, uint64_t k, std::string& parameters) {
        auto buf = vector.request();
        if (buf.format != "H") {
            throw std::invalid_argument("vector must be uint16 (raw bfloat16 bits) array");
        }
        if (buf.ndim != 1) {
            throw std::invalid_argument("vector must be 1-dimensional");
        }

        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(to_int64(buf.shape[0]))
            ->Float16Vectors(static_cast<const uint16_t*>(buf.ptr))
            ->Owner(false);

        uint64_t ids_shape[1]{k};
        uint64_t ids_strides[1]{sizeof(int64_t)};
        uint64_t dists_shape[1]{k};
        uint64_t dists_strides[1]{sizeof(float)};

        auto ids = py::array_t<int64_t>(ids_shape, ids_strides);
        auto dists = py::array_t<float>(dists_shape, dists_strides);
        auto ids_view = ids.mutable_unchecked<1>();
        auto dists_view = dists.mutable_unchecked<1>();
        for (uint32_t i = 0; i < k; ++i) {
            ids_view(i) = -1;
            dists_view(i) = -1.0f;
        }

        if (auto result = index_->KnnSearch(query, to_int64(k), parameters); result.has_value()) {
            const auto* vsag_ids = result.value()->GetIds();
            const auto* vsag_distances = result.value()->GetDistances();
            const auto count = static_cast<uint32_t>(result.value()->GetDim());
            for (uint32_t i = 0; i < k && i < count; ++i) {
                ids_view(i) = vsag_ids[i];
                dists_view(i) = vsag_distances[i];
            }
        }

        return py::make_tuple(ids, dists);
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

    py::object
    RangeSearch(py::array_t<float> point, float threshold, std::string& parameters) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(to_int64(point.size()))
            ->Float32Vectors(point.mutable_data())
            ->Owner(false);

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

    py::object
    RangeSearchFloat16(py::array point, float threshold, std::string& parameters) {
        auto buf = point.request();
        if (buf.format != "e") {
            throw std::invalid_argument("point must be float16 (numpy.float16) array");
        }
        if (buf.ndim != 1) {
            throw std::invalid_argument("point must be 1-dimensional");
        }

        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(to_int64(buf.shape[0]))
            ->Float16Vectors(static_cast<const uint16_t*>(buf.ptr))
            ->Owner(false);

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

    py::object
    RangeSearchBfloat16(py::array point, float threshold, std::string& parameters) {
        auto buf = point.request();
        if (buf.format != "H") {
            throw std::invalid_argument("point must be uint16 (raw bfloat16 bits) array");
        }
        if (buf.ndim != 1) {
            throw std::invalid_argument("point must be 1-dimensional");
        }

        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(to_int64(buf.shape[0]))
            ->Float16Vectors(static_cast<const uint16_t*>(buf.ptr))
            ->Owner(false);

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
    Add(py::array_t<float> vectors, py::array_t<int64_t> ids, uint64_t num_elements, uint64_t dim) {
        auto dataset = vsag::Dataset::Make();
        dataset->Owner(false)
            ->Dim(to_int64(dim))
            ->NumElements(to_int64(num_elements))
            ->Ids(ids.mutable_data())
            ->Float32Vectors(vectors.mutable_data());

        auto result = index_->Add(dataset);
        if (!result.has_value()) {
            throw std::runtime_error("Failed to add vectors to index");
        }
    }

    void
    AddFloat16(py::array vectors, py::array_t<int64_t> ids, uint64_t num_elements, uint64_t dim) {
        auto buf = vectors.request();
        if (buf.format != "e") {
            throw std::invalid_argument("vectors must be float16 (numpy.float16) array");
        }
        if (buf.ndim != 1) {
            throw std::invalid_argument("vectors must be 1-dimensional");
        }
        if (static_cast<uint64_t>(buf.shape[0]) != num_elements * dim) {
            throw std::invalid_argument(
                fmt::format("vectors length ({}) must equal num_elements * dim ({})",
                            buf.shape[0],
                            num_elements * dim));
        }
        auto buf_ids = ids.request();
        if (buf_ids.ndim != 1 || static_cast<uint64_t>(buf_ids.shape[0]) != num_elements) {
            throw std::invalid_argument(fmt::format(
                "ids length ({}) must equal num_elements ({})", buf_ids.shape[0], num_elements));
        }
        auto dataset = vsag::Dataset::Make();
        dataset->Owner(false)
            ->Dim(to_int64(dim))
            ->NumElements(to_int64(num_elements))
            ->Ids(ids.data())
            ->Float16Vectors(static_cast<const uint16_t*>(buf.ptr));

        auto result = index_->Add(dataset);
        if (!result.has_value()) {
            throw std::runtime_error("Failed to add fp16 vectors to index");
        }
    }

    void
    AddBfloat16(py::array vectors, py::array_t<int64_t> ids, uint64_t num_elements, uint64_t dim) {
        auto buf = vectors.request();
        if (buf.format != "H") {
            throw std::invalid_argument("vectors must be uint16 (raw bfloat16 bits) array");
        }
        if (buf.ndim != 1) {
            throw std::invalid_argument("vectors must be 1-dimensional");
        }
        if (static_cast<uint64_t>(buf.shape[0]) != num_elements * dim) {
            throw std::invalid_argument(
                fmt::format("vectors length ({}) must equal num_elements * dim ({})",
                            buf.shape[0],
                            num_elements * dim));
        }
        auto buf_ids = ids.request();
        if (buf_ids.ndim != 1 || static_cast<uint64_t>(buf_ids.shape[0]) != num_elements) {
            throw std::invalid_argument(fmt::format(
                "ids length ({}) must equal num_elements ({})", buf_ids.shape[0], num_elements));
        }
        auto dataset = vsag::Dataset::Make();
        dataset->Owner(false)
            ->Dim(to_int64(dim))
            ->NumElements(to_int64(num_elements))
            ->Ids(ids.data())
            ->BFloat16Vectors(static_cast<const uint16_t*>(buf.ptr));

        auto result = index_->Add(dataset);
        if (!result.has_value()) {
            throw std::runtime_error("Failed to add bf16 vectors to index");
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
    CalDistanceById(const py::array_t<float>& query, const py::array_t<int64_t>& ids) {
        auto buf_query = query.request();
        auto buf_ids = ids.request();

        if (buf_query.ndim != 1 || buf_ids.ndim != 1) {
            throw std::invalid_argument("query and ids must be 1-dimensional");
        }

        const int64_t count = buf_ids.shape[0];
        auto distances = py::array_t<float>(count);
        auto dist_view = distances.mutable_unchecked<1>();

        for (int64_t i = 0; i < count; ++i) {
            dist_view(i) = -1.0F;
        }

        auto result = index_->CalDistanceById(query.data(), ids.data(), count);

        if (result.has_value()) {
            const auto* dist_data = result.value()->GetDistances();
            for (int64_t i = 0; i < count; ++i) {
                dist_view(i) = dist_data[i];
            }
        }

        return distances;
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

        Supported data types:
        - float32: standard 32-bit floating point vectors
        - float16: 16-bit IEEE 754 half-precision vectors (use build_fp16, knn_search_fp16, etc.)
        - bfloat16: 16-bit brain floating point vectors (use build_bf16, knn_search_bf16, etc.)
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
         Build index from dense float32 vectors.

         Args:
             vectors (numpy.ndarray): 1D array of float32 values with total size num_elements * dim
             ids (numpy.ndarray): 1D array of int64 values with shape (num_elements,)
             num_elements (int): Number of vectors in the dataset
             dim (int): Dimensionality of each vector

         Note:
             - The vectors array should contain num_elements * dim consecutive float32 values
         )pbdoc")
        .def("build_fp16",
             &Index::BuildFloat16,
             py::arg("vectors"),
             py::arg("ids"),
             py::arg("num_elements"),
             py::arg("dim"),
             R"pbdoc(
         Build index from dense float16 (FP16) vectors.

         Args:
             vectors (numpy.ndarray): 1D array of numpy.float16 values with total size num_elements * dim
             ids (numpy.ndarray): 1D array of int64 values with shape (num_elements,)
             num_elements (int): Number of vectors in the dataset
             dim (int): Dimensionality of each vector

         Note:
             - The vectors array must have dtype numpy.float16
             - The index parameters must specify "dtype": "float16"
             - Memory layout of numpy.float16 is identical to uint16, so no conversion is needed
         )pbdoc")
        .def("build_bf16",
             &Index::BuildBfloat16,
             py::arg("vectors"),
             py::arg("ids"),
             py::arg("num_elements"),
             py::arg("dim"),
             R"pbdoc(
         Build index from dense bfloat16 (BF16) vectors.

         Args:
             vectors (numpy.ndarray): 1D array of uint16 values (raw bfloat16 bits) with total size num_elements * dim
             ids (numpy.ndarray): 1D array of int64 values with shape (num_elements,)
             num_elements (int): Number of vectors in the dataset
             dim (int): Dimensionality of each vector

         Note:
             - The vectors array must have dtype numpy.uint16, representing raw bfloat16 bit patterns
             - The index parameters must specify "dtype": "bfloat16"
             - Since numpy has no native bfloat16 dtype, users must convert bfloat16 data to uint16
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
             vector (numpy.ndarray): 1D array of float32 values representing the query vector
             k (int): Number of nearest neighbors to retrieve
             parameters (str): JSON-formatted string containing search-specific parameters

         Returns:
             tuple: (ids, distances) where:
                 - ids: numpy.ndarray of int64 with shape (k,) containing neighbor IDs
                 - distances: numpy.ndarray of float32 with shape (k,) containing distances to neighbors

         Note:
             - Distance metric depends on the index configuration (typically L2 or inner product)
             - Results are sorted by distance (closest first)
         )pbdoc")
        .def("knn_search_fp16",
             &Index::KnnSearchFloat16,
             py::arg("vector"),
             py::arg("k"),
             py::arg("parameters"),
             R"pbdoc(
         Perform k-nearest neighbors search on a single float16 (FP16) query vector.

         Args:
             vector (numpy.ndarray): 1D array of numpy.float16 values representing the query vector
             k (int): Number of nearest neighbors to retrieve
             parameters (str): JSON-formatted string containing search-specific parameters

         Returns:
             tuple: (ids, distances) where:
                 - ids: numpy.ndarray of int64 with shape (k,) containing neighbor IDs
                 - distances: numpy.ndarray of float32 with shape (k,) containing distances to neighbors

         Note:
             - The vector array must have dtype numpy.float16
             - The index must be built with "dtype": "float16"
         )pbdoc")
        .def("knn_search_bf16",
             &Index::KnnSearchBfloat16,
             py::arg("vector"),
             py::arg("k"),
             py::arg("parameters"),
             R"pbdoc(
         Perform k-nearest neighbors search on a single bfloat16 (BF16) query vector.

         Args:
             vector (numpy.ndarray): 1D array of uint16 values (raw bfloat16 bits) representing the query vector
             k (int): Number of nearest neighbors to retrieve
             parameters (str): JSON-formatted string containing search-specific parameters

         Returns:
             tuple: (ids, distances) where:
                 - ids: numpy.ndarray of int64 with shape (k,) containing neighbor IDs
                 - distances: numpy.ndarray of float32 with shape (k,) containing distances to neighbors

         Note:
             - The vector array must have dtype numpy.uint16, representing raw bfloat16 bit patterns
             - The index must be built with "dtype": "bfloat16"
         )pbdoc")
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
             point (numpy.ndarray): 1D array of float32 values representing the query vector
             threshold (float): Maximum distance threshold for inclusion in results
             parameters (str): JSON-formatted string containing search-specific parameters

         Returns:
             tuple: (ids, distances) where:
                 - ids: numpy.ndarray of int64 containing corresponding IDs
                 - distances: numpy.ndarray of float32 containing distances to all qualifying vectors

         Note:
             - The number of returned results varies based on data distribution and threshold
             - Results are not guaranteed to be sorted by distance
         )pbdoc")
        .def("range_search_fp16",
             &Index::RangeSearchFloat16,
             py::arg("point"),
             py::arg("threshold"),
             py::arg("parameters"),
             R"pbdoc(
         Perform range search on a single float16 (FP16) query vector.

         Args:
             point (numpy.ndarray): 1D array of numpy.float16 values representing the query vector
             threshold (float): Maximum distance threshold for inclusion in results
             parameters (str): JSON-formatted string containing search-specific parameters

         Returns:
             tuple: (ids, distances) where:
                 - ids: numpy.ndarray of int64 containing corresponding IDs
                 - distances: numpy.ndarray of float32 containing distances to all qualifying vectors

         Note:
             - The point array must have dtype numpy.float16
             - The index must be built with "dtype": "float16"
         )pbdoc")
        .def("range_search_bf16",
             &Index::RangeSearchBfloat16,
             py::arg("point"),
             py::arg("threshold"),
             py::arg("parameters"),
             R"pbdoc(
         Perform range search on a single bfloat16 (BF16) query vector.

         Args:
             point (numpy.ndarray): 1D array of uint16 values (raw bfloat16 bits) representing the query vector
             threshold (float): Maximum distance threshold for inclusion in results
             parameters (str): JSON-formatted string containing search-specific parameters

         Returns:
             tuple: (ids, distances) where:
                 - ids: numpy.ndarray of int64 containing corresponding IDs
                 - distances: numpy.ndarray of float32 containing distances to all qualifying vectors

         Note:
             - The point array must have dtype numpy.uint16, representing raw bfloat16 bit patterns
             - The index must be built with "dtype": "bfloat16"
         )pbdoc")
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
         Add new vectors to the index dynamically.

         Args:
             vectors (numpy.ndarray): 1D array of float32 values with total size num_elements * dim
             ids (numpy.ndarray): 1D array of int64 values with shape (num_elements,)
             num_elements (int): Number of vectors to add
             dim (int): Dimensionality of each vector

         Raises:
             RuntimeError: If the add operation fails.
         )pbdoc")
        .def("add_fp16",
             &Index::AddFloat16,
             py::arg("vectors"),
             py::arg("ids"),
             py::arg("num_elements"),
             py::arg("dim"),
             R"pbdoc(
         Add new float16 (FP16) vectors to the index dynamically.

         Args:
             vectors (numpy.ndarray): 1D array of numpy.float16 values with total size num_elements * dim
             ids (numpy.ndarray): 1D array of int64 values with shape (num_elements,)
             num_elements (int): Number of vectors to add
             dim (int): Dimensionality of each vector

         Raises:
             RuntimeError: If the add operation fails.
         )pbdoc")
        .def("add_bf16",
             &Index::AddBfloat16,
             py::arg("vectors"),
             py::arg("ids"),
             py::arg("num_elements"),
             py::arg("dim"),
             R"pbdoc(
         Add new bfloat16 (BF16) vectors to the index dynamically.

         Args:
             vectors (numpy.ndarray): 1D array of uint16 values (raw bfloat16 bits) with total size num_elements * dim
             ids (numpy.ndarray): 1D array of int64 values with shape (num_elements,)
             num_elements (int): Number of vectors to add
             dim (int): Dimensionality of each vector

         Raises:
             RuntimeError: If the add operation fails.
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
        .def("cal_distance_by_id",
             &Index::CalDistanceById,
             py::arg("query"),
             py::arg("ids"),
             R"pbdoc(
         Calculate distances between a query vector and vectors specified by IDs.

         Args:
             query (numpy.ndarray): 1D array of float32 values representing the query vector
             ids (numpy.ndarray): 1D array of int64 IDs to calculate distances for

         Returns:
             numpy.ndarray: Array of float32 distances corresponding to each ID.
         )pbdoc");
}
