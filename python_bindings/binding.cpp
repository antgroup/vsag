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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <filesystem>
#include <fstream>
#include <map>

#include "fmt/format.h"
#include "iostream"
#include "vsag/dataset.h"
#include "vsag/vsag.h"

namespace py = pybind11;

void
SetLoggerOff() {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kOFF);
}

void
SetLoggerInfo() {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kINFO);
}

void
SetLoggerDebug() {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kDEBUG);
}

template <typename T>
static void
writeBinaryPOD(std::ostream& out, const T& podRef) {
    out.write((char*)&podRef, sizeof(T));
}

template <typename T>
static void
readBinaryPOD(std::istream& in, T& podRef) {
    in.read((char*)&podRef, sizeof(T));
}

struct SparseVectors {
    std::vector<vsag::SparseVector> sparse_vectors;
    uint32_t num_elements;
    uint32_t num_non_zeros;

    SparseVectors(uint32_t num_elements)
        : sparse_vectors(num_elements), num_elements(num_elements), num_non_zeros(0) {
    }
};

SparseVectors
BuildSparseVectorsFromCSR(py::array_t<uint32_t> index_pointers,
                          py::array_t<uint32_t> indices,
                          py::array_t<float> values) {
    auto buf_ptr = index_pointers.request();
    auto buf_idx = indices.request();
    auto buf_val = values.request();

    if (buf_ptr.ndim != 1 || buf_idx.ndim != 1 || buf_val.ndim != 1) {
        throw std::invalid_argument("all inputs must be 1-dimensional");
    }

    if (buf_ptr.shape[0] < 2) {
        throw std::invalid_argument("index_pointers length must be at least 2");
    }
    uint32_t num_elements = buf_ptr.shape[0] - 1;

    const uint32_t* ptr_data = index_pointers.data();
    const uint32_t* idx_data = indices.data();
    const float* val_data = values.data();

    uint32_t num_non_zeros = ptr_data[num_elements];

    if (static_cast<size_t>(num_non_zeros) != buf_idx.shape[0]) {
        throw std::invalid_argument(
            fmt::format("Size of 'indices'({}) must equal index_pointers[last]",
                        buf_idx.shape[0],
                        num_non_zeros));
    }
    if (static_cast<size_t>(num_non_zeros) != buf_val.shape[0]) {
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

    SparseVectors svs(num_elements);
    svs.num_non_zeros = num_non_zeros;

    for (uint32_t i = 0; i < num_elements; ++i) {
        uint32_t start = ptr_data[i];
        uint32_t end = ptr_data[i + 1];
        uint32_t len = end - start;

        svs.sparse_vectors[i].len_ = len;
        svs.sparse_vectors[i].ids_ = const_cast<uint32_t*>(idx_data + start);
        svs.sparse_vectors[i].vals_ = const_cast<float*>(val_data + start);
    }

    return svs;
}

class Index {
public:
    Index(std::string name, const std::string& parameters) {
        if (auto index = vsag::Factory::CreateIndex(name, parameters)) {
            index_ = index.value();
        } else {
            vsag::Error error_code = index.error();
            if (error_code.type == vsag::ErrorType::UNSUPPORTED_INDEX) {
                throw std::runtime_error("error type: UNSUPPORTED_INDEX");
            } else if (error_code.type == vsag::ErrorType::INVALID_ARGUMENT) {
                throw std::runtime_error("error type: invalid_parameter");
            } else {
                throw std::runtime_error("error type: unexpectedError");
            }
        }
    }

public:
    void
    Build(py::array_t<float> vectors, py::array_t<int64_t> ids, size_t num_elements, size_t dim) {
        auto dataset = vsag::Dataset::Make();
        dataset->Owner(false)
            ->Dim(dim)
            ->NumElements(num_elements)
            ->Ids(ids.mutable_data())
            ->Float32Vectors(vectors.mutable_data());
        index_->Build(dataset);
    }

    void
    SparseBuild(py::array_t<uint32_t> index_pointers,
                py::array_t<uint32_t> indices,
                py::array_t<float> values,
                py::array_t<int64_t> ids) {
        auto batch = BuildSparseVectorsFromCSR(index_pointers, indices, values);

        auto buf_id = ids.request();
        if (buf_id.ndim != 1) {
            throw std::invalid_argument("all inputs must be 1-dimensional");
        }
        if (batch.num_elements != buf_id.shape[0]) {
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
    KnnSearch(py::array_t<float> vector, size_t k, std::string& parameters) {
        auto query = vsag::Dataset::Make();
        size_t data_num = 1;
        query->NumElements(data_num)
            ->Dim(vector.size())
            ->Float32Vectors(vector.mutable_data())
            ->Owner(false);

        size_t ids_shape[1]{k};
        size_t ids_strides[1]{sizeof(int64_t)};
        size_t dists_shape[1]{k};
        size_t dists_strides[1]{sizeof(float)};

        auto ids = py::array_t<int64_t>(ids_shape, ids_strides);
        auto dists = py::array_t<float>(dists_shape, dists_strides);
        if (auto result = index_->KnnSearch(query, k, parameters); result.has_value()) {
            auto ids_view = ids.mutable_unchecked<1>();
            auto dists_view = dists.mutable_unchecked<1>();

            auto vsag_ids = result.value()->GetIds();
            auto vsag_distances = result.value()->GetDistances();
            for (uint32_t i = 0; i < data_num * k; ++i) {
                ids_view(i) = vsag_ids[i];
                dists_view(i) = vsag_distances[i];
            }
        }

        return py::make_tuple(ids, dists);
    }

    py::tuple
    SparseKnnSearch(py::array_t<uint32_t> index_pointers,
                    py::array_t<uint32_t> indices,
                    py::array_t<float> values,
                    uint32_t k,
                    const std::string& parameters) {
        auto batch = BuildSparseVectorsFromCSR(index_pointers, indices, values);

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
        size_t data_num = 1;
        query->NumElements(data_num)
            ->Dim(point.size())
            ->Float32Vectors(point.mutable_data())
            ->Owner(false);

        py::array_t<int64_t> labels;
        py::array_t<float> dists;
        if (auto result = index_->RangeSearch(query, threshold, parameters); result.has_value()) {
            auto ids = result.value()->GetIds();
            auto distances = result.value()->GetDistances();
            auto k = result.value()->GetDim();
            labels.resize({k});
            dists.resize({k});
            auto labels_data = labels.mutable_data();
            auto dists_data = dists.mutable_data();
            for (uint32_t i = 0; i < data_num * k; ++i) {
                labels_data[i] = ids[i];
                dists_data[i] = distances[i];
            }
        }

        return py::make_tuple(labels, dists);
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

PYBIND11_MODULE(_pyvsag, m) {
    m.def("set_logger_off", &SetLoggerOff, "SetLoggerOff");
    m.def("set_logger_info", &SetLoggerInfo, "SetLoggerInfo");
    m.def("set_logger_debug", &SetLoggerDebug, "SetLoggerDebug");
    py::class_<Index>(m, "Index")
        .def(py::init<std::string, std::string&>(), py::arg("name"), py::arg("parameters"))
        .def("build",
             &Index::Build,
             py::arg("vectors"),
             py::arg("ids"),
             py::arg("num_elements"),
             py::arg("dim"))
        .def("build",
             &Index::SparseBuild,
             py::arg("index_pointers"),
             py::arg("indices"),
             py::arg("values"),
             py::arg("ids"))
        .def(
            "knn_search", &Index::KnnSearch, py::arg("vector"), py::arg("k"), py::arg("parameters"))
        .def("knn_search",
             &Index::SparseKnnSearch,
             py::arg("index_pointers"),
             py::arg("indices"),
             py::arg("values"),
             py::arg("k"),
             py::arg("parameters"))
        .def("range_search",
             &Index::RangeSearch,
             py::arg("vector"),
             py::arg("threshold"),
             py::arg("parameters"))
        .def("save", &Index::Save, py::arg("filename"))
        .def("load", &Index::Load, py::arg("filename"));
}
