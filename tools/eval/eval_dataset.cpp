
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

#include "eval_dataset.h"

namespace vsag::eval {

void
parse_sparse_vectors(const char* src_data,
                     size_t data_size,
                     std::vector<SparseVector>& parsed_vectors) {
    // parse the sparse vectors with ordered keys
    const char* ptr = src_data;
    const char* end = src_data + data_size;
    while (ptr < end) {
        SparseVector vec;

        if (ptr + sizeof(uint32_t) > end)
            break;
        memcpy(&vec.len_, ptr, sizeof(uint32_t));
        ptr += sizeof(uint32_t);

        if (vec.len_ == 0) {
            parsed_vectors.push_back(vec);
            continue;
        }

        const size_t keys_size = vec.len_ * sizeof(uint32_t);
        const size_t vals_size = vec.len_ * sizeof(float);

        if (ptr + keys_size + vals_size > end)
            break;

        vec.ids_ = new uint32_t[vec.len_];
        vec.vals_ = new float[vec.len_];

        memcpy(vec.ids_, ptr, keys_size);
        ptr += keys_size;

        memcpy(vec.vals_, ptr, vals_size);
        ptr += vals_size;

        std::vector<uint32_t> indices(vec.len_);
        for (uint32_t i = 0; i < vec.len_; ++i) indices[i] = i;

        std::sort(indices.begin(), indices.end(), [&](uint32_t a, uint32_t b) {
            return vec.ids_[a] < vec.ids_[b];
        });

        auto* sorted_ids = new uint32_t[vec.len_];
        auto* sorted_vals = new float[vec.len_];

        for (uint32_t i = 0; i < vec.len_; ++i) {
            sorted_ids[i] = vec.ids_[indices[i]];
            sorted_vals[i] = vec.vals_[indices[i]];
        }

        delete[] vec.ids_;
        delete[] vec.vals_;
        vec.ids_ = sorted_ids;
        vec.vals_ = sorted_vals;

        parsed_vectors.push_back(vec);
    }
    if (ptr != end) {
        throw std::runtime_error("parse_sparse_vectors: fail to parse sparse vectors");
    }
}

float
get_distance(const SparseVector* vector1, const SparseVector* vector2, const void* qty_ptr) {
    float sum = 0.0f;
    uint32_t i = 0, j = 0;
    while (i < vector1->len_ && j < vector2->len_) {
        const uint32_t id1 = vector1->ids_[i];
        const uint32_t id2 = vector2->ids_[j];
        if (id1 == id2) {
            sum += vector1->vals_[i] * vector2->vals_[j];
            i++;
            j++;
        } else if (id1 < id2) {
            i++;
        } else {
            j++;
        }
    }
    return sum;
}

EvalDatasetPtr
EvalDataset::Load(const std::string& filename) {
    H5::H5File file(filename, H5F_ACC_RDONLY);

    // check datasets exist
    bool has_labels = false;
    bool has_valid_ratio = false;
    {
        auto datasets = get_datasets(file);
        assert(datasets.count("train"));
        assert(datasets.count("test"));
        assert(datasets.count("neighbors"));
        assert(datasets.count("distances"));
        has_labels = datasets.count("train_labels") && datasets.count("test_labels");
        has_valid_ratio = datasets.count("valid_ratios") > 0;
    }

    // get and (should check shape)
    auto train_shape = get_shape(file, "train");
    spdlog::debug("train.shape: " + to_string(train_shape));
    auto test_shape = get_shape(file, "test");
    spdlog::debug("test.shape: " + to_string(test_shape));
    auto neighbors_shape = get_shape(file, "neighbors");
    spdlog::debug("neighbors.shape: " + to_string(neighbors_shape));
    assert(train_shape.second == test_shape.second);

    auto obj = std::make_shared<EvalDataset>();
    obj->file_path_ = filename;
    obj->train_shape_ = train_shape;
    obj->test_shape_ = test_shape;
    obj->neighbors_shape_ = neighbors_shape;
    obj->dim_ = train_shape.second;
    obj->number_of_base_ = train_shape.first;
    obj->number_of_query_ = test_shape.first;

    try {
        H5::Attribute attr = file.openAttribute("type");
        H5::StrType str_type = attr.getStrType();
        std::string type;
        attr.read(str_type, type);
        if (type == "dense") {
            obj->vector_type_ = DENSE_VECTORS;
        } else if (type == "sparse") {
            obj->vector_type_ = SPARSE_VECTORS;
        }
    } catch (H5::Exception& err) {
        throw std::runtime_error("fail to read metric: there is no 'type' in the dataset");
    }

    try {
        H5::Attribute attr = file.openAttribute("distance");
        H5::StrType str_type = attr.getStrType();
        std::string metric;
        attr.read(str_type, metric);
        if (obj->vector_type_ == DENSE_VECTORS) {
            if (metric == "euclidean") {
                // the distance in the ground truth (provided by public datasets), is L2 distance,
                // which cannot be compared with L2Sqr distance (from VSAG) directly
                obj->distance_func_ =
                    [](const void* query1, const void* query2, const void* qty_ptr) -> float {
                    return sqrt(vsag::L2Sqr(query1, query2, qty_ptr));
                };
            } else if (metric == "ip") {
                if (obj->train_data_type_ == vsag::DATATYPE_FLOAT32) {
                    obj->distance_func_ = vsag::InnerProductDistance;
                } else if (obj->train_data_type_ == vsag::DATATYPE_INT8) {
                    obj->distance_func_ = vsag::INT8InnerProductDistance;
                }
            } else if (metric == "angular") {
                obj->distance_func_ =
                    [](const void* query1, const void* query2, const void* qty_ptr) -> float {
                    return 1 - vsag::InnerProduct(query1, query2, qty_ptr) /
                                   std::sqrt(vsag::InnerProduct(query1, query1, qty_ptr) *
                                             vsag::InnerProduct(query2, query2, qty_ptr));
                };
            }
        } else {
            if (metric == "ip") {
                obj->distance_func_ =
                    [](const void* query1, const void* query2, const void* qty_ptr) -> float {
                    return 1 - get_distance((const SparseVector*)query1,
                                            (const SparseVector*)query2,
                                            qty_ptr);
                };
            } else {
                throw std::runtime_error("no support for sparse vectors with " + metric +
                                         " distance");
            }
        }
    } catch (H5::Exception& err) {
        throw std::runtime_error("fail to read metric: there is no 'distance' in the dataset");
    }

    if (obj->vector_type_ == DENSE_VECTORS) {
        // read from file
        {
            H5::DataSet dataset = file.openDataSet("/train");
            H5::DataSpace dataspace = dataset.getSpace();
            auto data_type = dataset.getDataType();
            H5::PredType type = H5::PredType::ALPHA_I8;
            if (data_type.getClass() == H5T_INTEGER && data_type.getSize() == 1) {
                obj->train_data_type_ = vsag::DATATYPE_INT8;
                type = H5::PredType::ALPHA_I8;
                obj->train_data_size_ = 1;
            } else if (data_type.getClass() == H5T_FLOAT) {
                obj->train_data_type_ = vsag::DATATYPE_FLOAT32;
                type = H5::PredType::NATIVE_FLOAT;
                obj->train_data_size_ = 4;
            } else {
                throw std::runtime_error(
                    fmt::format("wrong data type, data type ({}), data size ({})",
                                (int)data_type.getClass(),
                                data_type.getSize()));
            }
            obj->train_ = std::shared_ptr<char[]>(
                new char[train_shape.first * train_shape.second * obj->train_data_size_]);
            dataset.read(obj->train_.get(), type, dataspace);
        }

        {
            H5::DataSet dataset = file.openDataSet("/test");
            H5::DataSpace dataspace = dataset.getSpace();
            auto data_type = dataset.getDataType();
            H5::PredType type = H5::PredType::ALPHA_I8;
            if (data_type.getClass() == H5T_INTEGER && data_type.getSize() == 1) {
                obj->test_data_type_ = vsag::DATATYPE_INT8;
                type = H5::PredType::ALPHA_I8;
                obj->test_data_size_ = 1;
            } else if (data_type.getClass() == H5T_FLOAT) {
                obj->test_data_type_ = vsag::DATATYPE_FLOAT32;
                type = H5::PredType::NATIVE_FLOAT;
                obj->test_data_size_ = 4;
            } else {
                throw std::runtime_error("wrong data type");
            }
            obj->test_ = std::shared_ptr<char[]>(
                new char[test_shape.first * test_shape.second * obj->test_data_size_]);
            dataset.read(obj->test_.get(), type, dataspace);
        }
    } else {
        {
            H5::PredType type = H5::PredType::ALPHA_I8;
            H5::DataSet dataset = file.openDataSet("/train");
            H5::DataSpace dataspace = dataset.getSpace();
            hsize_t dims_out[2];
            dataspace.getSimpleExtentDims(dims_out, NULL);
            obj->train_data_size_ = dims_out[0];
            obj->train_.reset(new char[obj->train_data_size_]);
            dataset.read(obj->train_.get(), type, dataspace);
            parse_sparse_vectors(obj->train_.get(), obj->train_data_size_, obj->sparse_train_);
            obj->train_.reset();
            obj->number_of_base_ = obj->sparse_train_.size();
        }
        {
            H5::PredType type = H5::PredType::ALPHA_I8;
            H5::DataSet dataset = file.openDataSet("/test");
            H5::DataSpace dataspace = dataset.getSpace();
            hsize_t dims_out[2];
            dataspace.getSimpleExtentDims(dims_out, NULL);
            obj->test_data_size_ = dims_out[0];
            obj->test_.reset(new char[obj->test_data_size_]);
            dataset.read(obj->test_.get(), type, dataspace);
            parse_sparse_vectors(obj->test_.get(), obj->test_data_size_, obj->sparse_test_);
            obj->test_.reset();
            obj->number_of_query_ = obj->sparse_test_.size();
        }
    }

    {
        obj->neighbors_ =
            std::shared_ptr<int64_t[]>(new int64_t[neighbors_shape.first * neighbors_shape.second]);
        H5::DataSet dataset = file.openDataSet("/neighbors");
        H5::DataSpace dataspace = dataset.getSpace();
        H5::FloatType datatype(H5::PredType::NATIVE_INT64);
        dataset.read(obj->neighbors_.get(), datatype, dataspace);
    }

    {
        obj->distances_ =
            std::shared_ptr<float[]>(new float[neighbors_shape.first * neighbors_shape.second]);
        H5::DataSet dataset = file.openDataSet("/distances");
        H5::DataSpace dataspace = dataset.getSpace();
        H5::FloatType datatype(H5::PredType::NATIVE_FLOAT);
        dataset.read(obj->distances_.get(), datatype, dataspace);
    }

    if (has_labels) {
        H5::FloatType datatype(H5::PredType::NATIVE_INT64);

        H5::DataSet train_labels_dataset = file.openDataSet("/train_labels");
        H5::DataSpace train_labels_dataspace = train_labels_dataset.getSpace();
        obj->train_labels_ = std::shared_ptr<int64_t[]>(new int64_t[obj->number_of_base_]);
        train_labels_dataset.read(obj->train_labels_.get(), datatype, train_labels_dataspace);

        H5::DataSet test_labels_dataset = file.openDataSet("/test_labels");
        H5::DataSpace test_labels_dataspace = test_labels_dataset.getSpace();
        obj->test_labels_ = std::shared_ptr<int64_t[]>(new int64_t[obj->number_of_query_]);
        test_labels_dataset.read(obj->test_labels_.get(), datatype, test_labels_dataspace);

        if (has_valid_ratio) {
            H5::FloatType ratio_datatype(H5::PredType::NATIVE_FLOAT);
            H5::DataSet valid_ratio_dataset = file.openDataSet("/valid_ratios");
            H5::DataSpace valid_ratio_dataspace = valid_ratio_dataset.getSpace();
            hsize_t dims_out[1];
            int ndims = valid_ratio_dataspace.getSimpleExtentDims(dims_out, NULL);
            obj->number_of_label_ = dims_out[0];
            obj->valid_ratio_ = std::shared_ptr<float[]>(new float[obj->number_of_label_]);
            valid_ratio_dataset.read(
                obj->valid_ratio_.get(), ratio_datatype, valid_ratio_dataspace);
        }
    }

    return obj;
}
}  // namespace vsag::eval
