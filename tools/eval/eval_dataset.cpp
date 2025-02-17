
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
EvalDatasetPtr
EvalDataset::Load(const std::string& filename) {
    H5::H5File file(filename, H5F_ACC_RDONLY);

    // check datasets exist
    bool has_labels = false;
    {
        auto datasets = get_datasets(file);
        assert(datasets.count("train"));
        assert(datasets.count("test"));
        assert(datasets.count("neighbors"));
        assert(datasets.count("distances"));
        has_labels = datasets.count("train_labels") && datasets.count("test_labels");
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
            throw std::runtime_error(fmt::format("wrong data type, data type ({}), data size ({})",
                                                 (int)data_type.getClass(),
                                                 data_type.getSize()));
        }
        obj->train_ = std::shared_ptr<char[]>(
            new char[train_shape.first * train_shape.second * obj->train_data_size_]);
        dataset.read(obj->train_.get(), type, dataspace);
    }

    try {
        H5::Attribute attr = file.openAttribute("distance");
        H5::StrType str_type = attr.getStrType();
        std::string metric;
        attr.read(str_type, metric);
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
    } catch (H5::Exception& err) {
        throw std::runtime_error("fail to read metric: there is no 'distance' in the dataset");
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
    }

    return obj;
}
}  // namespace vsag::eval
