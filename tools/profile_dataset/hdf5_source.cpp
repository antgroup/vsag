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

#include "hdf5_source.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <numeric>

namespace vsag::dataset_profile {

namespace {

constexpr uint64_t RECORD_HEADER_BYTES = sizeof(uint32_t);
constexpr uint64_t BYTES_PER_ENTRY = sizeof(uint32_t) + sizeof(float);

[[noreturn]] void
rethrow_hdf5(const std::string& context, const H5::Exception& error) {
    throw ProfileError(context + ": " + error.getDetailMsg());
}

H5::H5File
open_file(const std::string& path) {
    H5::Exception::dontPrint();
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error)) {
        throw ProfileError("dataset file does not exist or is not a regular file: " + path);
    }
    try {
        if (static_cast<int>(H5::H5File::isHdf5(path)) <= 0) {
            throw ProfileError("not an HDF5 file: " + path);
        }
        return {path, H5F_ACC_RDONLY};
    } catch (const H5::Exception& e) {
        rethrow_hdf5("cannot open HDF5 file " + path, e);
    }
}

std::string
read_string_attribute(const H5::H5File& file, const char* name) {
    if (!file.attrExists(name)) {
        return "";
    }
    try {
        H5::Attribute attribute = file.openAttribute(name);
        if (attribute.getTypeClass() != H5T_STRING) {
            throw ProfileError(std::string("HDF5 attribute '") + name + "' must be a string");
        }
        std::string value;
        attribute.read(attribute.getStrType(), value);
        while (!value.empty() && (value.back() == '\0' || value.back() == ' ')) {
            value.pop_back();
        }
        return value;
    } catch (const H5::Exception& e) {
        rethrow_hdf5(std::string("cannot read HDF5 attribute '") + name + "'", e);
    }
}

std::vector<hsize_t>
dataset_shape(const H5::DataSet& dataset) {
    H5::DataSpace space = dataset.getSpace();
    const int rank = space.getSimpleExtentNdims();
    std::vector<hsize_t> dims(static_cast<uint64_t>(std::max(rank, 0)));
    if (rank > 0) {
        space.getSimpleExtentDims(dims.data());
    }
    return dims;
}

std::string
element_type_name(const H5::DataSet& dataset) {
    const H5T_class_t type_class = dataset.getTypeClass();
    const uint64_t size = dataset.getDataType().getSize();
    if (type_class == H5T_FLOAT) {
        return "float" + std::to_string(size * 8);
    }
    if (type_class == H5T_INTEGER) {
        const bool is_unsigned = dataset.getIntType().getSign() == H5T_SGN_NONE;
        return std::string(is_unsigned ? "uint" : "int") + std::to_string(size * 8);
    }
    return "unsupported";
}

void
inspect_dense_split(const H5::H5File& file, SplitInfo& split, DatasetInfo& info) {
    H5::DataSet dataset = file.openDataSet(split.name);
    const auto shape = dataset_shape(dataset);
    if (shape.size() != 2) {
        throw ProfileError("dense dataset '/" + split.name + "' must be two-dimensional (N, D)");
    }
    split.rows = shape[0];
    split.dim = shape[1];
    split.storage_type = element_type_name(dataset);
    split.stored_bytes = dataset.getStorageSize();
    if (split.rows == 0 || split.dim == 0) {
        throw ProfileError("dense dataset '/" + split.name + "' must not be empty");
    }
    if (split.storage_type == "unsupported") {
        throw ProfileError("dense dataset '/" + split.name +
                           "' must contain integer or floating-point elements");
    }
    if (split.storage_type == "uint8") {
        info.eval_compatibility_issues.push_back(
            "/" + split.name +
            " stores uint8; eval_performance reads every 1-byte integer as signed int8");
    } else if (split.storage_type != "float32" && split.storage_type != "int8" &&
               split.storage_type != "float64") {
        info.eval_compatibility_issues.push_back("/" + split.name + " stores " +
                                                 split.storage_type +
                                                 "; eval_performance accepts float32 or int8");
    }
}

void
inspect_sparse_split(const H5::H5File& file, SplitInfo& split, DatasetInfo& info) {
    H5::DataSet dataset = file.openDataSet(split.name);
    const auto shape = dataset_shape(dataset);
    if (shape.size() != 1 || dataset.getTypeClass() != H5T_INTEGER ||
        dataset.getDataType().getSize() != 1) {
        throw ProfileError("sparse dataset '/" + split.name +
                           "' must be a one-dimensional 1-byte integer byte stream");
    }
    split.storage_type = "sparse byte stream";
    split.stored_bytes = shape[0];
    const std::string offsets_name = split.name + "_offsets";
    if (file.nameExists(offsets_name)) {
        const auto offsets_shape = dataset_shape(file.openDataSet(offsets_name));
        if (offsets_shape.size() != 1 || offsets_shape[0] == 0) {
            throw ProfileError("'/" + offsets_name + "' must be a non-empty (N + 1,) array");
        }
        split.has_offsets = true;
        split.rows = offsets_shape[0] - 1;
    }
    (void)info;
}

}  // namespace

DatasetInfo
inspect_dataset(const std::string& path) {
    DatasetInfo info;
    info.path = path;
    info.file_name = std::filesystem::path(path).filename().string();
    H5::H5File file = open_file(path);
    std::error_code error;
    info.file_size_bytes = std::filesystem::file_size(path, error);
    try {
        for (hsize_t i = 0; i < file.getNumObjs(); ++i) {
            if (file.getObjTypeByIdx(i) == H5G_DATASET) {
                info.dataset_names.push_back(file.getObjnameByIdx(i));
            }
        }
        std::sort(info.dataset_names.begin(), info.dataset_names.end());
        auto has = [&info](const std::string& name) {
            return std::binary_search(info.dataset_names.begin(), info.dataset_names.end(), name);
        };

        info.type_attribute = read_string_attribute(file, "type");
        info.distance_attribute = read_string_attribute(file, "distance");
        if (info.type_attribute.empty() || info.type_attribute == "dense") {
            info.kind = VectorKind::DENSE;
        } else if (info.type_attribute == "sparse") {
            info.kind = VectorKind::SPARSE;
        } else if (info.type_attribute == "multi_vector") {
            throw ProfileError(
                "multi_vector datasets are not supported by profile_dataset; profile dense or "
                "sparse files");
        } else {
            throw ProfileError("unsupported HDF5 dataset type attribute '" + info.type_attribute +
                               "'; expected dense or sparse");
        }

        info.train.name = "train";
        info.test.name = "test";
        info.train.present = has("train");
        info.test.present = has("test");
        if (!info.train.present && !info.test.present) {
            throw ProfileError("HDF5 file contains neither /train nor /test");
        }
        for (SplitInfo* split : {&info.train, &info.test}) {
            if (!split->present) {
                continue;
            }
            if (info.kind == VectorKind::DENSE) {
                inspect_dense_split(file, *split, info);
            } else {
                inspect_sparse_split(file, *split, info);
            }
        }
        if (info.train.present && info.test.present && info.kind == VectorKind::DENSE) {
            if (info.train.dim != info.test.dim) {
                throw ProfileError("train and test vector dimensions must match: train=" +
                                   std::to_string(info.train.dim) +
                                   ", test=" + std::to_string(info.test.dim));
            }
            if (info.train.storage_type != info.test.storage_type) {
                info.eval_compatibility_issues.push_back(
                    "/train and /test use different element types (" + info.train.storage_type +
                    " vs " + info.test.storage_type + ")");
            }
        }
        if (!info.train.present || !info.test.present) {
            info.eval_compatibility_issues.emplace_back(
                "eval_performance requires both /train and /test");
        }

        info.has_neighbors = has("neighbors");
        info.has_distances = has("distances");
        if (info.has_neighbors != info.has_distances) {
            info.eval_compatibility_issues.emplace_back(
                "/neighbors and /distances must be provided together");
        }
        if (!info.has_neighbors || !info.has_distances) {
            info.eval_compatibility_issues.emplace_back(
                "eval_performance requires /neighbors and /distances ground truth");
        }
        if (info.has_neighbors) {
            const auto shape = dataset_shape(file.openDataSet("neighbors"));
            if (shape.size() != 2 || shape[1] == 0) {
                throw ProfileError("/neighbors must have shape (query_count, K) with K > 0");
            }
            info.ground_truth_rows = shape[0];
            info.ground_truth_k = shape[1];
            if (info.has_distances) {
                const auto distance_shape = dataset_shape(file.openDataSet("distances"));
                if (distance_shape != shape) {
                    info.eval_compatibility_issues.emplace_back(
                        "/distances shape does not match /neighbors");
                    info.has_distances = false;
                }
            }
        }
        if (info.distance_attribute.empty()) {
            info.eval_compatibility_issues.emplace_back(
                "missing global 'distance' attribute (euclidean, ip or angular)");
        } else {
            Metric metric = Metric::L2;
            if (!metric_from_distance_attribute(info.distance_attribute, metric)) {
                info.eval_compatibility_issues.push_back("unknown 'distance' attribute '" +
                                                         info.distance_attribute + "'");
            } else if (info.kind == VectorKind::SPARSE && metric != Metric::IP) {
                info.eval_compatibility_issues.emplace_back(
                    "sparse datasets support only distance='ip' in eval_performance");
            }
        }
        info.has_train_labels = has("train_labels");
        info.has_test_labels = has("test_labels");
        info.has_valid_ratios = has("valid_ratios");
        if (info.has_train_labels != info.has_test_labels) {
            info.eval_compatibility_issues.emplace_back(
                "train_labels and test_labels must be provided together");
        }
    } catch (const H5::Exception& e) {
        rethrow_hdf5("cannot inspect HDF5 file " + path, e);
    }
    return info;
}

DenseSplitReader::DenseSplitReader(const std::string& path, const std::string& name)
    : file_(open_file(path)) {
    try {
        dataset_ = file_.openDataSet(name);
        const auto shape = dataset_shape(dataset_);
        if (shape.size() != 2) {
            throw ProfileError("dense dataset '/" + name + "' must be two-dimensional");
        }
        rows_ = shape[0];
        dim_ = shape[1];
        if (dataset_.getTypeClass() == H5T_INTEGER && dataset_.getDataType().getSize() == 1) {
            const bool is_unsigned = dataset_.getIntType().getSign() == H5T_SGN_NONE;
            unsigned_bytes_ = is_unsigned;
            signed_bytes_ = !is_unsigned;
        }
    } catch (const H5::Exception& e) {
        rethrow_hdf5("cannot open dense dataset /" + name, e);
    }
}

void
DenseSplitReader::ReadRows(uint64_t begin, uint64_t count, float* out) {
    if (count == 0) {
        return;
    }
    if (begin + count > rows_) {
        throw ProfileError("row range exceeds the dataset size");
    }
    try {
        const hsize_t offset[2] = {begin, 0};
        const hsize_t extent[2] = {count, dim_};
        H5::DataSpace file_space = dataset_.getSpace();
        file_space.selectHyperslab(H5S_SELECT_SET, extent, offset);
        H5::DataSpace memory_space(2, extent);
        if (signed_bytes_ || unsigned_bytes_) {
            byte_buffer_.resize(count * dim_);
            dataset_.read(byte_buffer_.data(),
                          signed_bytes_ ? H5::PredType::NATIVE_INT8 : H5::PredType::NATIVE_UINT8,
                          memory_space,
                          file_space);
            for (uint64_t i = 0; i < count * dim_; ++i) {
                out[i] = signed_bytes_ ? static_cast<float>(static_cast<int8_t>(byte_buffer_[i]))
                                       : static_cast<float>(byte_buffer_[i]);
            }
        } else {
            dataset_.read(out, H5::PredType::NATIVE_FLOAT, memory_space, file_space);
        }
    } catch (const H5::Exception& e) {
        rethrow_hdf5("cannot read dense rows", e);
    }
}

void
SparseBatch::Clear() {
    first_row = 0;
    offsets.assign(1, 0);
    ids.clear();
    values.clear();
    unsorted_rows = 0;
    duplicate_id_rows = 0;
}

SparseSplitReader::SparseSplitReader(const std::string& path,
                                     const std::string& name,
                                     uint64_t chunk_bytes)
    : file_(open_file(path)), chunk_bytes_(std::max<uint64_t>(chunk_bytes, 1024)) {
    try {
        dataset_ = file_.openDataSet(name);
        const auto shape = dataset_shape(dataset_);
        if (shape.size() != 1) {
            throw ProfileError("sparse dataset '/" + name + "' must be one-dimensional");
        }
        total_bytes_ = shape[0];
    } catch (const H5::Exception& e) {
        rethrow_hdf5("cannot open sparse dataset /" + name, e);
    }
}

bool
SparseSplitReader::ensure_available(uint64_t bytes) {
    if (buffer_end_ - buffer_begin_ >= bytes) {
        return true;
    }
    const uint64_t consumed = buffer_stream_offset_ + buffer_begin_;
    if (total_bytes_ - consumed < bytes) {
        return false;
    }
    // Move the unread tail to the front and read enough for `bytes` plus one chunk.
    const uint64_t unread = buffer_end_ - buffer_begin_;
    if (unread > 0 && buffer_begin_ > 0) {
        std::memmove(buffer_.data(), buffer_.data() + buffer_begin_, unread);
    }
    buffer_stream_offset_ = consumed;
    buffer_begin_ = 0;
    buffer_end_ = unread;
    const uint64_t remaining = total_bytes_ - consumed - unread;
    const uint64_t to_read = std::min(remaining, std::max(bytes - unread, chunk_bytes_));
    if (buffer_.size() < unread + to_read) {
        buffer_.resize(unread + to_read);
    }
    try {
        const hsize_t offset[1] = {consumed + unread};
        const hsize_t extent[1] = {to_read};
        H5::DataSpace file_space = dataset_.getSpace();
        file_space.selectHyperslab(H5S_SELECT_SET, extent, offset);
        H5::DataSpace memory_space(1, extent);
        dataset_.read(buffer_.data() + unread, H5::PredType::NATIVE_CHAR, memory_space, file_space);
    } catch (const H5::Exception& e) {
        rethrow_hdf5("cannot read sparse byte stream", e);
    }
    buffer_end_ = unread + to_read;
    return buffer_end_ >= bytes;
}

bool
SparseSplitReader::NextBatch(uint64_t max_rows, SparseBatch& batch) {
    batch.Clear();
    batch.first_row = next_row_;
    while (batch.Rows() < max_rows) {
        const uint64_t record_offset = buffer_stream_offset_ + buffer_begin_;
        if (record_offset == total_bytes_) {
            break;
        }
        if (!ensure_available(RECORD_HEADER_BYTES)) {
            throw ProfileError("truncated sparse record header at byte " +
                               std::to_string(record_offset) + " (row " +
                               std::to_string(next_row_) + ")");
        }
        uint32_t length = 0;
        std::memcpy(&length, buffer_.data() + buffer_begin_, sizeof(length));
        const uint64_t record_bytes = RECORD_HEADER_BYTES + BYTES_PER_ENTRY * length;
        if (record_bytes > total_bytes_ - record_offset || !ensure_available(record_bytes)) {
            throw ProfileError("truncated sparse record at byte " + std::to_string(record_offset) +
                               " (row " + std::to_string(next_row_) + ", length " +
                               std::to_string(length) + ")");
        }
        const char* payload = buffer_.data() + buffer_begin_ + RECORD_HEADER_BYTES;
        scratch_ids_.resize(length);
        scratch_values_.resize(length);
        // An empty first record leaves the scratch buffers without storage (null data()).
        if (length > 0) {
            std::memcpy(scratch_ids_.data(), payload, sizeof(uint32_t) * length);
            std::memcpy(scratch_values_.data(),
                        payload + sizeof(uint32_t) * static_cast<uint64_t>(length),
                        sizeof(float) * length);
        }
        bool ascending = true;
        for (uint32_t i = 1; i < length; ++i) {
            if (scratch_ids_[i] <= scratch_ids_[i - 1]) {
                ascending = false;
                break;
            }
        }
        if (ascending) {
            batch.ids.insert(batch.ids.end(), scratch_ids_.begin(), scratch_ids_.end());
            batch.values.insert(batch.values.end(), scratch_values_.begin(), scratch_values_.end());
        } else {
            ++batch.unsorted_rows;
            order_.resize(length);
            std::iota(order_.begin(), order_.end(), 0U);
            std::stable_sort(order_.begin(), order_.end(), [this](uint32_t a, uint32_t b) {
                return scratch_ids_[a] < scratch_ids_[b];
            });
            bool repeated = false;
            for (uint32_t i = 0; i < length; ++i) {
                const uint32_t id = scratch_ids_[order_[i]];
                if (i > 0 && id == batch.ids.back()) {
                    repeated = true;
                }
                batch.ids.push_back(id);
                batch.values.push_back(scratch_values_[order_[i]]);
            }
            if (repeated) {
                ++batch.duplicate_id_rows;
            }
        }
        batch.offsets.push_back(batch.ids.size());
        buffer_begin_ += record_bytes;
        ++next_row_;
    }
    return batch.Rows() > 0;
}

uint64_t
count_sparse_rows(const std::string& path, const std::string& name, uint64_t chunk_bytes) {
    SparseSplitReader reader(path, name, chunk_bytes);
    SparseBatch batch;
    uint64_t rows = 0;
    while (reader.NextBatch(BLOCK_ROWS, batch)) {
        rows += batch.Rows();
    }
    return rows;
}

void
for_each_int64_chunk(const std::string& path,
                     const std::string& name,
                     uint64_t chunk_rows,
                     const std::function<void(uint64_t, const std::vector<int64_t>&)>& visit) {
    H5::H5File file = open_file(path);
    try {
        H5::DataSet dataset = file.openDataSet(name);
        const auto shape = dataset_shape(dataset);
        if (shape.size() != 1) {
            throw ProfileError("'/" + name + "' must be one-dimensional");
        }
        std::vector<int64_t> values;
        for (uint64_t begin = 0; begin < shape[0]; begin += chunk_rows) {
            const uint64_t count = std::min<uint64_t>(chunk_rows, shape[0] - begin);
            const hsize_t offset[1] = {begin};
            const hsize_t extent[1] = {count};
            H5::DataSpace file_space = dataset.getSpace();
            file_space.selectHyperslab(H5S_SELECT_SET, extent, offset);
            H5::DataSpace memory_space(1, extent);
            values.resize(count);
            dataset.read(values.data(), H5::PredType::NATIVE_INT64, memory_space, file_space);
            visit(begin, values);
        }
    } catch (const H5::Exception& e) {
        rethrow_hdf5("cannot read /" + name, e);
    }
}

void
read_ground_truth_rows(const std::string& path,
                       const std::vector<uint64_t>& rows,
                       uint64_t k,
                       std::vector<int64_t>& neighbors,
                       std::vector<float>& distances) {
    H5::H5File file = open_file(path);
    neighbors.assign(rows.size() * k, 0);
    distances.assign(rows.size() * k, 0.0F);
    try {
        H5::DataSet neighbor_set = file.openDataSet("neighbors");
        H5::DataSet distance_set = file.openDataSet("distances");
        const hsize_t extent[2] = {1, k};
        H5::DataSpace memory_space(2, extent);
        for (uint64_t i = 0; i < rows.size(); ++i) {
            const hsize_t offset[2] = {rows[i], 0};
            H5::DataSpace neighbor_space = neighbor_set.getSpace();
            neighbor_space.selectHyperslab(H5S_SELECT_SET, extent, offset);
            neighbor_set.read(
                neighbors.data() + i * k, H5::PredType::NATIVE_INT64, memory_space, neighbor_space);
            H5::DataSpace distance_space = distance_set.getSpace();
            distance_space.selectHyperslab(H5S_SELECT_SET, extent, offset);
            distance_set.read(
                distances.data() + i * k, H5::PredType::NATIVE_FLOAT, memory_space, distance_space);
        }
    } catch (const H5::Exception& e) {
        rethrow_hdf5("cannot read ground truth", e);
    }
}

std::vector<float>
read_float_vector(const std::string& path, const std::string& name) {
    H5::H5File file = open_file(path);
    try {
        H5::DataSet dataset = file.openDataSet(name);
        const auto shape = dataset_shape(dataset);
        if (shape.size() != 1) {
            throw ProfileError("'/" + name + "' must be one-dimensional");
        }
        std::vector<float> values(shape[0]);
        if (!values.empty()) {
            dataset.read(values.data(), H5::PredType::NATIVE_FLOAT);
        }
        return values;
    } catch (const H5::Exception& e) {
        rethrow_hdf5("cannot read /" + name, e);
    }
}

}  // namespace vsag::dataset_profile
