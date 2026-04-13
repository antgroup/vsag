/**
 * @file bucket_interface.h
 * @brief Bucket interface for IVF-based vector quantization and storage.
 *
 * This file defines the abstract interface for bucket-based data structures
 * used in Inverted File (IVF) index implementations.
 */

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

#include <string>

#include "algorithm/ivf_partition/ivf_partition_strategy.h"
#include "bucket_datacell_parameter.h"
#include "index_common_param.h"
#include "quantization/computer.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "typing.h"
#include "utils/pointer_define.h"

namespace vsag {
DEFINE_POINTER(BucketInterface);

/**
 * @brief Abstract interface for bucket-based vector storage.
 *
 * BucketInterface provides operations for IVF-based vector storage where
 * vectors are partitioned into buckets. It supports training, insertion,
 * querying, and serialization of bucket-organized quantized data.
 */
class BucketInterface {
public:
    BucketInterface() = default;

    /**
     * @brief Creates a BucketInterface instance based on parameters.
     *
     * @param param Bucket data cell configuration parameters.
     * @param common_param Common index parameters.
     * @return Shared pointer to the created BucketInterface instance.
     */
    static BucketInterfacePtr
    MakeInstance(const BucketDataCellParamPtr& param, const IndexCommonParam& common_param);

public:
    /**
     * @brief Scans all vectors in a bucket and computes distances.
     *
     * @param result_dists Output array for distance results.
     * @param computer The computer interface for distance computation.
     * @param bucket_id The bucket ID to scan.
     */
    virtual void
    ScanBucketById(float* result_dists,
                   const ComputerInterfacePtr& computer,
                   const BucketIdType& bucket_id) = 0;

    /**
     * @brief Queries the distance for a specific vector in a bucket.
     *
     * @param computer The computer interface for distance computation.
     * @param bucket_id The bucket ID.
     * @param offset_id The offset ID within the bucket.
     * @return The computed distance.
     */
    virtual float
    QueryOneById(const ComputerInterfacePtr& computer,
                 const BucketIdType& bucket_id,
                 const InnerIdType& offset_id) = 0;

    /**
     * @brief Creates a computer for distance computation.
     *
     * @param query The query vector data.
     * @return Shared pointer to the computer interface.
     */
    virtual ComputerInterfacePtr
    FactoryComputer(const void* query) = 0;

    /**
     * @brief Trains the quantizer with sample data.
     *
     * @param data Pointer to training data.
     * @param count Number of training vectors.
     */
    virtual void
    Train(const void* data, uint64_t count) = 0;

    /**
     * @brief Inserts a vector into a specific bucket.
     *
     * @param vector Pointer to the vector data.
     * @param bucket_id The bucket ID to insert into.
     * @param inner_id The internal ID for the vector.
     * @return The offset ID within the bucket.
     */
    virtual InnerIdType
    InsertVector(const void* vector, BucketIdType bucket_id, InnerIdType inner_id) = 0;

    /**
     * @brief Gets all internal IDs in a bucket.
     *
     * @param bucket_id The bucket ID to query.
     * @return Pointer to array of internal IDs.
     */
    virtual InnerIdType*
    GetInnerIds(BucketIdType bucket_id) = 0;

    /**
     * @brief Prefetches data for cache optimization.
     *
     * @param bucket_id The bucket ID.
     * @param offset_id The offset ID within the bucket.
     */
    virtual void
    Prefetch(BucketIdType bucket_id, InnerIdType offset_id) = 0;

    /**
     * @brief Copies quantized codes for a specific vector.
     *
     * @param bucket_id The bucket ID.
     * @param offset_id The offset ID within the bucket.
     * @param data Output buffer for the quantized codes.
     */
    virtual void
    GetCodesById(BucketIdType bucket_id, InnerIdType offset_id, uint8_t* data) const = 0;

    /**
     * @brief Gets the name of the quantizer used.
     *
     * @return Quantizer name string.
     */
    [[nodiscard]] virtual std::string
    GetQuantizerName() = 0;

    /**
     * @brief Gets the metric type used for distance computation.
     *
     * @return The metric type.
     */
    [[nodiscard]] virtual MetricType
    GetMetricType() = 0;

    /**
     * @brief Checks if residual quantization is used.
     *
     * @return True if using residual quantization.
     */
    [[nodiscard]] virtual bool
    UseResidual() const {
        return this->use_residual_;
    }

    /**
     * @brief Gets the number of vectors in a bucket.
     *
     * @param bucket_id The bucket ID to query.
     * @return Number of vectors in the bucket.
     */
    [[nodiscard]] virtual InnerIdType
    GetBucketSize(BucketIdType bucket_id) = 0;

    /**
     * @brief Exports the quantization model to another instance.
     *
     * @param other The target bucket interface to export to.
     */
    virtual void
    ExportModel(const BucketInterfacePtr& other) const = 0;

    /**
     * @brief Merges another bucket storage into this one.
     *
     * @param other The bucket storage to merge.
     * @param bias ID offset for the merged vectors.
     */
    virtual void
    MergeOther(const BucketInterfacePtr& other, InnerIdType bias) = 0;

    /**
     * @brief Sets the IVF partition strategy.
     *
     * @param strategy The partition strategy to use.
     */
    virtual void
    SetStrategy(const IVFPartitionStrategyPtr& strategy) {
        strategy_ = strategy;
    }

public:
    /**
     * @brief Prefetches a bucket (delegates to Prefetch with offset 0).
     *
     * @param bucket_id The bucket ID to prefetch.
     */
    virtual void
    Prefetch(BucketIdType bucket_id) {
        return this->Prefetch(bucket_id, 0);
    }

    /**
     * @brief Gets the total number of buckets.
     *
     * @return Number of buckets.
     */
    [[nodiscard]] virtual BucketIdType
    GetBucketCount() {
        return this->bucket_count_;
    }

    /**
     * @brief Serializes the bucket storage to a stream writer.
     *
     * @param writer The stream writer for output.
     */
    virtual void
    Serialize(StreamWriter& writer) {
        StreamWriter::WriteObj(writer, this->bucket_count_);
        StreamWriter::WriteObj(writer, this->code_size_);
    }

    /**
     * @brief Deserializes the bucket storage from a stream reader.
     *
     * @param reader The stream reader for input.
     */
    virtual void
    Deserialize(lvalue_or_rvalue<StreamReader> reader) {
        StreamReader::ReadObj(reader, this->bucket_count_);
        StreamReader::ReadObj(reader, this->code_size_);
    }

    /**
     * @brief Packages the bucket data (for storage optimization).
     */
    virtual void
    Package(){};

    /**
     * @brief Unpacks the bucket data (for storage optimization).
     */
    virtual void
    Unpack(){};

    /**
     * @brief Gets the memory usage of the bucket storage.
     *
     * @return Memory usage in bytes.
     */
    [[nodiscard]] virtual int64_t
    GetMemoryUsage() const = 0;

public:
    /// Total number of buckets
    BucketIdType bucket_count_{0};
    /// Size of each quantized code in bytes
    uint32_t code_size_{0};
    /// IVF partition strategy for bucket assignment
    IVFPartitionStrategyPtr strategy_{nullptr};
    /// Flag indicating if residual quantization is used
    bool use_residual_{false};
};

}  // namespace vsag