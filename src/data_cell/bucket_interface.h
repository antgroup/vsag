
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

#include "bucket_datacell_parameter.h"
#include "index/index_common_param.h"
#include "quantization/computer.h"
#include "stream_reader.h"
#include "stream_writer.h"
#include "typing.h"

namespace vsag {
class BucketInterface;
using BucketInterfacePtr = std::shared_ptr<BucketInterface>;

class BucketInterface {
public:
    BucketInterface() = default;

    static BucketInterfacePtr
    MakeInstance(const BucketDataCellParamPtr& param, const IndexCommonParam& common_param);

public:
    virtual void
    ScanBucketById(float* result_dists,
                   const ComputerInterfacePtr& computer,
                   const BucketIdType& bucket_id) = 0;

    virtual float
    QueryOneById(const ComputerInterfacePtr& computer,
                 const BucketIdType& bucket_id,
                 const InnerIdType& offset_id) = 0;

    virtual ComputerInterfacePtr
    FactoryComputer(const void* query) = 0;

    virtual void
    Train(const void* data, uint64_t count) = 0;

    virtual void
    InsertVector(const void* vector, BucketIdType bucket_id, LabelType label) = 0;

    virtual LabelType*
    GetLabel(BucketIdType bucket_id) = 0;

    virtual void
    Prefetch(BucketIdType bucket_id, InnerIdType offset_id) = 0;

    [[nodiscard]] virtual std::string
    GetQuantizerName() = 0;

    [[nodiscard]] virtual MetricType
    GetMetricType() = 0;

    [[nodiscard]] virtual InnerIdType
    GetBucketSize(BucketIdType bucket_id) = 0;

public:
    virtual void
    Prefetch(BucketIdType bucket_id) {
        return this->Prefetch(bucket_id, 0);
    }

    [[nodiscard]] virtual BucketIdType
    GetBucketCount() {
        return this->bucket_count_;
    }

    virtual void
    Serialize(StreamWriter& writer) {
        StreamWriter::WriteObj(writer, this->bucket_count_);
        StreamWriter::WriteObj(writer, this->code_size_);
    }

    virtual void
    Deserialize(StreamReader& reader) {
        StreamReader::ReadObj(reader, this->bucket_count_);
        StreamReader::ReadObj(reader, this->code_size_);
    }

public:
    BucketIdType bucket_count_{0};
    uint32_t code_size_{0};
};

}  // namespace vsag
