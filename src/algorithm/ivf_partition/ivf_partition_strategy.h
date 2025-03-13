
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

#include <vector>

#include "stream_reader.h"
#include "stream_writer.h"
#include "vsag/dataset.h"

namespace vsag {
class IVFPartitionStrategy {
public:
    explicit IVFPartitionStrategy(Allocator* allocator, BucketIdType bucket_count, int64_t dim)
        : allocator_(allocator), bucket_count_(bucket_count), dim_(dim){};

    virtual void
    Train(const DatasetPtr dataset) = 0;

    virtual Vector<BucketIdType>
    ClassifyDatas(const void* datas, int64_t count, BucketIdType buckets_per_data) = 0;

    virtual void
    Serialize(StreamWriter& writer) {
        StreamWriter::WriteObj(writer, this->is_trained_);
        StreamWriter::WriteObj(writer, this->bucket_count_);
        StreamWriter::WriteObj(writer, this->dim_);
    }

    virtual void
    Deserialize(StreamReader& reader) {
        StreamReader::ReadObj(reader, this->is_trained_);
        StreamReader::ReadObj(reader, this->bucket_count_);
        StreamReader::ReadObj(reader, this->dim_);
    }

public:
    bool is_trained_{false};

    Allocator* const allocator_{nullptr};

    BucketIdType bucket_count_{0};

    int64_t dim_{-1};
};

using IVFPartitionStrategyPtr = std::shared_ptr<IVFPartitionStrategy>;

}  // namespace vsag
