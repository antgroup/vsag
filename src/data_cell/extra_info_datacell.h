
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

#include <algorithm>
#include <limits>
#include <memory>

#include "byte_buffer.h"
#include "extra_info_interface.h"
#include "io/basic_io.h"
#include "io/memory_block_io.h"
#include "quantization/quantizer.h"

namespace vsag {
/*
* thread unsafe
*/
class ExtraInfoDataCell : public ExtraInfoInterface {
public:
    ExtraInfoDataCell() = default;

    explicit ExtraInfoDataCell(const ExtraInfoDataCellParamPtr& io_param,
                               const IndexCommonParam& common_param);

    void
    InsertExtraInfo(const char* extra_info, InnerIdType idx) override;

    void
    BatchInsertExtraInfo(const char* extra_infos, InnerIdType count, InnerIdType* idx) override;

    void
    SetMaxCapacity(InnerIdType capacity) override {
        this->max_capacity_ = std::max(capacity, this->total_count_);
    }

    void
    Prefetch(InnerIdType id) override {
        if (this->force_in_memory_) {
            force_in_memory_io_->Prefetch(id * extra_info_size_, extra_info_size_);
        } else {
            io_->Prefetch(id * extra_info_size_, extra_info_size_);
        }
    };

    [[nodiscard]] bool
    InMemory() const override;

    void
    EnableForceInMemory() override;

    void
    DisableForceInMemory() override;

    const char*
    GetExtraInfoById(InnerIdType id, bool& need_release) const override;

    bool
    GetExtraInfoById(InnerIdType id, char* extra_info) const override;

    void
    Serialize(StreamWriter& writer) override;

    void
    Deserialize(StreamReader& reader) override;

    inline void
    SetIO(std::shared_ptr<BasicIO<MemoryBlockIO>> io) {
        this->io_ = io;
    }

public:
    std::shared_ptr<BasicIO<MemoryBlockIO>> io_{nullptr};

    Allocator* const allocator_{nullptr};

private:
    void
    trans_from_memory_io();

private:
    bool force_in_memory_{false};

    std::shared_ptr<MemoryBlockIO> force_in_memory_io_{};
};
}  // namespace vsag
