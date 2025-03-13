
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

#include "extra_info_datacell.h"

namespace vsag {
ExtraInfoDataCell::ExtraInfoDataCell(const ExtraInfoDataCellParamPtr& param,
                                     const IndexCommonParam& common_param)
    : allocator_(common_param.allocator_.get()) {
    this->extra_info_size_ = param->extra_info_size;
    this->io_ = std::make_shared<MemoryBlockIO>(param->io_parameter, common_param);
    this->force_in_memory_io_ =
        std::make_shared<MemoryBlockIO>(allocator_, Options::Instance().block_size_limit());
}

void
ExtraInfoDataCell::InsertExtraInfo(const char* extra_info, InnerIdType idx) {
    if (idx == std::numeric_limits<InnerIdType>::max()) {
        idx = total_count_;
        ++total_count_;
    } else {
        total_count_ = std::max(total_count_, idx + 1);
    }

    if (this->force_in_memory_) {
        force_in_memory_io_->Write(
            reinterpret_cast<const uint8_t*>(extra_info),
            extra_info_size_,
            static_cast<uint64_t>(idx) * static_cast<uint64_t>(extra_info_size_));
    } else {
        io_->Write(reinterpret_cast<const uint8_t*>(extra_info),
                   extra_info_size_,
                   static_cast<uint64_t>(idx) * static_cast<uint64_t>(extra_info_size_));
    }
}

void
ExtraInfoDataCell::BatchInsertExtraInfo(const char* extra_infos,
                                        InnerIdType count,
                                        InnerIdType* idx) {
    if (idx == nullptr) {
        // length of extra info is fixed currently
        if (this->force_in_memory_) {
            force_in_memory_io_->Write(
                reinterpret_cast<const uint8_t*>(extra_infos),
                static_cast<uint64_t>(count) * static_cast<uint64_t>(extra_info_size_),
                static_cast<uint64_t>(total_count_) * static_cast<uint64_t>(extra_info_size_));
        } else {
            io_->Write(
                reinterpret_cast<const uint8_t*>(extra_infos),
                static_cast<uint64_t>(count) * static_cast<uint64_t>(extra_info_size_),
                static_cast<uint64_t>(total_count_) * static_cast<uint64_t>(extra_info_size_));
        }
        total_count_ += count;
    } else {
        for (int64_t i = 0; i < count; ++i) {
            this->InsertExtraInfo(extra_infos + extra_info_size_ * i, idx[i]);
        }
    }
}

void
ExtraInfoDataCell::trans_from_memory_io() {
    int64_t max_size = static_cast<int64_t>(this->extra_info_size_) * this->total_count_;
    constexpr uint64_t block_size = 1024 * 1024;
    uint64_t offset = 0;
    bool need_release = false;
    while (max_size > block_size) {
        auto data = this->force_in_memory_io_->Read(block_size, offset, need_release);
        this->io_->Write(data, block_size, offset);
        max_size -= block_size;
        offset += block_size;
        if (need_release) {
            this->force_in_memory_io_->Release(data);
        }
    }
    if (max_size > 0) {
        auto data = this->force_in_memory_io_->Read(max_size, offset, need_release);
        this->io_->Write(data, max_size, offset);
        if (need_release) {
            this->force_in_memory_io_->Release(data);
        }
    }
}

void
ExtraInfoDataCell::EnableForceInMemory() {
    if (this->TotalCount() != 0) {
        throw std::runtime_error("EnableForceInMemory must with empty extra info datacell");
    }
    this->force_in_memory_ = true;
}

void
ExtraInfoDataCell::DisableForceInMemory() {
    this->force_in_memory_ = false;
    this->trans_from_memory_io();
}

bool
ExtraInfoDataCell::InMemory() const {
    return this->io_->InMemory();
}

const char*
ExtraInfoDataCell::GetExtraInfoById(InnerIdType id, bool& need_release) const {
    if (force_in_memory_) {
        return reinterpret_cast<const char*>(force_in_memory_io_->Read(
            extra_info_size_,
            static_cast<uint64_t>(id) * static_cast<uint64_t>(extra_info_size_),
            need_release));
    } else {
        return reinterpret_cast<const char*>(
            io_->Read(extra_info_size_,
                      static_cast<uint64_t>(id) * static_cast<uint64_t>(extra_info_size_),
                      need_release));
    }
}

bool
ExtraInfoDataCell::GetExtraInfoById(InnerIdType id, char* extra_info) const {
    if (force_in_memory_) {
        return force_in_memory_io_->Read(
            extra_info_size_,
            static_cast<uint64_t>(id) * static_cast<uint64_t>(extra_info_size_),
            reinterpret_cast<uint8_t*>(extra_info));
    } else {
        return io_->Read(extra_info_size_,
                         static_cast<uint64_t>(id) * static_cast<uint64_t>(extra_info_size_),
                         reinterpret_cast<uint8_t*>(extra_info));
    }
}

void
ExtraInfoDataCell::Serialize(StreamWriter& writer) {
    ExtraInfoInterface::Serialize(writer);
    this->io_->Serialize(writer);
}

void
ExtraInfoDataCell::Deserialize(StreamReader& reader) {
    ExtraInfoInterface::Deserialize(reader);
    this->io_->Deserialize(reader);
}

}  // namespace vsag
