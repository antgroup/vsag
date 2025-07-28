
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

#include <streambuf>

#include "append_buffer.h"
#include "cv_struct.h"
#include "meta_table.h"

class RecordWriteStreamBuf : public std::streambuf {
public:
    RecordWriteStreamBuf(MetaTable* meta_table, AppendBuffer* append_buffer, CvStruct& cv_struct)
        : meta_table_(meta_table),
          cursor_(0),
          append_buffer_(append_buffer),
          cv_struct_(cv_struct) {
    }

    uint64_t
    GetTotalSize() {
        return this->total_size_;
    }

protected:
    std::streamsize
    xsputn(const char* s, std::streamsize n) override;

    pos_type
    seekoff(off_type off,
            std::ios_base::seekdir dir,
            std::ios_base::openmode which = std::ios_base::out) override;

    pos_type
    seekpos(pos_type sp, std::ios_base::openmode which = std::ios_base::out) override;

private:
    MetaTable* meta_table_{nullptr};
    uint64_t cursor_{0};
    AppendBuffer* append_buffer_{nullptr};

    CvStruct& cv_struct_;

    uint64_t total_size_{0};
};
