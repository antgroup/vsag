
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

#include <chrono>
#include <thread>

#include "append_buffer.h"
#include "message_queue.h"
#include "meta_table.h"
#include "record_write_stream.h"
#include "vsag/index.h"

enum class SenderStatus {
    kInit,
    kDataPrepared,
    kFinished,
};

class Sender {
public:
    Sender(vsag::Index* index, MessageQueue* send_queue, MessageQueue* recv_queue)
        : index_(index), send_queue_(send_queue), recv_queue_(recv_queue) {
        append_buffer_ = new AppendBuffer();
        meta_table_ = new MetaTable();
    }
    ~Sender() = default;

    void
    Init();

    void
    Run();

    void
    Pause();

    bool
    IsFinished() {
        return this->status_ == SenderStatus::kFinished;
    }

private:
    void
    work();

    void
    send_first_message();

private:
    vsag::Index* index_;
    AppendBuffer* append_buffer_;
    MetaTable* meta_table_;

    CvStruct cv_struct_;
    std::thread* work_thread_{nullptr};

    RecordWriteStreamBuf* record_write_stream_buf_;
    std::ostream* record_write_stream_;

    SenderStatus status_{SenderStatus::kInit};

    MessageQueue* send_queue_;
    MessageQueue* recv_queue_;
};