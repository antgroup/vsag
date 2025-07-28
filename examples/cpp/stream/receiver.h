
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

#include "cv_struct.h"
#include "message_queue.h"
#include "read_stream.h"
#include "vsag/index.h"

enum class ReceiverStatus {
    kInit,
    kDataReceive,
    kFinished,
};

class Receiver {
public:
    Receiver(vsag::Index* index, MessageQueue* send_queue, MessageQueue* recv_queue)
        : index_(index), send_queue_(send_queue), recv_queue_(recv_queue) {
    }
    ~Receiver() = default;

    void
    Init();

    void
    Run();

    void
    Pause();

    bool
    IsFinished() {
        return this->status_ == ReceiverStatus::kFinished;
    }

private:
    void
    work();

    void
    receive_first_message();

    void
    send_last_message();

private:
    vsag::Index* index_;

    MessageQueue* send_queue_;
    MessageQueue* recv_queue_;

    CvStruct cv_struct_;

    ReadStreamBuf* read_stream_buf_;
    std::istream* read_stream_;

    ReceiverStatus status_{ReceiverStatus::kInit};

    uint64_t total_size_{0};

    std::thread* work_thread_{nullptr};
};