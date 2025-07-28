
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

#include "sender.h"

#include <cstring>

void
Sender::Run() {
    if (work_thread_ == nullptr) {
        work_thread_ = new std::thread([this]() { this->work(); });
    }
    {
        std::unique_lock<std::mutex> lk(this->cv_struct_.mu_);
        this->cv_struct_.pause_signal_.store(false);
    }
    this->cv_struct_.cv_.notify_one();
}

void
Sender::Pause() {
    {
        std::unique_lock<std::mutex> lk(this->cv_struct_.mu_);
        this->cv_struct_.pause_signal_.store(true);
    }
    this->cv_struct_.cv_.notify_one();
}

void
Sender::Init() {
    this->record_write_stream_buf_ =
        new RecordWriteStreamBuf(meta_table_, append_buffer_, this->cv_struct_);
    this->record_write_stream_ = new std::ostream(this->record_write_stream_buf_);
}

void
Sender::send_first_message() {
    Message* message = new Message();
    message->type = MessageType::kStart;
    message->size = this->record_write_stream_buf_->GetTotalSize();
    send_queue_->SendMessage(message);
}

void
Sender::work() {
    if (status_ == SenderStatus::kInit) {
        this->index_->Serialize(*record_write_stream_);
        status_ = SenderStatus::kDataPrepared;
    }
    if (status_ == SenderStatus::kDataPrepared) {
        send_first_message();
        while (true) {
            if (this->cv_struct_.pause_signal_.load()) {
                std::unique_lock<std::mutex> lk(this->cv_struct_.mu_);
                this->cv_struct_.cv_.wait(
                    lk, [this]() { return !this->cv_struct_.pause_signal_.load(); });
            }
            Message* message = recv_queue_->ReceiveMessage();
            if (message == nullptr) {
                continue;
            }
            if (message->type == MessageType::kEnd) {
                break;
            }
            if (message->type == MessageType::kData) {
                const auto* data_uint64 = reinterpret_cast<const uint64_t*>(message->data);
                uint64_t offset = data_uint64[0];
                uint64_t size = data_uint64[1];
                const char* data = meta_table_->GetDataPtr(offset, size);
                if (data == nullptr) {
                    exit(-1);
                }
                memcpy(message->data, data, size);
                message->size = size;
                send_queue_->SendMessage(message);
            }
        }
        status_ = SenderStatus::kFinished;
    }
}