
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

#include "read_stream.h"

#include <cstring>

std::streamsize
ReadStreamBuf::xsgetn(char* s, std::streamsize n) {
    std::streamsize ret = 0;
    if (n > 4096) {
        for (int64_t i = 0; i < n; i += 4096) {
            auto size = std::min(4096L, n - i);
            ret += xsgetn(s + i, size);
        }
    } else {
        bool have_send = false;
        while (true) {
            if (cv_struct_.pause_signal_.load() == true) {
                std::unique_lock<std::mutex> lk(cv_struct_.mu_);
                cv_struct_.cv_.wait(lk, [this]() { return !cv_struct_.pause_signal_.load(); });
            }
            if (not have_send) {
                send_message_.type = MessageType::kData;
                uint64_t offset = cursor_;
                uint64_t size = n;
                memcpy(send_message_.data, &offset, sizeof(uint64_t));
                memcpy(send_message_.data + 8, &size, sizeof(uint64_t));
                send_queue_->SendMessage(&send_message_);
                have_send = true;
            }
            Message* message = recv_queue_->ReceiveMessage();
            if (message == nullptr) {
                continue;
            }
            if (message->type == MessageType::kData) {
                memcpy(s, message->data, n);
                break;
            }
        }
        cursor_ += n;
        ret += n;
    }
    return ret;
}

ReadStreamBuf::pos_type
ReadStreamBuf::seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which) {
    if (which != std::ios_base::in) {
        return pos_type(off_type(-1));
    }

    switch (dir) {
        case std::ios_base::beg:
            cursor_ = static_cast<uint64_t>(off);
            break;
        case std::ios_base::cur:
            if (off >= 0) {
                cursor_ += static_cast<uint64_t>(off);
            } else {
                if (static_cast<uint64_t>(-off) > cursor_) {
                    return pos_type(off_type(-1));
                }
                cursor_ -= static_cast<uint64_t>(-off);
            }
            break;
        case std::ios_base::end:
            cursor_ = this->max_size_ + off;
            return pos_type(this->max_size_);
        default:
            return pos_type(off_type(-1));
    }

    return pos_type(static_cast<off_type>(cursor_));
}

ReadStreamBuf::pos_type
ReadStreamBuf::seekpos(pos_type sp, std::ios_base::openmode which) {
    if (which != std::ios_base::in) {
        return pos_type(off_type(-1));
    }

    if (sp < 0) {
        return pos_type(off_type(-1));
    }

    cursor_ = static_cast<uint64_t>(sp);
    return pos_type(static_cast<off_type>(cursor_));
}