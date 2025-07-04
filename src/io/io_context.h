
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

#include "libaio.h"
#include "utils/resource_object.h"
#include "utils/resource_object_pool.h"

namespace vsag {
class IOContext : public ResourceObject {
public:
    IOContext() {
        memset(&ctx_, 0, sizeof(ctx_));
        io_setup(DEFAULT_REQUEST_COUNT, &this->ctx_);
        for (int i = 0; i < DEFAULT_REQUEST_COUNT; ++i) {
            this->cb_[i] = static_cast<iocb*>(malloc(sizeof(struct iocb)));
        }
    }

    ~IOContext() override {
        io_destroy(this->ctx_);
        for (int i = 0; i < DEFAULT_REQUEST_COUNT; ++i) {
            free(this->cb_[i]);
        }
    };

    void
    Reset() override{};

public:
    static constexpr int64_t DEFAULT_REQUEST_COUNT = 100;

    io_context_t ctx_;

    struct iocb* cb_[DEFAULT_REQUEST_COUNT];

    struct io_event events_[DEFAULT_REQUEST_COUNT];
};

using IOContextPool = ResourceObjectPool<IOContext>;

}  // namespace vsag
