
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

#include <memory>

#include "./serialization.h"
#include "./stream_reader.h"
#include "./stream_writer.h"
#include "vsag/binaryset.h"
#include "vsag/readerset.h"

namespace vsag {

// TODO(wxyu): move to typing.h later
template <typename T>
struct non_copyable {
public:
    non_copyable(const non_copyable&) = delete;
    non_copyable&
    operator=(const non_copyable&) = delete;

protected:
    non_copyable() = default;
    ~non_copyable() = default;
};

// internal serialize interfaces
struct serializable;
using serializable_ptr = std::shared_ptr<serializable>;
struct serializable : public non_copyable<serializable> {
public:
    virtual void
    Serialize(Serial& serial) const = 0;

    virtual void
    Deserialize(Serial& serial) = 0;
};

};  // namespace vsag

// namespace vsag
