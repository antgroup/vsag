
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

#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "typing.h"
#include "vsag/attribute.h"

namespace vsag {

class AttrTypeSchema {
public:
    explicit AttrTypeSchema(Allocator* allocator);

    virtual ~AttrTypeSchema() = default;

    AttrValueType
    GetTypeOfField(const std::string& field_name);

    void
    SetTypeOfField(const std::string& field_name, AttrValueType type);

    void
    Serialize(StreamWriter& writer);

    void
    Deserialize(lvalue_or_rvalue<StreamReader> reader);

private:
    UnorderedMap<std::string, AttrValueType> schema_;

    Allocator* const allocator_{nullptr};
};

}  // namespace vsag
