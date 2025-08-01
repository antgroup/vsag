
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

#include "attr_type_schema.h"

#include "vsag_exception.h"

namespace vsag {
AttrTypeSchema::AttrTypeSchema(Allocator* allocator) : allocator_(allocator), schema_(allocator) {
}

AttrValueType
AttrTypeSchema::GetTypeOfField(const std::string& field_name) {
    auto iter = this->schema_.find(field_name);
    if (iter == this->schema_.end()) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            fmt::format("field not found: {}", field_name));
    }
    return iter->second;
}

void
AttrTypeSchema::SetTypeOfField(const std::string& field_name, AttrValueType type) {
    schema_[field_name] = type;
}

void
AttrTypeSchema::Serialize(StreamWriter& writer) {
    auto size = this->schema_.size();
    StreamWriter::WriteObj(writer, size);
    for (const auto& [k, v] : this->schema_) {
        StreamWriter::WriteString(writer, k);
        StreamWriter::WriteObj(writer, static_cast<int64_t>(v));
    }
}

void
AttrTypeSchema::Deserialize(lvalue_or_rvalue<StreamReader> reader) {
    uint64_t size;
    StreamReader::ReadObj(reader, size);
    this->schema_.reserve(size);
    std::string key;
    int64_t value;
    for (int64_t i = 0; i < size; ++i) {
        key = StreamReader::ReadString(reader);
        StreamReader::ReadObj(reader, value);
        this->schema_[key] = static_cast<AttrValueType>(value);
    }
}

}  // namespace vsag
