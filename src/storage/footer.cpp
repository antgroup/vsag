
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

#include "footer.h"

namespace vsag {

SerializationFooter::SerializationFooter() {
    this->Clear();
}

void
SerializationFooter::Clear() {
    json_.Clear();
    this->SetMetadata(SERIALIZE_MAGIC_NUM, MAGIC_NUM);
    this->SetMetadata(SERIALIZE_VERSION, VERSION);
}

void
SerializationFooter::SetMetadata(const std::string& key, const std::string& value) {
    bool has_key = json_.Contains(key);
    std::string old_value;
    if (has_key) {
        old_value = json_[key].GetString();
    }

    json_[key].SetString(value);
    std::string new_json_str = json_.Dump();

    if (new_json_str.size() >= FOOTER_SIZE - sizeof(uint32_t)) {
        if (has_key) {
            json_[key].SetString(old_value);
        } else {
            json_.Erase(key);
        }
        throw std::runtime_error("Serialized footer size exceeds 4KB");
    }
}

std::string
SerializationFooter::GetMetadata(const std::string& key) const {
    if (not json_.Contains(key)) {
        throw std::runtime_error(fmt::format("Footer doesn't contain key ({})", key));
    }
    return json_[key].GetString();
}

void
SerializationFooter::Serialize(std::ostream& out_stream) const {
    std::string serialized_data = json_.Dump();
    uint32_t serialized_data_size = serialized_data.size();

    out_stream.write(reinterpret_cast<const char*>(&serialized_data_size),
                     sizeof(serialized_data_size));
    out_stream.write(serialized_data.data(), serialized_data_size);

    size_t padding_size = FOOTER_SIZE - sizeof(uint32_t) - serialized_data_size;

    for (size_t i = 0; i < padding_size; ++i) {
        out_stream.put(0);
    }

    out_stream.flush();
}

void
SerializationFooter::Deserialize(StreamReader& in_stream) {
    // read json size
    uint32_t serialized_data_size;
    in_stream.Read(reinterpret_cast<char*>(&serialized_data_size), sizeof(serialized_data_size));
    if (serialized_data_size > FOOTER_SIZE - sizeof(uint32_t)) {
        throw std::runtime_error("Serialized footer size exceeds 4KB");
    }

    // read footer
    std::vector<char> buffer(FOOTER_SIZE - sizeof(uint32_t));
    in_stream.Read(buffer.data(), FOOTER_SIZE - sizeof(uint32_t));

    // parse json
    std::string serialized_data(buffer.begin(), buffer.begin() + FOOTER_SIZE - sizeof(uint32_t));
    json_ = JsonType::Parse(serialized_data, false);
    if (json_.IsDiscarded()) {
        throw std::runtime_error("Failed to parse JSON data");
    }

    // check
    std::string magic_num = this->GetMetadata(SERIALIZE_MAGIC_NUM);
    if (magic_num != MAGIC_NUM) {
        throw std::runtime_error(fmt::format("Incorrect footer.MAGIC_NUM: {}", magic_num));
    }

    std::string version = this->GetMetadata(SERIALIZE_VERSION);
    if (version != VERSION) {
        throw std::runtime_error(fmt::format("Incorrect footer.VERSION: {}", version));
    }
}

}  // namespace vsag
