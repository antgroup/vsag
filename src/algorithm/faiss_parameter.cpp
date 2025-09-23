
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

#include "faiss_parameter.h"

#include "inner_string_params.h"

namespace vsag {

FaissParameter::FaissParameter() = default;

void
FaissParameter::FromJson(const JsonType& json) {
    if (json.Contains(FAISS_STRING_KEY)) {
        this->faiss_string = json[FAISS_STRING_KEY].GetString();
    }
    if (json.Contains(FAISS_INDEX_PATH_KEY)) {
        this->index_path = json[FAISS_INDEX_PATH_KEY].GetString();
    }
}

JsonType
FaissParameter::ToJson() const {
    JsonType json;
    json["type"].SetString(INDEX_FAISS);
    json[FAISS_STRING_KEY].SetString(this->faiss_string);
    json[FAISS_INDEX_PATH_KEY].SetString(this->index_path);
    return json;
}

bool
FaissParameter::CheckCompatibility(const ParamPtr& other) const {
    return this->faiss_string == std::dynamic_pointer_cast<FaissParameter>(other)->faiss_string;
}
}  // namespace vsag
