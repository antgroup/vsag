
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

#include "transform_quantizer_parameter.h"

#include "impl/logger/logger.h"

namespace vsag {

TransformQuantizerParameter::TransformQuantizerParameter()
    : QuantizerParameter(QUANTIZATION_TYPE_VALUE_TQ) {
}

void
TransformQuantizerParameter::FromJson(const JsonType& json) {
    std::string chain_str;
    if (json.Contains(TQ_CHAIN_KEY)) {
        chain_str = json[TQ_CHAIN_KEY].GetString();
        this->tq_chain_ = SplitString(chain_str);
    }
    if (this->tq_chain_.size() <= 1) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            fmt::format("tq_chain: ({}) must contains 1 or more transformer and 1 quantizer, "
                        "e.g., tq_chain: \"rom, fp32\"",
                        chain_str));
    }

    auto quantizer_type = tq_chain_.back();
    if (not TransformQuantizerParameter::IsValidQuantizationType(quantizer_type)) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("base quantizer: \"{}\" is invalid", quantizer_type));
    }

    base_quantizer_json_ = json;
    base_quantizer_json_[TYPE_KEY].SetString(quantizer_type);
    tq_chain_.pop_back();
}

std::vector<std::string>
TransformQuantizerParameter::SplitString(const std::string& input, char delimiter) {
    std::vector<std::string> result;
    std::stringstream ss(input);
    std::string item;

    while (std::getline(ss, item, delimiter)) {
        item.erase(item.begin(), std::find_if(item.begin(), item.end(), [](unsigned char ch) {
                       return std::isspace(ch) == 0;
                   }));
        item.erase(
            std::find_if(
                item.rbegin(), item.rend(), [](unsigned char ch) { return std::isspace(ch) == 0; })
                .base(),
            item.end());

        if (!item.empty()) {
            result.push_back(item);
        }
    }

    return result;
}

std::string
TransformQuantizerParameter::MergeStrings(const std::vector<std::string>& vec, char delimiter) {
    std::ostringstream oss;
    for (size_t i = 0; i < vec.size(); ++i) {
        oss << vec[i];
        if (i != vec.size() - 1) {
            oss << delimiter;
        }
    }
    return oss.str();
}

JsonType
TransformQuantizerParameter::ToJson() const {
    JsonType json = base_quantizer_json_;
    auto tmp_tq_chain = tq_chain_;
    tmp_tq_chain.emplace_back(json[TYPE_KEY].GetString());
    json[TQ_CHAIN_KEY].SetString(MergeStrings(tmp_tq_chain));
    json[TYPE_KEY].SetString(QUANTIZATION_TYPE_VALUE_TQ);
    return json;
}

bool
TransformQuantizerParameter::CheckCompatibility(const ParamPtr& other) const {
    auto tq_param = std::dynamic_pointer_cast<TransformQuantizerParameter>(other);
    if (not tq_param) {
        logger::error(
            "TransformQuantizerParameter::CheckCompatibility: other parameter is not a "
            "TransformQuantizerParameter");
        return false;
    }
    if (tq_param->tq_chain_.size() != this->tq_chain_.size()) {
        return false;
    }
    for (auto i = 0; i < tq_param->tq_chain_.size(); i++) {
        if (this->tq_chain_[i] != tq_param->tq_chain_[i]) {
            return false;
        }
    }
    return this->base_quantizer_json_[TYPE_KEY].GetString() ==
           tq_param->base_quantizer_json_[TYPE_KEY].GetString();
}
}  // namespace vsag
