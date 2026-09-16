
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

#include "impl/transform/vector_transformer_parameter.h"
#include "utils/param_compat_macros.h"

namespace vsag {

TransformQuantizerParamPtr
TransformQuantizerParameter::CreateDefault(const std::string& chain) {
    auto chain_items = SplitString(chain);
    CHECK_ARGUMENT(chain_items.size() > 1,
                   "tq_chain must contain at least one transformer and one quantizer");
    const auto bottom_type = chain_items.back();
    auto json = QuantizerParameter::CreateDefault(bottom_type)->ToJson();
    const auto transformer_json = VectorTransformerParameter().ToJson();
    json[INPUT_DIM_KEY].SetJson(transformer_json[INPUT_DIM_KEY]);
    json[PCA_DIM_KEY].SetJson(transformer_json[PCA_DIM_KEY]);
    json[MRLE_DIM_KEY].SetJson(transformer_json[MRLE_DIM_KEY]);
    json[TQ_CHAIN_KEY].SetString(chain);
    auto parameter = std::make_shared<TransformQuantizerParameter>();
    parameter->FromJson(json);
    return parameter;
}

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
    for (uint64_t i = 0; i < vec.size(); ++i) {
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
    PARAM_CAST_OR_RETURN(TransformQuantizerParameter, p, other);
    CHECK_FIELD_EQ(*this, *p, tq_chain_);

    auto transformer_param = std::make_shared<VectorTransformerParameter>();
    transformer_param->FromJson(this->base_quantizer_json_);
    auto other_transformer_param = std::make_shared<VectorTransformerParameter>();
    other_transformer_param->FromJson(p->base_quantizer_json_);
    if (not transformer_param->CheckCompatibility(other_transformer_param)) {
        return false;
    }

    auto bottom_param = QuantizerParameter::GetQuantizerParameterByJson(this->base_quantizer_json_);
    auto other_bottom_param =
        QuantizerParameter::GetQuantizerParameterByJson(p->base_quantizer_json_);
    return bottom_param->CheckCompatibility(other_bottom_param);
}
}  // namespace vsag
