
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

#include "vector_transformer_parameter.h"

#include "inner_string_params.h"

namespace vsag {

void
VectorTransformerParameter::FromJson(const JsonType& json) {
    if (json.contains(INPUT_DIM)) {
        input_dim_ = json[INPUT_DIM];
    }

    if (json.contains(PCA_DIM)) {
        pca_dim_ = json[PCA_DIM];
    }
}

JsonType
VectorTransformerParameter::ToJson() const {
    JsonType json;
    json[PCA_DIM] = pca_dim_;
    json[INPUT_DIM] = input_dim_;
    return json;
}

bool
VectorTransformerParameter::CheckCompatibility(const ParamPtr& other) const {
    auto param = std::dynamic_pointer_cast<VectorTransformerParameter>(other);
    if (not param) {
        logger::error(
            "VectorTransformerParameter::CheckCompatibility: other parameter is not a "
            "VectorTransformerParameter");
        return false;
    }
    if (pca_dim_ != param->pca_dim_) {
        return false;
    }
    if (input_dim_ != param->input_dim_) {
        return false;
    }
    return true;
}

}  // namespace vsag
