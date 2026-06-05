
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
#include "utils/param_compat_macros.h"

namespace vsag {

void
VectorTransformerParameter::FromJson(const JsonType& json) {
    if (json.Contains(INPUT_DIM_KEY)) {
        input_dim_ = json[INPUT_DIM_KEY].GetInt();
    }

    if (json.Contains(PCA_DIM_KEY)) {
        pca_dim_ = json[PCA_DIM_KEY].GetInt();
    }

    if (json.Contains(MRLE_DIM_KEY)) {
        mrle_dim_ = json[MRLE_DIM_KEY].GetInt();
    }
}

JsonType
VectorTransformerParameter::ToJson() const {
    JsonType json;
    json[PCA_DIM_KEY].SetInt(pca_dim_);
    json[MRLE_DIM_KEY].SetInt(mrle_dim_);
    json[INPUT_DIM_KEY].SetInt(input_dim_);
    return json;
}

bool
VectorTransformerParameter::CheckCompatibility(const ParamPtr& other) const {
    PARAM_CAST_OR_RETURN(VectorTransformerParameter, p, other);
    CHECK_FIELD_EQ(*this, *p, pca_dim_);
    CHECK_FIELD_EQ(*this, *p, input_dim_);
    CHECK_FIELD_EQ(*this, *p, mrle_dim_);
    return true;
}

}  // namespace vsag
