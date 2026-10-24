#pragma once

#include "inner_string_params.h"
#include "quantization/quantizer_parameter.h"

namespace vsag {

class SQ4LocalSharedQuantizerParameter : public QuantizerParameter {
public:
    SQ4LocalSharedQuantizerParameter()
        : QuantizerParameter(QUANTIZATION_TYPE_VALUE_SQ4_LOCAL_SHARED) {
    }

    void
    FromJson(const JsonType&) override {
    }

    [[nodiscard]] JsonType
    ToJson() const override {
        JsonType json;
        json[TYPE_KEY].SetString(this->GetTypeName());
        return json;
    }
};

}  // namespace vsag
