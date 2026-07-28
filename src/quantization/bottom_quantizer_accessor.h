// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "quantization/transform_quantization/transform_quantizer.h"

namespace vsag {

template <typename QuantizerT>
struct BottomQuantizerAccessor {
    using BottomQuantizerType = QuantizerT;
    using BottomComputerType = Computer<QuantizerT>;

    static QuantizerT&
    GetQuantizer(QuantizerT& quantizer) {
        return quantizer;
    }

    static const QuantizerT&
    GetQuantizer(const QuantizerT& quantizer) {
        return quantizer;
    }

    static BottomComputerType&
    GetComputer(Computer<QuantizerT>& computer) {
        return computer;
    }

    static const float*
    PrepareBottomInput(const QuantizerT&, const float* input, Vector<float>&) {
        return input;
    }
};

template <typename InnerQuantizerT, MetricType metric>
struct BottomQuantizerAccessor<TransformQuantizer<InnerQuantizerT, metric>> {
    using TransformQuantizerType = TransformQuantizer<InnerQuantizerT, metric>;
    using BottomQuantizerType = InnerQuantizerT;
    using BottomComputerType = Computer<InnerQuantizerT>;

    static InnerQuantizerT&
    GetQuantizer(TransformQuantizerType& quantizer) {
        return *quantizer.quantizer_;
    }

    static const InnerQuantizerT&
    GetQuantizer(const TransformQuantizerType& quantizer) {
        return *quantizer.quantizer_;
    }

    static BottomComputerType&
    GetComputer(Computer<TransformQuantizerType>& computer) {
        return *computer.inner_computer_;
    }

    static const float*
    PrepareBottomInput(const TransformQuantizerType& quantizer,
                       const float* input,
                       Vector<float>& scratch) {
        scratch.resize(quantizer.GetTransformedDim());
        quantizer.TransformBaseVector(input, scratch.data());
        return scratch.data();
    }
};

}  // namespace vsag
