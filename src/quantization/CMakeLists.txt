
# Copyright 2024-present the vsag project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


set (QUANTIZER_SRC
        quantizer_adapter.cpp
        fp32_quantizer.cpp
        int8_quantizer.cpp
        scalar_quantization/scalar_quantizer.cpp
        scalar_quantization/bf16_quantizer.cpp
        scalar_quantization/fp16_quantizer.cpp
        scalar_quantization/sq4_uniform_quantizer.cpp
        scalar_quantization/sq8_uniform_quantizer.cpp
        product_quantization/pq_fastscan_quantizer.cpp
        product_quantization/product_quantizer.cpp
        rabitq_quantization/rabitq_quantizer.cpp
        quantizer_parameter.cpp
        fp32_quantizer_parameter.cpp
        int8_quantizer_parameter.cpp
        scalar_quantization/sq8_uniform_quantizer_parameter.cpp
        scalar_quantization/sq4_uniform_quantizer_parameter.cpp
        scalar_quantization/bf16_quantizer_parameter.cpp
        scalar_quantization/scalar_quantization_trainer.cpp
        scalar_quantization/fp16_quantizer_parameter.cpp
        rabitq_quantization/rabitq_quantizer_parameter.cpp
        product_quantization/product_quantizer_parameter.cpp
        product_quantization/pq_fastscan_quantizer_parameter.cpp
        transform_quantization/transform_quantizer_parameter.cpp
)

add_library (quantizer OBJECT ${QUANTIZER_SRC})
target_link_libraries(quantizer PUBLIC fmt::fmt coverage_config)

maybe_add_dependencies (quantizer spdlog openblas mkl)

if (ENABLE_TESTS)
    file (GLOB_RECURSE QUANTIZER_TESTS "*_test.cpp")
    add_library (quantizer_test STATIC ${QUANTIZER_TESTS})
    target_link_libraries (quantizer_test PRIVATE Catch2::Catch2 vsag)
    add_dependencies (quantizer_test spdlog Catch2)
endif ()
