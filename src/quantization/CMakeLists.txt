
set (QUANTIZER_SRC
        fp32_quantizer.cpp
        scalar_quantization/bf16_quantizer.cpp
        scalar_quantization/fp16_quantizer.cpp
        scalar_quantization/sq4_quantizer.cpp
        scalar_quantization/sq4_uniform_quantizer.cpp
        scalar_quantization/sq8_quantizer.cpp
        scalar_quantization/sq8_uniform_quantizer.cpp
        product_quantization/pq_fastscan_quantizer.cpp
        product_quantization/product_quantizer.cpp
        rabitq_quantization/rabitq_quantizer.cpp
        transform_quantization/transform_quantizer.cpp
        quantizer_parameter.cpp
        fp32_quantizer_parameter.cpp
        scalar_quantization/sq8_quantizer_parameter.cpp
        scalar_quantization/sq8_uniform_quantizer_parameter.cpp
        scalar_quantization/sq4_quantizer_parameter.cpp
        scalar_quantization/sq4_uniform_quantizer_parameter.cpp
        scalar_quantization/bf16_quantizer_parameter.cpp
        scalar_quantization/scalar_quantization_trainer.cpp
        scalar_quantization/fp16_quantizer_parameter.cpp
        rabitq_quantization/rabitq_quantizer_parameter.cpp
        product_quantization/product_quantizer_parameter.cpp
        product_quantization/pq_fastscan_quantizer_parameter.cpp
        transform_quantization/transform_quantizer_parameter.cpp
        quantizer_interface.cpp
)

add_library (quantizer OBJECT ${QUANTIZER_SRC})
target_link_libraries(quantizer PUBLIC fmt::fmt coverage_config)

maybe_add_dependencies (quantizer spdlog openblas mkl)
