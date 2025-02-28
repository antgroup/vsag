
set (QUANTIZER_SRC
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
)

add_library (quantizer OBJECT ${QUANTIZER_SRC})
