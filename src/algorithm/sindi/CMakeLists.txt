
set (SINDI_SRCS
        sindi.cpp
        sindi_parameter.cpp
)

add_library (sindi OBJECT ${SINDI_SRCS})
target_link_libraries (sindi PUBLIC coverage_config)
maybe_add_dependencies (sindi spdlog fmt::fmt mkl openblas)
