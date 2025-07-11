
set (IVF_PARTITION_SRCS
        gno_imi_parameter.cpp
        ivf_nearest_partition.cpp
        gno_imi_partition.cpp
        ivf_partition_strategy_parameter.cpp
        ivf_partition_strategy.cpp
)

add_library (ivf_partition OBJECT ${IVF_PARTITION_SRCS})
target_link_libraries (ivf_partition PUBLIC coverage_config)
maybe_add_dependencies (ivf_partition spdlog fmt::fmt mkl openblas)
