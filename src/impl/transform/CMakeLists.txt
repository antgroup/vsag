
set (TRANSFORM_SRC
        vector_transformer.cpp
        vector_transformer.h
        random_orthogonal_transformer.cpp
        random_orthogonal_transformer.h
        fht_kac_rotate_transformer.cpp
        fht_kac_rotate_transformer.h
        pca_transformer.cpp
        pca_transformer.h
)
add_library (transform OBJECT ${TRANSFORM_SRC})
target_link_libraries (transform PUBLIC fmt::fmt coverage_config)
maybe_add_dependencies (transform spdlog openblas mkl)
