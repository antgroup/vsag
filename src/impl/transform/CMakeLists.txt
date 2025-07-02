
set (TRANSFORM_SRC
        vector_transformer.cpp
        random_orthogonal_transformer.cpp
        fht_kac_rotate_transformer.cpp
        pca_transformer.cpp
)
add_library (transform OBJECT ${TRANSFORM_SRC})
target_link_libraries (transform PUBLIC fmt::fmt-header-only)
maybe_add_dependencies (transform spdlog openblas mkl)
