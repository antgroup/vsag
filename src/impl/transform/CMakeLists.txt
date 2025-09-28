
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


set (TRANSFORM_SRC
        vector_transformer.cpp
        vector_transformer.h
        random_orthogonal_transformer.cpp
        random_orthogonal_transformer.h
        fht_kac_rotate_transformer.cpp
        fht_kac_rotate_transformer.h
        pca_transformer.cpp
        pca_transformer.h
        vector_transformer_parameter.cpp
        vector_transformer_parameter.h
)
add_library (transform OBJECT ${TRANSFORM_SRC})
target_link_libraries (transform PUBLIC fmt::fmt coverage_config)
maybe_add_dependencies (transform spdlog openblas mkl)
