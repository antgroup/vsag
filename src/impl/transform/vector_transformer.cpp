
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "vector_transformer.h"

#include "vsag_exception.h"

namespace vsag {

VectorTransformer::VectorTransformer(Allocator* allocator, int64_t input_dim, int64_t output_dim)
    : allocator_(allocator), input_dim_(input_dim), output_dim_(output_dim) {
}
void
VectorTransformer::InverseTransform(const float* input_vec, float* output_vec) const {
    throw VsagException(ErrorType::INTERNAL_ERROR, "InverseTransform not implement");
}

}  // namespace vsag