
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

#include <cstdint>
#include <cstring>

#include "vector_transformer.h"

namespace vsag {

struct FHTMeta : public TransformerMeta {};

class FhtKacRotator : public VectorTransformer {
public:
    explicit FhtKacRotator(Allocator* allocator, int64_t dim);

    ~FhtKacRotator() override = default;

    TransformerMetaPtr
    Transform(const float* data, float* rotated_vec) const override;

    void
    InverseTransform(const float* data, float* rotated_vec) const override;

    void
    Serialize(StreamWriter& writer) const override;

    void
    Deserialize(StreamReader& reader) override;

    void
    Train(const float* data, uint64_t count) override;

    void
    Train();

    void
    CopyFlip(uint8_t* out_flip) const;

    constexpr static size_t BYTE_LEN = 8;
    constexpr static int ROUND = 4;

private:
    size_t flip_offset_{0};

    std::vector<uint8_t> flip_;

    size_t trunc_dim_{0};

    float fac_{0};
};
}  //namespace vsag
