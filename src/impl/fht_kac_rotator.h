
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

#include <stdint.h>

#include <cstring>
#include <random>

#include "../logger.h"
#include "matrix_rotator.h"
#include "simd/simd.h"
#include "stream_reader.h"
#include "stream_writer.h"

namespace vsag {
class FhtKacRotator : public MatrixRotator {
public:
    FhtKacRotator(uint64_t dim, Allocator* allocator);

    ~FhtKacRotator() = default;

    void
    Transform(const float* data, float* rotated_vec) const override;

    void
    InverseTransform(const float* data, float* rotated_vec) const override;

    void
    Serialize(StreamWriter& writer) override;

    void
    Deserialize(StreamReader& reader) override;

    bool
    Build() override;

    void
    CopyFlip(uint8_t* out_flip) const;

    const size_t kByteLen_ = 8;
    const int round_ = 4;

private:
    const uint64_t dim_{0};
    size_t flip_offset_ = 0;
    Allocator* const allocator_{nullptr};

    std::vector<uint8_t> flip_;

    size_t trunc_dim_ = 0;

    float fac_ = 0;
};
}  //namespace vsag
