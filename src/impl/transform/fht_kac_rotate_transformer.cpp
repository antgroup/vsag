
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

#include "fht_kac_rotate_transformer.h"

#include <random>

#include "simd/rabitq_simd.h"

namespace vsag {

inline size_t
floor_log2(size_t x) {  //smaller or equal
    size_t ret = 0;
    while (x > 1) {
        ret++;
        x >>= 1;
    }
    return ret;
}

void
FhtKacRotator::CopyFlip(uint8_t* out_flip) const {
    std::copy(flip_.data(), flip_.data() + flip_.size(), out_flip);
}

FhtKacRotator::FhtKacRotator(Allocator* allocator, int64_t dim)
    : VectorTransformer(allocator, dim) {
    this->type_ = VectorTransformerType::FHT;
    flip_offset_ = (this->input_dim_ + 7) / BYTE_LEN;
    flip_.resize(ROUND * flip_offset_);
    size_t bottom_log_dim = floor_log2(dim);
    trunc_dim_ = 1 << bottom_log_dim;
    fac_ = 1.0F / std::sqrt(static_cast<float>(trunc_dim_));
}

void
FhtKacRotator::Train() {
    std::random_device rd;   // Seed
    std::mt19937 gen(rd());  // Mersenne Twister RNG
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto& i : flip_) {
        i = static_cast<uint8_t>(dist(gen));
    }
}

void
FhtKacRotator::Train(const float* data, uint64_t count) {
    this->Train();
}

TransformerMetaPtr
FhtKacRotator::Transform(const float* data, float* rotated_vec) const {
    auto meta = std::make_shared<FHTMeta>();
    auto dim = static_cast<uint64_t>(this->input_dim_);
    std::memcpy(rotated_vec, data, sizeof(float) * dim);
    if (trunc_dim_ == dim) {
        for (int flip_time = 0; flip_time < ROUND; flip_time++) {
            FlipSign(flip_.data() + flip_time * flip_offset_, rotated_vec, dim);
            FHTRotate(rotated_vec, trunc_dim_);
            VecRescale(rotated_vec, trunc_dim_, fac_);
        }
        return meta;
    }

    size_t start = dim - trunc_dim_;

    for (int flip_time = 0; flip_time < ROUND; flip_time += 2) {
        FlipSign(flip_.data() + flip_time * flip_offset_, rotated_vec, dim);
        FHTRotate(rotated_vec, trunc_dim_);
        VecRescale(rotated_vec, trunc_dim_, fac_);
        KacsWalk(rotated_vec, dim);

        FlipSign(flip_.data() + (flip_time + 1) * flip_offset_, rotated_vec, dim);
        FHTRotate(rotated_vec + start, trunc_dim_);
        VecRescale(rotated_vec + start, trunc_dim_, fac_);
        KacsWalk(rotated_vec, dim);
    }
    VecRescale(rotated_vec, dim, 0.25F);
    //origin vec(x,y), after kacs_walk_generic() -> (x+y, x-y),should be resize by sqrt(0.5) for each KacsWalk() to make the len of vector consistent

    return meta;
}
void
FhtKacRotator::InverseTransform(float const* data, float* rotated_vec) const {
    auto dim = static_cast<uint64_t>(this->input_dim_);

    std::memcpy(rotated_vec, data, sizeof(float) * dim);
    if (trunc_dim_ == dim) {
        for (int flip_time = ROUND - 1; flip_time >= 0; flip_time--) {
            FHTRotate(rotated_vec, trunc_dim_);
            VecRescale(rotated_vec, trunc_dim_, fac_);
            FlipSign(flip_.data() + flip_time * flip_offset_, rotated_vec, dim);
        }
        return;
    }

    size_t start = dim - trunc_dim_;

    VecRescale(rotated_vec, dim, 0.25F);
    for (int flip_time = ROUND - 1; flip_time > 0; flip_time -= 2) {
        KacsWalk(rotated_vec, dim);
        FHTRotate(rotated_vec + start, trunc_dim_);
        VecRescale(rotated_vec + start, trunc_dim_, fac_);
        FlipSign(flip_.data() + flip_time * flip_offset_, rotated_vec, dim);

        KacsWalk(rotated_vec, dim);
        FHTRotate(rotated_vec, trunc_dim_);
        VecRescale(rotated_vec, trunc_dim_, fac_);
        FlipSign(flip_.data() + (flip_time - 1) * flip_offset_, rotated_vec, dim);
    }
}

void
FhtKacRotator::Serialize(StreamWriter& writer) const {
    StreamWriter::WriteVector(writer, this->flip_);
}

void
FhtKacRotator::Deserialize(StreamReader& reader) {
    StreamReader::ReadVector(reader, this->flip_);
}

}  // namespace vsag
