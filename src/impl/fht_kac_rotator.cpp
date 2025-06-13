#include "fht_kac_rotator.h"
#include <immintrin.h> // AVX-512 intrinsics
#include <iostream>
namespace vsag{

static size_t ceil_log2(size_t val) {
    size_t res = 0;
    for (size_t i = 0; i < 31; ++i) {
        if ((1U << i) >= val) {
            res = i;
            break;
        }
    }
    return 1 << res;
}
static inline void flip_array(const uint8_t* flip, float* data, size_t dim) {//generic
    for (size_t i = 0; i < dim; i ++) {
        bool mask = flip[i/8] & (1 << (i%8));
        if (mask) {
          data[i] = -data[i];
        }
    }
}
static inline void flip_sign(const uint8_t* flip, float* data, size_t dim) {//翻转符号
    constexpr size_t kFloatsPerChunk = 64;  // Process 64 floats per iteration
    // constexpr size_t bits_per_chunk = floats_per_chunk;  // 64 bits = 8 bytes

    static_assert(
        kFloatsPerChunk % 16 == 0,
        "floats_per_chunk must be divisible by AVX512 register width"
    );

    for (size_t i = 0; i < dim; i += kFloatsPerChunk) {
        // Load 64 bits (8 bytes) from the bit sequence
        uint64_t mask_bits;
        std::memcpy(&mask_bits, &flip[i / 8], sizeof(mask_bits));

        // Split into four 16-bit mask segments
        const __mmask16 mask0 = _cvtu32_mask16(static_cast<uint32_t>(mask_bits & 0xFFFF));
        const __mmask16 mask1 =
            _cvtu32_mask16(static_cast<uint32_t>((mask_bits >> 16) & 0xFFFF));
        const __mmask16 mask2 =
            _cvtu32_mask16(static_cast<uint32_t>((mask_bits >> 32) & 0xFFFF));
        const __mmask16 mask3 =
            _cvtu32_mask16(static_cast<uint32_t>((mask_bits >> 48) & 0xFFFF));

        // Prepare sign-flip constant
        const __m512 sign_flip = _mm512_castsi512_ps(_mm512_set1_epi32(0x80000000));

        // Process 16 floats at a time with each mask segment
        __m512 vec0 = _mm512_loadu_ps(&data[i]);
        vec0 = _mm512_mask_xor_ps(vec0, mask0, vec0, sign_flip);
        _mm512_storeu_ps(&data[i], vec0);

        __m512 vec1 = _mm512_loadu_ps(&data[i + 16]);
        vec1 = _mm512_mask_xor_ps(vec1, mask1, vec1, sign_flip);
        _mm512_storeu_ps(&data[i + 16], vec1);

        __m512 vec2 = _mm512_loadu_ps(&data[i + 32]);
        vec2 = _mm512_mask_xor_ps(vec2, mask2, vec2, sign_flip);
        _mm512_storeu_ps(&data[i + 32], vec2);

        __m512 vec3 = _mm512_loadu_ps(&data[i + 48]);
        vec3 = _mm512_mask_xor_ps(vec3, mask3, vec3, sign_flip);
        _mm512_storeu_ps(&data[i + 48], vec3);
    }
}
    inline size_t floor_log2(size_t x) {//小于等于
        size_t ret = 0;
        while (x > 1) {
            ret++;
            x >>= 1;
        }
        return ret;
    }
    inline void vec_rescale(float* data, size_t dim, float val) {
        for(int i = 0; i < dim; i++){
            data[i] *= val;
        }
    }
    
    inline size_t round_up_to_multiple_of(size_t x, size_t div){
        size_t res = x / div;
        if(x % div != 0){
            res+=1;
        }
        return res * div;
    }
    void
    FhtKacRotator::CopyFlip(uint8_t* out_flip) const {
        std::copy(flip_.data(), flip_.data() + flip_.size(), out_flip);
    }

    FhtKacRotator::FhtKacRotator(uint64_t dim, Allocator* allocator)
    :dim_(dim),  allocator_(allocator)
    {
        flip_offset_ = (dim_ + 7) / kByteLen_;
        flip_.resize(round_ * flip_offset_);//原作者中round_ = 4
        std::random_device rd;   // Seed
        std::mt19937 gen(rd());  // Mersenne Twister RNG

        // Uniform distribution in the range [0, 255]
        std::uniform_int_distribution<int> dist(0, 255);

        // Generate a single random uint8_t value
        for (auto& i : flip_) {
            i = static_cast<uint8_t>(dist(gen));
        }

        size_t bottom_log_dim = floor_log2(dim);
        trunc_dim_ = 1 << bottom_log_dim;//2次幂
        fac_ = 1.0F / std::sqrt(static_cast<float>(trunc_dim_));
    }
    FhtKacRotator::~FhtKacRotator() {}
    static void kacs_walk(float* data, size_t len) {
        // ! len % 32 == 0;
        int base = len % 2;
        if (base == 0) {
            int offset = base + (len / 2); // for odd dim
            for (size_t i = 0; i < len / 2; i += 16) {
                __m512 x = _mm512_loadu_ps(&data[i]);
                __m512 y = _mm512_loadu_ps(&data[i + offset]);

                __m512 new_x = _mm512_add_ps(x, y);
                __m512 new_y = _mm512_sub_ps(x, y);

                _mm512_storeu_ps(&data[i], new_x);
                _mm512_storeu_ps(&data[i + offset], new_y);
            }
        } else {
            std::vector<float> tmp(len + 1, 0);
            memcpy(tmp.data(), data, len * sizeof(float));
            // copy
            int offset = base + (len / 2); // for odd dim
            for (size_t i = 0; i < len / 2; i += 16) {
                __m512 x = _mm512_loadu_ps(&data[i]);
                __m512 y = _mm512_loadu_ps(&data[i + offset]);

                __m512 new_x = _mm512_add_ps(x, y);
                __m512 new_y = _mm512_sub_ps(x, y);

                _mm512_storeu_ps(&data[i], new_x);
                _mm512_storeu_ps(&data[i + offset], new_y);
            }
            // copy back
            memcpy(data, tmp.data(), len * sizeof(float));
        }
    }
    static void kacs_walk_generic(float* data, size_t len) {
        // ! len % 32 == 0;
        int base = len % 2;
        int offset = base + (len / 2); // for odd dim
        for (size_t i = 0; i < len / 2; i ++) {
            float add = data[i] + data[i + offset];
            float sub = data[i] - data[i + offset];
            data[i] = add;
            data[i + offset] = sub;
        }
        if(base){
            data[len / 2] *= sqrt(2);
        }
    }

    void FhtKacRotator::fht_float_(float* data) const{
        int n = trunc_dim_;
        int step = 1;
        // 逐步合并
        while (step < n) {
            for (int i = 0; i < n; i += step * 2) {
                // if(step>16 && step % 16 == 0){
                //     for (int j = 0; j < step; j+=16) {
                //         __m512 g1 = _mm512_loadu_ps(&data[i + j]);
                //         __m512 g2 = _mm512_loadu_ps(&data[i + j + step]);
                //         _mm512_storeu_ps(&data[i + j],_mm512_add_ps(g1, g2));
                //         _mm512_storeu_ps(&data[i + j + step],_mm512_sub_ps(g1, g2));
                //     }
                // } else//generic
                {
                    for (int j = 0; j < step; j++) {
                        // 合并操作
                        float even = data[i + j];
                        float odd = data[i + j + step];
                        // 更新数组
                        data[i + j] = even + odd;         // 相加
                        data[i + j + step] = even - odd; // 相减
                    }
                }
            }
            step *= 2; // 增加步长
        }
    }

    void FhtKacRotator::Transform(const float* data, float* rotated_vec) const{
        std::memcpy(rotated_vec, data, sizeof(float) * dim_);
        std::fill(rotated_vec + dim_, rotated_vec + dim_, 0);
        if (trunc_dim_ == dim_) {
            flip_array(flip_.data() + 0 * flip_offset_, rotated_vec, dim_);
            fht_float_(rotated_vec);
            vec_rescale(rotated_vec, trunc_dim_, fac_);

            flip_array(flip_.data()+ 1 * flip_offset_, rotated_vec, dim_);
            fht_float_(rotated_vec);
            vec_rescale(rotated_vec, trunc_dim_, fac_);

            flip_array(flip_.data()+ 2 * flip_offset_, rotated_vec, dim_);
            fht_float_(rotated_vec);
            vec_rescale(rotated_vec, trunc_dim_, fac_);

            flip_array(flip_.data()+ 3 * flip_offset_, rotated_vec, dim_);
            fht_float_(rotated_vec);
            vec_rescale(rotated_vec, trunc_dim_, fac_);

            return;
        }

        size_t start = dim_ - trunc_dim_;

        flip_array(flip_.data() + 0 * flip_offset_, rotated_vec, dim_);
        fht_float_(rotated_vec);
        vec_rescale(rotated_vec, trunc_dim_, fac_);
        kacs_walk_generic(rotated_vec, dim_);
        vec_rescale(rotated_vec, dim_, sqrt(0.5));

        flip_array(flip_.data() + flip_offset_, rotated_vec, dim_);
        fht_float_(rotated_vec + start);
        vec_rescale(rotated_vec + start, trunc_dim_, fac_);
        kacs_walk_generic(rotated_vec, dim_);
        vec_rescale(rotated_vec, dim_, sqrt(0.5));

        flip_array(flip_.data() + 2 * flip_offset_, rotated_vec, dim_);
        fht_float_(rotated_vec);
        vec_rescale(rotated_vec, trunc_dim_, fac_);
        kacs_walk_generic(rotated_vec, dim_);
        vec_rescale(rotated_vec, dim_, sqrt(0.5));

        flip_array(flip_.data() + 3 * flip_offset_, rotated_vec, dim_);
        fht_float_(rotated_vec + start);
        vec_rescale(rotated_vec + start, trunc_dim_, fac_);
        kacs_walk_generic(rotated_vec, dim_);
        vec_rescale(rotated_vec, dim_, sqrt(0.5));

    }
    void FhtKacRotator::InverseTransform(float const*data, float*rotated_vec) const{
        std::memcpy(rotated_vec, data, sizeof(float) * dim_);
        std::fill(rotated_vec + dim_, rotated_vec + dim_, 0);
        if (trunc_dim_ == dim_) {
            fht_float_(rotated_vec);
            vec_rescale(rotated_vec, trunc_dim_, fac_);
            flip_array(flip_.data() + 3 * flip_offset_, rotated_vec, dim_);
            
            fht_float_(rotated_vec);
            vec_rescale(rotated_vec, trunc_dim_, fac_);
            flip_array(flip_.data()+ 2 * flip_offset_, rotated_vec, dim_);

            fht_float_(rotated_vec);
            vec_rescale(rotated_vec, trunc_dim_, fac_);
            flip_array(flip_.data()+ 1 * flip_offset_, rotated_vec, dim_);

            fht_float_(rotated_vec);
            vec_rescale(rotated_vec, trunc_dim_, fac_);
            flip_array(flip_.data() + 0 * flip_offset_, rotated_vec, dim_);

            return;
        }

        size_t start = dim_ - trunc_dim_;

        kacs_walk_generic(rotated_vec,dim_);
        vec_rescale(rotated_vec, dim_, sqrt(0.5));
        fht_float_(rotated_vec + start);
        vec_rescale(rotated_vec + start, trunc_dim_, fac_);
        flip_array(flip_.data() + 3 * flip_offset_,rotated_vec, dim_);

        kacs_walk_generic(rotated_vec, dim_);
        vec_rescale(rotated_vec, dim_, sqrt(0.5));
        fht_float_(rotated_vec);
        vec_rescale(rotated_vec, trunc_dim_, fac_);
        flip_array(flip_.data() + 2 * flip_offset_,rotated_vec, dim_);

        kacs_walk_generic(rotated_vec,dim_);
        vec_rescale(rotated_vec, dim_, sqrt(0.5));
        fht_float_(rotated_vec + start);
        vec_rescale(rotated_vec + start, trunc_dim_, fac_);
        flip_array(flip_.data() + 1 * flip_offset_,rotated_vec, dim_);

        kacs_walk_generic(rotated_vec, dim_);
        vec_rescale(rotated_vec, dim_, sqrt(0.5));
        fht_float_(rotated_vec);
        vec_rescale(rotated_vec, trunc_dim_, fac_);
        flip_array(flip_.data() + 0 * flip_offset_,rotated_vec, dim_);

    }

    void FhtKacRotator::Serialize(StreamWriter& writer) {
        StreamWriter::WriteVector(writer, this->flip_);
    }

    void FhtKacRotator::Deserialize(StreamReader& reader) {
        StreamReader::ReadVector(reader, this->flip_);
    }


}
