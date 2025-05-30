#include "hadamard_matrix.h"
#include <immintrin.h> // AVX-512 intrinsics
namespace vsag{
    // void HadamardMatrix::CopyOrthogonalMatrix() const{

    // }
    void HadamardMatrix::Transform(const float* original_vec, float* transformed_vec) const{
        int n = dim_;
        int step = 1;
        // 逐步合并
        while (step < n) {
            for (int i = 0; i < n; i += step * 2) {
                if(step>16 && step % 16 == 0){
                    for (int j = 0; j < step; j+=16) {
                        __m512 g1 = _mm512_loadu_ps(&original_vec[i + j]);
                        __m512 g2 = _mm512_loadu_ps(&original_vec[i + j + step]);
                        _mm512_storeu_ps(&transformed_vec[i + j],_mm512_add_ps(g1, g2));
                        _mm512_storeu_ps(&transformed_vec[i + j + step],_mm512_sub_ps(g1, g2));
                    }
                }else{
                    for (int j = 0; j < step; j++) {
                        // 合并操作
                        float even = original_vec[i + j];
                        float odd = original_vec[i + j + step];
                        // 更新数组
                        transformed_vec[i + j] = even + odd;         // 相加
                        transformed_vec[i + j + step] = even - odd; // 相减
                    }
                }
            }
            step *= 2; // 增加步长
        }
    }

    void HadamardMatrix::InverseTransform(const float* transformed_vec, float* original_vec) const{
        Transform(transformed_vec,original_vec);
        for(int i = 0; i < dim_; i++){
            original_vec[i]/=dim_;
        }
        //利用Hadamard矩阵的特性做逆运算
    }

    bool HadamardMatrix::GenerateHadamardMatrix(){
        return true;
    };

    void HadamardMatrix::Serialize(StreamWriter& writer){};

    void HadamardMatrix::Deserialize(StreamReader& reader){};

}//namespace vsag