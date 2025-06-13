#include "matrix_rotator.h"
#include <stdint.h>
#include <random>
#include <cstring>
#include "stream_reader.h"
#include "stream_writer.h"
#include "../logger.h"

namespace vsag{
    class FhtKacRotator : public MatrixRotator {
         public:
        FhtKacRotator(uint64_t dim, Allocator* allocator);
        virtual ~FhtKacRotator();
        // void CopyOrthogonalMatrix() const;

        void Transform(const float* original_vec, float* transformed_vec) const override;

        void InverseTransform(const float* transformed_vec, float* original_vec) const override;

        bool GenerateRandomOrthogonalMatrix() override { return true; }

        void GenerateRandomOrthogonalMatrixWithRetry() override {GenerateRandomOrthogonalMatrix();}

        double ComputeDeterminant() const override { return 1.0; }

        void Serialize(StreamWriter& writer) override;

        void Deserialize(StreamReader& reader) override ;

        void fht_float_(float *data) const;
        void calculate() const;

        void random_flip();

        void swap_data(float* data)const;

        void CopyFlip(uint8_t* out_flip) const;

        const size_t kByteLen_ = 8;
        const size_t round_ = 4;


        private:
            const uint64_t dim_{0};
            size_t flip_offset_ = 0;
            Allocator* const allocator_{nullptr};

            std::vector<uint8_t> flip_;

            size_t trunc_dim_ = 0;

            float fac_ = 0;
        
        };
}