#include <cblas.h>
#include <lapacke.h>

#include "stream_reader.h"
#include "stream_writer.h"
#include <random>
#include "../logger.h"
// #include "typing.h"
// #include "vsag/allocator.h"
namespace vsag{
    
class HadamardMatrix{
    public:
    HadamardMatrix(uint64_t dim,Allocator* allocator)
    :dim_(dim),
    allocator_(allocator),
    hadamard_matrix_(allocator)

    {}
    // void CopyOrthogonalMatrix() const;

    void Transform(const float* original_vec, float* transformed_vec) const;

    void InverseTransform(const float* transformed_vec, float* original_vec) const;

    bool GenerateHadamardMatrix();

    void Serialize(StreamWriter& writer);

    void Deserialize(StreamReader& reader);


    private:
        const uint64_t dim_{0};

        vsag::Vector<float> hadamard_matrix_;

        Allocator* const allocator_{nullptr};


};

}