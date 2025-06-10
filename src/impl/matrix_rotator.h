#pragma once

#include "stream_reader.h"
#include "stream_writer.h"

class MatrixRotator
{
    public:
    MatrixRotator() {}
    virtual ~MatrixRotator() {}

    virtual void Transform(const float* original_vec, float* transformed_vec) const = 0;

    virtual void InverseTransform(const float* transformed_vec, float* original_vec) const = 0;

    virtual bool GenerateRandomOrthogonalMatrix() = 0;

    virtual void GenerateRandomOrthogonalMatrixWithRetry() = 0;

    virtual double ComputeDeterminant() const  = 0;

    virtual void Serialize(StreamWriter& writer) = 0;

    virtual void Deserialize(StreamReader& reader) = 0;
};