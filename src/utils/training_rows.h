// Copyright 2024-present the vsag project
// Licensed under the Apache License, Version 2.0.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>

#include "data_type.h"
#include "utils/half_input.h"
#include "vsag_exception.h"
namespace vsag {
// Non-owning training view. Stride and extent are bytes, including any row padding.
// Decoding is caller-buffered; this view never allocates a count-by-dim mirror.
class TrainingRows {
public:
    using RowReader = void (*)(const void* context, uint64_t row, float* output);
    using ColumnReader = void (*)(const void*, uint64_t, uint64_t, uint64_t, float*);

    // Borrowed synchronous provider: fills exactly Dim() FP32 values; may run
    // concurrently. Reader must be repeatable/thread-safe and must not retain
    // output. Context (possibly null) and dependencies must outlive Run().
    TrainingRows(uint64_t count,
                 int32_t dim,
                 RowReader reader,
                 const void* context,
                 ColumnReader columns = nullptr)
        : data_(nullptr),
          count_(count),
          dim_(dim),
          type_(DataTypes::DATA_TYPE_FLOAT),
          stride_(0),
          reader_(reader),
          context_(context),
          columns_(columns) {
        if (count == 0 || dim <= 0 || reader == nullptr) {
            throw VsagException(ErrorType::INVALID_ARGUMENT, "invalid kmeans row provider");
        }
    }

    uint64_t
    Count() const {
        return count_;
    }
    int32_t
    Dim() const {
        return dim_;
    }

    TrainingRows(const void* data,
                 uint64_t count,
                 int32_t dim,
                 DataTypes type,
                 uint64_t stride_bytes,
                 uint64_t extent_bytes)
        : data_(static_cast<const uint8_t*>(data)),
          count_(count),
          dim_(dim),
          type_(type),
          stride_(stride_bytes) {
        if (type != DataTypes::DATA_TYPE_FLOAT && type != DataTypes::DATA_TYPE_FP16 &&
            type != DataTypes::DATA_TYPE_BF16) {
            throw VsagException(ErrorType::INVALID_ARGUMENT, "unsupported kmeans input type");
        }
        const uint64_t width =
            type == DataTypes::DATA_TYPE_FLOAT ? sizeof(float) : sizeof(uint16_t);
        if (data == nullptr || count == 0 || dim <= 0) {
            throw VsagException(ErrorType::INVALID_ARGUMENT, "invalid typed kmeans input");
        }
        const uint64_t row_bytes = static_cast<uint64_t>(dim) * width;
        if (stride_ < row_bytes ||
            count - 1 > (std::numeric_limits<uint64_t>::max() - row_bytes) / stride_ ||
            extent_bytes < (count - 1) * stride_ + row_bytes) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                "invalid typed kmeans stride or extent");
        }
        const uint64_t span = (count - 1) * stride_ + row_bytes;
        if (span > static_cast<uint64_t>(PTRDIFF_MAX))
            throw VsagException(ErrorType::INVALID_ARGUMENT, "training span exceeds PTRDIFF_MAX");
        if (span - 1 > std::numeric_limits<uintptr_t>::max() - reinterpret_cast<uintptr_t>(data_)) {
            throw VsagException(ErrorType::INVALID_ARGUMENT, "kmeans address span overflow");
        }
    }

    void
    Read(uint64_t row, float* output) const {
        if (row >= count_ || output == nullptr) {
            throw VsagException(ErrorType::INVALID_ARGUMENT, "invalid kmeans row read");
        }
        if (reader_ != nullptr) {
            reader_(context_, row, output);
            return;
        }
        const auto* source = data_ + row * stride_;
        if (type_ == DataTypes::DATA_TYPE_FLOAT) {
            std::memcpy(output, source, static_cast<uint64_t>(dim_) * sizeof(float));
            return;
        }
        for (int32_t j = 0; j < dim_; ++j) {
            uint16_t bits;
            std::memcpy(&bits, source + static_cast<uint64_t>(j) * sizeof(bits), sizeof(bits));
            if (type_ == DataTypes::DATA_TYPE_FP16) {
                output[j] = half_input::FP16ToFloat(bits);
            } else {
                uint32_t expanded = static_cast<uint32_t>(bits) << 16;
                std::memcpy(output + j, &expanded, sizeof(expanded));
            }
        }
    }

    // Columns must exactly match Read(row)[begin:begin+columns], including FP32 bits.
    // Provider contexts are synchronous, repeatable and thread-safe (KMeans parallel reads).
    bool
    HasColumnReader() const {
        return reader_ == nullptr || columns_ != nullptr;
    }
    const float*
    ContiguousFP32() const {
        return reader_ == nullptr && type_ == DataTypes::DATA_TYPE_FLOAT &&
                       reinterpret_cast<uintptr_t>(data_) % alignof(float) == 0 &&
                       stride_ == static_cast<uint64_t>(dim_) * sizeof(float)
                   ? reinterpret_cast<const float*>(data_)
                   : nullptr;
    }
    void
    ReadColumns(uint64_t row, uint64_t begin, uint64_t columns, float* output) const {
        if (row >= count_ || begin > static_cast<uint64_t>(dim_) ||
            columns > static_cast<uint64_t>(dim_) - begin || (columns && !output))
            throw VsagException(ErrorType::INVALID_ARGUMENT, "invalid training column read");
        if (columns == 0)
            return;
        if (reader_) {
            if (!columns_)
                throw VsagException(ErrorType::INVALID_ARGUMENT, "provider has no column reader");
            columns_(context_, row, begin, columns, output);
            return;
        }
        const uint64_t width =
            type_ == DataTypes::DATA_TYPE_FLOAT ? sizeof(float) : sizeof(uint16_t);
        const auto* source = data_ + row * stride_ + begin * width;
        if (type_ == DataTypes::DATA_TYPE_FLOAT) {
            std::memcpy(output, source, columns * sizeof(float));
            return;
        }
        for (uint64_t j = 0; j < columns; ++j) {
            uint16_t bits;
            std::memcpy(&bits, source + j * sizeof(bits), sizeof(bits));
            if (type_ == DataTypes::DATA_TYPE_FP16)
                output[j] = half_input::FP16ToFloat(bits);
            else {
                const uint32_t value = static_cast<uint32_t>(bits) << 16;
                std::memcpy(output + j, &value, sizeof(value));
            }
        }
    }

    void
    ReadRows(uint64_t first, uint64_t rows, float* output) const {
        if (first > count_ || rows > count_ - first || (rows != 0 && output == nullptr) ||
            rows > std::numeric_limits<uint64_t>::max() / sizeof(float) /
                       static_cast<uint64_t>(dim_)) {
            throw VsagException(ErrorType::INVALID_ARGUMENT, "invalid kmeans row range");
        }
        const uint64_t bytes = rows * static_cast<uint64_t>(dim_) * sizeof(float);
        if (bytes != 0 && bytes - 1 > std::numeric_limits<uintptr_t>::max() -
                                          reinterpret_cast<uintptr_t>(output)) {
            throw VsagException(ErrorType::INVALID_ARGUMENT, "kmeans output span overflow");
        }
        for (uint64_t i = 0; i < rows; ++i) {
            Read(first + i, output + i * static_cast<uint64_t>(dim_));
        }
    }

private:
    const uint8_t* data_;
    const uint64_t count_;
    const int32_t dim_;
    const DataTypes type_;
    const uint64_t stride_;
    RowReader reader_{nullptr};
    const void* context_{nullptr};
    ColumnReader columns_{nullptr};
};

}  // namespace vsag
