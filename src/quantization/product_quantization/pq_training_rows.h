#pragma once
#include "simd/normalize.h"
#include "typing.h"
#include "utils/training_rows.h"
namespace vsag::detail {
// Synchronous borrowed source. Only a full-D row, S norms and the caller's
// single packed subspace exist; no full normalized sample matrix.
class PQTrainingRows {
public:
    PQTrainingRows(const TrainingRows& rows, bool normalize, Allocator* allocator)
        : rows_(rows),
          count_(std::min<uint64_t>(rows.Count(), 65536UL)),
          normalize_(normalize),
          allocator_(allocator),
          norms_(allocator) {
        if (normalize_) {
            Vector<float> row(rows.Dim(), allocator);
            norms_.resize(count_);
            for (uint64_t j = 0; j < count_; ++j) {
                rows_.Read(j, row.data());
                norms_[j] = Normalize(row.data(), row.data(), rows_.Dim());
            }
        }
    }
    uint64_t
    Count() const {
        return count_;
    }
    void
    Pack(uint64_t begin, uint64_t width, float* slice) const {
        if (begin > static_cast<uint64_t>(rows_.Dim()) ||
            width > static_cast<uint64_t>(rows_.Dim()) - begin || (width && !slice) ||
            (width && count_ > static_cast<uint64_t>(PTRDIFF_MAX) / sizeof(float) / width))
            throw std::invalid_argument("invalid PQ packing range");
        if (width == 0)
            return;
        Vector<float> row(allocator_);
        if (!rows_.HasColumnReader())
            row.resize(rows_.Dim());
        for (uint64_t j = 0; j < count_; ++j) {
            float* out = slice + j * width;
            if (rows_.HasColumnReader())
                rows_.ReadColumns(j, begin, width, out);
            else {
                rows_.Read(j, row.data());
                std::memcpy(out, row.data() + begin, width * sizeof(float));
            }
            if (normalize_)
                DivScalar(out, out, width, norms_[j]);
        }
    }

private:
    const TrainingRows& rows_;
    const uint64_t count_;
    const bool normalize_;
    Allocator* allocator_;  // Borrowed for the synchronous training lifetime.
    Vector<float> norms_;
};
}  // namespace vsag::detail
