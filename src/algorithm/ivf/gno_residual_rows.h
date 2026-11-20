#pragma once
#include "impl/blas/blas_function.h"
#include "simd/normalize.h"
#include "utils/training_rows.h"
#include "utils/util_functions.h"
namespace vsag {
// Borrowed original input; owns O(N + K*D) latest-stage metadata, never N*D residuals.
// Call Replace only after all users of the previous immutable generation have joined.
class GNOResidualRows {
public:
    GNOResidualRows(const TrainingRows& original, bool normalize, Allocator* allocator)
        : original_(original),
          normalize_(normalize),
          allocator_(allocator),
          centroids_(allocator),
          labels_(allocator) {
    }
    static void
    Read(const void* context, uint64_t row, float* output) {
        const auto& self = *static_cast<const GNOResidualRows*>(context);
        if (row >= self.original_.Count() || output == nullptr)
            throw VsagException(ErrorType::INVALID_ARGUMENT, "invalid GNO residual row read");
        self.original_.Read(row, output);
        if (self.normalize_)
            Normalize(output, output, self.original_.Dim());
        if (!self.labels_.empty()) {
            const auto label = self.labels_[row];
            BlasFunction::Saxpy(
                self.original_.Dim(),
                -1.0,
                self.centroids_.data() + static_cast<uint64_t>(label) * self.original_.Dim(),
                1,
                output,
                1);
        }
    }
    void
    Replace(const float* centroids, uint64_t count, Vector<BucketIdType> labels) {
        if (!centroids || count == 0 || labels.size() != original_.Count() ||
            count > static_cast<uint64_t>(PTRDIFF_MAX) / sizeof(float) / original_.Dim())
            throw VsagException(ErrorType::INVALID_ARGUMENT, "invalid GNO residual generation");
        for (const auto label : labels)
            if (label >= count)
                throw VsagException(ErrorType::INVALID_ARGUMENT, "invalid GNO BF centroid ID");
        Vector<float> next(centroids, centroids + count * original_.Dim(), allocator_);
        centroids_.swap(next);
        labels_.swap(labels);
    }

private:
    TrainingRows original_;
    bool normalize_;
    Allocator* allocator_;
    Vector<float> centroids_;
    Vector<BucketIdType> labels_;
};
}  // namespace vsag
