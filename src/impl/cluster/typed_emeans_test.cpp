#include <cmath>
#include <cstring>

#include "impl/allocator/safe_allocator.h"
#include "impl/cluster/e_means_cluster.h"
#include "unittest.h"
TEST_CASE("Typed e-means preserves seeded FLOAT algorithm", "[TypedEMeans]") {
    using namespace vsag;
    const bool bf = GENERATE(false, true);
    const int dim = GENERATE(7, 33);
    const int batch = GENERATE(1, 17, 100);
    constexpr uint64_t n = 100;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    std::vector<uint16_t> half(n * dim);
    std::vector<float> rounded(n * dim);
    for (uint64_t i = 0; i < half.size(); ++i) {
        // Exactly representable finite values, independently expanded.
        half[i] = static_cast<uint16_t>((bf ? 0x3e00 : 0x3400) + i % 127);
        if (bf) {
            uint32_t bits = uint32_t(half[i]) << 16;
            std::memcpy(&rounded[i], &bits, 4);
        } else {
            const auto bits = half[i];
            rounded[i] = std::ldexp(1.0F + float(bits & 1023) / 1024, int((bits >> 10) & 31) - 15);
        }
    }
    TrainingRows rows(half.data(),
                      n,
                      dim,
                      bf ? DataTypes::DATA_TYPE_BF16 : DataTypes::DATA_TYPE_FP16,
                      dim * 2,
                      half.size() * 2);
    EMeansCluster direct(dim, allocator.get()), typed(dim, allocator.get());
    double a = 0, b = 0;
    const auto expected = direct.Run(4, rounded.data(), n, 3, &a, 2026, batch);
    const auto actual = typed.Run(4, rows, 3, &b, 2026, batch);
    REQUIRE(expected == actual);
    REQUIRE(a == b);
    REQUIRE(std::memcmp(direct.k_centroids_, typed.k_centroids_, 4 * dim * sizeof(float)) == 0);
    TrainingRows bad(
        n,
        dim,
        [](const void*, uint64_t, float*) { throw std::runtime_error("read failure"); },
        nullptr);
    std::vector<float> saved(typed.k_centroids_, typed.k_centroids_ + 4 * dim);
    REQUIRE_THROWS(typed.Run(4, bad, 3, nullptr, 2026, batch));
    REQUIRE(std::memcmp(saved.data(), typed.k_centroids_, saved.size() * sizeof(float)) == 0);
}
TEST_CASE("Typed e-means rejects invalid shape before model mutation", "[TypedEMeans]") {
    using namespace vsag;
    auto a = SafeAllocator::FactoryDefaultAllocator();
    EMeansCluster typed(7, a.get());
    TrainingRows wrong(2, 8, [](const void*, uint64_t, float*) {}, nullptr);
    REQUIRE_THROWS(typed.Run(1, wrong));
    REQUIRE(typed.k_centroids_ == nullptr);
}
