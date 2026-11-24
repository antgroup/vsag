
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

#include "pca_transformer.h"

#include "impl/allocator/safe_allocator.h"
#include "storage/serialization_template_test.h"
#include "unittest.h"
#include "vsag_exception.h"
using namespace vsag;

void
TestCentralize(PCATransformer& pca, uint64_t dim) {
    uint32_t count = 1000;
    std::vector<float> mean(dim, 0);
    std::vector<float> vec = fixtures::generate_vectors(count, dim);
    std::vector<float> centralized_single_vec(dim, 0);

    for (uint64_t i = count / 2; i < count; i++) {
        for (uint64_t d = 0; d < dim; d++) {
            vec[i * dim + d] = vec[(i - count / 2) * dim + d] * -1 + d;
        }
    }

    pca.ComputeColumnMean(vec.data(), count);
    pca.CopyMean(mean.data());

    for (uint64_t d = 0; d < dim; d++) {
        float expected_mean = d / 2.0;
        REQUIRE(std::abs(mean[d] - expected_mean) < 1e-2);
    }

    for (uint64_t i = 0; i < count; ++i) {
        auto single_vec = vec.data() + i * dim;
        pca.CentralizeData(single_vec, centralized_single_vec.data());
        for (uint64_t d = 0; d < dim; d++) {
            float expected_mean = d / 2.0;
            REQUIRE(std::abs(single_vec[d] - expected_mean - centralized_single_vec[d]) < 1e-2);
        }
    }
}

void
TestPerformEigenDecomposition() {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    const uint64_t original_dim = 3;
    const uint64_t target_dim = 2;
    std::vector<float> pca_matrix(target_dim * original_dim, 0);

    // eigen_value = 3, 2, 1
    std::vector<float> covariance_matrix = {
        3.0f,
        0.0f,
        0.0f,  // eigen_vec[2] = [1, 0, 0]
        0.0f,
        2.0f,
        0.0f,  // eigen_vec[1] = [0, 1, 0]
        0.0f,
        0.0f,
        1.0f  // eigen_vec[0] = [0, 0, 1]
    };

    vsag::PCATransformer pca(allocator.get(), original_dim, target_dim);

    pca.PerformEigenDecomposition(covariance_matrix.data());

    pca.CopyPCAMatrix(pca_matrix.data());

    std::vector<float> expected_pca_matrix = {1.0f,
                                              0.0f,
                                              0.0f,  // eigen_vec[2]
                                              0.0f,
                                              1.0f,
                                              0.0f};  // eigen_vec[1]

    for (uint64_t i = 0; i < original_dim * target_dim; ++i) {
        REQUIRE(std::abs(pca_matrix[i] - expected_pca_matrix[i]) < 1e-5);
    }
}

void
TestPerformEigenDecompositionWithNonDiagonalCovariance() {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr uint64_t dim = 3;
    constexpr float inverse_sqrt_two = 0.7071067811865475F;
    constexpr float inverse_sqrt_three = 0.5773502691896258F;
    constexpr float inverse_sqrt_six = 0.4082482904638630F;
    const std::vector<float> eigenvectors = {
        inverse_sqrt_two,
        -inverse_sqrt_six,
        inverse_sqrt_three,
        inverse_sqrt_two,
        inverse_sqrt_six,
        -inverse_sqrt_three,
        0.0F,
        2.0F * inverse_sqrt_six,
        inverse_sqrt_three,
    };
    const std::vector<float> eigenvalues = {9.0F, 4.0F, 1.0F};
    std::vector<float> covariance(dim * dim, 0.0F);
    for (uint64_t row = 0; row < dim; ++row) {
        for (uint64_t column = 0; column < dim; ++column) {
            for (uint64_t component = 0; component < dim; ++component) {
                covariance[row * dim + column] += eigenvectors[row * dim + component] *
                                                  eigenvalues[component] *
                                                  eigenvectors[column * dim + component];
            }
        }
    }

    PCATransformer pca(allocator.get(), dim, dim);
    REQUIRE(pca.PerformEigenDecomposition(covariance.data()));
    std::vector<float> projection(dim * dim, 0.0F);
    pca.CopyPCAMatrix(projection.data());

    for (uint64_t left = 0; left < dim; ++left) {
        for (uint64_t right = 0; right < dim; ++right) {
            float projected_covariance = 0.0F;
            for (uint64_t row = 0; row < dim; ++row) {
                for (uint64_t column = 0; column < dim; ++column) {
                    projected_covariance += projection[left * dim + row] *
                                            covariance[row * dim + column] *
                                            projection[right * dim + column];
                }
            }
            const float expected = left == right ? eigenvalues[left] : 0.0F;
            REQUIRE(std::abs(projected_covariance - expected) < 1e-4F);
        }
    }
}

void
TestComputeCovarianceMatrix() {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    uint64_t count = 2;
    uint64_t original_dim = 2;
    std::vector<float> centralized_data = {1.0f, -1.0f, -1.0f, 1.0f};

    std::vector<float> covariance_matrix(original_dim * original_dim, 0.0f);

    vsag::PCATransformer pca(allocator.get(), original_dim, 1);

    pca.ComputeCovarianceMatrix(centralized_data.data(), count, covariance_matrix.data());

    // equal to centralized_data * 2
    std::vector<float> expected_covariance_matrix = {2.0f, -2.0f, -2.0f, 2.0f};

    for (uint64_t i = 0; i < original_dim * original_dim; ++i) {
        REQUIRE(std::abs(covariance_matrix[i] - expected_covariance_matrix[i]) < 1e-6);
    }
}

void
TestTransform() {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    uint64_t original_dim = 3;
    uint64_t target_dim = 2;
    std::vector<float> mean = {3.0f, 4.0f, 5.0f};
    std::vector<float> pca_matrix = {1.0f,
                                     0.0f,
                                     0.0f,  // eigen_vec[-1]
                                     0.0f,
                                     0.0f,
                                     1.0f};  // eigen_vec[-2]
    PCATransformer pca(allocator.get(), original_dim, target_dim);
    pca.SetMeanForTest(mean.data());
    pca.SetPCAMatrixForTest(pca_matrix.data());

    std::vector<float> input = {4.0f, 6.0f, 8.0f};  // centralized: [1, 2, 3]
    std::vector<float> output(target_dim, 0);
    std::vector<float> expected = {1.0f,   // eigen_vec[-1] * centralized
                                   3.0f};  // eigen_vec[-2] * centralized

    pca.Transform(input.data(), output.data());
    for (uint64_t i = 0; i < target_dim; i++) {
        REQUIRE(std::abs(output[i] - expected[i]) < 1e-5);
    }
}

void
TestTrain() {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    const uint64_t original_dim = 2;
    const uint64_t target_dim = 2;
    const uint64_t sample_count = 4;

    std::vector<float> data = {
        3.0f,
        0.0f,  // first dim has large var
        1.0f,
        0.0f,
        -1.0f,
        0.0f,
        -3.0f,
        0.0f  // second dim has small var
    };

    PCATransformer pca(allocator.get(), original_dim, target_dim);
    pca.Train(data.data(), sample_count);

    std::vector<float> pca_matrix(target_dim * original_dim);
    pca.CopyPCAMatrix(pca_matrix.data());
    std::vector<float> expected = {1.0f, 0.0f};

    for (uint64_t i = 0; i < target_dim; i++) {
        REQUIRE(std::abs(pca_matrix[i] - expected[i]) < 1e-5);
    }
}

void
TestTrainMinSampleCount() {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    const uint64_t original_dim = 2;
    const uint64_t target_dim = 2;

    PCATransformer pca(allocator.get(), original_dim, target_dim);

    std::vector<float> single_sample = {1.0f, 2.0f};
    REQUIRE_THROWS_AS(pca.Train(single_sample.data(), 1), VsagException);
    REQUIRE_THROWS_AS(pca.Train(single_sample.data(), 0), VsagException);
    const float* null_data = nullptr;
    REQUIRE_THROWS_AS(pca.Train(null_data, 2), VsagException);
}

TEST_CASE("PCA Basic Test", "[ut][PCA]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    const auto dims = fixtures::get_common_used_dims();

    TestPerformEigenDecomposition();
    TestPerformEigenDecompositionWithNonDiagonalCovariance();
    TestComputeCovarianceMatrix();
    TestTransform();
    TestTrain();
    TestTrainMinSampleCount();

    for (auto dim : dims) {
        PCATransformer pca(allocator.get(), dim, dim);
        TestCentralize(pca, dim);
    }
}

TEST_CASE("PCA Serialize / Deserialize Test", "[ut][PCA]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    const auto dims = fixtures::get_common_used_dims();
    uint32_t count = 1000;

    for (auto dim : dims) {
        // prepare pca1 and pca2
        uint64_t target_dim = (dim + 1) / 2;
        PCATransformer pca1(allocator.get(), dim, target_dim);
        PCATransformer pca2(allocator.get(), dim, target_dim);
        std::vector<float> vec = fixtures::generate_vectors(count, dim);
        pca1.Train(vec.data(), count);

        // copy pca1 -> pca2
        test_serializion(pca1, pca2);
        // validate pca1 == pca2
        std::vector<float> mean1(dim, 0);
        std::vector<float> mean2(dim, 0);
        std::vector<float> pca_matrix1(target_dim * dim, 0);
        std::vector<float> pca_matrix2(target_dim * dim, 0);
        pca1.CopyPCAMatrix(pca_matrix1.data());
        pca1.CopyMean(mean1.data());

        pca2.CopyPCAMatrix(pca_matrix2.data());
        pca2.CopyMean(mean2.data());

        for (auto i = 0; i < pca_matrix1.size(); i++) {
            REQUIRE(std::abs(pca_matrix1[i] - pca_matrix2[i]) < 1e-5);
        }

        for (auto i = 0; i < mean1.size(); i++) {
            REQUIRE(std::abs(mean1[i] - mean2[i]) < 1e-5);
        }
    }
}
