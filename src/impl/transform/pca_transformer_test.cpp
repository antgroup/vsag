
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

#include <catch2/catch_test_macros.hpp>

#include "fixtures.h"
#include "impl/allocator/safe_allocator.h"

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
    pca.CopyMeanForTest(mean.data());

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

    pca.CopyPCAMatrixForTest(pca_matrix.data());

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
    pca.SetPCAMatrixForText(pca_matrix.data());

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
    pca.CopyPCAMatrixForTest(pca_matrix.data());
    std::vector<float> expected = {1.0f, 0.0f};

    for (uint64_t i = 0; i < target_dim; i++) {
        REQUIRE(std::abs(pca_matrix[i] - expected[i]) < 1e-5);
    }
}

TEST_CASE("PCA Basic Test", "[ut][PCA]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    const auto dims = fixtures::get_common_used_dims();

    TestPerformEigenDecomposition();
    TestComputeCovarianceMatrix();
    TestTransform();
    TestTrain();

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
        fixtures::TempDir dir("pca");
        auto filename = dir.GenerateRandomFile();
        std::ofstream outfile(filename.c_str(), std::ios::binary);
        IOStreamWriter writer(outfile);
        pca1.Serialize(writer);
        outfile.close();

        std::ifstream infile(filename.c_str(), std::ios::binary);
        IOStreamReader reader(infile);
        pca2.Deserialize(reader);
        infile.close();

        // validate pca1 == pca2
        std::vector<float> mean1(dim, 0);
        std::vector<float> mean2(dim, 0);
        std::vector<float> pca_matrix1(target_dim * dim, 0);
        std::vector<float> pca_matrix2(target_dim * dim, 0);
        pca1.CopyPCAMatrixForTest(pca_matrix1.data());
        pca1.CopyMeanForTest(mean1.data());

        pca2.CopyPCAMatrixForTest(pca_matrix2.data());
        pca2.CopyMeanForTest(mean2.data());

        for (auto i = 0; i < pca_matrix1.size(); i++) {
            REQUIRE(std::abs(pca_matrix1[i] - pca_matrix2[i]) < 1e-5);
        }

        for (auto i = 0; i < mean1.size(); i++) {
            REQUIRE(std::abs(mean1[i] - mean2[i]) < 1e-5);
        }
    }
}
