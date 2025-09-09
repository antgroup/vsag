# BruteForce Index

## Definition
BruteForce Index is a **linear search** index structure that performs exact nearest neighbor search by computing distances between the query vector and all vectors in the database. It guarantees 100% recall rate without any approximation or optimization techniques, providing the most accurate search results at the cost of computational complexity.

## Working Principle
1. **Index Construction Phase**:
   Simply stores all vector data in memory without any preprocessing, compression, or indexing structure. Each vector maintains its original precision and dimensionality.

2. **Search Phase**:
   When a query vector is received, the system calculates the distance between the query and every single vector in the database using the specified distance metric (L2, IP, or cosine). All distances are computed and then sorted to return the top-K most similar vectors.

3. **Computational Process**:
   The algorithm has O(n) complexity where n is the number of vectors. Despite its linear complexity, it can achieve reasonable performance for small to medium datasets through optimized implementations using SIMD instructions and multi-threading.

## Suitable Scenarios
1. **Small Datasets**: When the dataset contains less than 1 million vectors where linear scan remains computationally feasible.
2. **Maximum Accuracy**: When 100% accuracy is mandatory and no approximation error can be tolerated.
3. **Baseline Evaluation**: As a golden standard for evaluating the accuracy and performance of other approximate search algorithms.
4. **Development and Testing**: During development phases where simplicity and accuracy are prioritized over performance optimization.


## Usage
For examples, refer to [105_index_brute_force.cpp](https://github.com/antgroup/vsag/blob/main/examples/cpp/105_index_brute_force.cpp).

## Factory Parameter Overview Table

| **Category** | **Parameter** | **Type** | **Default Value** | **Required** | **Description** |
|--------------|---------------|----------|-------------|--------------|-----------------|
| **Basic** | dtype | string | "float32" | Yes | Data type (only float32 supported) |
| **Basic** | metric_type | string | "l2" | Yes | Distance metric: l2, ip, cosine |
| **Basic** | dim | int | - | Yes | Vector dimension [1, 4096] |

## Detailed Explanation of Building Parameters

### dtype
- **Parameter Type**: string
- **Parameter Description**: Data type for vector elements
- **Optional Values**: "float32" (currently only supports float32)
- **Default Value**: "float32"

### metric_type
- **Parameter Type**: string
- **Parameter Description**: Distance metric type for similarity calculation
- **Optional Values**: "l2", "ip", "cosine"
- **Default Value**: "l2"

### dim
- **Parameter Type**: int
- **Parameter Description**: Vector dimension
- **Optional Values**: 1 to 4096
- **Default Value**: Must be provided (no default value)
