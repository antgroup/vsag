import h5py
import numpy as np
import struct
import os

def write_fvecs(file_path, vectors):
    """
    将向量数据写入 .fvecs 文件。

    :param file_path: 输出的 .fvecs 文件路径
    :param vectors: 向量数据 (NumPy 数组，形状为 [num_vectors, dim])
    """
    num_vectors, dim = vectors.shape

    # 确保向量数据是 float32 类型
    vectors = vectors.astype(np.float32)

    with open(file_path, 'wb') as f:
        for i in range(num_vectors):
            # 写入向量维度 (4 字节整数，小端格式)
            f.write(struct.pack('<i', dim))
            # 写入向量数据 (d * 4 字节，小端格式)
            f.write(vectors[i].tobytes())

def read_fvecs(file_path):
    """
    读取 .fvecs 文件并返回向量数据。

    :param file_path: .fvecs 文件路径
    :return: 向量数据 (NumPy 数组，形状为 [num_vectors, dim])
    """
    with open(file_path, 'rb') as f:
        vectors = []
        while True:
            # 读取向量维度 (4 字节整数，小端格式)
            dim_data = f.read(4)
            if not dim_data:
                break  # 文件结束
            dim = struct.unpack('<i', dim_data)[0]

            # 读取向量数据 (d * 4 字节，小端格式)
            vector_data = f.read(dim * 4)
            vector = np.frombuffer(vector_data, dtype=np.float32)
            if dim == 512:
                norm = np.linalg.norm(vector)
                if np.abs(norm - 1.0) > 1e-3:
                    raise ValueError(f"norm is {norm}")

            vectors.append(vector)

        return np.array(vectors)

def validate_fvecs(file_path, expected_num_vectors, expected_dim):
    """
    校验 .fvecs 文件的大小和内容是否正确。

    :param file_path: .fvecs 文件路径
    :param expected_num_vectors: 预期的向量数量
    :param expected_dim: 预期的向量维度
    :return: 是否校验通过
    """
    # 计算预期文件大小
    expected_file_size = expected_num_vectors * (4 + expected_dim * 4)

    # 检查文件大小
    actual_file_size = os.path.getsize(file_path)
    if actual_file_size != expected_file_size:
        print(f"文件大小不匹配！预期: {expected_file_size} 字节, 实际: {actual_file_size} 字节")
        return False

    # 检查文件内容
    vectors = read_fvecs(file_path)
    actual_num_vectors, actual_dim = vectors.shape
    if actual_num_vectors != expected_num_vectors or actual_dim != expected_dim:
        print(f"向量数量或维度不匹配！预期: ({expected_num_vectors}, {expected_dim}), 实际: ({actual_num_vectors}, {actual_dim})")
        return False

    print("校验通过！文件大小和内容均正确。")
    return True

# 主程序
if __name__ == "__main__":
    file_path = '/tbase-project/ann-benchmarks/data/security-512-5m-ip.hdf5'
    with h5py.File(file_path, 'r') as f:
        print("Keys in the HDF5 file:", list(f.keys()))

        for dataset_name in list(f.keys()):
            vectors = f[dataset_name][()]
            if dataset_name == "train":
                dataset_name = "learn"
            elif dataset_name == "test":
                dataset_name = "query"
            else:
                continue

            #
            # # 处理不同数据集类型
            # if dataset_name == "neighbors":
            #     vectors = vectors.astype(np.int32)
            #     print(f"Dataset '{dataset_name}' is of type int32. Skipping conversion to .fvecs.")
            #     continue  # 跳过非浮点数数据集
            # else:
            #     vectors = vectors.astype(np.float32)
            #
            # # 写入 .fvecs 文件
            fvecs_file_path = f"/tbase-project/ann-benchmarks/data/security-512-euclidean/{dataset_name}.fvecs"
            # write_fvecs(fvecs_file_path, vectors)
            #
            # # 校验 .fvecs 文件
            expected_num_vectors, expected_dim = vectors.shape
            # print(f"Dataset '{dataset_name}' shape: {vectors.shape}")
            validate_fvecs(fvecs_file_path, expected_num_vectors, expected_dim)
