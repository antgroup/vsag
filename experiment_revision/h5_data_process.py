import h5py
import numpy as np
import struct
import os

def write_fbin(file_path, vectors):
    """
    将向量数据写入 .fbin 文件。

    :param file_path: 输出的 .fbin 文件路径
    :param vectors: 向量数据 (NumPy 数组，形状为 [num_vectors, dim])
    """
    num_vectors, dim = vectors.shape

    # 打开文件并写入数据
    with open(file_path, 'wb') as f:
        # 写入向量数量 (4 字节整数)
        f.write(struct.pack('i', num_vectors))
        # 写入向量维度 (4 字节整数)
        f.write(struct.pack('i', dim))
        # 写入所有向量数据 (每个 float 占用 4 字节)
        vectors.tofile(f)

def read_fbin(file_path):
    """
    读取 .fbin 文件并返回向量数据。

    :param file_path: .fbin 文件路径
    :return: 向量数据 (NumPy 数组，形状为 [num_vectors, dim])
    """
    with open(file_path, 'rb') as f:
        # 读取向量数量
        num_vectors = struct.unpack('i', f.read(4))[0]
        # 读取向量维度
        dim = struct.unpack('i', f.read(4))[0]
        # 读取所有向量数据
        vectors = np.fromfile(f, dtype=np.float32).reshape(num_vectors, dim)
    return vectors

def validate_fbin(file_path, expected_num_vectors, expected_dim):
    """
    校验 .fbin 文件的大小和内容是否正确。

    :param file_path: .fbin 文件路径
    :param expected_num_vectors: 预期的向量数量
    :param expected_dim: 预期的向量维度
    :return: 是否校验通过
    """
    # 计算预期文件大小
    expected_file_size = 4 * 2 + expected_num_vectors * expected_dim * 4

    # 检查文件大小
    actual_file_size = os.path.getsize(file_path)
    if actual_file_size != expected_file_size:
        print(f"文件大小不匹配！预期: {expected_file_size} 字节, 实际: {actual_file_size} 字节")
        return False

    # 检查文件内容
    vectors = read_fbin(file_path)
    actual_num_vectors, actual_dim = vectors.shape
    if actual_num_vectors != expected_num_vectors or actual_dim != expected_dim:
        print(f"向量数量或维度不匹配！预期: ({expected_num_vectors}, {expected_dim}), 实际: ({actual_num_vectors}, {actual_dim})")
        return False

    print("校验通过！文件大小和内容均正确。")
    return True



file_path = '/root/internetsearch-qa-768-10m-euclidean.hdf5'
with h5py.File(file_path, 'r') as f:
    print("Keys in the HDF5 file:", list(f.keys()))

    for dataset_name in list(f.keys()):
        vectors = f[dataset_name][()]

        if dataset_name == "neighbors":
            vectors = vectors.astype(np.int32)
        else:
            vectors = vectors.astype(np.float32)

        fbin_file_path = f"/root/internet_search_{dataset_name}.fbin"
        write_fbin(fbin_file_path, vectors)

        expected_num_vectors, expected_dim = vectors.shape
        print(vectors.shape)
        validate_fbin(fbin_file_path, expected_num_vectors, expected_dim)


