import numpy as np
import faiss
import scann
import time

is_10m = False
data_dir = "/tbase-project/vsag/data/internet_search/"
base_path = data_dir + "internet_search_train.fbin"
query_path = data_dir + "internet_search_test.fbin"
gt_path = data_dir + "internet_search_neighbors.fbin"

if not is_10m:
    base_path = data_dir + "msmar_base.fbin"
    query_path = data_dir + "msmar_query.fbin"
    gt_path = data_dir + "msmarc_groundtruth.ibin"

topk = 10

query_args = [[1, 30], [2, 30], [4, 30], [8, 30], [30, 120], [35, 100], [40, 80],
              [45, 80], [50, 80], [55, 95], [60, 110], [65, 110], [75, 110],
             [90, 110], [110, 120], [130, 150], [150, 200], [170, 200], [200, 300],
             [220, 500], [250, 500], [310, 300], [400, 300], [500, 500], [800, 1000]]
class Scann():
    def __init__(self, n_leaves=2000, avq_threshold=0.2, dims_per_block=2, dist="squared_l2"):
        self.name = "scann n_leaves={} avq_threshold={:.02f} dims_per_block={}".format(
            n_leaves, avq_threshold, dims_per_block
        )
        self.n_leaves = n_leaves
        self.avq_threshold = avq_threshold
        self.dims_per_block = dims_per_block
        self.dist = dist

    def fit(self, X):
        if self.dist == "dot_product":
            spherical = True
            X[np.linalg.norm(X, axis=1) == 0] = 1.0 / np.sqrt(X.shape[1])
            X /= np.linalg.norm(X, axis=1)[:, np.newaxis]
        else:
            spherical = False

        self.searcher = (
            scann.scann_ops_pybind.builder(X, 10, self.dist)
            .tree(self.n_leaves, 1, training_sample_size=len(X), spherical=spherical, quantize_centroids=True)
            .score_ah(self.dims_per_block, anisotropic_quantization_threshold=self.avq_threshold)
            .reorder(1)
            .build()
        )

    def set_query_arguments(self, leaves_reorder):
        self.leaves_to_search, self.reorder = leaves_reorder

    def query(self, v, n):
        return self.searcher.search(v, n, self.reorder, self.leaves_to_search)[0]

def load_memmap_file(filename, data_type=np.float32, n_rows=None):
    data = np.memmap(filename, dtype=np.uint8, mode='r')
    header = data[:8].view(dtype=np.int32)
    count, dim = header[0], header[1]

    if n_rows is not None:
        count = min(count, n_rows)

    data_size = np.dtype(data_type).itemsize
    byte_offset = 8 + count * dim * data_size

    rest_data = data[8:byte_offset].view(dtype=data_type)
    array = rest_data.reshape(count, dim)

    return dim, count, array

def load_memmap_file_original(filename):
    data = np.memmap(filename, dtype=np.uint8, mode='r')
    header = data[:8].view(dtype=np.int32)
    count, dim = header[0], header[1]
    rest_data = data[8:].view(dtype=np.float32)
    array = rest_data.reshape(count, dim)
    return dim, count, array

def evaluate_index(index, query, gt, topk):
    query_count = len(query)
    total_recall = 0.0
    start_time = time.time()

    while True:
        for i in range(query_count):
            result = index.query(query[i], topk)

        gt_set = set(gt[i])

        hit_count = sum(1 for id in result if id in gt_set)
        recall = hit_count / topk
        total_recall += recall

    end_time = time.time()
    elapsed_time = end_time - start_time

    qps = query_count / elapsed_time
    avg_recall = total_recall / query_count

    return qps, avg_recall

if __name__ == "__main__":
    query_dim, query_count, query = load_memmap_file(query_path)
    print(query_dim, query_count, query.shape)

    base_dim, base_count, base = load_memmap_file(base_path, n_rows=10000)
    # base_dim, base_count, base = load_memmap_file_original(base_path)
    print(base_dim, base_count, base.shape)

    gt_dim, gt_count, gt = load_memmap_file(gt_path, np.int32)
    print(gt_dim, gt_count, gt.shape)

    index = Scann()
    index.fit(base)

    for query_param in query_args:
        index.set_query_arguments(query_param)
        qps, recall = evaluate_index(index, query, gt, topk=10)
        print(f"QPS: {qps:.2f}, Recall: {recall:.4f}")