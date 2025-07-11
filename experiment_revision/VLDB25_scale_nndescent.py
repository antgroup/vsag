import numpy as np
import nndescent
import scipy.sparse
import time

data_dir = "/tbase-project/vsag/data/internet_search/"
base_path = data_dir + "internet_search_train.fbin"
query_path = data_dir + "internet_search_test.fbin"
gt_path = data_dir + "internet_search_neighbors.fbin"

topk = 10

query_args = [0.0, 0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36]

class NNDescent():
    def __init__(self, metric="euclidean", index_param_dict = []):
        if "n_neighbors" in index_param_dict:
            self.n_neighbors = int(index_param_dict["n_neighbors"])
        else:
            self.n_neighbors = 40

        if "pruning_degree_multiplier" in index_param_dict:
            self.pruning_degree_multiplier = float(
                index_param_dict["pruning_degree_multiplier"]
            )
        else:
            self.pruning_degree_multiplier = 2.0

        if "pruning_prob" in index_param_dict:
            self.pruning_prob = float(index_param_dict["pruning_prob"])
        else:
            self.pruning_prob = 0.0

        if "leaf_size" in index_param_dict:
            self.leaf_size = int(index_param_dict["leaf_size"])
        else:
            self.leaf_size = 36

        self.is_sparse = metric in ["jaccard"]

        self.nnd_metric = {
            "angular": "dot",
            "euclidean": "euclidean",
            "hamming": "hamming",
            "jaccard": "jaccard",
        }[metric]

    def fit(self, X):
        if self.is_sparse:
            # Convert to sparse matrix format
            if type(X) == list:
                sizes = [len(x) for x in X]
                n_cols = max([max(x) for x in X]) + 1
                matrix = scipy.sparse.csr_matrix(
                    (len(X), n_cols), dtype=np.float32
                )
                matrix.indices = np.hstack(X).astype(np.int32)
                matrix.indptr = np.concatenate([[0], np.cumsum(sizes)]).astype(
                    np.int32
                )
                matrix.data = np.ones(
                    matrix.indices.shape[0], dtype=np.float32
                )
                matrix.sort_indices()
                X = matrix
            else:
                X = scipy.sparse.csr_matrix(X)

            self.query_matrix = scipy.sparse.csr_matrix(
                (1, X.shape[1]), dtype=np.float32
            )
        elif not isinstance(X, np.ndarray) or X.dtype != np.float32:
            print("Convert data to float32")
            X = np.asarray(X, dtype=np.float32)

        # nndescent uses pointers to the data. Make shure X does not change
        # outside of this scope.
        self.X = X
        self.index = nndescent.NNDescent(
            self.X,
            n_threads=128,
            n_neighbors=self.n_neighbors,
            metric=self.nnd_metric,
            leaf_size=self.leaf_size,
            pruning_degree_multiplier=self.pruning_degree_multiplier,
            pruning_prob=self.pruning_prob,
            verbose=True,
        )
        # Make a dummy query to prepare the search graph.
        if self.is_sparse:
            empty_mtx = np.empty((0, X.shape[0]), dtype=np.float32)
            empty_csr = scipy.sparse.csr_matrix(empty_mtx)
            self.index.query(empty_csr, k=1, epsilon=0.1)
        else:
            empty_mtx = np.empty((0, X.shape[0]), dtype=np.float32)
            self.index.query(empty_mtx, k=1, epsilon=0.1)

    def set_query_arguments(self, epsilon=0.1):
        self.epsilon = float(epsilon)

    def query(self, v, n):
        if self.is_sparse:
            # Convert index array to sparse matrix format and query; the
            # overhead of direct conversion is high for single queries
            # (converting the entire test dataset and sending single rows is
            # better), so we just populate the required structures.
            if v.dtype == np.bool_:
                self.query_matrix.indices = np.flatnonzero(v).astype(np.int32)
            else:
                self.query_matrix.indices = v.astype(np.int32)
            size = self.query_matrix.indices.shape[0]
            self.query_matrix.indptr = np.array([0, size], dtype=np.int32)
            self.query_matrix.data = np.ones(size, dtype=np.float32)
            ind, dist = self.index.query(
                self.query_matrix, k=n, epsilon=self.epsilon
            )
        else:
            ind, dist = self.index.query(
                v.reshape(1, -1).astype("float32"), k=n, epsilon=self.epsilon
            )
        return ind[0]

    def __str__(self):
        return (
            f"NNDescent(n_neighbors={self.n_neighbors}, "
            f"pruning_mult={self.pruning_degree_multiplier:.2f}, "
            f"pruning_prob={self.pruning_prob:.3f}, "
            f"epsilon={self.epsilon:.3f}, "
            f"leaf_size={self.leaf_size:02d})"
        )

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

    for i in range(query_count):
        result = index.query(query[i], topk)

        gt_set = set(gt[i][:topk])

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

    index = NNDescent()
    index.fit(base)

    for query_param in query_args:
        index.set_query_arguments(query_param)
        qps, recall = evaluate_index(index, query, gt, topk=10)
        print(f"QPS: {qps:.2f}, Recall: {recall:.4f}")