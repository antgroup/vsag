import faiss
import numpy as np

import time

is_10m = True
data_dir = "/tbase-project/vsag/data/internet_search/"
base_path = data_dir + "internet_search_train.fbin"
query_path = data_dir + "internet_search_test.fbin"
gt_path = data_dir + "internet_search_neighbors.fbin"

if not is_10m:
    base_path = data_dir + "msmar_base.fbin"
    query_path = data_dir + "msmar_query.fbin"
    gt_path = data_dir + "msmarc_groundtruth.ibin"

topk = 10

n_probes = [1, 5, 10, 50, 100, 200]
k_rerank = [0, 10, 100, 1000]

class Faiss():
    def query(self, v, n):
        if self._metric == "angular":
            v /= np.linalg.norm(v)
        D, I = self.index.search(np.expand_dims(v, axis=0).astype(np.float32), n)
        return I[0]

    def get_batch_results(self):
        D, L = self.res
        res = []
        for i in range(len(D)):
            r = []
            for l, d in zip(L[i], D[i]):
                if l != -1:
                    r.append(l)
            res.append(r)
        return res

class FaissIVFPQfs(Faiss):
    def __init__(self, metric, n_list):
        self._n_list = n_list
        self._metric = metric

    def fit(self, X):
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        if self._metric == "angular":
            faiss.normalize_L2(X)

        d = X.shape[1]
        faiss_metric = faiss.METRIC_INNER_PRODUCT if self._metric == "angular" else faiss.METRIC_L2
        factory_string = f"IVF{self._n_list},PQ{d//2}x4fs"
        faiss.omp_set_num_threads(128)
        index = faiss.index_factory(d, factory_string, faiss_metric)
        index.train(X)
        index.add(X)
        index_refine = faiss.IndexRefineFlat(index, faiss.swig_ptr(X))
        self.base_index = index
        self.refine_index = index_refine
        faiss.omp_set_num_threads(1)

    def set_query_arguments(self, n_probe, k_reorder):
        faiss.cvar.indexIVF_stats.reset()
        self._n_probe = n_probe
        self._k_reorder = k_reorder
        self.base_index.nprobe = self._n_probe
        self.refine_index.k_factor = self._k_reorder
        if self._k_reorder == 0:
            self.index = self.base_index
        else:
            self.index = self.refine_index
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

    # base_dim, base_count, base = load_memmap_file(base_path, n_rows=200000)
    base_dim, base_count, base = load_memmap_file_original(base_path)
    print(base_dim, base_count, base.shape)

    gt_dim, gt_count, gt = load_memmap_file(gt_path, np.int32)
    print(gt_dim, gt_count, gt.shape)

    index = FaissIVFPQfs("euclidean", 4096)
    index.fit(base)

    for n_pro in n_probes:
        for k in k_rerank:
            index.set_query_arguments(n_pro, k)
            qps, recall = evaluate_index(index, query, gt, topk=10)
            print(f"QPS: {qps:.2f}, Recall: {recall:.4f}")