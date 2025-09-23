import h5py
import faiss
import time
import numpy as np

def load_data(gt_k=100):
    with h5py.File('/home/tianlan.lht/data/gist-960-euclidean.hdf5', 'r') as f:
        x = np.array(f['train']).astype('float32')
        q = np.array(f['test']).astype('float32')
        gt = np.array(f['neighbors'][:, :gt_k])  # 直接读取真值
    return x, q, gt

def build_ivf_index(x):
    d = x.shape[1]
    print(d)
    index = faiss.index_factory(d, "IVF5000(HNSW32),PQ320x4fs,RFlat", faiss.METRIC_L2)
    index.train(x)
    index.add(x)
    return index

def evaluate_performance(index, query, gt, k=10):    
    # 预热缓存（确保计时准确）
    _ = index.search(query[:10], k)
    faiss.omp_set_num_threads(1)
    # 精确计时
    start = time.perf_counter()
    while True:
        _, I = index.search(query, k)
    search_time = time.perf_counter() - start
    
    # 计算召回率
    recall = np.mean([len(np.intersect1d(gt[i][:k], I[i]))/k for i in range(len(query))])
    
    # 获取实际距离计算次数（通过Faiss统计）
    stats = faiss.cvar.indexIVF_stats
    compute_count = stats.ndis
    
    return search_time, recall, compute_count

if __name__ == "__main__":
    # 加载数据（包含真值）
    x, q, gt = load_data()
    nprobe = 210
    # 构建索引
    # index = build_ivf_index(x)
    # faiss.write_index(index, "./faiss_IVF5000(HNSW32),PQ320x4fs,RFlat.index")
    # exit()
    index = faiss.read_index("./faiss_IVF5000(HNSW32),PQ320x4fs,RFlat.index")
    index_ivf = faiss.downcast_index(faiss.downcast_index(index).base_index)
    index.k_factor = 10
    quantizer = index_ivf.quantizer
    faiss.downcast_index(quantizer).hnsw.ef_search = nprobe * 1.2
    index_ivf.nprobe = nprobe

    
    # 性能评估
    total_time, recall, compute_count = evaluate_performance(
        index, q, gt, k=10
    )
    
    # 输出结果
    print(f"搜索耗时：{total_time:.2f}s")
    print(f"QPS：{len(q)/total_time:.2f} queries/second")
    print(f"召回率@{10}：{recall:.4f}")
    print(f"实际距离计算次数：{compute_count:,}")
