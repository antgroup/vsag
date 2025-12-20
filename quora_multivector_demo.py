"""
Quora数据集的多向量索引演示

使用流程：
1. 加载Quora数据集
2. 将问题编码为多向量
3. 构建多向量索引
4. 搜索和评估
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import time


# 简化版的多向量索引（不依赖pyvsag，先做原型）
class SimpleMultiVectorIndex:
    """
    简单的多向量索引实现
    用于快速原型和测试，不依赖复杂的图索引
    """

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.doc_vectors = {}  # doc_id -> np.ndarray [num_vecs, dim]
        self.doc_ids = []      # 文档ID列表

    def add_documents(self, doc_ids: List[str], doc_vectors_list: List[np.ndarray]):
        """批量添加文档"""
        for doc_id, vectors in zip(doc_ids, doc_vectors_list):
            # 归一化
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            vectors = vectors / norms

            self.doc_vectors[doc_id] = vectors.astype(np.float32)
            self.doc_ids.append(doc_id)

    def search(self, query_vectors: np.ndarray, k: int = 10) -> Tuple[List[str], List[float]]:
        """
        MaxSim搜索策略

        对于每个文档，计算：
        score = sum(max(cosine(q_i, d_j) for d_j in doc_vectors) for q_i in query_vectors)
        """
        # 归一化查询向量
        norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        query_vectors = query_vectors / norms

        doc_scores = {}

        # 遍历所有文档
        for doc_id in self.doc_ids:
            doc_vecs = self.doc_vectors[doc_id]

            # 计算query和doc的所有向量对的相似度
            # shape: [num_query_vecs, num_doc_vecs]
            similarities = np.dot(query_vectors, doc_vecs.T)

            # MaxSim: 对每个query向量，取最大的doc向量相似度
            max_sims = np.max(similarities, axis=1)  # shape: [num_query_vecs]

            # 对所有query向量求和
            total_score = np.sum(max_sims)

            doc_scores[doc_id] = float(total_score)

        # 排序返回top-k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        if not sorted_docs:
            return [], []

        doc_ids, scores = zip(*sorted_docs)
        return list(doc_ids), list(scores)


def encode_text_simple(text: str, dim: int = 128) -> np.ndarray:
    """
    简单的文本编码器（演示用）

    策略：将文本分词，每个词生成一个向量
    使用词的hash作为种子，生成确定性的"假"向量

    实际应用中应该替换为：
    - ColBERT模型
    - BERT + token embeddings
    - Sentence-BERT + 分句
    """
    # 简单分词
    tokens = text.lower().split()

    if not tokens:
        return np.random.randn(1, dim).astype(np.float32)

    vectors = []
    for token in tokens[:30]:  # 限制最多30个token
        # 使用token hash生成确定性向量
        seed = abs(hash(token)) % (2**31)
        rng = np.random.RandomState(seed)
        vec = rng.randn(dim)
        vectors.append(vec)

    return np.array(vectors, dtype=np.float32)


def load_quora_data(data_dir: str):
    """加载Quora数据集"""
    data_dir = Path(data_dir)

    print("Loading Quora dataset...")

    # 加载corpus
    corpus = {}
    with open(data_dir / "corpus.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line.strip())
            corpus[doc['_id']] = doc['text']

    # 加载queries
    queries = {}
    with open(data_dir / "queries.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            query = json.loads(line.strip())
            queries[query['_id']] = query['text']

    # 加载qrels
    qrels = defaultdict(list)
    qrels_file = data_dir / "qrels" / "dev.tsv"
    with open(qrels_file, 'r', encoding='utf-8') as f:
        next(f)  # 跳过header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                query_id, doc_id = parts[0], parts[1]
                qrels[query_id].append(doc_id)

    print(f"  Corpus: {len(corpus)} documents")
    print(f"  Queries: {len(queries)} queries")
    print(f"  Qrels: {len(qrels)} query-doc pairs")

    return corpus, queries, dict(qrels)


def evaluate_retrieval(results: Dict[str, List[str]], qrels: Dict[str, List[str]], k: int = 10):
    """
    评估检索结果

    计算：
    - Recall@k: 在top-k中找到相关文档的比例
    - MRR (Mean Reciprocal Rank): 第一个相关文档的倒数排名的平均值
    """
    recalls = []
    reciprocal_ranks = []

    for query_id, retrieved_docs in results.items():
        if query_id not in qrels:
            continue

        relevant_docs = set(qrels[query_id])
        retrieved_set = set(retrieved_docs[:k])

        # Recall@k
        if len(relevant_docs) > 0:
            recall = len(retrieved_set & relevant_docs) / len(relevant_docs)
            recalls.append(recall)

        # MRR
        for rank, doc_id in enumerate(retrieved_docs[:k], 1):
            if doc_id in relevant_docs:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)

    metrics = {
        f'Recall@{k}': np.mean(recalls) if recalls else 0.0,
        'MRR': np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        'num_queries': len(results)
    }

    return metrics


def main():
    print("=" * 60)
    print("Quora Multi-Vector Index Demo")
    print("=" * 60)

    # 配置
    DATA_DIR = "quora"  # Quora数据集目录
    DIM = 128           # 向量维度
    K = 100             # 返回top-k结果
    MAX_DOCS = 1000     # 限制文档数量（用于快速测试）

    # 1. 加载数据
    corpus, queries, qrels = load_quora_data(DATA_DIR)

    # 2. 限制数据量（可选，用于快速测试）
    if MAX_DOCS:
        corpus_ids = list(corpus.keys())[:MAX_DOCS]
        corpus = {doc_id: corpus[doc_id] for doc_id in corpus_ids}
        print(f"\n[INFO] Limited to {MAX_DOCS} documents for quick testing")

    # 3. 编码文档为多向量
    print(f"\nEncoding {len(corpus)} documents to multi-vectors...")
    doc_ids = []
    doc_vectors_list = []

    start_time = time.time()
    for i, (doc_id, text) in enumerate(corpus.items(), 1):
        vectors = encode_text_simple(text, dim=DIM)
        doc_ids.append(doc_id)
        doc_vectors_list.append(vectors)

        if i % 100 == 0:
            print(f"  Encoded {i}/{len(corpus)} documents...")

    encoding_time = time.time() - start_time
    print(f"Encoding completed in {encoding_time:.2f}s")

    # 统计向量数量
    total_vectors = sum(v.shape[0] for v in doc_vectors_list)
    avg_vectors = total_vectors / len(doc_vectors_list)
    print(f"  Total vectors: {total_vectors}")
    print(f"  Avg vectors per document: {avg_vectors:.2f}")

    # 4. 构建索引
    print(f"\nBuilding multi-vector index...")
    start_time = time.time()

    index = SimpleMultiVectorIndex(dim=DIM)
    index.add_documents(doc_ids, doc_vectors_list)

    index_time = time.time() - start_time
    print(f"Index built in {index_time:.2f}s")

    # 5. 搜索
    print(f"\nSearching for {len(queries)} queries...")
    results = {}

    start_time = time.time()
    for i, (query_id, query_text) in enumerate(queries.items(), 1):
        query_vectors = encode_text_simple(query_text, dim=DIM)
        retrieved_docs, scores = index.search(query_vectors, k=K)
        results[query_id] = retrieved_docs

        if i % 10 == 0:
            print(f"  Searched {i}/{len(queries)} queries...")

    search_time = time.time() - start_time
    print(f"Search completed in {search_time:.2f}s")
    print(f"  Avg time per query: {search_time / len(queries) * 1000:.2f}ms")

    # 6. 评估
    print(f"\nEvaluating results...")
    metrics = evaluate_retrieval(results, qrels, k=K)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Recall@{K}:  {metrics[f'Recall@{K}']:.4f}")
    print(f"MRR:         {metrics['MRR']:.4f}")
    print(f"Queries:     {metrics['num_queries']}")

    # 7. 显示一些示例结果
    print("\n" + "=" * 60)
    print("Sample Results")
    print("=" * 60)

    for i, (query_id, query_text) in enumerate(list(queries.items())[:3], 1):
        print(f"\nQuery {i}: {query_text}")
        print(f"  (Query ID: {query_id})")

        retrieved = results.get(query_id, [])[:5]
        relevant = qrels.get(query_id, [])

        print(f"  Relevant docs: {relevant}")
        print(f"  Retrieved (top-5):")
        for rank, doc_id in enumerate(retrieved, 1):
            is_relevant = "✓" if doc_id in relevant else " "
            doc_text = corpus.get(doc_id, "")[:60]
            print(f"    {rank}. [{is_relevant}] {doc_id}: {doc_text}...")

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("\nNOTE: This demo uses RANDOM vectors for demonstration.")
    print("      For real experiments, replace encode_text_simple() with:")
    print("      - ColBERT model")
    print("      - BERT tokenizer + embeddings")
    print("      - Sentence-BERT + sentence splitting")
    print("=" * 60)


if __name__ == "__main__":
    main()
