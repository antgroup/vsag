"""
Multi-Vector Index for VSAG
支持多向量表示的文档检索系统
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import pyvsag


class MultiVectorIndex:
    """
    多向量索引实现

    每个文档由多个向量表示，搜索时使用MaxSim策略：
    score(query, doc) = sum(max_sim(q_vec, doc_vecs) for q_vec in query_vecs)
    """

    def __init__(self, dim: int = 128, index_type: str = "hnsw", max_degree: int = 16):
        """
        初始化多向量索引

        Args:
            dim: 向量维度
            index_type: 底层索引类型 (hnsw, diskann, hgraph)
            max_degree: HNSW的最大度数
        """
        self.dim = dim
        self.index_type = index_type

        # VSAG索引参数
        index_params = json.dumps({
            "dtype": "float32",
            "metric_type": "ip",  # 内积（用于余弦相似度）
            "dim": dim,
            "hnsw": {
                "max_degree": max_degree,
                "ef_construction": 200
            }
        })

        # 创建VSAG索引（存储所有向量）
        self.vsag_index = pyvsag.Index(index_type, index_params)

        # 映射关系
        self.doc_to_vectors = {}  # doc_id -> [vector_ids]
        self.vector_to_doc = {}   # vector_id -> doc_id
        self.next_vector_id = 0

        # 统计信息
        self.num_docs = 0
        self.num_vectors = 0

    def add_document(self, doc_id: str, vectors: np.ndarray):
        """
        添加一个文档的多个向量

        Args:
            doc_id: 文档ID
            vectors: shape [num_vectors, dim] 的向量数组
        """
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        assert vectors.shape[1] == self.dim, f"Vector dim mismatch: {vectors.shape[1]} vs {self.dim}"

        # 归一化向量（用于余弦相似度）
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # 避免除零
        vectors = vectors / norms

        # 为每个向量分配唯一ID
        vector_ids = []
        for vec in vectors:
            vec_id = self.next_vector_id
            vector_ids.append(vec_id)
            self.vector_to_doc[vec_id] = doc_id
            self.next_vector_id += 1

        # 添加到VSAG索引
        vector_ids_array = np.array(vector_ids, dtype=np.int64)
        self.vsag_index.build(
            vectors=vectors.astype(np.float32),
            ids=vector_ids_array,
            num_elements=len(vector_ids),
            dim=self.dim
        )

        # 更新映射
        if doc_id in self.doc_to_vectors:
            self.doc_to_vectors[doc_id].extend(vector_ids)
        else:
            self.doc_to_vectors[doc_id] = vector_ids
            self.num_docs += 1

        self.num_vectors += len(vector_ids)

    def search(self, query_vectors: np.ndarray, k: int = 10,
               ef_search: int = 100) -> Tuple[List[str], List[float]]:
        """
        多向量搜索（MaxSim策略）

        Args:
            query_vectors: shape [num_query_vectors, dim]
            k: 返回top-k文档
            ef_search: HNSW搜索参数

        Returns:
            (doc_ids, scores): 排序后的文档ID和分数
        """
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)

        # 归一化查询向量
        norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        query_vectors = query_vectors / norms

        # 搜索参数
        search_params = json.dumps({"hnsw": {"ef_search": ef_search}})

        # 对每个查询向量搜索最相似的文档向量
        doc_scores = defaultdict(float)

        for q_vec in query_vectors:
            # 搜索top-k*10个向量（因为多个向量可能属于同一文档）
            result_ids, distances = self.vsag_index.knn_search(
                vector=q_vec.astype(np.float32),
                k=min(k * 20, self.num_vectors),  # 搜索更多候选
                parameters=search_params
            )

            # 将距离转换为相似度（内积已经是相似度）
            similarities = distances

            # MaxSim: 对每个文档，取其所有向量中与当前查询向量最相似的
            for vec_id, sim in zip(result_ids, similarities):
                doc_id = self.vector_to_doc.get(vec_id)
                if doc_id is not None:
                    # 只保留每个文档的最大相似度
                    doc_scores[doc_id] = max(doc_scores[doc_id], sim)

        # 聚合所有查询向量的分数（sum of max similarities）
        # 注意：上面的循环已经实现了MaxSim，这里直接排序

        # 排序并返回top-k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        if not sorted_docs:
            return [], []

        doc_ids, scores = zip(*sorted_docs)
        return list(doc_ids), list(scores)

    def get_stats(self) -> Dict:
        """返回索引统计信息"""
        return {
            "num_documents": self.num_docs,
            "num_vectors": self.num_vectors,
            "avg_vectors_per_doc": self.num_vectors / max(self.num_docs, 1),
            "dim": self.dim,
            "index_type": self.index_type
        }


def load_corpus(corpus_file: str) -> Dict[str, str]:
    """加载corpus文件"""
    corpus = {}
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line.strip())
            doc_id = doc['_id']
            text = doc.get('text', '') or doc.get('title', '')
            corpus[doc_id] = text
    return corpus


def load_queries(queries_file: str) -> Dict[str, str]:
    """加载queries文件"""
    queries = {}
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            query = json.loads(line.strip())
            query_id = query['_id']
            text = query['text']
            queries[query_id] = text
    return queries


def load_qrels(qrels_file: str) -> Dict[str, List[str]]:
    """加载qrels文件"""
    qrels = defaultdict(list)
    with open(qrels_file, 'r', encoding='utf-8') as f:
        next(f)  # 跳过header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                query_id, doc_id = parts[0], parts[1]
                qrels[query_id].append(doc_id)
    return dict(qrels)


# 文本向量化的占位函数（需要用实际的embedding模型替换）
def encode_text_to_multi_vectors(text: str, dim: int = 128) -> np.ndarray:
    """
    将文本编码为多个向量

    TODO: 这是占位实现，需要替换为真实的模型
    实际应用中可以使用：
    1. ColBERT模型（token-level embeddings）
    2. Sentence-BERT + 分句
    3. 其他多向量编码器

    Args:
        text: 输入文本
        dim: 向量维度

    Returns:
        shape [num_vectors, dim] 的向量数组
    """
    # 简单的占位实现：按空格分词，每个词生成一个随机向量
    # 实际使用时应该替换为真实的embedding模型
    tokens = text.lower().split()

    if not tokens:
        return np.random.randn(1, dim).astype(np.float32)

    # 为演示目的，使用确定性的"假"向量（基于token hash）
    vectors = []
    for i, token in enumerate(tokens[:20]):  # 限制最多20个token
        # 使用token的hash作为随机种子，保证可重复性
        seed = hash(token) % (2**31)
        rng = np.random.RandomState(seed)
        vec = rng.randn(dim)
        vectors.append(vec)

    return np.array(vectors, dtype=np.float32)


if __name__ == "__main__":
    # 示例用法
    print("Multi-Vector Index for VSAG")
    print("=" * 50)

    # 创建一个小示例
    index = MultiVectorIndex(dim=128)

    # 添加文档
    doc1_text = "What is Python programming?"
    doc1_vecs = encode_text_to_multi_vectors(doc1_text)
    index.add_document("doc1", doc1_vecs)

    doc2_text = "How to learn machine learning?"
    doc2_vecs = encode_text_to_multi_vectors(doc2_text)
    index.add_document("doc2", doc2_vecs)

    # 搜索
    query_text = "Python programming tutorial"
    query_vecs = encode_text_to_multi_vectors(query_text)
    doc_ids, scores = index.search(query_vecs, k=2)

    print(f"\nQuery: {query_text}")
    print(f"Results:")
    for doc_id, score in zip(doc_ids, scores):
        print(f"  {doc_id}: {score:.4f}")

    print(f"\nIndex stats: {index.get_stats()}")
