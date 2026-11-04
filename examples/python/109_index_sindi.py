#  Copyright 2024-present the vsag project
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Build and search a SINDI index from readable term-weight mappings.

SINDI consumes sparse vectors as CSR (Compressed Sparse Row) arrays whose
column indices are integer term IDs. Real applications usually start from
human-readable terms, such as tokens, tags, or keywords. This example shows
the complete pipeline:

    documents with string terms -> vocabulary -> CSR arrays -> build -> search

``pyvsag.sparse`` provides the reusable helpers; the same functions work for
both the document corpus and query terms.
"""

import json

import numpy as np
import pyvsag

from pyvsag.sparse import build_term_vocabulary, term_weights_to_csr

# Each document carries business metadata plus a {term: weight} mapping.
# Weights can be TF-IDF, SPLADE activations, BM25 term frequencies, or any
# other non-negative value produced by your encoder.
DOCUMENTS = [
    {
        "id": 1001,
        "title": "Guide to vector search",
        "terms": {"vector": 1.0, "search": 1.0, "guide": 0.6},
    },
    {
        "id": 1002,
        "title": "Sparse retrieval with SINDI",
        "terms": {"sparse": 1.0, "retrieval": 1.0, "sindi": 0.9},
    },
    {
        "id": 1003,
        "title": "Vector database indexing",
        "terms": {"vector": 1.0, "database": 0.8, "indexing": 0.9},
    },
    {
        "id": 1004,
        "title": "Full-text search with keywords",
        "terms": {"full": 0.7, "text": 0.8, "search": 1.0, "keyword": 1.0},
    },
    {
        "id": 1005,
        "title": "Approximate nearest neighbor search",
        "terms": {
            "approximate": 1.0,
            "nearest": 1.0,
            "neighbor": 0.9,
            "search": 0.8,
        },
    },
    {
        "id": 1006,
        "title": "Sparse vector index tuning",
        "terms": {"sparse": 1.0, "vector": 0.9, "index": 1.0, "tuning": 0.7},
    },
]


def print_results(index, vocabulary, query_terms, topk, search_params, titles):
    """Convert query terms to CSR, run KNN, and print readable results."""
    missing = sorted(set(query_terms) - set(vocabulary))
    if missing:
        raise ValueError(f"query terms missing from the vocabulary: {missing}")

    query_pointers, query_indices, query_values = term_weights_to_csr(
        [query_terms], vocabulary
    )
    result_ids, result_distances = index.knn_search(
        query_pointers, query_indices, query_values, topk, search_params
    )

    print(f"\nQuery terms: {query_terms}")
    for rank, (doc_id, distance) in enumerate(
        zip(result_ids[0], result_distances[0]), start=1
    ):
        doc_id = int(doc_id)
        if doc_id < 0:
            print(f"  {rank}. <no result>")
            continue
        print(f"  {rank}. {titles[doc_id]} (id={doc_id}, distance={float(distance):.4f})")


def sindi_term_weight_example():
    term_weights = [document["terms"] for document in DOCUMENTS]
    ids = np.array([document["id"] for document in DOCUMENTS], dtype=np.int64)
    titles = {document["id"]: document["title"] for document in DOCUMENTS}

    # Step 1: assign a stable uint32 term ID to every string term. Terms are
    # sorted lexicographically, so the vocabulary does not depend on dict
    # insertion order or document order.
    vocabulary = build_term_vocabulary(term_weights)
    print("Term vocabulary:")
    for term, term_id in vocabulary.items():
        print(f"  {term_id:>2} -> {term}")

    # Step 2: convert documents to the CSR arrays consumed by pyvsag.
    index_pointers, indices, values = term_weights_to_csr(term_weights, vocabulary)
    print("\nCSR preview:")
    print("  index_pointers:", index_pointers.tolist())
    print("  indices:       ", indices.tolist())
    # The vocabulary from the 6 documents includes: approximate, database, full,
    # guide, indexing, keyword, nearest, neighbor, retrieval, sparse, text, tuning,
    # vector.  With terms sorted lexicographically the mapping is:
    #   approximate:0, database:1, full:2, guide:3, indexing:4, keyword:5,
    #   nearest:6, neighbor:7, retrieval:8, sparse:9, text:10, tuning:11,
    #   vector:12
    # Document 1 (vector=1.0, search=1.0, guide=0.6) -> terms vector(12), search(13),
    # but wait - let's just print the actual values:
    print("  values:        ", [float(v) for v in values.tolist()])

    # `dim` is the maximum number of non-zero terms per vector, not the
    # vocabulary size. `term_id_limit` must cover the largest term ID + 1.
    max_terms_per_document = max(len(weights) for weights in term_weights)
    index_params = json.dumps(
        {
            "dtype": "sparse",
            "dim": max_terms_per_document,
            "metric_type": "ip",
            "index_param": {
                "term_id_limit": max(1, len(vocabulary)),
                "window_size": 50000,
                "doc_prune_ratio": 0.0,
                "use_reorder": True,
                "rerank_type": "fp32",
            },
        }
    )
    index = pyvsag.Index("sindi", index_params)

    # Step 3: build the index from the CSR arrays.
    index.build(index_pointers=index_pointers, indices=indices, values=values, ids=ids)
    print(f"\nBuilt SINDI index with {index.get_num_elements()} documents")

    # Step 4: search with readable query terms. The same helper converts the
    # query to CSR; unknown terms must be filtered or mapped beforehand.
    search_params = json.dumps({"sindi": {"query_prune_ratio": 0.0, "n_candidate": 6}})
    query_terms = {"vector": 1.0, "search": 0.8}
    print_results(index, vocabulary, query_terms, 3, search_params, titles)

    # Step 5: persist the index and confirm the loaded copy returns the same
    # neighbours.
    filename = "./python_example_sindi.index"
    index.save(filename)
    restored = pyvsag.Index("sindi", index_params)
    restored.load(filename)
    print(f"\nReloaded index from {filename}")

    # Compare results before and after round-trip
    search_params = json.dumps({"sindi": {"query_prune_ratio": 0.0, "n_candidate": 6}})
    query_terms = {"vector": 1.0, "search": 0.8}
    _, result_ids_before, result_distances_before = index.knn_search(
        index_pointers, indices, values, 3, search_params
    )
    _, result_ids_after, result_distances_after = restored.knn_search(
        index_pointers, indices, values, 3, search_params
    )
    ids_match = (result_ids_before == result_ids_after).all()
    dists_match = np.allclose(result_distances_before, result_distances_after)
    print(f"\nSave/Load round-trip test:")
    print(f"  result_ids match:    {ids_match}")
    print(f"  result_distances match: {dists_match}")
    if ids_match and dists_match:
        print("  PASS: save/load round-trip preserves results")
    else:
        print("  FAIL: save/load round-trip changed results")
        if not ids_match:
            print(f"    before IDs:   {result_ids_before.tolist()}")
            print(f"    after IDs:    {result_ids_after.tolist()}")
        if not dists_match:
            print(f"    before dists: {result_distances_before.tolist()}")
            print(f"    after dists:  {result_distances_after.tolist()}")


if __name__ == "__main__":
    sindi_term_weight_example()
