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


def print_results(result_ids, result_distances, titles):
    """Print the neighbours of one query in a readable form."""
    for rank, (doc_id, distance) in enumerate(
        zip(result_ids, result_distances), start=1
    ):
        doc_id = int(doc_id)
        if doc_id not in titles:
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
    # `values` holds one weight per non-zero term, ordered by term ID within
    # each row; the vocabulary printed above maps each term to its ID.
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
                "window_size": 10000,
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
    topk = 3
    search_params = json.dumps({"sindi": {"query_prune_ratio": 0.0, "n_candidate": 6}})
    query_terms = {"vector": 1.0, "search": 0.8}
    missing = sorted(set(query_terms) - set(vocabulary))
    if missing:
        raise ValueError(f"query terms missing from the vocabulary: {missing}")

    # Convert the query once and reuse the same CSR arrays before and after
    # the save/load round-trip, so both searches really use one query.
    query_pointers, query_indices, query_values = term_weights_to_csr(
        [query_terms], vocabulary
    )
    result_ids, result_distances = index.knn_search(
        query_pointers, query_indices, query_values, topk, search_params
    )
    print(f"\nQuery terms: {query_terms}")
    print_results(result_ids[0], result_distances[0], titles)

    # Step 5: persist the index and confirm the loaded copy returns the same
    # neighbours for the same query.
    filename = "./python_example_sindi.index"
    index.save(filename)
    restored = pyvsag.Index("sindi", index_params)
    restored.load(filename)
    print(f"\nReloaded index from {filename}")

    restored_ids, restored_distances = restored.knn_search(
        query_pointers, query_indices, query_values, topk, search_params
    )
    ids_match = bool(np.array_equal(result_ids, restored_ids))
    dists_match = bool(np.allclose(result_distances, restored_distances))
    print("\nSave/Load round-trip test:")
    print(f"  result_ids match:       {ids_match}")
    print(f"  result_distances match: {dists_match}")
    if not (ids_match and dists_match):
        print("  FAIL: save/load round-trip changed results")
        if not ids_match:
            print(f"    before IDs:   {result_ids.tolist()}")
            print(f"    after IDs:    {restored_ids.tolist()}")
        if not dists_match:
            print(f"    before dists: {result_distances.tolist()}")
            print(f"    after dists:  {restored_distances.tolist()}")
        raise SystemExit(1)
    print("  PASS: save/load round-trip preserves results")


if __name__ == "__main__":
    sindi_term_weight_example()
