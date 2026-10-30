# Copyright 2024-present the vsag project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the LangChain VectorStore adapter."""

import numpy as np
import pytest

pytest.importorskip("langchain_core")

from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings  # noqa: E402

from pyvsag.langchain import VSAGVectorStore  # noqa: E402

DIM = 4


class TableEmbedding(Embeddings):
    """Deterministic embedding backed by an explicit text -> vector mapping."""

    def __init__(self, mapping):
        self._mapping = {
            key: np.asarray(value, dtype=np.float32) for key, value in mapping.items()
        }

    def embed_documents(self, texts):
        return [self._mapping[text].copy() for text in texts]

    def embed_query(self, text):
        return self._mapping[text].copy()


def _store(metric="l2", index_type="brute_force", mapping=None):
    mapping = mapping or {
        "alpha": [1.0, 0.0, 0.0, 0.0],
        "beta": [0.0, 1.0, 0.0, 0.0],
        "gamma": [1.0, 1.0, 0.0, 0.0],
        "query": [1.0, 2.0, 0.0, 0.0],
    }
    return VSAGVectorStore(
        TableEmbedding(mapping), index_type=index_type, metric_type=metric
    )


def test_from_texts_assigns_string_ids():
    store = VSAGVectorStore.from_texts(
        ["alpha", "beta"], DeterministicFakeEmbedding(size=DIM)
    )
    assert store.index is not None
    documents = store.similarity_search("alpha", k=2)
    assert {doc.page_content for doc in documents} == {"alpha", "beta"}
    assert all(isinstance(doc.id, str) and doc.id for doc in documents)


def test_add_texts_with_user_ids_and_metadata():
    store = _store()
    ids = store.add_texts(
        ["alpha", "beta"], metadatas=[{"source": "a"}, {"source": "b"}], ids=["1", "2"]
    )
    assert ids == ["1", "2"]
    documents = store.get_by_ids(["1", "2"])
    assert [doc.id for doc in documents] == ["1", "2"]
    assert documents[0].metadata == {"source": "a"}


def test_add_documents_inherited_preserves_ids():
    store = _store()
    ids = store.add_documents(
        [Document(id="foo", page_content="alpha"), Document(page_content="beta")]
    )
    assert ids[0] == "foo"
    assert len(ids[1]) > 0
    assert store.get_by_ids(["foo"])[0].page_content == "alpha"


def test_incremental_add_uses_existing_index():
    store = _store()
    store.add_texts(["alpha"], ids=["1"])
    index = store.index
    store.add_texts(["beta"], ids=["2"])
    assert store.index is index
    assert {doc.page_content for doc in store.similarity_search("gamma", k=2)} == {
        "alpha",
        "beta",
    }


def test_search_returns_original_text_metadata_and_id():
    store = _store()
    store.add_texts(["alpha"], metadatas=[{"k": "v"}], ids=["doc-1"])
    documents = store.similarity_search("query", k=1)
    assert documents == [
        Document(id="doc-1", page_content="alpha", metadata={"k": "v"})
    ]


def test_similarity_search_by_vector_matches_text_query():
    store = _store()
    store.add_texts(["alpha", "beta", "gamma"], ids=["1", "2", "3"])
    by_text = store.similarity_search("query", k=2)
    by_vector = store.similarity_search_by_vector([1.0, 2.0, 0.0, 0.0], k=2)
    assert [doc.id for doc in by_vector] == [doc.id for doc in by_text]


@pytest.mark.parametrize("metric", ["l2", "ip", "cosine"])
def test_scores_match_vsag_metric_semantics(metric):
    mapping = {
        "d1": [1.0, 0.0, 0.0, 0.0],
        "d2": [0.0, 1.0, 0.0, 0.0],
        "d3": [1.0, 1.0, 0.0, 0.0],
        "query": [1.0, 2.0, 0.0, 0.0],
    }
    store = _store(metric=metric, mapping=mapping)
    store.add_texts(["d1", "d2", "d3"], ids=["1", "2", "3"])

    results = store.similarity_search_with_score("query", k=3)
    assert [doc.page_content for doc, _ in results] == ["d3", "d2", "d1"]

    query = np.asarray(mapping["query"], dtype=np.float32)
    for document, distance in results:
        vector = np.asarray(mapping[document.page_content], dtype=np.float32)
        if metric == "l2":
            expected = float(((vector - query) ** 2).sum())
        elif metric == "ip":
            expected = float(1.0 - np.dot(vector, query))
        else:
            expected = float(
                1.0
                - np.dot(vector, query)
                / (np.linalg.norm(vector) * np.linalg.norm(query))
            )
        assert distance == pytest.approx(expected, abs=1e-4)


def test_empty_store_operations_are_noops():
    store = _store()
    assert store.similarity_search("alpha", k=1) == []
    assert store.similarity_search_with_score("alpha", k=1) == []
    assert store.similarity_search_by_vector([1.0, 0.0, 0.0, 0.0], k=1) == []
    assert store.get_by_ids(["missing"]) == []
    store.delete(["missing"])
    assert store.index is None


def test_k_is_clamped_to_live_documents():
    store = _store()
    store.add_texts(["alpha", "beta"], ids=["1", "2"])
    assert len(store.similarity_search("query", k=10)) == 2


def test_upsert_replaces_document_content():
    store = _store()
    store.add_texts(["alpha"], metadatas=[{"version": 1}], ids=["1"])
    store.add_texts(["beta"], metadatas=[{"version": 2}], ids=["1"])

    documents = store.get_by_ids(["1"])
    assert len(documents) == 1
    assert documents[0] == Document(id="1", page_content="beta", metadata={"version": 2})
    assert {doc.id for doc in store.similarity_search("query", k=2)} == {"1"}


def test_delete_removes_vectors_and_mappings():
    store = _store()
    store.add_texts(["alpha", "beta"], ids=["1", "2"])
    assert store.delete(["1", "missing"]) is True

    assert store.get_by_ids(["1"]) == []
    assert [doc.id for doc in store.get_by_ids(["2"])] == ["2"]
    assert {doc.id for doc in store.similarity_search("query", k=2)} == {"2"}


def test_re_add_after_delete_works():
    store = _store()
    store.add_texts(["alpha"], ids=["1"])
    store.delete(["1"])
    assert store.add_texts(["beta"], ids=["1"]) == ["1"]
    assert store.get_by_ids(["1"])[0].page_content == "beta"


def test_delete_all_documents():
    store = _store()
    store.add_texts(["alpha", "beta"], ids=["1", "2"])
    assert store.delete() is True
    assert store.get_by_ids(["1", "2"]) == []
    assert store.similarity_search("query", k=2) == []


def test_invalid_inputs_raise():
    store = _store()
    with pytest.raises(ValueError):
        store.add_texts(["alpha", "beta"], ids=["1", "1"])
    with pytest.raises(ValueError):
        store.add_texts(["alpha", "beta"], ids=["1"])
    with pytest.raises(ValueError):
        store.add_texts(["alpha"], metadatas=[{}, {}])
    with pytest.raises(ValueError):
        store.similarity_search("alpha", k=0)

    store.add_texts(["alpha"], ids=["1"])
    with pytest.raises(NotImplementedError):
        store.similarity_search("alpha", k=1, filter={"source": "a"})
    assert store.similarity_search("alpha", k=1, unsupported=True)[0].id == "1"
    store.add_texts(["beta"], ids=["2"], timeout=1)
    store.delete(["missing"], timeout=1)

    other = _store(mapping={"alpha": [1.0, 0.0, 0.0, 0.0], "wide": [1.0] * 5})
    other.add_texts(["alpha"], ids=["1"])
    with pytest.raises(ValueError):
        other.add_texts(["wide"], ids=["2"])
    with pytest.raises(ValueError):
        other.similarity_search_by_vector([1.0] * 5, k=1)


def test_invalid_metric_and_missing_search_params():
    with pytest.raises(ValueError):
        VSAGVectorStore(DeterministicFakeEmbedding(size=DIM), metric_type="dot")
    with pytest.raises(ValueError):
        VSAGVectorStore(DeterministicFakeEmbedding(size=DIM), index_type="ivf")
    with pytest.raises(ValueError):
        VSAGVectorStore(
            DeterministicFakeEmbedding(size=DIM),
            index_param={"support_remove": False},
        )


def test_json_parameters_and_caller_dict_are_safe():
    store = VSAGVectorStore(
        TableEmbedding(
            {"alpha": [1.0, 0.0, 0.0, 0.0], "query": [1.0, 0.0, 0.0, 0.0]}
        ),
        index_param='{"max_degree": 16}',
        search_params='{"hgraph": {"ef_search": 32}}',
    )
    store.add_texts(["alpha"], ids=["1"])

    per_call = {"hgraph": {}}
    store.similarity_search("query", k=1, search_params=per_call)
    assert per_call == {"hgraph": {}}


def test_hgraph_default_search_params():
    store = VSAGVectorStore(
        TableEmbedding(
            {
                "alpha": [1.0, 0.0, 0.0, 0.0],
                "beta": [0.0, 1.0, 0.0, 0.0],
                "query": [1.0, 0.0, 0.0, 0.0],
            }
        )
    )
    store.add_texts(["alpha", "beta"], ids=["1", "2"])
    assert store.similarity_search("query", k=1)[0].id == "1"
    assert store.similarity_search("query", k=1, ef_search=32)[0].id == "1"


def test_as_retriever_uses_inherited_behaviour():
    store = _store()
    store.add_texts(["alpha", "beta", "gamma"], ids=["1", "2", "3"])
    documents = store.as_retriever(search_kwargs={"k": 2}).invoke("query")
    assert [doc.page_content for doc in documents] == ["gamma", "beta"]


class FaultIndex:
    """Deterministic engine double for partial writes and stale search candidates."""

    def __init__(self, *args):
        self.vectors = {}
        self.refuse_remove = set()
        self.raise_remove = set()
        self.keep_stale = False
        self.fail_add = False
        self.search_limits = []

    def build(self, vectors, ids, count, dim):
        self.add(vectors, ids, count, dim)

    def add(self, vectors, ids, count, dim):
        self.vectors.update({int(i): v.copy() for i, v in zip(ids, vectors)})
        if self.fail_add:
            raise RuntimeError("injected partial add failure")

    def remove(self, ids):
        removed = 0
        for int_id in ids.tolist():
            if int_id in self.raise_remove:
                raise RuntimeError("injected remove failure")
            if int_id not in self.refuse_remove and int_id in self.vectors:
                removed += 1
                if not self.keep_stale:
                    del self.vectors[int_id]
        return removed

    def check_id_exist(self, int_id):
        return int_id in self.vectors

    def get_num_elements(self):
        return len(self.vectors)

    def knn_search(self, vector, k, parameters):
        self.search_limits.append(k)
        candidates = sorted(
            (float(np.sum((v - vector) ** 2)), i) for i, v in self.vectors.items()
        )[:k]
        return (
            np.asarray([i for _, i in candidates], dtype=np.int64),
            np.asarray([d for d, _ in candidates], dtype=np.float32),
        )


@pytest.fixture
def fault_store(monkeypatch):
    monkeypatch.setattr("pyvsag.Index", FaultIndex)
    return _store()


def test_regression_delete_preserves_unremoved_documents_and_allows_retry(fault_store):
    store = fault_store
    store.add_texts(["alpha", "beta"], ids=["1", "2"])
    store.index.refuse_remove.add(0)
    with pytest.raises(RuntimeError, match="could not remove vectors"):
        store.delete(["1", "2"])
    assert store.get_by_ids(["1", "2"]) == [Document(id="1", page_content="alpha")]
    assert store._id_to_int == {"1": 0}
    assert store._int_to_id == {0: "1"}
    assert store.similarity_search("alpha", k=2)[0].id == "1"
    store.index.refuse_remove.clear()
    assert store.delete(["1"]) is True
    assert store.delete(["1"]) is True
    assert store.get_by_ids(["1"]) == []


@pytest.mark.parametrize("raises", [False, True])
def test_regression_upsert_preserves_old_document_on_remove_failure(fault_store, raises):
    store = fault_store
    store.add_texts(["alpha"], ids=["1"], metadatas=[{"version": 1}])
    failures = store.index.raise_remove if raises else store.index.refuse_remove
    failures.add(0)
    with pytest.raises(RuntimeError):
        store.add_texts(["beta"], ids=["1"], metadatas=[{"version": 2}])
    assert store.get_by_ids(["1"]) == [
        Document(id="1", page_content="alpha", metadata={"version": 1})
    ]
    assert set(store.index.vectors) == {0}
    assert store._int_to_id == {0: "1"}
    failures.clear()
    store.add_texts(["beta"], ids=["1"])
    assert store.similarity_search("beta", k=1)[0].page_content == "beta"


def test_regression_upsert_partial_batch_publishes_only_confirmed_replacements(fault_store):
    store = fault_store
    store.add_texts(["alpha", "beta"], ids=["1", "2"])
    store.index.refuse_remove.add(1)
    with pytest.raises(RuntimeError):
        store.add_texts(["gamma", "gamma"], ids=["1", "2"])
    assert [doc.page_content for doc in store.get_by_ids(["1", "2"])] == ["gamma", "beta"]
    assert store._int_to_id == {1: "2", 2: "1"}
    assert set(store.index.vectors) == {1, 2}


def test_regression_failed_cleanup_is_tracked_filtered_and_retried(fault_store):
    store = fault_store
    store.add_texts(["alpha", "gamma"], ids=["1", "2"])
    store.index.refuse_remove.update({0, 2})
    with pytest.raises(RuntimeError):
        store.add_texts(["beta"], ids=["1"])
    assert store._pending_removals == {2}
    results = store.similarity_search("beta", k=2)
    assert {doc.id for doc in results} == {"1", "2"}
    assert len(results) == 2
    store.index.refuse_remove.clear()
    store.add_texts(["gamma"], ids=["3"])
    assert store._pending_removals == set()
    assert 2 not in store.index.vectors


def test_regression_partial_add_never_publishes_or_reuses_staged_ids(fault_store):
    store = fault_store
    store.add_texts(["alpha"], ids=["1"])
    store.index.fail_add = True
    with pytest.raises(RuntimeError, match="partial add"):
        store.add_texts(["beta"], ids=["2"])
    assert store.get_by_ids(["2"]) == []
    assert set(store.index.vectors) == {0}
    store.index.fail_add = False
    store.add_texts(["beta"], ids=["2"])
    assert store._id_to_int["2"] == 2


def test_regression_search_expands_past_stale_candidates(fault_store):
    store = fault_store
    store.add_texts(["alpha", "beta", "gamma"], ids=["1", "2", "3"])
    store.index.keep_stale = True
    store.delete(["1"])
    results = store.similarity_search_with_score("alpha", k=2)
    assert [doc.id for doc, _ in results] == ["3", "2"]
    assert store.index.search_limits == [2, 3]
    assert len(store.similarity_search("alpha", k=20)) == 2


def test_regression_upsert_search_returns_only_current_unique_documents(fault_store):
    store = fault_store
    store.add_texts(["alpha", "beta"], ids=["1", "2"])
    store.index.keep_stale = True
    store.add_texts(["gamma"], ids=["1"])
    results = store.similarity_search("alpha", k=2)
    assert {doc.id for doc in results} == {"1", "2"}
    assert len(results) == 2
    assert next(doc for doc in results if doc.id == "1").page_content == "gamma"


def test_regression_search_serializes_native_call_with_upsert(fault_store, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    store = fault_store
    store.add_texts(["alpha", "beta"], ids=["1", "2"])
    entered, release, mutation_started, added = Event(), Event(), Event(), Event()
    original_search = store.index.knn_search
    original_add = store.index.add

    def blocked_search(*args):
        entered.set()
        assert release.wait(5)
        return original_search(*args)

    def observed_add(*args):
        added.set()
        return original_add(*args)

    def upsert():
        mutation_started.set()
        return store.add_texts(["gamma"], ids=["1"])

    monkeypatch.setattr(store.index, "knn_search", blocked_search)
    monkeypatch.setattr(store.index, "add", observed_add)
    with ThreadPoolExecutor(max_workers=2) as executor:
        search = executor.submit(store.similarity_search, "alpha", k=2)
        try:
            assert entered.wait(5)
            mutation = executor.submit(upsert)
            assert mutation_started.wait(5)
            assert not added.wait(0.1)
        finally:
            release.set()
        assert search.result(timeout=5)[0].page_content == "alpha"
        assert mutation.result(timeout=5) == ["1"]
    assert store.get_by_ids(["1"])[0].page_content == "gamma"


@pytest.mark.parametrize("metadata", ["abc", [], [("source", "a")], 42])
def test_regression_metadata_rejects_non_mappings_before_embedding(fault_store, metadata):
    with pytest.raises(TypeError, match=r"metadatas\[0\] must be a Mapping"):
        fault_store.add_texts(["not-in-embedding-table"], metadatas=[metadata])
    assert fault_store.index is None


def test_regression_metadata_accepts_mapping_and_none(fault_store):
    from collections import UserDict

    fault_store.add_texts(
        ["alpha", "beta"], ids=["1", "2"], metadatas=[UserDict({"a": 1}), None]
    )
    assert [doc.metadata for doc in fault_store.get_by_ids(["1", "2"])] == [{"a": 1}, {}]


def test_regression_zero_width_embeddings_fail_before_index_creation(fault_store):
    fault_store._embedding = TableEmbedding({"empty": []})
    with pytest.raises(ValueError, match="dimension > 0"):
        fault_store.add_texts(["empty"])
    assert fault_store.index is None


@pytest.mark.parametrize("populated", [False, True])
@pytest.mark.parametrize("options", [
    {"search_type": "mmr"}, {"search_type": "similarity_score_threshold"},
    {"score_threshold": 0.0}, {"fetch_k": 10}, {"lambda_mult": 0.5},
])
def test_regression_unsupported_search_options_raise(fault_store, populated, options):
    if populated:
        fault_store.add_texts(["alpha"], ids=["1"])
    with pytest.raises(NotImplementedError):
        fault_store.similarity_search("alpha", **options)
    with pytest.raises(NotImplementedError):
        fault_store.similarity_search_with_score_by_vector([1.0] * DIM, **options)
    with pytest.raises(NotImplementedError):
        fault_store.as_retriever(search_kwargs=options)


@pytest.mark.parametrize("search_type", ["mmr", "similarity_score_threshold"])
def test_regression_unsupported_retriever_types_raise(fault_store, search_type):
    with pytest.raises(NotImplementedError, match="only supports similarity"):
        fault_store.as_retriever(search_type=search_type)


def test_regression_pyramid_requires_paths():
    with pytest.raises(ValueError, match="pyramid requires path metadata"):
        VSAGVectorStore(
            DeterministicFakeEmbedding(size=DIM), index_type="pyramid",
            search_params={"pyramid": {"ef_search": 100}},
        )


def test_regression_underfilled_native_search_stops(fault_store, monkeypatch):
    store = fault_store
    store.add_texts(["alpha", "beta"], ids=["1", "2"])
    store._next_int_id = 1000
    limits = []

    def underfilled(vector, k, parameters):
        limits.append(k)
        return np.array([0, -1]), np.array([0.0, -1.0])

    monkeypatch.setattr(store.index, "knn_search", underfilled)
    assert [doc.id for doc in store.similarity_search("alpha", k=2)] == ["1"]
    assert limits == [2]


def test_regression_cleanup_discards_absent_ids_and_keeps_live_failures(fault_store):
    store = fault_store
    store.add_texts(["alpha"], ids=["1"])
    store.index.vectors[2] = np.zeros(DIM, dtype=np.float32)
    store.index.refuse_remove.add(2)
    store._pending_removals.update({1, 2})
    store._cleanup_pending_removals()
    assert store._pending_removals == {2}


@pytest.mark.parametrize("by_vector", [False, True])
def test_regression_numpy_k_is_serializable(fault_store, by_vector):
    store = fault_store
    store._is_hgraph = True
    store.add_texts(["alpha"] * 101, ids=[str(i) for i in range(101)])
    if by_vector:
        result = store.similarity_search_with_score_by_vector([1.0, 0.0, 0.0, 0.0], k=np.int64(101))
    else:
        result = store.similarity_search_with_score("alpha", k=np.int64(101))
    assert len(result) == 101


@pytest.mark.parametrize("per_call", [False, True])
def test_regression_numpy_ef_search_in_mapping(fault_store, per_call):
    store = fault_store
    store._is_hgraph = True
    params = {"hgraph": {"ef_search": np.int64(120)}}
    store.add_texts(["alpha"], ids=["1"])
    if per_call:
        results = store.similarity_search("alpha", search_params=params)
    else:
        store._search_params = params
        results = store.similarity_search("alpha")
    assert results[0].id == "1"


def test_regression_failed_reservations_do_not_expand_search(fault_store, monkeypatch):
    store = fault_store
    store.add_texts(["alpha", "beta"], ids=["1", "2"])
    store._next_int_id = 1000000
    limits = []

    def stale_results(vector, k, parameters):
        limits.append(k)
        return np.array([999999] * k), np.zeros(k)

    monkeypatch.setattr(store.index, "knn_search", stale_results)
    assert store.similarity_search("alpha", k=2) == []
    assert limits == [2]
