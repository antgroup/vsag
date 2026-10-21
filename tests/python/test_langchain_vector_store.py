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
