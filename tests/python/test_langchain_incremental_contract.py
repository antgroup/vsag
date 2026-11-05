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

"""Independent contract and edge-case companion tests for PR #3119.

The LangChain adapter is proposed by #3119; it is not functionality provided by
this branch or the current upstream base. These cases intentionally skip until
that adapter is available. The small index double below only replaces the
public ``pyvsag.Index`` backend boundary; it does not implement the adapter.
"""

import pytest

_langchain = pytest.importorskip(
    "pyvsag.langchain", reason="companion tests run after PR #3119 provides the adapter"
)
np = pytest.importorskip("numpy")
Document = pytest.importorskip("langchain_core.documents").Document
VSAGVectorStore = _langchain.VSAGVectorStore


class CountingEmbeddings:
    """Deterministic embeddings with an observable document-call count."""

    def __init__(self, vectors):
        self.vectors = vectors
        self.document_calls = 0

    def embed_documents(self, texts):
        self.document_calls += 1
        return [self.vectors[text] for text in texts]

    def embed_query(self, text):
        return self.vectors[text]


class FakeIndex:
    """Minimal public Index double recording writes, removals, and searches."""

    def __init__(self, *args):
        self.vectors = {}
        self.build_ids = []
        self.add_ids = []
        self.remove_ids = []
        self.search_calls = []

    @staticmethod
    def _int_ids(ids, count):
        return [int(value) for value in ids[:count].tolist()]

    def _write(self, vectors, int_ids):
        self.vectors.update(
            {int_id: vector.copy() for int_id, vector in zip(int_ids, vectors)}
        )

    def build(self, vectors, ids, count, dim):
        int_ids = self._int_ids(ids, count)
        self.build_ids.append(int_ids)
        self._write(vectors, int_ids)

    def add(self, vectors, ids, count, dim):
        int_ids = self._int_ids(ids, count)
        self.add_ids.append(int_ids)
        self._write(vectors, int_ids)

    def remove(self, ids):
        int_ids = [int(value) for value in ids.tolist()]
        self.remove_ids.append(int_ids)
        removed = 0
        for int_id in int_ids:
            if int_id in self.vectors:
                del self.vectors[int_id]
                removed += 1
        return removed

    def check_id_exist(self, int_id):
        return int_id in self.vectors

    def get_num_elements(self):
        return len(self.vectors)

    def knn_search(self, vector, k, parameters):
        self.search_calls.append((k, parameters))
        candidates = sorted(
            (float(np.sum((stored - vector) ** 2)), int_id)
            for int_id, stored in self.vectors.items()
        )[:k]
        return (
            np.asarray([int_id for _, int_id in candidates], dtype=np.int64),
            np.asarray([distance for distance, _ in candidates], dtype=np.float32),
        )


@pytest.fixture(autouse=True)
def fake_backend(monkeypatch):
    monkeypatch.setattr("pyvsag.Index", FakeIndex)


def _make_store(vectors):
    return VSAGVectorStore(CountingEmbeddings(vectors), index_type="brute_force")


def test_delete_and_re_add_preserves_user_ids_without_recycling_backend_ids():
    ids = ["0", "文档🚀", " id with spaces "]
    store = _make_store(
        {
            "zero": [1.0, 0.0],
            "unicode": [0.0, 1.0],
            "space": [1.0, 1.0],
            "zero-new": [2.0, 0.0],
            "unicode-new": [0.0, 2.0],
            "query": [0.0, 0.0],
        }
    )
    assert store.add_texts(
        ["zero", "unicode", "space"],
        metadatas=[{"slot": "zero"}, {"slot": "unicode"}, {"slot": "untouched"}],
        ids=ids,
    ) == ids
    backend = store.index
    original_int_ids = {int_id for batch in backend.build_ids for int_id in batch}

    assert store.delete(ids[:2]) is True
    assert store.add_texts(
        ["zero-new", "unicode-new"],
        metadatas=[{"slot": "zero-new"}, {"slot": "unicode-new"}],
        ids=ids[:2],
    ) == ids[:2]

    added_int_ids = {int_id for batch in backend.add_ids for int_id in batch}
    assert added_int_ids
    assert added_int_ids.isdisjoint(original_int_ids)
    expected = {
        "0": Document(id="0", page_content="zero-new", metadata={"slot": "zero-new"}),
        "文档🚀": Document(
            id="文档🚀", page_content="unicode-new", metadata={"slot": "unicode-new"}
        ),
        " id with spaces ": Document(
            id=" id with spaces ", page_content="space", metadata={"slot": "untouched"}
        ),
    }
    assert store.get_by_ids(ids) == [expected[doc_id] for doc_id in ids]
    assert {doc.id: doc for doc in store.similarity_search("query", k=3)} == expected


def test_rejected_dimension_change_preserves_existing_document():
    store = _make_store(
        {"old": [1.0, 0.0], "wide": [1.0, 0.0, 0.0], "query": [1.0, 0.0]}
    )
    store.add_texts(["old"], metadatas=[{"version": "original"}], ids=["stable"])

    with pytest.raises(ValueError, match="dimension 3 does not match index dimension 2"):
        store.add_texts(["wide"], ids=["rejected"])

    expected = Document(
        id="stable", page_content="old", metadata={"version": "original"}
    )
    assert store.get_by_ids(["stable", "rejected"]) == [expected]
    assert store.similarity_search("query", k=2) == [expected]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"metadatas": [{}]}, "metadatas length must match"),
        ({"ids": ["only-one"]}, "ids length must match"),
    ],
)
def test_length_mismatch_fails_before_embedding(options, message):
    store = _make_store({"first": [1.0, 0.0], "second": [0.0, 1.0]})

    with pytest.raises(ValueError, match=message):
        store.add_texts(["first", "second"], **options)

    assert store.embeddings.document_calls == 0
    assert store.index is None


def test_custom_ids_coexist_with_later_generated_document_ids():
    custom_id = "00000000-0000-0000-0000-000000000000"
    embedding = CountingEmbeddings(
        {
            "custom": [1.0, 0.0],
            "generated-a": [0.0, 1.0],
            "generated-b": [1.0, 1.0],
            "query": [0.0, 0.0],
        }
    )
    store = VSAGVectorStore.from_texts(
        ["custom"], embedding, ids=[custom_id], index_type="brute_force"
    )

    generated_ids = store.add_documents(
        [Document(page_content="generated-a"), Document(page_content="generated-b")]
    )

    assert len(generated_ids) == 2
    assert all(isinstance(doc_id, str) and doc_id for doc_id in generated_ids)
    assert len(set(generated_ids)) == 2
    assert custom_id not in generated_ids
    expected = [
        Document(id=custom_id, page_content="custom", metadata={}),
        Document(id=generated_ids[0], page_content="generated-a", metadata={}),
        Document(id=generated_ids[1], page_content="generated-b", metadata={}),
    ]
    assert store.get_by_ids([custom_id, *generated_ids]) == expected
    assert {doc.id for doc in store.similarity_search("query", k=3)} == {
        custom_id,
        *generated_ids,
    }
