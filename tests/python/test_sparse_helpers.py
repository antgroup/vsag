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

"""Unit tests for the sparse term-weight helpers in ``pyvsag.sparse``."""

import numpy as np
import pytest

from pyvsag import sparse
from pyvsag.sparse import TermVocabulary, build_term_vocabulary, term_weights_to_csr


def test_build_term_vocabulary_is_lexicographic_and_starts_at_zero():
    documents = [
        {"zebra": 1.0, "apple": 0.5},
        {"mango": 2.0, "apple": 1.0},
    ]

    vocabulary = build_term_vocabulary(documents)

    assert vocabulary == {"apple": 0, "mango": 1, "zebra": 2}
    assert list(vocabulary) == ["apple", "mango", "zebra"]
    assert list(vocabulary.values()) == [0, 1, 2]


def test_build_term_vocabulary_is_insertion_and_document_order_independent():
    first = [
        {"b": 1.0, "a": 2.0},
        {"d": 1.0, "c": 1.0},
    ]
    second = [
        {"c": 1.0, "d": 1.0},
        {"a": 2.0, "b": 1.0},
    ]

    assert build_term_vocabulary(first) == build_term_vocabulary(second)


def test_build_term_vocabulary_is_deterministic_for_generators():
    documents = ({"term": float(index % 2)} for index in range(4))

    first = build_term_vocabulary(documents)
    second = build_term_vocabulary([{"term": 1.0}])

    assert first == second == {"term": 0}


def test_build_term_vocabulary_empty_input():
    assert build_term_vocabulary([]) == {}


def test_build_term_vocabulary_rejects_empty_documents():
    with pytest.raises(ValueError, match="empty"):
        build_term_vocabulary([{}, {"term": 1.0}])


def test_build_term_vocabulary_accepts_unicode_terms():
    vocabulary = build_term_vocabulary([{"向量": 1.0, "检索": 0.5}])

    assert vocabulary == {"向量": 0, "检索": 1}


def test_build_term_vocabulary_rejects_non_string_terms():
    with pytest.raises(TypeError, match="non-string term"):
        build_term_vocabulary([{1: 1.0}])


def test_build_term_vocabulary_rejects_non_mapping_documents():
    with pytest.raises(TypeError, match="must be a mapping"):
        build_term_vocabulary([["not", "a", "mapping"]])


def test_term_weights_to_csr_builds_expected_arrays():
    documents = [
        {"search": 0.5, "vector": 1.0},
        {"index": 0.75},
    ]
    vocabulary = build_term_vocabulary(documents)
    assert vocabulary == {"index": 0, "search": 1, "vector": 2}

    index_pointers, indices, values = term_weights_to_csr(documents, vocabulary)

    assert index_pointers.dtype == np.uint32
    assert indices.dtype == np.uint32
    assert values.dtype == np.float32
    assert index_pointers.tolist() == [0, 2, 3]
    assert indices.tolist() == [1, 2, 0]
    np.testing.assert_allclose(values, [0.5, 1.0, 0.75])
    assert index_pointers.shape == (len(documents) + 1,)
    assert indices.shape == values.shape == (3,)


def test_term_weights_to_csr_is_insertion_order_independent():
    vocabulary = build_term_vocabulary([{"a": 1.0, "b": 1.0, "c": 1.0}])
    first = term_weights_to_csr([{"c": 3.0, "a": 1.0, "b": 2.0}], vocabulary)
    second = term_weights_to_csr([{"a": 1.0, "b": 2.0, "c": 3.0}], vocabulary)

    for first_array, second_array in zip(first, second):
        np.testing.assert_array_equal(first_array, second_array)

    # Rows are ordered by term ID, not by insertion order.
    assert first[1].tolist() == [0, 1, 2]
    np.testing.assert_allclose(first[2], [1.0, 2.0, 3.0])


def test_term_weights_to_csr_empty_input():
    index_pointers, indices, values = term_weights_to_csr([], {})

    assert index_pointers.dtype == np.uint32
    assert index_pointers.tolist() == [0]
    assert indices.shape == values.shape == (0,)
    assert indices.dtype == np.uint32
    assert values.dtype == np.float32


@pytest.mark.parametrize("vocabulary_type", [dict, TermVocabulary])
def test_term_weights_to_csr_rejects_empty_documents(vocabulary_type):
    vocabulary = vocabulary_type({"term": 0, "other": 1})
    with pytest.raises(ValueError, match="empty"):
        term_weights_to_csr([{}, {"term": 1.0}], vocabulary)


def test_term_weights_to_csr_rejects_duplicate_vocab_ids():
    vocabulary = {"a": 0, "b": 0}
    with pytest.raises(ValueError, match="duplicate term ID"):
        term_weights_to_csr([{"a": 1.0, "b": 2.0}], vocabulary)


def test_term_weights_to_csr_rejects_duplicate_ids_across_types():
    # ``1.0 == 1`` hashes identically, so this is a duplicate even though
    # ``1.0`` alone would be rejected as a non-integer ID.
    with pytest.raises(ValueError, match="duplicate term ID"):
        term_weights_to_csr([{"a": 1.0}], {"a": 1, "b": 1.0})
    with pytest.raises(ValueError, match="duplicate term ID"):
        term_weights_to_csr([{"a": 1.0}], {"a": 1, "b": True})


class _IndexLike:
    """Integer-like object that hashes by identity (unlike ``int``)."""

    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


@pytest.mark.parametrize(
    "duplicate_id",
    [
        _IndexLike(0),
        np.array(0, dtype=np.int64),
        np.array(0, dtype=np.uint32),
    ],
)
def test_term_weights_to_csr_rejects_duplicate_ids_after_normalization(
    duplicate_id,
):
    # 0-d integer arrays are unhashable and ``__index__`` objects hash by
    # identity, so the raw-value lookup misses them. The IDs collide once
    # normalized, so both paths must still reject the vocabulary.
    vocabulary = {"a": 0, "b": duplicate_id}

    with pytest.raises(ValueError, match="duplicate term ID"):
        term_weights_to_csr([{"a": 1.0}], vocabulary)
    with pytest.raises(ValueError, match="duplicate term ID"):
        TermVocabulary(vocabulary)


def test_term_weights_to_csr_reports_invalid_ids_with_descriptive_error():
    # Duplicate detection must not swallow the descriptive TypeError for
    # unhashable / non-integer vocabulary IDs.
    with pytest.raises(TypeError, match="must be an integer, got list"):
        term_weights_to_csr([{"a": 1.0}], {"a": [1]})
    with pytest.raises(TypeError, match="must be an integer, got float"):
        term_weights_to_csr([{"a": 1.0}], {"a": 1.0})


def test_term_vocabulary_matches_plain_mapping():
    plain = {"a": 0, "b": 1}
    vocabulary = TermVocabulary(plain)

    assert vocabulary == plain
    assert plain == vocabulary
    assert vocabulary["b"] == 1
    assert list(vocabulary) == ["a", "b"]
    assert list(vocabulary.items()) == [("a", 0), ("b", 1)]
    assert len(vocabulary) == 2


def test_term_vocabulary_rejects_invalid_mappings():
    with pytest.raises(TypeError, match="must be a mapping"):
        TermVocabulary([("a", 0)])
    with pytest.raises(TypeError, match="non-string term"):
        TermVocabulary({1: 0})
    with pytest.raises(ValueError, match="duplicate term ID"):
        TermVocabulary({"a": 0, "b": 0})


def test_term_weights_to_csr_accepts_term_vocabulary():
    documents = [{"search": 0.5, "vector": 1.0}, {"index": 0.75}]
    plain = build_term_vocabulary(documents)

    first = term_weights_to_csr(documents, TermVocabulary(plain))
    second = term_weights_to_csr(documents, plain)

    for first_array, second_array in zip(first, second):
        np.testing.assert_array_equal(first_array, second_array)


def test_term_vocabulary_is_validated_once(monkeypatch):
    documents = [{"a": 1.0}, {"b": 2.0}]
    vocabulary = TermVocabulary(build_term_vocabulary(documents))

    calls = []
    original = sparse._validated_term_ids

    def counting_validator(mapping):
        calls.append(mapping)
        return original(mapping)

    monkeypatch.setattr(sparse, "_validated_term_ids", counting_validator)

    term_weights_to_csr(documents, vocabulary)
    term_weights_to_csr([{"a": 1.0}], vocabulary)
    assert calls == []

    # A raw mapping is still validated on every call.
    term_weights_to_csr([{"a": 1.0}], dict(vocabulary))
    assert len(calls) == 1


def test_term_weights_to_csr_accepts_generator_documents():
    vocabulary = {"a": 0, "b": 1}
    documents = (document for document in [{"b": 2.0}, {"a": 1.0}])

    index_pointers, indices, values = term_weights_to_csr(documents, vocabulary)

    assert index_pointers.tolist() == [0, 1, 2]
    assert indices.tolist() == [1, 0]
    np.testing.assert_allclose(values, [2.0, 1.0])


def test_term_weights_to_csr_accepts_maximum_uint32_term_id():
    vocabulary = {"max": 2**32 - 1}

    _, indices, _ = term_weights_to_csr([{"max": 1.0}], vocabulary)

    assert indices.dtype == np.uint32
    assert indices.tolist() == [2**32 - 1]


def test_term_weights_to_csr_accepts_numpy_integer_term_ids():
    vocabulary = {"term": np.uint32(7)}

    _, indices, _ = term_weights_to_csr([{"term": 1.0}], vocabulary)

    assert indices.tolist() == [7]


@pytest.mark.parametrize("term_id", [2**32, -(2**31), -1])
def test_term_weights_to_csr_rejects_out_of_range_term_ids(term_id):
    with pytest.raises(ValueError, match="uint32 range"):
        term_weights_to_csr([{"term": 1.0}], {"term": term_id})


@pytest.mark.parametrize("term_id", [1.0, "1", True])
def test_term_weights_to_csr_rejects_non_integer_term_ids(term_id):
    with pytest.raises(TypeError, match="must be an integer"):
        term_weights_to_csr([{"term": 1.0}], {"term": term_id})


def test_term_weights_to_csr_rejects_unknown_terms():
    with pytest.raises(KeyError, match="missing from the vocabulary"):
        term_weights_to_csr([{"unknown": 1.0}], {"known": 0})


@pytest.mark.parametrize(
    "weight", [float("nan"), float("inf"), float("-inf"), 1e300, 10**400]
)
def test_term_weights_to_csr_rejects_non_finite_or_overflowing_weights(weight):
    with pytest.raises(ValueError):
        term_weights_to_csr([{"term": weight}], {"term": 0})


@pytest.mark.parametrize("weight", ["1.0", None, object()])
def test_term_weights_to_csr_rejects_non_numeric_weights(weight):
    with pytest.raises(TypeError, match="must be numeric"):
        term_weights_to_csr([{"term": weight}], {"term": 0})


@pytest.mark.parametrize(
    "weight",
    [
        1 + 2j,
        np.complex64(1 + 2j),
        np.complex128(1 + 2j),
        np.complex128(complex(1.0, float("inf"))),
    ],
)
def test_term_weights_to_csr_rejects_complex_weights(weight):
    with pytest.raises(TypeError, match="must be numeric"):
        term_weights_to_csr([{"term": weight}], {"term": 0})
