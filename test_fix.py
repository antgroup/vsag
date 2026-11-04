# Test the sparse.py functions directly without importing the C extension
import math
import operator
import struct
from collections.abc import Mapping as MappingABC
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np

# Constants from sparse.py
_UINT32_MAX = (1 << 32) - 1
_MAX_TERM_COUNT = 1 << 32
_FLOAT32_MAX = float(np.finfo(np.float32).max)

def _validate_term_id(term: str, value: Any) -> int:
    '''Return ``value`` as an int, rejecting IDs outside the uint32 range.'''
    if isinstance(value, bool):
        raise TypeError(
            f'vocabulary ID for term {term!r} must be an integer, not bool'
        )
    try:
        term_id = operator.index(value)
    except TypeError as exc:
        raise TypeError(
            f'vocabulary ID for term {term!r} must be an integer, got '
            f'{type(value).__name__}'
        ) from exc
    if term_id < 0 or term_id > _UINT32_MAX:
        raise ValueError(
            f'vocabulary ID for term {term!r} is {term_id}, outside the '
            f'uint32 range [0, {_UINT32_MAX}]'
        )
    return term_id

def _validate_weight(term: str, value: Any) -> float:
    '''Return ``value`` as a finite float that fits in float32.'''
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(
            f'weight for term {term!r} must be numeric, got '
            f'{type(value).__name__}'
        )
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f'weight for term {term!r} must be numeric, got '
            f'{type(value).__name__}'
        ) from exc
    except OverflowError as exc:
        raise ValueError(
            f'weight for term {term!r} does not fit in float32: {value!r}'
        ) from exc
    if not math.isfinite(weight):
        raise ValueError(
            f'weight for term {term!r} must be finite, got {weight!r}'
        )
    if weight < -_FLOAT32_MAX or weight > _FLOAT32_MAX:
        raise ValueError(
            f'weight for term {term!r} does not fit in float32: {weight!r}'
        )
    return weight

def build_term_vocabulary(documents):
    '''Assign a stable ``uint32``-compatible ID to every term.'''
    terms = set()
    for position, document in enumerate(documents):
        if not isinstance(document, MappingABC):
            raise TypeError(
                f'documents[{position}] must be a mapping of str to numeric '
                f'weight, got {type(document).__name__}'
            )
        for term in document:
            if not isinstance(term, str):
                raise TypeError(
                    f'documents[{position}] contains a non-string term '
                    f'{term!r} ({type(term).__name__})'
                )
            terms.add(term)
    if len(terms) > _MAX_TERM_COUNT:
        raise ValueError(
            f'vocabulary has {len(terms)} terms, exceeding the uint32 term '
            f'ID space (maximum {_MAX_TERM_COUNT})'
        )
    return {term: index for index, term in enumerate(sorted(terms))}

def term_weights_to_csr(documents, vocabulary):
    '''Convert term-weight mappings into pyvsag sparse CSR arrays.'''
    if not isinstance(vocabulary, MappingABC):
        raise TypeError(
            'vocabulary must be a mapping of str to uint32-compatible integer, '
            f'got {type(vocabulary).__name__}'
        )
    
    # Validate that vocabulary IDs are unique
    seen_ids = set()
    id_to_term = {}
    for term, vid in vocabulary.items():
        if vid in seen_ids:
            raise ValueError(
                f'vocabulary has duplicate term ID {vid} for terms '
                f'{term!r} and {id_to_term[vid]!r}'
            )
        seen_ids.add(vid)
        id_to_term[vid] = term
    
    index_pointers = [0]
    indices: List[int] = []
    values: List[float] = []
    resolved_ids: Dict[str, int] = {}
    
    for position, document in enumerate(documents):
        if not isinstance(document, MappingABC):
            raise TypeError(
                f'documents[{position}] must be a mapping of str to numeric '
                f'weight, got {type(document).__name__}'
            )
 
        # Reject empty documents - SINDI::add skips empty documents
        # and SINDI::KnnSearch rejects empty queries
        if len(document) == 0:
            raise ValueError(
                f"documents[{position}] is empty; empty documents are not "
                f"supported by SINDI (they are silently dropped during "
                f"add and rejected by knn_search)"
            )
 
        row = []
        for term, weight in document.items():
            if not isinstance(term, str):
                raise TypeError(
                    f"documents[{position}] contains a non-string term "
                    f"{term!r} ({type(term).__name__})"
                )
            if term not in vocabulary:
                raise KeyError(
                    f'term {term!r} in documents[{position}] is missing from '
                    'the vocabulary'
                )
            term_id = resolved_ids.get(term)
            if term_id is None:
                term_id = _validate_term_id(term, vocabulary[term])
                resolved_ids[term] = term_id
            row.append((term_id, term, _validate_weight(term, weight)))
 
        # Sort by ID so rows are deterministic; the term breaks ties if a
        # caller passes a vocabulary with duplicate IDs.
        row.sort(key=lambda item: (item[0], item[1]))
        for term_id, _, weight in row:
            indices.append(term_id)
            values.append(weight)
        index_pointers.append(len(indices))
 
    if len(indices) > _UINT32_MAX:
        raise ValueError(
            f'CSR has {len(indices)} stored terms, exceeding the uint32 '
            f'offset space (maximum {_UINT32_MAX})'
        )
 
    return (
        np.asarray(index_pointers, dtype=np.uint32),
        np.asarray(indices, dtype=np.uint32),
        np.asarray(values, dtype=np.float32),
    )

# Test 1: build_term_vocabulary basic
documents = [{'zebra': 1.0, 'apple': 0.5}, {'mango': 2.0, 'apple': 1.0}]
vocabulary = build_term_vocabulary(documents)
assert vocabulary == {'apple': 0, 'mango': 1, 'zebra': 2}, f'Got {vocabulary}'
print('Test 1 passed: build_term_vocabulary is lexicographic and starts at zero')

# Test 2: duplicate vocab IDs raise ValueError
try:
    vocab = {'a': 0, 'b': 0}
    term_weights_to_csr([{'a': 1.0, 'b': 2.0}], vocab)
    print('Test 2 FAILED: should have raised ValueError for duplicate vocab IDs')
except ValueError as e:
    print(f'Test 2 passed: duplicate vocab IDs raise ValueError: {e}')

# Test 3: empty documents raise ValueError
try:
    vocab = {'a': 0, 'b': 1}
    term_weights_to_csr([{}, {'a': 1.0}], vocab)
    print('Test 3 FAILED: should have raised ValueError for empty document')
except ValueError as e:
    print(f'Test 3 passed: empty documents raise ValueError: {e}')

# Test 4: normal operation
documents = [{'search': 0.5, 'vector': 1.0}, {'index': 0.75}]
vocabulary = build_term_vocabulary(documents)
assert vocabulary == {'index': 0, 'search': 1, 'vector': 2}, f'Got {vocabulary}'
index_pointers, indices, values = term_weights_to_csr(documents, vocabulary)
assert index_pointers.tolist() == [0, 2, 3], f'Got {index_pointers.tolist()}'
assert indices.tolist() == [1, 2, 0], f'Got {indices.tolist()}'
np.testing.assert_allclose(values, [0.5, 1.0, 0.75])
print('Test 4 passed: normal operation works correctly')

# Test 5: generator documents
vocabulary = {'a': 0, 'b': 1}
documents = (document for document in [{'b': 2.0}, {'a': 1.0}])
index_pointers, indices, values = term_weights_to_csr(documents, vocabulary)
assert index_pointers.tolist() == [0, 1, 2], f'Got {index_pointers.tolist()}'
assert indices.tolist() == [1, 0], f'Got {indices.tolist()}'
np.testing.assert_allclose(values, [2.0, 1.0])
print('Test 5 passed: generator documents work')

# Test 6: empty input
index_pointers, indices, values = term_weights_to_csr([], {})
assert index_pointers.tolist() == [0]
assert indices.shape == values.shape == (0,)
assert indices.dtype == np.uint32
assert values.dtype == np.float32
print('Test 6 passed: empty input works')

# Test 7: maximum uint32 term id
vocabulary = {'max': 2**32 - 1}
_, indices, _ = term_weights_to_csr([{'max': 1.0}], vocabulary)
assert indices.tolist() == [2**32 - 1]
print('Test 7 passed: maximum uint32 term id')

# Test 8: numpy integer term ids
vocabulary = {'term': np.uint32(7)}
_, indices, _ = term_weights_to_csr([{'term': 1.0}], vocabulary)
assert indices.tolist() == [7]
print('Test 8 passed: numpy integer term ids')

print()
print('All manual tests passed!')