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

"""Tests for deterministic sampling helpers."""

from __future__ import annotations

import numpy as np
import pytest

from dataset_profiler.sampling import make_rng, sample_indices, take_prefix_sample


def test_sample_indices_is_deterministic_for_a_seed():
    first = sample_indices(1000, 37, make_rng(42))
    second = sample_indices(1000, 37, make_rng(42))
    assert np.array_equal(first, second)


def test_sample_indices_changes_with_seed():
    first = sample_indices(10000, 64, make_rng(1))
    second = sample_indices(10000, 64, make_rng(2))
    assert not np.array_equal(first, second)


def test_sample_indices_are_distinct_and_sorted():
    sample = sample_indices(500, 50, make_rng(7))
    assert sample.size == 50
    assert len(set(sample.tolist())) == 50
    assert np.all(sample[1:] > sample[:-1])
    assert int(sample.min()) >= 0
    assert int(sample.max()) < 500


def test_sample_indices_returns_everything_when_k_ge_n():
    sample = sample_indices(10, 10, make_rng(3))
    assert sample.tolist() == list(range(10))
    sample = sample_indices(4, 100, make_rng(3))
    assert sample.tolist() == list(range(4))


def test_sample_indices_rejects_negative_arguments():
    with pytest.raises(ValueError):
        sample_indices(-1, 5, make_rng(0))
    with pytest.raises(ValueError):
        sample_indices(10, -1, make_rng(0))


def test_take_prefix_sample_bounds_and_validates():
    sample = np.arange(20)
    assert take_prefix_sample(sample, 5).tolist() == [0, 1, 2, 3, 4]
    assert take_prefix_sample(sample, None).tolist() == list(range(20))
    with pytest.raises(ValueError):
        take_prefix_sample(sample, 0)


def test_make_rng_does_not_touch_global_state():
    np.random.seed(12345)
    rng = make_rng(99)
    values = rng.integers(0, 1_000_000, size=4)
    np.random.seed(12345)
    control = np.random.randint(0, 1_000_000, size=4)
    assert not np.array_equal(values, control)
