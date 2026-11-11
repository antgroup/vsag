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

"""pyvsag shares a process with other OpenMP users (torch, scikit-learn).

Each of them ships or loads its own OpenMP runtime. Every import order runs in a fresh
interpreter, because the runtime that gets loaded first can decide whether a later one conflicts.
"""

import itertools
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

LIBRARIES = ("pyvsag", "torch", "sklearn")

WORKLOAD = textwrap.dedent(
    """
    import json
    import threading

    import numpy as np
    import pyvsag
    import sklearn.cluster
    import torch

    rng = np.random.default_rng(7)
    dim, count = 32, 2000
    vectors = rng.random((count, dim), dtype=np.float32)
    index = pyvsag.Index(
        "brute_force",
        json.dumps(
            {
                "dtype": "float32",
                "metric_type": "l2",
                "dim": dim,
                "index_param": {
                    "base_quantization_type": "fp32",
                    "precise_quantization_type": "fp32",
                },
            }
        ),
    )
    index.build(vectors, np.arange(count, dtype=np.int64), count, dim)

    failures = []

    def search():
        for i in range(0, count, 40):
            ids, _ = index.knn_search(vectors[i], 1, "{}")
            if ids[0] != i:
                failures.append(i)

    worker = threading.Thread(target=search)
    worker.start()
    torch.set_num_threads(4)
    for _ in range(5):
        product = torch.randn(256, 256) @ torch.randn(256, 256)
        assert torch.isfinite(product).all()
    clusters = sklearn.cluster.KMeans(n_clusters=4, n_init=2, random_state=0).fit(vectors[:, :8])
    assert len(set(clusters.labels_)) == 4
    worker.join()
    search()
    assert not failures, failures
    print("OPENMP_COEXISTENCE_OK")
    """
)


@pytest.mark.parametrize("order", list(itertools.permutations(LIBRARIES)), ids="-".join)
def test_shares_a_process_with_other_openmp_runtimes(order):
    imports = "\n".join("import " + name for name in order)
    result = subprocess.run(
        [sys.executable, "-c", imports + "\n" + WORKLOAD],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OPENMP_COEXISTENCE_OK" in result.stdout.splitlines()
