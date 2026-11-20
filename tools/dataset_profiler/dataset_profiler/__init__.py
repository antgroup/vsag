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

"""Deterministic profiling and visualization for VSAG-compatible HDF5 datasets.

The public entry point is :func:`profile_dataset` in :mod:`dataset_profiler.profiler`;
the command-line wrapper lives in :mod:`dataset_profiler.cli`.
"""

from __future__ import annotations

TOOL_NAME = "vsag-dataset-profiler"
__version__ = "0.1.0"
SCHEMA_VERSION = "1.0"

from .profiler import profile_dataset  # noqa: E402  (after constants: report.py imports them back)
from .report import build_report  # noqa: E402

__all__ = ["SCHEMA_VERSION", "TOOL_NAME", "__version__", "build_report", "profile_dataset"]
