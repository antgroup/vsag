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

"""Command-line interface for the VSAG dataset profiler.

The deterministic JSON and HTML reports are always produced offline.  AI
interpretation is opt-in through ``--ai`` (or ``VSAG_DATASET_PROFILER_AI=1``)
and configured exclusively through environment variables so API keys never
appear in arguments, reports, or logs.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from . import __version__
from .ai import analyze_report
from .hdf5 import DEFAULT_CHUNK_BYTES, DatasetFormatError
from .html_report import render_html
from .profiler import VALID_METRICS, profile_dataset
from .stats import dump_json

_TRUE_VALUES = ("1", "true", "yes", "on")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dataset_profiler",
        description=(
            "Profile and visualize the feature characteristics of a "
            "VSAG-compatible HDF5 dataset (dense or sparse)."
        ),
    )
    parser.add_argument("input", help="path to the .hdf5 dataset file")
    parser.add_argument("--split", default="train", help="dataset split to profile (default: train)")
    parser.add_argument("--seed", type=int, default=42, help="explicit random seed (default: 42)")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1024,
        help="max vectors drawn for sampled metrics (default: 1024)",
    )
    parser.add_argument(
        "--pairwise-sample-size",
        type=int,
        default=512,
        help="max vectors used for the bounded pairwise-distance matrix (default: 512)",
    )
    parser.add_argument(
        "--metric",
        choices=VALID_METRICS,
        default=None,
        help="distance metric for pairwise statistics (default: file 'distance' attribute)",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=0,
        help="optional k-means cluster count on the PCA-2 projection (0 disables)",
    )
    parser.add_argument(
        "--near-duplicate-epsilon",
        type=float,
        default=1e-3,
        help="relative pairwise-distance threshold for near-duplicate indicators (default: 1e-3)",
    )
    parser.add_argument(
        "--max-chunk-bytes",
        type=int,
        default=DEFAULT_CHUNK_BYTES,
        help="streaming read chunk size for sparse records (default: 64 MiB)",
    )
    parser.add_argument(
        "--max-dimension-points",
        type=int,
        default=1024,
        help="max per-dimension points stored per series (default: 1024)",
    )
    parser.add_argument(
        "--out-json",
        metavar="PATH",
        default=None,
        help="write the JSON report here; pass '-' to print to stdout (default: <input>.profile.json)",
    )
    parser.add_argument(
        "--out-html",
        metavar="PATH",
        default=None,
        help="write the self-contained HTML report here (default: <input>.profile.html)",
    )
    parser.add_argument(
        "--ai",
        action="store_true",
        help="enable optional AI-assisted interpretation (also enabled by VSAG_DATASET_PROFILER_AI=1)",
    )
    parser.add_argument("--ai-model", default=None, help="AI model name (default: $VSAG_AI_MODEL)")
    parser.add_argument(
        "--ai-base-url", default=None, help="OpenAI-compatible base URL (default: $VSAG_AI_BASE_URL)"
    )
    parser.add_argument(
        "--ai-key-env",
        default=None,
        help="name of the environment variable holding the API key (default: $VSAG_AI_KEY_ENV or VSAG_AI_API_KEY)",
    )
    parser.add_argument(
        "--ai-timeout", type=float, default=60.0, help="AI request timeout in seconds (default: 60)"
    )
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)
    return parser


def _ai_requested(args: argparse.Namespace, env: Dict[str, str]) -> bool:
    if args.ai:
        return True
    return env.get("VSAG_DATASET_PROFILER_AI", "").strip().lower() in _TRUE_VALUES


def _default_output_path(input_path: str, suffix: str) -> Path:
    return Path(input_path).with_name(Path(input_path).stem + ".profile." + suffix)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(argv: Optional[List[str]] = None, env: Optional[Dict[str, str]] = None) -> int:
    env = os.environ if env is None else env
    args = build_parser().parse_args(argv)

    try:
        report = profile_dataset(
            args.input,
            split=args.split,
            seed=args.seed,
            sample_size=args.sample_size,
            pairwise_sample_size=args.pairwise_sample_size,
            metric=args.metric,
            clusters=args.clusters,
            near_duplicate_relative_epsilon=args.near_duplicate_epsilon,
            max_chunk_bytes=args.max_chunk_bytes,
            max_dimension_points=args.max_dimension_points,
        )
    except (DatasetFormatError, ValueError, OSError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1

    if _ai_requested(args, env):
        analysis = analyze_report(
            report,
            model=args.ai_model,
            base_url=args.ai_base_url,
            key_env=args.ai_key_env,
            timeout=args.ai_timeout,
            env=env,
        )
        report["ai_analysis"] = analysis
        if analysis.get("status") != "ok":
            print(
                "warning: AI analysis failed (%s); the deterministic JSON and HTML "
                "reports were still generated" % analysis.get("error", "unknown error"),
                file=sys.stderr,
            )

    for warning in report.get("warnings", []):
        print("warning: %s" % warning, file=sys.stderr)

    json_text = dump_json(report)
    if args.out_json == "-":
        sys.stdout.write(json_text)
    else:
        json_path = Path(args.out_json) if args.out_json else _default_output_path(args.input, "json")
        _write_text(json_path, json_text)
        print("wrote %s" % json_path, file=sys.stderr)

    if args.out_html:
        html_path = Path(args.out_html)
    elif args.out_json is None:
        html_path = _default_output_path(args.input, "html")
    else:
        html_path = None
    if html_path is not None:
        _write_text(html_path, render_html(report))
        print("wrote %s" % html_path, file=sys.stderr)

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
