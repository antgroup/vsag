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

"""``pyvsag-analyze-index`` console entry point.

The command is a thin wrapper around :func:`pyvsag.analyze_index`. It writes the JSON report
to stdout, sends diagnostics and errors to stderr, and exits non-zero when analysis fails.
"""

import argparse
import json
import sys

from .analyzer import analyze_index

EXIT_SUCCESS = 0
EXIT_FAILURE = 1


def build_parser():
    parser = argparse.ArgumentParser(
        prog="pyvsag-analyze-index",
        description=(
            "Inspect a serialized VSAG index. Prints a JSON report to stdout; "
            "diagnostics and errors go to stderr."
        ),
    )
    parser.add_argument(
        "--index_path",
        "--index-path",
        "-i",
        required=True,
        help="Path to the serialized VSAG index file (required).",
    )
    parser.add_argument(
        "--build_parameter",
        "--build-parameter",
        "-bp",
        default="default",
        help=(
            "Build parameter JSON used to reload the index. Defaults to the parameters "
            "embedded in the index file."
        ),
    )
    parser.add_argument(
        "--query_path",
        "--query-path",
        "-qp",
        default=None,
        help="Binary query dataset path. Omit for static analysis only.",
    )
    parser.add_argument(
        "--query_data_type",
        "--query-data-type",
        default="auto",
        choices=["auto", "dense", "sparse"],
        help="Query dataset type (default: auto).",
    )
    parser.add_argument(
        "--base_path",
        "--base-path",
        default=None,
        help="Optional sparse CSR base dataset for SINDI analysis.",
    )
    parser.add_argument(
        "--groundtruth_path",
        "--groundtruth-path",
        default=None,
        help="Optional ground-truth file for dynamic analysis.",
    )
    parser.add_argument(
        "--save_groundtruth_path",
        "--save-groundtruth-path",
        default=None,
        help="Optional output path for generated ground truth.",
    )
    parser.add_argument(
        "--search_parameter",
        "--search-parameter",
        "-sp",
        default="default",
        help="Search parameter JSON used by dynamic analysis.",
    )
    parser.add_argument(
        "--topk",
        "-k",
        type=int,
        default=100,
        help="Top-K used by dynamic analysis (default: 100).",
    )
    return parser


def _silence_native_logs():
    """Keep the native library from writing INFO logs to stdout."""
    try:
        import pyvsag

        pyvsag.set_logger_off()
    except Exception:  # pragma: no cover - logging is best-effort
        pass


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    _silence_native_logs()

    try:
        report = analyze_index(
            index_path=args.index_path,
            build_parameter=args.build_parameter,
            query_path=args.query_path,
            query_data_type=args.query_data_type,
            base_path=args.base_path,
            groundtruth_path=args.groundtruth_path,
            save_groundtruth_path=args.save_groundtruth_path,
            search_parameter=args.search_parameter,
            topk=args.topk,
        )
    except (TypeError, ValueError, RuntimeError, OSError) as error:
        print("pyvsag-analyze-index: error: {}".format(error), file=sys.stderr)
        return EXIT_FAILURE

    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
