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

"""The ``pyvsag-analyze-index`` command: print the JSON report of ``pyvsag.analyze_index()``.

The report is the only output on stdout; diagnostics and errors go to stderr. Exit status: 0
on success, 1 when the analysis fails, 2 for an invalid command line.

This module is installed next to the ``pyvsag`` package rather than inside it: importing
``pyvsag`` loads the native library, which logs its start-up banner to stdout, so stdout has to
be taken over before ``pyvsag`` is imported.
"""

import argparse
import io
import json
import os
import sys

PROG = "pyvsag-analyze-index"
EXIT_ANALYSIS_FAILED = 1
EXIT_USAGE = 2


def build_parser():
    """Return the parser of the command; the options match the C++ ``analyze_index`` tool."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        allow_abbrev=False,
        description="Analyze a serialized VSAG index file and print a JSON report to stdout. "
        "Diagnostics and errors go to stderr.",
        epilog="Exit status: 0 on success, 1 when the analysis fails, 2 for an invalid "
        "command line.",
    )

    def add(*names, **kwargs):
        # Accept the C++ tool spellings (--index_path) and dashed ones (--index-path).
        dashed = tuple(n.replace("_", "-") for n in names if n.startswith("--") and "_" in n)
        parser.add_argument(*(names + dashed), **kwargs)

    add("-i", "--index_path", help="serialized index file to analyze (required)")
    add(
        "-bp",
        "--build_parameter",
        default="default",
        help='build parameters (JSON) used to reload the index; "default" uses the parameters '
        "recorded in the file",
    )
    add("-qp", "--query_path", help="query dataset for dynamic analysis")
    add(
        "--query_data_type",
        default="auto",
        choices=("auto", "dense", "sparse"),
        help="query file layout (default: auto)",
    )
    add("--base_path", help="sindi only: sparse CSR base dataset")
    add("--groundtruth_path", help="sindi only: existing ground-truth file")
    add("--save_groundtruth_path", help="sindi only: where to save generated ground truth")
    add(
        "-sp",
        "--search_parameter",
        default="default",
        help='search parameters (JSON) for dynamic analysis; "default" uses index defaults',
    )
    add("-k", "--topk", type=int, default=100, help="neighbors evaluated per query (default: 100)")
    parser.add_argument("--compact", action="store_true", help="print the report on one line")
    parser.add_argument("--version", action="store_true", help="print the pyvsag version")
    return parser


def _take_over_stdout():
    """Point file descriptor 1 at stderr and return a text stream on the original stdout.

    Every later write to stdout, native or Python, then reaches stderr; only the report is
    written to the returned stream.
    """
    sys.stdout.flush()
    try:
        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, ValueError, io.UnsupportedOperation):
        # Replaced streams without descriptors (embedding, test capture) are left alone.
        return sys.stdout
    report_fd = os.dup(stdout_fd)
    os.dup2(stderr_fd, stdout_fd)
    return io.open(report_fd, "w", encoding="utf-8", closefd=True)


def _error(message):
    sys.stderr.write("{}: error {}\n".format(PROG, message))
    sys.stderr.flush()


def main(argv=None):
    """Run the command and return its exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.index_path is None and not args.version:
        parser.error("the following arguments are required: -i/--index_path")
    output = _take_over_stdout()
    try:
        import pyvsag
    except ImportError as error:
        _error("[import_error]: cannot import pyvsag: {}".format(error))
        return EXIT_ANALYSIS_FAILED
    if args.version:
        output.write("pyvsag {}\n".format(pyvsag.__version__))
        output.flush()
        return 0
    try:
        report = pyvsag.analyze_index(
            args.index_path,
            build_parameter=args.build_parameter,
            query_path=args.query_path,
            query_data_type=args.query_data_type,
            base_path=args.base_path,
            groundtruth_path=args.groundtruth_path,
            save_groundtruth_path=args.save_groundtruth_path,
            search_parameter=args.search_parameter,
            topk=args.topk,
        )
    except pyvsag.IndexAnalysisError as error:
        _error("[{}]: {}".format(error.code, error.message))
        return EXIT_ANALYSIS_FAILED
    except KeyboardInterrupt:
        return 130
    try:
        json.dump(report, output, indent=None if args.compact else 2)
        output.write("\n")
        output.flush()
    except BrokenPipeError:
        # The reader went away (for example `| head`); nothing else can be reported.
        return EXIT_ANALYSIS_FAILED
    return 0


if __name__ == "__main__":
    sys.exit(main())
