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

"""Command line entry point: ``pyvsag-eval job.yaml`` or ``python -m pyvsag.eval job.yaml``."""

import argparse
import sys

from .job import run_job


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="pyvsag-eval",
        description="Run an eval_performance YAML job with the native VSAG measurement core.",
    )
    parser.add_argument("job", help="path of the YAML job file (see tools/eval/eval_template.yaml)")
    args = parser.parse_args(argv)
    try:
        run_job(args.job, export=True)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        sys.stderr.write("pyvsag-eval: %s\n" % error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
