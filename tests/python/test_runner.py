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

"""Regressions for full test discovery and failure propagation."""

import os
import subprocess
import sys

import pytest
import run_test


def test_runner_discovers_directory_and_preserves_status(monkeypatch):
    calls = []

    def fake_main(args):
        calls.append(args)
        return 7

    monkeypatch.setattr(run_test.pytest, "main", fake_main)
    assert run_test.run() == 7
    assert calls == [["-v", os.path.dirname(os.path.abspath(run_test.__file__))]]


@pytest.mark.parametrize("exit_code", [0, 7])
def test_script_exit_status(tmpdir, exit_code):
    # Replace only pytest in a child process; no pyvsag or real suite is needed.
    tmpdir.join("pytest.py").write(
        "def main(args):\n    return {}\n".format(exit_code)
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmpdir)
    result = subprocess.call([sys.executable, run_test.__file__], cwd=str(tmpdir), env=env)
    assert result == exit_code


@pytest.mark.parametrize("exit_code", [0, 7])
def test_local_build_wheel_preserves_pytest_status_and_cleanup(tmpdir, exit_code):
    # Execute the real script, but confine every build/install/cleanup operation
    # to a disposable repository with fake tool commands. Never use the real
    # source tree as cwd: the script intentionally removes generated files.
    script = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "scripts", "python", "local_build_wheel.sh"
    ))
    repo = tmpdir.mkdir("repo")
    tests_dir = repo.ensure("tests", "python", dir=True)
    prepare = repo.ensure("scripts", "python", "prepare_python_build.sh")
    prepare.write("#!/bin/bash\nexit 0\n")
    fake_bin = tmpdir.mkdir("bin")
    stub = tmpdir.join("fake_python.py")
    stub.write("""import os
import sys
from pathlib import Path

root = Path(os.environ["TEST_REPO"])
args = sys.argv[1:]
if args[:1] == ["-c"]:
    sys.exit(0)
if args[:2] == ["-m", "build"]:
    assert Path.cwd() == root
    (root / "wheelhouse" / "pyvsag-test.whl").write_text("fake wheel")
    sys.exit(0)
if args[:2] == ["-m", "pytest"]:
    assert Path.cwd() == root / "tests" / "python"
    assert args[2:] == [".", "-v", "--tb=short"]
    # Create artifacts after the script's pre-build cleanup, so only the
    # post-test cleanup can remove them. Nested canaries must remain untouched.
    for base in [root, Path.cwd()]:
        (base / "python" / "build").mkdir(parents=True)
        (base / "python" / "pyvsag").mkdir()
        (base / "python" / "pyvsag" / "_version.py").write_text("canary")
        (base / "python" / "artifact.so").write_text("canary")
        (base / "python" / "pyvsag" / "artifact.so").write_text("canary")
    print("fake pytest stdout")
    print("fake pytest stderr", file=sys.stderr)
    sys.exit(int(os.environ["TEST_PYTEST_EXIT"]))
raise AssertionError("unexpected Python invocation: " + repr(args))
""")
    for name in ["python", "python3"]:
        command = fake_bin.join(name)
        command.write('#!/bin/sh\nexec "$TEST_PYTHON" "$TEST_PYTHON_STUB" "$@"\n')
        command.chmod(0o755)
    for name, code in [("pip", 0), ("docker", 1)]:
        command = fake_bin.join(name)
        command.write("#!/bin/sh\nexit {}\n".format(code))
        command.chmod(0o755)
    env = os.environ.copy()
    env.update({
        "PATH": str(fake_bin) + os.pathsep + env.get("PATH", ""),
        "VIRTUAL_ENV": str(tmpdir.join("fake-venv")),
        "CONDA_PREFIX": "",
        "TEST_PYTHON": sys.executable,
        "TEST_PYTHON_STUB": str(stub),
        "TEST_REPO": str(repo),
        "TEST_PYTEST_EXIT": str(exit_code),
    })
    process = subprocess.Popen(
        ["bash", script, "3.10"], cwd=str(repo), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True
    )
    output, _ = process.communicate()
    assert process.returncode == exit_code, output
    log = tests_dir.join("test-output.log").read()
    for message in ["fake pytest stdout", "fake pytest stderr"]:
        assert message in log
        assert message in output
    assert ("Successfully built wheel" in output) == (exit_code == 0)
    assert not repo.join("python", "build").exists()
    for relative in ["pyvsag/_version.py", "artifact.so", "pyvsag/artifact.so"]:
        assert not repo.join("python", relative).exists()
        assert tests_dir.join("python", relative).exists()
    assert tests_dir.join("python", "build").isdir()
