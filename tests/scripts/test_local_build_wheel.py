#!/usr/bin/env python3
"""Exercise native wheel routing, repair and failure propagation without compiling VSAG."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/python/local_build_wheel.sh"
FAKE_TOOL = r'''
import json
import os
from pathlib import Path
import sys

name = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["COMMAND_LOG"], "a") as log:
    log.write(json.dumps([name, *args]) + "\n")
if name == "uname":
    print("Darwin" if args == ["-s"] else "arm64")
elif name == "python" and args[:2] == ["-m", "build"]:
    if os.environ.get("BUILD_FAIL"):
        sys.exit(13)
    out = Path(args[args.index("--outdir") + 1])
    (out / "pyvsag-1.0-py3-none-macosx_14_0_arm64.whl").write_text("unrepaired")
elif name == "delocate-wheel":
    if os.environ.get("REPAIR_FAIL"):
        sys.exit(9)
    out = Path(args[args.index("--wheel-dir") + 1])
    out.mkdir()
    (out / Path(args[-1]).name).write_text("repaired")
elif name == "python" and args[:1] == ["scripts/python/check_macos_wheel.py"]:
    assert Path(args[1]).read_text() == "repaired"
    sys.exit(int(os.environ.get("AUDIT_EXIT", "0")))
elif name == "pip" and args == ["install", "delocate"]:
    sys.exit(int(os.environ.get("DELOCATE_INSTALL_EXIT", "0")))
elif name == "pip" and args[:1] == ["install"] and args[1].endswith(".whl"):
    assert Path(args[1]).read_text() == "repaired", "installed an unrepaired or stale wheel"
elif name == "python" and args[:2] == ["-m", "pytest"]:
    sys.exit(int(os.environ.get("PYTEST_EXIT", "0")))
'''


class LocalWheelTest(unittest.TestCase):
    def run_build(self, *, stale_wheel=True, **extra_env):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for directory in ("bin", "scripts/python", "tests/python", "python/pyvsag"):
                (root / directory).mkdir(parents=True)
            (root / "scripts/python/prepare_python_build.sh").write_text(":\n")
            if stale_wheel:
                (root / "wheelhouse").mkdir()
                (root / "wheelhouse/pyvsag-9.0-py3-none-macosx_14_0_arm64.whl").write_text("stale")
            for tool in ("python", "python3", "pip", "docker", "uname", "delocate-wheel"):
                path = root / "bin" / tool
                path.write_text(f"#!{sys.executable}\n" + FAKE_TOOL)
                path.chmod(0o755)
            log = root / "commands.jsonl"
            env = dict(os.environ, PATH=f"{root / 'bin'}:{os.environ['PATH']}",
                       VIRTUAL_ENV=str(root / "venv"), COMMAND_LOG=str(log), **extra_env)
            result = subprocess.run(["bash", str(SCRIPT)], cwd=root, env=env,
                                    capture_output=True, text=True, timeout=60)
            commands = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertFalse(list((root / "wheelhouse").glob(".native-build.*")))
            return result, commands

    def test_macos_repairs_native_wheel_even_with_docker_available(self):
        result, commands = self.run_build()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(any(command[0] == "docker" for command in commands))
        self.assertTrue(any(command[:3] == ["delocate-wheel", "--require-archs", "arm64"]
                            for command in commands))
        self.assertTrue(any(command[:3] == ["python", "-m", "pytest"] for command in commands))

    def test_prerequisite_failure_prevents_build(self):
        result, commands = self.run_build(DELOCATE_INSTALL_EXIT="23")
        self.assertEqual(result.returncode, 23)
        self.assertFalse(any(command[:3] == ["python", "-m", "build"] for command in commands))

    def test_failure_immediately_after_mktemp_cleans_directory(self):
        result, commands = self.run_build(stale_wheel=False, BUILD_FAIL="1")
        self.assertEqual(result.returncode, 13)
        self.assertTrue(any(command[:3] == ["python", "-m", "build"] for command in commands))
        self.assertFalse(any(command[0] == "delocate-wheel" for command in commands))

    def test_audit_failure_prevents_installation(self):
        result, commands = self.run_build(AUDIT_EXIT="19")
        self.assertEqual(result.returncode, 19)
        self.assertFalse(any(command[0] == "pip" and any(arg.endswith(".whl") for arg in command)
                             for command in commands))

    def test_native_build_creates_wheelhouse(self):
        result, _ = self.run_build(stale_wheel=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_pytest_failure_survives_tee(self):
        result, _ = self.run_build(PYTEST_EXIT="17")
        self.assertEqual(result.returncode, 17, result.stdout + result.stderr)
        self.assertNotIn("Successfully built", result.stdout)

    def test_repair_failure_prevents_installation(self):
        result, commands = self.run_build(REPAIR_FAIL="1")
        self.assertEqual(result.returncode, 9, result.stdout + result.stderr)
        self.assertFalse(any(command[0] == "pip" and any(arg.endswith(".whl") for arg in command)
                             for command in commands))


if __name__ == "__main__":
    unittest.main()
