#!/usr/bin/env python3

import glob
import importlib.util
from pathlib import Path
import sysconfig
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock
import zipfile


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "check_wheel_libs", ROOT / "scripts/python/check_wheel_libs.py"
)
check_wheel_libs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_wheel_libs)


def make_wheel(directory, names):
    path = Path(directory) / "pyvsag-0-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as wheel:
        for name in names:
            wheel.writestr(name, b"")
    return str(path)


def fake_readelf(returncode=0):
    """Pretends readelf exists and reports no NEEDED entries, for wheels with placeholder members."""
    result = mock.Mock(returncode=returncode, stdout="")
    return (
        mock.patch.object(check_wheel_libs.shutil, "which", return_value="readelf"),
        mock.patch.object(check_wheel_libs.subprocess, "run", return_value=result),
    )


class CheckWheelLibsTest(unittest.TestCase):
    def test_clean_wheel_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel = make_wheel(tmp, ["pyvsag/__init__.py", "pyvsag.libs/libgomp-abc.so.1.0.0"])
            which, run = fake_readelf()
            with which, run:
                self.assertEqual(check_wheel_libs.check_wheel(wheel), [])

    def test_unreadable_or_unverifiable_members_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel = make_wheel(tmp, ["pyvsag/_ext.so"])
            which, run = fake_readelf(returncode=1)
            with which, run:
                self.assertEqual(len(check_wheel_libs.check_wheel(wheel)), 1)
            with mock.patch.object(check_wheel_libs.shutil, "which", return_value=None):
                self.assertEqual(len(check_wheel_libs.check_wheel(wheel)), 1)

    def test_bundled_forbidden_libraries_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel = make_wheel(
                tmp,
                [
                    "pyvsag.libs/libgfortran-1234.so.5.0.0",
                    "pyvsag.libs/libquadmath-1234.so.0.0.0",
                    "pyvsag.libs/libaio-1234.so.1.0.1",
                ],
            )
            which, run = fake_readelf()
            with which, run:
                self.assertEqual(len(check_wheel_libs.check_wheel(wheel)), 3)

    def test_needed_libraries_are_parsed_from_readelf_output(self):
        out = (
            " 0x0000000000000001 (NEEDED)             Shared library: [libgfortran.so.5]\n"
            " 0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]\n"
        )
        needed = check_wheel_libs.needed_libraries(out)
        self.assertEqual(needed, ["libgfortran.so.5", "libc.so.6"])
        self.assertEqual([n for n in needed if check_wheel_libs.is_forbidden(n)], ["libgfortran.so.5"])

    def test_needed_entries_of_a_real_elf_are_reported(self):
        readelf = shutil.which("readelf")
        candidates = sorted(glob.glob(f"{sysconfig.get_path('stdlib')}/lib-dynload/*.so"))
        target = needed = None
        for so in candidates if readelf else []:
            out = subprocess.run([readelf, "-d", so], capture_output=True, text=True).stdout
            libs = check_wheel_libs.needed_libraries(out)
            if libs:
                target, needed = so, libs[0]
                break
        if target is None:
            self.skipTest("no readelf or stdlib extension module with NEEDED entries")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pyvsag-0-py3-none-any.whl"
            with zipfile.ZipFile(path, "w") as wheel:
                wheel.write(target, "pyvsag/_ext.so")
            original = check_wheel_libs.FORBIDDEN
            check_wheel_libs.FORBIDDEN = (needed,)
            try:
                problems = check_wheel_libs.check_wheel(str(path))
            finally:
                check_wheel_libs.FORBIDDEN = original
        self.assertTrue(any(f"needs {needed}" in p for p in problems), problems)


if __name__ == "__main__":
    unittest.main()
