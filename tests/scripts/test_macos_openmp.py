#!/usr/bin/env python3
"""Check custom OpenMP diagnostics without downloading or compiling a runtime."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(shutil.which("cmake"), "requires cmake")
class MacOSOpenMPTest(unittest.TestCase):
    def configure(self, *, minos="15.0", missing=False, legacy=False, platform="1", extra_commands="", source=False, disable_perl=False, banner=""):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = root / "custom"
            (prefix / "include").mkdir(parents=True)
            (prefix / "lib").mkdir()
            if not missing:
                (prefix / "include/omp.h").touch()
                (prefix / "lib/libomp.a").touch()
            (root / "bin").mkdir()
            # The tool version must not be mistaken for the deployment version.
            commands = (f"cmd LC_VERSION_MIN_MACOSX\n cmdsize 16\n version {minos}\n sdk 26.0\n"
                        if legacy else f"cmd LC_BUILD_VERSION\n platform {platform}\n minos {minos}\n sdk 26.0\n"
                        " tool 3\n version 1234.5\n")
            commands = banner + commands + extra_commands
            otool = root / "bin/otool"
            otool.write_text(f"#!{sys.executable}\nprint({commands!r})\n")
            otool.chmod(0o755)
            (root / "CMakeLists.txt").write_text(f'''
cmake_minimum_required(VERSION 3.20)
project(openmp_configuration NONE)
set(CMAKE_OSX_DEPLOYMENT_TARGET 15.0)
set(LIBOMP_PREFIX "{'' if source else prefix}")
set(NUM_BUILDING_JOBS 2)
set(CMAKE_DISABLE_FIND_PACKAGE_Perl {'TRUE' if disable_perl else 'FALSE'})
add_library(vsag_openmp INTERFACE)
include("{ROOT}/cmake/VSAGMacOSOpenMP.cmake")
vsag_configure_private_openmp()
get_target_property(library vsag_openmp INTERFACE_LINK_LIBRARIES)
if({str(source).upper()})
  get_property(command TARGET vsag_private_openmp_build PROPERTY _EP_BUILD_COMMAND)
  if(NOT command MATCHES "--parallel;2$")
    message(FATAL_ERROR "Source build did not receive the configured parallelism")
  endif()
elseif(NOT library STREQUAL "{prefix}/lib/libomp.a")
  message(FATAL_ERROR "Custom prefix was ignored")
endif()
''')
            env = dict(os.environ, PATH=f"{root / 'bin'}:{os.environ['PATH']}")
            return subprocess.run(["cmake", "-S", str(root), "-B", str(root / "build")],
                                  env=env, capture_output=True, text=True, timeout=30)

    def test_source_build_receives_parallelism(self):
        result = self.configure(source=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_missing_perl_has_actionable_diagnostic(self):
        result = self.configure(source=True, disable_perl=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires Perl", result.stderr)
        self.assertIn("brew install perl", " ".join(result.stderr.split()))

    def test_custom_prefix_is_used(self):
        result = self.configure()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_named_macos_platform_is_accepted(self):
        result = self.configure(platform="MACOS")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_non_macos_platforms_are_rejected(self):
        for platform in ("2", "7", "IOS", "IOSSIMULATOR", "unknown"):
            with self.subTest(platform=platform):
                result = self.configure(platform=platform)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("non-macOS", result.stderr)

    def test_mixed_archive_platform_is_rejected(self):
        result = self.configure(extra_commands="Load command 0\ncmd LC_BUILD_VERSION\n platform 7\n minos 15.0\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("non-macOS", result.stderr)

    def test_archive_path_list_delimiters_are_ignored(self):
        for banner in ("/tmp/lib[;omp.a(member.o):\n", "/tmp/lib];omp.a(member.o):\n"):
            with self.subTest(banner=banner):
                result = self.configure(banner=banner)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_archive_path_delimiters_do_not_hide_mixed_platform(self):
        for banner in ("/tmp/lib[;omp.a(member.o):\n", "/tmp/lib];omp.a(member.o):\n"):
            with self.subTest(banner=banner):
                result = self.configure(banner=banner, extra_commands=
                                        "Load command 0\ncmd LC_BUILD_VERSION\n platform 7\n minos 15.0\n")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("non-macOS", result.stderr)

    def test_newer_archive_is_rejected(self):
        result = self.configure(minos="26.0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("newer than target 15.0", result.stderr)

    def test_missing_archive_has_actionable_diagnostic(self):
        result = self.configure(missing=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unset LIBOMP_PREFIX", " ".join(result.stderr.split()))
        self.assertIn("brew install libomp", result.stderr)

    def test_legacy_deployment_command(self):
        result = self.configure(legacy=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
