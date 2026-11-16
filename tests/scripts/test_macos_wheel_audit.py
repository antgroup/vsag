#!/usr/bin/env python3
"""Portable audit regressions; no Mach-O tools or native execution required."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("audit", ROOT / "scripts/python/check_macos_wheel.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class MacOSWheelAuditTest(unittest.TestCase):
    def check(self, *, extra=None, dependency="/usr/lib/libSystem.B.dylib", minos="14.0",
              symbol="_vsag_init\n_vsag_destroy", license_expression=audit.LICENSE_EXPRESSION, omit_license=None, platform="1", magic="cffaedfe"):
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "pyvsag-1.0-py3-none-macosx_14_0_arm64.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr(audit.LIBRARY, bytes.fromhex(magic))
                archive.writestr("pyvsag/_vsag_c_api.h", "/* vsag_comment_only(); */\nvoid vsag_init(void);\nvoid vsag_destroy(void);\n")
                if extra:
                    archive.writestr(extra, bytes.fromhex("cffaedfe"))
                metadata = "License-Expression: " + license_expression + "\n"
                for name in ("LICENSE", "NOTICE", "licenses/LLVM-OpenMP.txt"):
                    metadata += f"License-File: {name}\n"
                    if name != omit_license:
                        archive.writestr("pyvsag-1.0.dist-info/licenses/" + name, "license text")
                archive.writestr("pyvsag-1.0.dist-info/METADATA", metadata)

            def run(args):
                if args[:2] == ["otool", "-L"]:
                    return ("libvsag_c.dylib:\n\t@rpath/libvsag_c.dylib (compatibility version 0.0.0)\n"
                            f"\t{dependency} (compatibility version 1.0.0)\n")
                if args[:2] == ["otool", "-l"]:
                    return f"cmd LC_BUILD_VERSION\n platform {platform}\n minos {minos}\n sdk 26.0\n"
                self.assertEqual(args[:2], ["nm", "-gUj"])
                return symbol + "\n"

            return audit.audit_wheel(wheel, run)

    def test_c_api_prototype_with_function_pointer_parameter(self):
        self.assertEqual(audit.required_c_api_symbols(
            "void vsag_set_callback(void (*callback)(void *), void *context);"),
            {"_vsag_set_callback"})

    def test_accepts_private_runtime_with_system_dependencies(self):
        self.assertIn("passed", self.check())

    def test_rejects_bundled_or_disguised_runtime(self):
        for member in ("pyvsag/.dylibs/libomp.dylib", "pyvsag/hidden-runtime.bin"):
            with self.subTest(member=member), self.assertRaisesRegex(ValueError, "native members"):
                self.check(extra=member)

    def test_rejects_external_and_loader_relative_dependencies(self):
        for name in ("/opt/homebrew/opt/libomp/lib/libomp.dylib", "@loader_path/libomp.dylib",
                     "@rpath/libopenblas.dylib"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "dependency"):
                self.check(dependency=name)

    def test_rejects_deployment_newer_than_tag(self):
        with self.assertRaisesRegex(ValueError, "exceeds wheel tag"):
            self.check(minos="26.0")

    def test_rejects_missing_deployment(self):
        with self.assertRaisesRegex(ValueError, "deployment target"):
            self.check(minos="unknown")

    def test_legacy_deployment_command(self):
        self.assertEqual(audit.deployment_versions(
            "cmd LC_VERSION_MIN_MACOSX\n cmdsize 16\n version 14.0\n sdk 26.0"), [(14, 0, 0)])

    def test_rejects_openmp_exports(self):
        with self.assertRaisesRegex(ValueError, "export only"):
            self.check(symbol="___kmpc_fork_call")

    def test_rejects_missing_or_extra_c_api_symbol(self):
        for symbol in ("_vsag_init", "_vsag_init\n_vsag_destroy\n_vsag_other"):
            with self.subTest(symbol=symbol), self.assertRaisesRegex(ValueError, "export mismatch"):
                self.check(symbol=symbol)

    def test_rejects_fat_binary_even_with_only_c_api_exports(self):
        with self.assertRaisesRegex(ValueError, "thin 64-bit"):
            self.check(magic="cafebabe")

    def test_rejects_non_macos_platform(self):
        for platform in ("2", "7", "IOS", "IOSSIMULATOR", "unknown"):
            with self.subTest(platform=platform), self.assertRaisesRegex(ValueError, "platform"):
                self.check(platform=platform)
        self.assertIn("passed", self.check(platform="MACOS"))

    def test_rejects_missing_license_metadata(self):
        with self.assertRaisesRegex(ValueError, "LLVM-exception"):
            self.check(license_expression="Apache-2.0")

    def test_rejects_missing_license_file(self):
        with self.assertRaises(KeyError):
            self.check(omit_license="licenses/LLVM-OpenMP.txt")


if __name__ == "__main__":
    unittest.main()
