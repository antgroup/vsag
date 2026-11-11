#!/usr/bin/env python3

from pathlib import Path
import subprocess
import re
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/linters/check-struct-names.py"


class CheckStructNamesTest(unittest.TestCase):
    def run_checker(self, source: str, filename: str = "fixture.h") -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / filename
            fixture.parent.mkdir(parents=True, exist_ok=True)
            fixture.write_text(source)
            return subprocess.run(
                [str(CHECKER), "--root", directory, str(fixture)],
                text=True,
                capture_output=True,
                check=False,
            )

    def test_accepts_camel_case_and_ignores_non_definitions(self):
        result = self.run_checker(
            """
            // struct commented_out {};
            struct external_type;
            struct alignas(64) Http2Config {};
            template <typename T> struct VectorTraits<T*> {};
            #define DECLARE_TRAIT(Name) struct Has##Name {};
            DECLARE_TRAIT(ReadImpl)
            """
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_c_abi_exception_is_limited_to_declared_handles_and_files(self):
        names = re.findall(r"typedef struct (vsag_\w+) vsag_\w+_t;",
                           (ROOT / "include/vsag/vsag_c_api.h").read_text())
        self.assertTrue(names, "public C API opaque declarations must be discovered")
        self.assertEqual(len(names), len(set(names)), "opaque tags must be unique")
        source = "\n".join(f"struct {name} {{}};" for name in names)
        for filename in ("src/vsag_c_api.cpp", "include/vsag/vsag_c_api.h"):
            with self.subTest(filename=filename):
                result = self.run_checker(source, filename)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(self.run_checker("struct vsag_new {};", filename).returncode, 1)
                self.assertEqual(self.run_checker("struct internal_name {};", filename).returncode, 1)
        self.assertEqual(self.run_checker(source).returncode, 1)
        # clang-tidy must recognize the same finite set of public tags.
        config = (ROOT / ".clang-tidy").read_text()
        expression = re.search(r"StructIgnoredRegexp, value: '([^']+)'", config).group(1)
        for name in names:
            self.assertRegex(name, expression)
        for name in ("vsag_new", "prefix_vsag_error", "vsag_error_suffix", "internal_name"):
            self.assertNotRegex(name, expression)

    def test_rejects_non_camel_case_definition(self):
        result = self.run_checker("struct invalid_name {};\n")
        self.assertEqual(result.returncode, 1)
        self.assertIn("fixture.h:1: struct 'invalid_name' is not CamelCase", result.stdout)

    def test_rejects_non_camel_case_token_pasted_definition(self):
        result = self.run_checker("#define DECLARE(Name) struct invalid_##Name {};\n")
        self.assertEqual(result.returncode, 1)
        self.assertIn("struct 'invalid_Name' is not CamelCase", result.stdout)


if __name__ == "__main__":
    unittest.main()
