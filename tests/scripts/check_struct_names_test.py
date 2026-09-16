#!/usr/bin/env python3

from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/linters/check-struct-names.py"


class CheckStructNamesTest(unittest.TestCase):
    def run_checker(self, source: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "fixture.h"
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
