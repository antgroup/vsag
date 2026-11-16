"""Keep the wheel export audit aligned with pyvsag's real cffi declarations."""

import ast
import importlib.util
from pathlib import Path
import re

import cffi


ROOT = Path(__file__).resolve().parents[2]


def test_wheel_export_parser_matches_real_cffi_header():
    spec = importlib.util.spec_from_file_location(
        "wheel_audit", ROOT / "scripts/python/check_macos_wheel.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)

    # Execute the binding's actual header-preparation function without loading libvsag_c.
    module = ast.parse((ROOT / "python/pyvsag/_lib.py").read_text())
    prepare = next(node for node in module.body
                   if isinstance(node, ast.FunctionDef) and node.name == "_cdef_from_header")
    namespace = {"re": re}
    exec(compile(ast.Module(body=[prepare], type_ignores=[]), "_lib.py", "exec"), namespace)
    header = ROOT / "include/vsag/vsag_c_api.h"
    ffi = cffi.FFI()
    ffi.cdef(namespace["_cdef_from_header"](header))
    # cffi has no public declaration-name iterator; inspect its parsed declarations in this test.
    functions = {"_" + name.removeprefix("function ") for name in ffi._parser._declarations
                 if name.startswith("function vsag_")}
    assert functions
    assert audit.required_c_api_symbols(header.read_text()) == functions
