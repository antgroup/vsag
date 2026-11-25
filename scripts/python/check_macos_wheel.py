#!/usr/bin/env python3
"""Audit a repaired macOS wheel without importing or executing its native library."""

import argparse
from email.parser import Parser
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile


LIBRARY = "pyvsag/libvsag_c.dylib"
LICENSE_EXPRESSION = "Apache-2.0 AND Apache-2.0 WITH LLVM-exception"
MACHO_MAGICS = {bytes.fromhex(value) for value in
               ("feedface", "feedfacf", "cefaedfe", "cffaedfe", "cafebabe", "bebafeca",
                "cafebabf", "bfbafeca")}


def deployment_versions(load_commands):
    for block in re.split(r"(?m)^\s*Load command", load_commands):
        if "cmd LC_BUILD_VERSION" in block and not re.search(r"\bplatform (?:1|MACOS)\s", block):
            raise ValueError("Non-macOS or unknown Mach-O platform")
    versions = re.findall(r"\bminos (\d+\.\d+(?:\.\d+)?)", load_commands)
    versions += re.findall(r"cmd LC_VERSION_MIN_MACOSX\s+cmdsize \d+\s+"
                           r"version (\d+\.\d+(?:\.\d+)?)", load_commands)
    if not versions:
        raise ValueError("Missing Mach-O macOS deployment target")
    return [tuple(map(int, version.split("."))) + (0,) * (3 - len(version.split(".")))
            for version in versions]


def audit_library(path, target, required_symbols, run):
    dependencies = run(["otool", "-L", str(path)])
    for line in dependencies.splitlines()[1:]:
        dependency = line.strip().split(" (compatibility version", 1)[0]
        if dependency == "@rpath/libvsag_c.dylib":
            continue  # LC_ID_DYLIB, not an external dependency.
        if not dependency.startswith(("/usr/lib/", "/System/Library/Frameworks/")):
            raise ValueError(f"Unexpected Mach-O dependency: {dependency}")
    versions = deployment_versions(run(["otool", "-l", str(path)]))
    if any(version > target for version in versions):
        raise ValueError(f"Mach-O deployment target {max(versions)} exceeds wheel tag {target}")
    # delocate --require-archs checks the requested architecture before this audit.
    # A thin Mach-O is required below, so nm output describes exactly that image.
    symbols = run(["nm", "-gUj", str(path)]).splitlines()
    if not symbols or any(not re.fullmatch(r"_vsag_[A-Za-z0-9_]+", name) for name in symbols):
        raise ValueError("The wheel must export only the VSAG C API (_vsag_*)")
    if set(symbols) != required_symbols:
        raise ValueError(f"C API export mismatch: missing={sorted(required_symbols - set(symbols))}, "
                         f"unexpected={sorted(set(symbols) - required_symbols)}")
    return versions, len(symbols)


def required_c_api_symbols(header):
    # Supports direct C prototypes (including callback parameters), not macro-wrapped declarations.
    header = re.sub(r"/\*.*?\*/|//[^\n]*", "", header, flags=re.S)
    required_symbols = {"_" + name for name in re.findall(
        r"\b(vsag_[A-Za-z0-9_]+)\s*\([^;{}]*\)\s*;", header)}
    if not required_symbols:
        raise ValueError("No C API declarations found in the packaged header")
    return required_symbols


def audit_wheel(wheel, run=None):
    wheel = Path(wheel)
    match = re.search(r"-py3-none-macosx_(\d+)_(\d+)_(arm64|x86_64)\.whl$", wheel.name)
    if not match:
        raise ValueError("Expected a single-architecture py3-none macOS wheel")
    target = (int(match[1]), int(match[2]), 0)
    if run is None:
        def run(args):
            return subprocess.check_output(args, text=True)
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate wheel members")
        native = []
        for name in names:
            with archive.open(name) as member:
                magic = member.read(4)
            if magic in MACHO_MAGICS or magic == b"\x7fELF" or name.endswith((".dylib", ".so", ".a")):
                native.append(name)
        if native != [LIBRARY]:
            raise ValueError(f"Expected only {LIBRARY}; found native members: {native}")
        metadata_paths = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            raise ValueError("Expected exactly one wheel METADATA file")
        metadata = Parser().parsestr(archive.read(metadata_paths[0]).decode("utf-8"))
        if metadata.get("License-Expression") != LICENSE_EXPRESSION:
            raise ValueError("Missing LLVM-exception in wheel License-Expression")
        license_root = metadata_paths[0].removesuffix("METADATA") + "licenses/"
        for license_file in ("LICENSE", "NOTICE", "licenses/LLVM-OpenMP.txt"):
            if license_file not in metadata.get_all("License-File", []):
                raise ValueError(f"Missing License-File metadata: {license_file}")
            if not archive.read(license_root + license_file).strip():
                raise ValueError(f"Empty license file: {license_file}")
        binary = archive.read(LIBRARY)
        if binary[:4] not in (bytes.fromhex("feedfacf"), bytes.fromhex("cffaedfe")):
            raise ValueError("Expected a thin 64-bit Mach-O library")
        header = archive.read("pyvsag/_vsag_c_api.h").decode("utf-8")
        required_symbols = required_c_api_symbols(header)
        eval_header = "pyvsag/_vsag_eval_c_api.h"
        # Older wheels did not ship eval. When present, audit its ABI as well.
        if eval_header in names or "pyvsag/eval/_native.py" in names:
            required_symbols |= required_c_api_symbols(archive.read(eval_header).decode("utf-8"))
        with tempfile.TemporaryDirectory(prefix="vsag-wheel-audit-") as directory:
            library = Path(directory) / "libvsag_c.dylib"
            library.write_bytes(binary)
            versions, symbols = audit_library(library, target, required_symbols, run)
    return f"macOS wheel audit passed: deployment={max(versions)}, exports={symbols}, system dependencies only"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    try:
        print(audit_wheel(args.wheel))
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError, zipfile.BadZipFile) as error:
        parser.exit(1, f"macOS wheel audit failed: {error}\n")


if __name__ == "__main__":
    main()
