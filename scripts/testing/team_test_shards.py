#!/usr/bin/env python3
"""Inventory, balance and verify Catch2 tests without sharing writable workspaces.

All manifests are consistency checks, not authorization to mount artifacts or
trust PR output. The caller must enforce run identity and storage permissions.
Each run output directory must be new. Catch2 3.7 XML is used rather than JUnit,
whose section-level records are not an inventory of top-level test cases.
"""
from __future__ import annotations

import argparse
import concurrent.futures
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path, PurePosixPath
import signal
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET


SCHEMA = 3
EXAMPLE_FALLBACK_SECONDS = 300.0


class ShardError(ValueError):
    """An invalid manifest or incomplete test execution."""


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sealed(value):
    value = dict(value)
    value["digest"] = digest(value)
    return value


def reject_nonfinite(constant):
    raise ShardError(f"Non-finite JSON value: {constant}")


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"),
                          parse_constant=reject_nonfinite)
    except (OSError, ValueError) as error:
        raise ShardError(f"Cannot read JSON {path}: {error}") from error


def verify_seal(value, kind):
    if not isinstance(value, dict) or value.get("kind") != kind or value.get("schema_version") != SCHEMA:
        raise ShardError(f"Expected schema {SCHEMA} {kind}")
    body = {key: item for key, item in value.items() if key != "digest"}
    if value.get("digest") != digest(body):
        raise ShardError(f"Invalid {kind} digest")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        json.dump(value, output, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        output.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def finite_positive(value, description):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ShardError(f"{description} must be a positive finite number")
    return value


def positive_int(value, description):
    if type(value) is not int or value <= 0:
        raise ShardError(f"{description} must be a positive integer")
    return value


def case_identity(test):
    """Catch2 lists and execution reports both preserve these exact fields."""
    return tuple(test[key] for key in ("name", "tags", "source_file", "source_line"))


def validate_case(test):
    for key in ("name", "tags", "source_file"):
        value = test.get(key)
        if not isinstance(value, str) or any(ord(char) < 32 for char in value):
            raise ShardError("Invalid test identity field: " + key)
    if not test["name"] or test["name"] != test["name"].strip() or not test["source_file"]:
        raise ShardError("Test name and source file must be nonempty")
    positive_int(test.get("source_line"), "Test source line")


def stable_id(binary, test):
    return digest([binary, *case_identity(test)])


def filter_name_key(name):
    # Catch2 3.7 uses byte-wise std::tolower. The test executables keep the
    # startup C locale, so only ASCII A-Z fold; Unicode casefold is too broad.
    return name.translate(str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"))


def case_groups(tests):
    # Catch2's name filter is case-insensitive. Keep the complete matching
    # group indivisible even when it exceeds the requested batch size.
    groups = {}
    for test in tests:
        groups.setdefault((test["binary"], filter_name_key(test["name"])), []).append(test)
    return list(groups.values())


def binary_path(root, relative):
    if not isinstance(relative, str) or "\\" in relative:
        raise ShardError("Binary must be a POSIX relative path")
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts or str(path) != relative:
        raise ShardError(f"Invalid binary path: {relative!r}")
    resolved = (root / relative).resolve()
    if root not in resolved.parents or not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ShardError(f"Binary must be executable within build root: {relative}")
    return resolved


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()



def relative_parts(value):
    if not isinstance(value, str) or "\\" in value or any(ord(c) < 32 for c in value):
        raise ShardError("Invalid example path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts or str(path) != value:
        raise ShardError("Example path must be a canonical relative path")
    return path.parts


def example_path(root, relative, directory=False):
    current = root
    parts = relative_parts(relative)
    for index, part in enumerate(parts):
        current = current / part
        info = current.lstat()
        is_directory = index < len(parts) - 1 or directory
        if stat.S_ISLNK(info.st_mode) or (is_directory and not stat.S_ISDIR(info.st_mode)):
            raise ShardError("Example path contains a symlink or non-directory")
        if not is_directory and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                                 or not os.access(current, os.X_OK)):
            raise ShardError("Example must be a regular executable with one link")
    return current


def example_id(relative, metadata):
    return digest(["example", relative, metadata["sha256"], metadata["size"]])


def discover_examples(root, directory):
    path = example_path(root, directory, directory=True)
    examples = {}
    for entry in sorted(path.iterdir()):
        if not re.match(r"^[0-9]{3}", entry.name):
            continue
        relative = str(PurePosixPath(directory) / entry.name)
        executable = example_path(root, relative)
        before = executable.stat()
        metadata = {"sha256": file_hash(executable), "size": before.st_size}
        after = executable.stat()
        stable_fields = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, name) != getattr(after, name) for name in stable_fields):
            raise ShardError("Example changed during inventory")
        examples[relative] = dict(metadata, id=example_id(relative, metadata))
    if not examples:
        raise ShardError("Example inventory is empty")
    return examples


def validate_examples(value):
    has_examples, has_directory = "examples" in value, "example_directory" in value
    if has_examples != has_directory:
        raise ShardError("Example inventory and directory must appear together")
    if not has_examples:
        return
    relative_parts(value["example_directory"])
    examples = value["examples"]
    if not isinstance(examples, dict) or not examples:
        raise ShardError("Example inventory must be nonempty")
    for relative, metadata in examples.items():
        parts = relative_parts(relative)
        if (str(PurePosixPath(relative).parent) != value["example_directory"]
                or not re.match(r"^[0-9]{3}", parts[-1])):
            raise ShardError("Example must be a numbered direct child of its directory")
        if not isinstance(metadata, dict) or set(metadata) != {"id", "sha256", "size"}:
            raise ShardError("Invalid example metadata")
        if (not isinstance(metadata["sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", metadata["sha256"])):
            raise ShardError("Invalid example SHA256")
        positive_int(metadata["size"], "Example size")
        if metadata["id"] != example_id(relative, metadata):
            raise ShardError("Invalid example ID")


def execute_example(relative, metadata, root, output, timeout, cancelled):
    directory = output / ("example-" + metadata["id"])
    directory.mkdir()
    cwd, temporary = directory / "cwd", directory / "tmp"
    cwd.mkdir()
    temporary.mkdir()
    result = {"binary": relative, **metadata, "directory": directory.name,
              "status": "failed", "exit_code": None, "seconds": 0.0}
    try:
        executable = example_path(root, relative)
        # The caller verifies the complete immutable inventory before dispatch.
        result.update(execute([str(executable)], cwd, temporary,
                              directory / "output.log", timeout, cancelled))
    except (OSError, ShardError) as error:
        result["status"] = "cancelled" if cancelled.is_set() else "launch_error"
        result["error"] = str(error)
    log = directory / "output.log"
    result["log"] = str(log.relative_to(output)) if log.is_file() else None
    # Keep every example visible in the Actions step, including failed or unstarted work.
    print("Example result: " + json.dumps({
        "example": relative, "status": result["status"], "exit_code": result["exit_code"],
        "log": str(Path(output.name) / result["log"]) if result["log"] else None,
    }, sort_keys=True), flush=True)
    return result


def mixed_tasks(batches, examples, durations, target_seconds):
    """Keep batches intact; every unknown example uses the same fixed hint."""
    tasks = []
    for index, batch in enumerate(batches):
        seconds = (sum(durations[t["id"]] for t in batch)
                   if all(t["id"] in durations for t in batch)
                   else (target_seconds or EXAMPLE_FALLBACK_SECONDS))
        tasks.append((seconds, f"catch2:{index:06d}", "catch2", index, batch))
    for relative, metadata in sorted(examples.items()):
        tasks.append((EXAMPLE_FALLBACK_SECONDS, "example:" + relative,
                      "example", relative, metadata))
    return sorted(tasks, key=lambda task: (-task[0], task[1]))


def xml_root(path, expected):
    try:
        raw = Path(path).read_bytes()
        if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
            raise ShardError("XML entities and DTDs are not accepted")
        root = ET.fromstring(raw)
    except (OSError, ET.ParseError) as error:
        raise ShardError(f"Missing or malformed XML report {path}: {error}") from error
    if root.tag != expected:
        raise ShardError(f"Expected {expected} XML, got {root.tag}")
    return root


def parse_inventory_xml(path):
    result = []
    for node in xml_root(path, "MatchingTests").findall("TestCase"):
        try:
            line = int(node.findtext("SourceInfo/Line", ""))
        except ValueError as error:
            raise ShardError("Invalid inventory source line") from error
        test = {"name": node.findtext("Name"), "tags": node.findtext("Tags") or "",
                "source_file": node.findtext("SourceInfo/File"), "source_line": line}
        validate_case(test)
        result.append(test)
    if len({case_identity(test) for test in result}) != len(result):
        raise ShardError("Duplicate test identities in Catch2 inventory")
    return result


def exact_filter(names):
    # Quotes protect tag/exclusion syntax; commas still need escaping in Catch2.
    # Catch2 also performs case folding and interprets boundary '*', so run() must
    # verify the selected inventory before executing this filter.
    return ",".join('"' + name.replace("\\", "\\\\").replace('"', '\\"').replace(",", "\\,") + '"'
                    for name in dict.fromkeys(names))


def kill_group(process, sig):
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def execute(command, cwd, temporary, log, timeout, cancelled):
    """Run and reap an entire process group, including on timeout/cancellation."""
    started = time.monotonic()
    environment = os.environ.copy()
    environment.update(TMPDIR=str(temporary), TMP=str(temporary), TEMP=str(temporary))
    if cancelled.is_set():
        return {"exit_code": None, "status": "cancelled", "seconds": 0.0}
    with Path(log).open("w", encoding="utf-8") as output:
        output.write("Command: " + json.dumps(command, ensure_ascii=False) + "\n")
        output.flush()
        try:
            process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as error:
            output.write(str(error) + "\n")
            return {"exit_code": None, "status": "launch_error", "seconds": time.monotonic() - started}
        reason = None
        try:
            while process.poll() is None:
                if cancelled.is_set():
                    reason = "cancelled"
                    break
                if time.monotonic() - started > timeout:
                    reason = "timeout"
                    break
                time.sleep(0.025)
            if reason:
                kill_group(process, signal.SIGTERM)
                try:
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    kill_group(process, signal.SIGKILL)
            code = process.wait()
        finally:
            # A test can exit while one of its children remains alive. Never let
            # that child outlive its isolated batch or keep an output file open.
            kill_group(process, signal.SIGKILL)
            process.wait()
    return {"exit_code": code, "status": reason or ("passed" if code == 0 else "failed"),
            "seconds": round(time.monotonic() - started, 6)}


def listing(binary, test_filter, seed, directory, timeout, cancelled):
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "tmp"
    temporary.mkdir(exist_ok=True)
    report = directory / "inventory.xml"
    command = [str(binary), test_filter, "--list-tests", "--reporter", "xml", "--out", str(report),
               "--order", "lex", "--rng-seed", str(seed)]
    status = execute(command, directory, temporary, directory / "inventory.log", timeout, cancelled)
    if status["status"] != "passed":
        raise ShardError(f"Test listing {status['status']} (exit {status['exit_code']}); see {directory}")
    return parse_inventory_xml(report)


def validate_inventory(value):
    verify_seal(value, "inventory")
    seed = value.get("seed")
    if type(seed) is not int or not 0 < seed < 2**32:
        raise ShardError("Seed must be an integer from 1 to 4294967295")
    if not isinstance(value.get("context"), dict):
        raise ShardError("Inventory context must be an object")
    binaries = value.get("binaries")
    tests = value.get("tests")
    if not isinstance(binaries, dict) or not binaries or not isinstance(tests, list) or not tests:
        raise ShardError("Inventory must contain binaries and tests")
    ids = set()
    for test in tests:
        if not isinstance(test, dict) or not isinstance(test.get("binary"), str):
            raise ShardError("Invalid inventory test")
        validate_case(test)
        binary = test["binary"]
        if binary not in binaries or test.get("id") != stable_id(binary, test) or test["id"] in ids:
            raise ShardError("Invalid or duplicate inventory ID")
        ids.add(test["id"])
    validate_examples(value)
    for value_hash in binaries.values():
        if not isinstance(value_hash, str) or len(value_hash) != 64 or any(c not in "0123456789abcdef" for c in value_hash):
            raise ShardError("Invalid binary SHA256")


def inventory(args):
    finite_positive(args.timeout, "Listing timeout")
    if not 0 < args.seed < 2**32:
        raise ShardError("Seed must be an integer from 1 to 4294967295")
    root = Path(args.build_root).resolve()
    if len(set(args.binary)) != len(args.binary):
        raise ShardError("Duplicate --binary")
    cancelled = threading.Event()
    received_signal, previous = [], {}

    def cancel(signum, _frame):
        received_signal.append(signum)
        cancelled.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.signal(sig, cancel)
    tests, hashes = [], {}
    try:
        with ExitStack() as stack:
            if args.diagnostics_dir:
                directory = Path(args.diagnostics_dir).resolve()
                if directory == root or root in directory.parents:
                    raise ShardError("Inventory diagnostics must be outside the build root")
                directory.mkdir(parents=True, exist_ok=False)
            else:
                directory = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="team-inventory-")))
            for index, relative in enumerate(args.binary):
                binary = binary_path(root, relative)
                before = file_hash(binary)
                selected = listing(binary, args.filter, args.seed, Path(directory) / str(index), args.timeout, cancelled)
                if not selected:
                    raise ShardError(f"No tests selected from {relative}")
                if before != file_hash(binary):
                    raise ShardError("Binary changed during inventory")
                hashes[relative] = before
                tests.extend(dict(test, binary=relative, id=stable_id(relative, test)) for test in selected)
    except ShardError:
        if received_signal:
            return 128 + received_signal[0]
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    extra = {}
    if args.examples_directory:
        extra = {"example_directory": args.examples_directory,
                 "examples": discover_examples(root, args.examples_directory)}
    result = sealed({"schema_version": SCHEMA, "kind": "inventory", "seed": args.seed,
                     "filter": args.filter, "context": load_json(args.context) if args.context else {},
                     "binaries": hashes, "tests": sorted(tests, key=lambda test: test["id"]), **extra})
    validate_inventory(result)
    write_json(args.output, result)
    return 0


def validate_durations(durations):
    if not isinstance(durations, dict):
        raise ShardError("Durations must map test IDs to seconds")
    for key, seconds in durations.items():
        if not isinstance(key, str):
            raise ShardError("Duration IDs must be strings")
        finite_positive(seconds, "Historical duration")


def make_batches(selected, batch_size, durations=None, target_seconds=None):
    """Keep filter groups atomic; optionally pack known short groups by duration."""
    positive_int(batch_size, "Batch size")
    groups = case_groups(selected)
    if target_seconds is None:
        batches = []
        for group in groups:
            if (batches and batches[-1][0]["binary"] == group[0]["binary"]
                    and len(batches[-1]) + len(group) <= batch_size):
                batches[-1].extend(group)
            else:
                batches.append(group)
        return batches
    finite_positive(target_seconds, "Batch target seconds")
    durations = {} if durations is None else durations
    validate_durations(durations)
    weighted = [(sum(durations[test["id"]] for test in group)
                 if all(test["id"] in durations for test in group) else None, group)
                for group in groups]
    # Unknown groups run alone: a missing estimate must not turn a new slow
    # case into a long serial batch. Stable IDs break all scheduling ties.
    def sort_key(pair):
        seconds, group = pair
        return (-(seconds if seconds is not None else target_seconds),
                tuple(sorted(test["id"] for test in group)))

    packed = []
    for seconds, group in sorted(weighted, key=sort_key):
        candidates = []
        if seconds is not None and seconds < target_seconds and len(group) <= batch_size:
            candidates = [index for index, (weight, batch) in enumerate(packed)
                          if weight is not None and weight < target_seconds
                          and batch[0]["binary"] == group[0]["binary"]
                          and weight + seconds <= target_seconds
                          and len(batch) + len(group) <= batch_size]
        if candidates:
            index = max(candidates, key=lambda candidate: (packed[candidate][0], -candidate))
            weight, batch = packed[index]
            packed[index] = (weight + seconds, batch + group)
        else:
            # A same-filter group can itself exceed either limit. It stays
            # complete and alone; never split or omit cases to meet a target.
            packed.append((seconds, group))
    return [batch for _seconds, batch in sorted(packed, key=sort_key)]


def example_assignments(shards, examples):
    """Keep Catch2 membership fixed; balance equal example hints on its load."""
    loads = []
    for shard in shards:
        seconds = shard.get("estimated_seconds")
        if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds) or seconds < 0):
            raise ShardError("Invalid shard estimate")
        loads.append(seconds)
    assigned = [[] for _ in shards]
    for relative in sorted(examples):
        target = min(range(len(shards)), key=lambda index: (loads[index], index))
        assigned[target].append(relative)
        loads[target] += EXAMPLE_FALLBACK_SECONDS
    return assigned, loads


def assigned_examples(value, index):
    inventory = value["inventory"].get("examples", {})
    return {relative: inventory[relative]
            for relative in value["shards"][index].get("example_binaries", [])}


def make_plan(inventory_value, shard_count, durations=None):
    validate_inventory(inventory_value)
    positive_int(shard_count, "Shard count")
    groups = case_groups(inventory_value["tests"])
    if shard_count > len(groups):
        raise ShardError("More shards than tests would create empty shards")
    durations = {} if durations is None else durations
    validate_durations(durations)
    # Build flags can change Catch2 source locations and thus stable IDs. Use
    # only exact identities from this binary inventory; stale history must not
    # affect even the fallback estimate, and never rewrites a case identity.
    current_ids = {test["id"] for test in inventory_value["tests"]}
    durations = {key: seconds for key, seconds in durations.items() if key in current_ids}
    known = sorted(durations.values())
    fallback = known[len(known) // 2] if known else 1.0
    shards = [{"index": index, "test_ids": [], "estimated_seconds": 0.0} for index in range(shard_count)]
    weighted = [(sum(durations.get(test["id"], fallback) for test in group),
                 sorted(test["id"] for test in group)) for group in groups]
    for seconds, test_ids in sorted(weighted, key=lambda pair: (-pair[0], pair[1])):
        target = min(shards, key=lambda shard: (shard["estimated_seconds"], shard["index"]))
        target["test_ids"].extend(test_ids)
        target["estimated_seconds"] += seconds
    if "examples" in inventory_value:
        assigned, loads = example_assignments(shards, inventory_value["examples"])
        for shard, relative, seconds in zip(shards, assigned, loads):
            shard["example_binaries"] = relative
            shard["estimated_total_seconds"] = seconds
    return sealed({"schema_version": SCHEMA, "kind": "plan", "inventory": inventory_value,
                   "shards": shards, "duration_history": durations})


def validate_plan(value):
    verify_seal(value, "plan")
    validate_inventory(value.get("inventory"))
    validate_durations(value.get("duration_history"))
    shards = value.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ShardError("Plan has no shards")
    if "example_shard" in value:
        raise ShardError("Legacy single-shard examples are not supported")
    all_ids = []
    for index, shard in enumerate(shards):
        if not isinstance(shard, dict) or shard.get("index") != index or not isinstance(shard.get("test_ids"), list) or not shard["test_ids"]:
            raise ShardError("Plan contains invalid or empty shard")
        if not all(isinstance(test_id, str) for test_id in shard["test_ids"]):
            raise ShardError("Plan test IDs must be strings")
        all_ids.extend(shard["test_ids"])
    expected = [test["id"] for test in value["inventory"]["tests"]]
    if Counter(all_ids) != Counter(expected):
        raise ShardError("Plan must cover every inventory test exactly once")
    owners = {test_id: shard["index"] for shard in shards for test_id in shard["test_ids"]}
    for group in case_groups(value["inventory"]["tests"]):
        if len({owners[test["id"]] for test in group}) != 1:
            raise ShardError("Same-name cases must stay in one indivisible shard group")
    if "examples" in value["inventory"]:
        expected_examples = value["inventory"]["examples"]
        assigned, loads = example_assignments(shards, expected_examples)
        all_examples = []
        for shard, expected, seconds in zip(shards, assigned, loads):
            actual = shard.get("example_binaries")
            if (not isinstance(actual, list) or not all(isinstance(name, str) for name in actual)
                    or actual != expected or shard.get("estimated_total_seconds") != seconds):
                raise ShardError("Examples must match the balanced shard assignment")
            all_examples.extend(actual)
        if Counter(all_examples) != Counter(expected_examples.keys()):
            raise ShardError("Plan must cover every example exactly once")
    elif any("example_binaries" in shard or "estimated_total_seconds" in shard for shard in shards):
        raise ShardError("Example assignment has no inventory")


def plan(args):
    history = load_json(args.durations) if args.durations else {}
    if isinstance(history, dict) and "duration_history" in history:
        history = history["duration_history"]
    value = make_plan(load_json(args.inventory), args.shards, history)
    write_json(args.output, value)
    return 0


def parse_execution_xml(path, expected):
    root = xml_root(path, "Catch2TestRun")
    cases = root.findall("TestCase")
    identities = []
    for case in cases:
        try:
            line = int(case.get("line", ""))
        except ValueError as error:
            raise ShardError("Invalid executed source line") from error
        identity = {"name": case.get("name"), "tags": case.get("tags", ""),
                    "source_file": case.get("filename"), "source_line": line}
        validate_case(identity)
        identities.append(identity)
    if Counter(case_identity(test) for test in identities) != Counter(case_identity(test) for test in expected):
        raise ShardError("Executed test identities do not match the planned batch exactly once")
    totals = root.find("OverallResultsCases")
    if totals is None:
        raise ShardError("Catch2 report lacks final test-case totals")
    try:
        totals_counts = [int(totals.get(key, "-1")) for key in ("successes", "failures", "expectedFailures", "skips")]
        if min(totals_counts) < 0 or sum(totals_counts) != len(expected):
            raise ShardError("Catch2 final test-case totals do not match inventory")
    except ValueError as error:
        raise ShardError("Invalid Catch2 test-case totals") from error
    by_identity = {case_identity(test): test for test in expected}
    result = []
    for case, identity in zip(cases, identities):
        overall = case.find("OverallResult")
        if overall is None or overall.get("success") not in ("true", "false"):
            raise ShardError("Test case lacks a final result")
        try:
            seconds = float(overall.get("durationInSeconds", "nan"))
            skips = int(overall.get("skips", "-1"))
        except ValueError as error:
            raise ShardError("Invalid Catch2 case timing or skips") from error
        if not math.isfinite(seconds) or seconds < 0 or skips < 0:
            raise ShardError("Invalid Catch2 case timing or skips")
        result.append({"id": by_identity[case_identity(identity)]["id"], **identity,
                       "seconds": seconds, "skips": skips, "success": overall.get("success") == "true"})
    return result, totals_counts[1] == 0, totals_counts[3] == len(expected)


def execute_batch(batch, index, root, output, seed, timeout, cancelled):
    directory = output / f"batch-{index:04d}"
    directory.mkdir()
    cwd, temporary = directory / "cwd", directory / "tmp"
    cwd.mkdir()
    temporary.mkdir()
    result = {"index": index, "binary": batch[0]["binary"], "test_ids": [test["id"] for test in batch],
              "tests": [], "status": "failed", "exit_code": None, "seconds": 0.0,
              "directory": directory.name}
    started = time.monotonic()
    try:
        binary = binary_path(root, batch[0]["binary"])
        selected = listing(binary, exact_filter([test["name"] for test in batch]), seed,
                           directory / "selection", min(timeout, 60), cancelled)
        if Counter(case_identity(test) for test in selected) != Counter(case_identity(test) for test in batch):
            raise ShardError("Exact-name filter selected unexpected, duplicate or missing cases")
        report = directory / "report.xml"
        command = [str(binary), exact_filter([test["name"] for test in batch]), "--reporter", "xml",
                   "--out", str(report), "--durations", "yes", "--order", "rand", "--rng-seed", str(seed)]
        process_result = execute(command, cwd, temporary, directory / "output.log", timeout, cancelled)
        result.update(process_result)
        if process_result["status"] not in ("timeout", "cancelled", "launch_error"):
            result["tests"], cases_passed, all_skipped = parse_execution_xml(report, batch)
            result["all_skipped"] = all_skipped
            # Catch2 returns 4 when every case in this small batch is skipped.
            # Preserve existing SKIP semantics, but only with complete XML proof.
            if (process_result["exit_code"] == 4 and all_skipped and cases_passed
                    and all(test["success"] and test["skips"] > 0 for test in result["tests"])):
                result["status"] = "passed"
            if not cases_passed:
                result["status"] = "failed"
    except (ShardError, OSError) as error:
        result["status"] = "cancelled" if cancelled.is_set() else "failed"
        result["error"] = str(error)
    result["seconds"] = round(time.monotonic() - started, 6)
    return result


def run(args):
    value = load_json(args.plan)
    validate_plan(value)
    positive_int(args.workers, "Workers")
    positive_int(args.batch_size, "Batch size")
    finite_positive(args.timeout, "Timeout")
    finite_positive(args.example_timeout, "Example timeout")
    if not 0 <= args.shard < len(value["shards"]):
        raise ShardError("Shard index out of range")
    root = Path(args.build_root).resolve()
    output = Path(args.output_dir).resolve()
    if output == root or root in output.parents:
        raise ShardError("Output directory must be outside the immutable build root")
    for relative, expected in value["inventory"]["binaries"].items():
        if file_hash(binary_path(root, relative)) != expected:
            raise ShardError(f"Binary digest mismatch: {relative}")
    examples = assigned_examples(value, args.shard)
    if "examples" in value["inventory"]:
        # Verify the complete sealed inventory, then execute only this shard's subset.
        actual = discover_examples(root, value["inventory"]["example_directory"])
        if actual != value["inventory"]["examples"]:
            raise ShardError("Example inventory differs from the immutable build")
    output.mkdir(parents=True, exist_ok=False)
    started, started_at = time.monotonic(), time.time()
    tests = {test["id"]: test for test in value["inventory"]["tests"]}
    selected = [tests[test_id] for test_id in value["shards"][args.shard]["test_ids"]]
    batches = make_batches(selected, args.batch_size, value["duration_history"], args.batch_target_seconds)
    cancelled = threading.Event()
    received_signal = []
    previous = {}

    def cancel(signum, _frame):
        received_signal.append(signum)
        cancelled.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.signal(sig, cancel)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            if examples:
                tasks = mixed_tasks(batches, examples, value["duration_history"],
                                    args.batch_target_seconds)
            else:
                # A shard with no examples retains its existing Catch2 queue order.
                tasks = [(0, "", "catch2", index, batch) for index, batch in enumerate(batches)]
            futures = []
            for _weight, _key, kind, index, task in tasks:
                if kind == "catch2":
                    future = executor.submit(execute_batch, task, index, root, output,
                                             value["inventory"]["seed"], args.timeout, cancelled)
                else:
                    future = executor.submit(execute_example, index, task, root, output,
                                             args.example_timeout, cancelled)
                futures.append((kind, future))
            completed = [(kind, future.result()) for kind, future in futures]
            results = sorted([result for kind, result in completed if kind == "catch2"],
                             key=lambda result: result["index"])
            example_results = sorted([result for kind, result in completed if kind == "example"],
                                     key=lambda result: result["binary"])
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    success = (not cancelled.is_set()
               and all(result["status"] == "passed" for result in results + example_results))
    extra = ({"examples": example_results,
              "example_binaries": value["shards"][args.shard]["example_binaries"]}
             if "examples" in value["inventory"] else {})
    result = sealed({"schema_version": SCHEMA, "kind": "result", "plan_digest": value["digest"],
                     "context": value["inventory"]["context"], "shard": args.shard,
                     "test_ids": [test["id"] for test in selected], "seed": value["inventory"]["seed"],
                     "workers": args.workers, "batch_size": args.batch_size,
                     "batch_target_seconds": args.batch_target_seconds, "batches": results,
                     "success": success, "started_at": started_at, "finished_at": time.time(),
                     "seconds": round(time.monotonic() - started, 6),
                     "signal": received_signal[0] if received_signal else None, **extra})
    write_json(output / "result.json", result)
    return 128 + received_signal[0] if received_signal else (0 if success else 1)


def merge_results(value, results):
    validate_plan(value)
    errors, seen_shards, observed, durations = [], [], [], {}
    observed_examples = []
    inventory_tests = {test["id"]: test for test in value["inventory"]["tests"]}
    for result in results:
        verify_seal(result, "result")
        if result.get("plan_digest") != value["digest"] or result.get("context") != value["inventory"]["context"]:
            raise ShardError("Result belongs to a different plan or run context")
        index = result.get("shard")
        if type(index) is not int or not 0 <= index < len(value["shards"]):
            raise ShardError("Invalid result shard")
        seen_shards.append(index)
        expected = value["shards"][index]["test_ids"]
        if result.get("test_ids") != expected or result.get("seed") != value["inventory"]["seed"]:
            errors.append(f"Shard {index} manifest mismatch")
        if result.get("success") is not True:
            errors.append(f"Shard {index} failed or was cancelled")
        shard_observed, assigned = [], []
        for batch in result.get("batches", []):
            assigned.extend(batch.get("test_ids", []))
            skipped_exit = (batch.get("exit_code") == 4 and batch.get("all_skipped") is True
                            and bool(batch.get("tests"))
                            and all(test.get("success") is True and type(test.get("skips")) is int
                                    and test["skips"] > 0 for test in batch["tests"]))
            if batch.get("status") != "passed" or not (batch.get("exit_code") == 0 or skipped_exit):
                errors.append(f"Shard {index} batch {batch.get('index')} did not pass")
            actual = [test.get("id") for test in batch.get("tests", [])]
            if Counter(actual) != Counter(batch.get("test_ids", [])):
                errors.append(f"Shard {index} batch {batch.get('index')} incomplete")
            for test in batch.get("tests", []):
                validate_case(test)
                if (test.get("success") is not True or type(test.get("skips")) is not int
                        or test["skips"] < 0):
                    errors.append(f"Shard {index} unsuccessful or invalid case result")
                expected_test = inventory_tests.get(test.get("id"))
                if (expected_test is None or case_identity(test) != case_identity(expected_test)
                        or batch.get("binary") != expected_test["binary"]):
                    errors.append(f"Shard {index} result case identity mismatch")
                seconds = test.get("seconds")
                if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
                    raise ShardError("Invalid result duration")
                durations[test["id"]] = max(seconds, 0.001)
            shard_observed.extend(actual)
        if Counter(assigned) != Counter(expected) or Counter(shard_observed) != Counter(expected):
            errors.append(f"Shard {index} missing, duplicated or unexpected tests")
        observed.extend(shard_observed)
        expected_examples = assigned_examples(value, index)
        if "examples" in value["inventory"]:
            if result.get("example_binaries") != value["shards"][index]["example_binaries"]:
                errors.append(f"Shard {index} example assignment manifest mismatch")
        elif "example_binaries" in result:
            errors.append(f"Shard {index} unexpected example assignment")
        example_results = result.get("examples", [])
        if not isinstance(example_results, list):
            raise ShardError("Example results must be a list")
        if "examples" in value["inventory"] and "examples" not in result:
            errors.append(f"Shard {index} has no explicit example results")
        actual_examples = []
        for example in example_results:
            if not isinstance(example, dict) or not isinstance(example.get("binary"), str):
                raise ShardError("Invalid example result")
            relative = example["binary"]
            actual_examples.append(relative)
            metadata = expected_examples.get(relative)
            if metadata is None or any(example.get(key) != val for key, val in metadata.items()):
                errors.append(f"Shard {index} example binary identity mismatch")
            seconds = example.get("seconds")
            if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                    or not math.isfinite(seconds) or seconds < 0):
                raise ShardError("Invalid example duration")
            if example.get("status") != "passed" or type(example.get("exit_code")) is not int or example["exit_code"] != 0:
                errors.append(f"Shard {index} example {relative} did not pass")
        if Counter(actual_examples) != Counter(expected_examples.keys()):
            errors.append(f"Shard {index} missing, duplicated or unexpected examples")
        observed_examples.extend(actual_examples)
    if Counter(seen_shards) != Counter(range(len(value["shards"]))):
        errors.append("Missing or duplicate shard results")
    expected_ids = [test["id"] for test in value["inventory"]["tests"]]
    if Counter(observed) != Counter(expected_ids):
        errors.append("Executed inventory differs from expected inventory")
    extra = {}
    if "examples" in value["inventory"]:
        expected_examples = value["inventory"]["examples"]
        if Counter(observed_examples) != Counter(expected_examples.keys()):
            errors.append("Executed examples differ from expected inventory")
        extra = {"expected_example_count": len(expected_examples),
                 "observed_example_count": len(observed_examples)}
    return sealed({"schema_version": SCHEMA, "kind": "summary", "plan_digest": value["digest"],
                   "context": value["inventory"]["context"], "success": not errors,
                   "expected_count": len(expected_ids), "observed_count": len(observed),
                   "errors": errors, "duration_history": durations,
                   "result_digests": [result["digest"] for result in results], **extra})


def merge(args):
    result = merge_results(load_json(args.plan), [load_json(path) for path in args.result])
    write_json(args.output, result)
    if result["errors"]:
        print("\n".join(result["errors"]), file=sys.stderr)
    return 0 if result["success"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("inventory", help="List non-daily Catch2 cases and hash binaries")
    command.add_argument("--build-root", required=True)
    command.add_argument("--binary", action="append", required=True, help="Executable relative to build root; repeatable")
    command.add_argument("--filter", default="~[daily]")
    command.add_argument("--examples-directory", help="Numbered executable directory relative to build root")
    command.add_argument("--seed", type=int, default=424242)
    command.add_argument("--timeout", type=float, default=60)
    command.add_argument("--context", help="JSON object binding the artifact to its run/attempt/SHA")
    command.add_argument("--diagnostics-dir", help="New directory outside build root; retain listing logs, including on failure")
    command.add_argument("--output", required=True)
    command.set_defaults(function=inventory)
    command = commands.add_parser("plan", help="Create deterministic duration-balanced shard manifests")
    command.add_argument("--inventory", required=True)
    command.add_argument("--shards", type=int, default=4)
    command.add_argument("--durations", help="ID-to-seconds mapping or previous summary JSON")
    command.add_argument("--output", required=True)
    command.set_defaults(function=plan)
    command = commands.add_parser("run", help="Execute one shard; output directory must be new")
    command.add_argument("--plan", required=True)
    command.add_argument("--build-root", required=True)
    command.add_argument("--shard", type=int, required=True)
    command.add_argument("--workers", type=int, default=1)
    command.add_argument("--batch-size", type=int, default=1, help="Maximum cases per batch, except an indivisible filter group")
    command.add_argument("--batch-target-seconds", type=float,
                         help="Pack known short groups up to this estimated duration; unknown or slow groups run alone")
    command.add_argument("--timeout", type=float, default=900, help="Per-batch execution timeout in seconds")
    command.add_argument("--example-timeout", type=float, default=1800, help="Per-example execution timeout in seconds")
    command.add_argument("--output-dir", required=True)
    command.set_defaults(function=run)
    command = commands.add_parser("merge", help="Require all shards and an exact completed test inventory")
    command.add_argument("--plan", required=True)
    command.add_argument("--result", action="append", required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(function=merge)
    args = parser.parse_args(argv)
    try:
        return args.function(args)
    except (ShardError, OSError) as error:
        print(f"team_test_shards: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
