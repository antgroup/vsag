"""Exercise sharding with real mock child processes; no build or network needed."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/testing/team_test_shards.py"
SPEC = importlib.util.spec_from_file_location("team_test_shards", SCRIPT)
SHARDS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SHARDS)

MOCK = r'''
import json, os, signal, subprocess, sys, time
from pathlib import Path
import xml.etree.ElementTree as ET
args = sys.argv[1:]
config = json.loads(Path(__file__).with_suffix('.json').read_text())
cases = config['cases']
for index, case in enumerate(cases):
    case.setdefault('tags', '')
    case.setdefault('source_file', 'fixture.cpp')
    case.setdefault('source_line', index + 1)
report = Path(args[args.index('--out') + 1])
selected = []
if args[0] == '~[daily]':
    selected = [case for case in cases if '[daily]' not in case.get('tags', '')]
else:
    names, token, escaped, quoted = [], '', False, False
    for char in args[0]:
        if escaped:
            token += char
            escaped = False
        elif char == '\\':
            escaped = True
        elif char == '"':
            quoted = not quoted
        elif char == ',' and not quoted:
            names.append(token)
            token = ''
        else:
            token += char
    names.append(token)
    for case in cases:
        for name in names:
            fold = str.maketrans('ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')
            pattern, actual = name.translate(fold).strip(), case['name'].translate(fold).strip()
            matches = actual == pattern
            if pattern.startswith('*') and pattern.endswith('*'):
                matches = pattern.strip('*') in actual
            elif pattern.startswith('*'):
                matches = actual.endswith(pattern[1:])
            elif pattern.endswith('*'):
                matches = actual.startswith(pattern[:-1])
            if matches:
                selected.append(case)
                break
if '--list-tests' in args:
    if config.get('listing_message'):
        print(config['listing_message'], flush=True)
    if config.get('listing_started'):
        Path(config['listing_started']).write_text(str(os.getpid()))
    if config.get('listing_sleep'):
        time.sleep(config['listing_sleep'])
    root = ET.Element('MatchingTests')
    if config.get('extra_selection') and args[0] != '~[daily]':
        selected += [{'name': 'unexpected'}]
    for case in selected:
        node = ET.SubElement(root, 'TestCase')
        ET.SubElement(node, 'Name').text = case['name']
        ET.SubElement(node, 'Tags').text = case.get('tags', '')
        source = ET.SubElement(node, 'SourceInfo')
        ET.SubElement(source, 'File').text = case.get('source_file', 'unexpected.cpp')
        ET.SubElement(source, 'Line').text = str(case.get('source_line', 1))
    ET.ElementTree(root).write(report)
    sys.exit(config.get('listing_exit', 0))
root = ET.Element('Catch2TestRun')
failures = 0
skipped = 0
for case in selected:
    event = {'name': case['name'], 'cwd': os.getcwd(), 'tmp': os.environ['TMPDIR'],
             'pid': os.getpid(), 'seed': args[args.index('--rng-seed') + 1]}
    Path(os.environ['TMPDIR'], 'invocation.json').write_text(json.dumps(event))
    if config.get('events'):
        with open(config['events'], 'a') as output:
            output.write(json.dumps(dict(event, phase='start', time=time.time())) + '\n')
    if case.get('spawn_child'):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        Path(config['child_pid']).write_text(str(child.pid))
    if case.get('crash'):
        sys.exit(17)
    if case.get('sleep'):
        time.sleep(case['sleep'])
    if not case.get('omit'):
        node = ET.SubElement(root, 'TestCase', name=case['name'], tags=case.get('report_tags', case['tags']),
                             filename=case.get('report_file', case['source_file']),
                             line=str(case.get('report_line', case['source_line'])))
        ET.SubElement(node, 'OverallResult', success='false' if case.get('fail') else 'true',
                      durationInSeconds=str(case.get('duration', 0.01)), skips='1' if case.get('skip') else '0')
        if case.get('duplicate'):
            root.append(ET.fromstring(ET.tostring(node)))
    failures += bool(case.get('fail'))
    skipped += bool(case.get('skip'))
    if config.get('events'):
        with open(config['events'], 'a') as output:
            output.write(json.dumps(dict(event, phase='end', time=time.time())) + '\n')
ET.SubElement(root, 'OverallResultsCases', successes=str(len(selected)-failures-skipped),
              failures=str(failures), expectedFailures='0', skips=str(skipped))
ET.ElementTree(root).write(report)
sys.exit(config.get('execution_exit', 1 if failures else (4 if skipped == len(selected) else 0)))
'''


def fixture_case(name, tags="[ut]", source_line=1):
    return {"name": name, "tags": tags, "source_file": "fixture.cpp", "source_line": source_line}


def fixture_inventory(names):
    binary = "tests/unittests"
    cases = [fixture_case(name) for name in names]
    return SHARDS.sealed({"schema_version": SHARDS.SCHEMA, "kind": "inventory", "seed": 424242,
                          "filter": "~[daily]", "context": {"attempt": 1},
                          "binaries": {binary: "a" * 64},
                          "tests": [{"id": SHARDS.stable_id(binary, case), "binary": binary,
                                     **case} for case in cases]})


class ManifestTest(unittest.TestCase):
    def test_stable_ids_depend_on_binary_and_complete_case_identity(self):
        case = fixture_case("x")
        self.assertEqual(SHARDS.stable_id("tests/a", case), SHARDS.stable_id("tests/a", case))
        self.assertNotEqual(SHARDS.stable_id("tests/a", case), SHARDS.stable_id("tests/b", case))
        for field, value in (("name", "X"), ("tags", "[other]"), ("source_file", "other.cpp"), ("source_line", 2)):
            self.assertNotEqual(SHARDS.stable_id("tests/a", case),
                                SHARDS.stable_id("tests/a", dict(case, **{field: value})))

    def test_duration_balance_is_deterministic_and_complete(self):
        inventory = fixture_inventory(["a", "b", "c", "d", "e"])
        history = {test["id"]: weight for test, weight in zip(inventory["tests"], [9, 8, 3, 2, 1])}
        plan = SHARDS.make_plan(inventory, 2, history)
        self.assertEqual(plan, SHARDS.make_plan(inventory, 2, history))
        self.assertEqual(sorted(shard["estimated_seconds"] for shard in plan["shards"]), [11, 12])
        SHARDS.validate_plan(plan)
        self.assertEqual(sorted(test for shard in plan["shards"] for test in shard["test_ids"]),
                         sorted(test["id"] for test in inventory["tests"]))

    def test_duration_hints_match_only_current_exact_inventory_ids(self):
        inventory = fixture_inventory(["a", "b", "c"])
        tests = inventory["tests"]
        changed = dict(tests[0], source_file="../../../__w/vsag/vsag/fixture.cpp")
        old_id = SHARDS.stable_id(changed["binary"], changed)
        self.assertNotEqual(old_id, tests[0]["id"])
        stale = {old_id: 900, "retired-case-id": 800}
        # Even the fallback estimate must not be skewed by unrelated identities.
        self.assertEqual(SHARDS.make_plan(inventory, 2, stale), SHARDS.make_plan(inventory, 2))
        current = {tests[1]["id"]: 3}
        plan = SHARDS.make_plan(inventory, 2, dict(stale, **current))
        self.assertEqual(plan["duration_history"], current)
        self.assertEqual(plan["inventory"], inventory)
        SHARDS.validate_plan(plan)
        self.assertEqual(sorted(test_id for shard in plan["shards"] for test_id in shard["test_ids"]),
                         sorted(test["id"] for test in tests))
        batches = SHARDS.make_batches(tests, 32, plan["duration_history"], 25)
        self.assertEqual(len(batches), 3)  # Unmatched groups stay standalone.

    def test_duration_batches_pack_short_cases_and_leave_slow_or_unknown_alone(self):
        tests = fixture_inventory(["slow", "eight", "seven", "six", "four", "unknown"])["tests"]
        history = {test["id"]: weight for test, weight in zip(tests, [30, 8, 7, 6, 4])}
        batches = SHARDS.make_batches(tests, 32, history, 25)
        self.assertEqual([[test["name"] for test in batch] for batch in batches],
                         [["slow"], ["eight", "seven", "six", "four"], ["unknown"]])
        self.assertEqual(batches, SHARDS.make_batches(list(reversed(tests)), 32, history, 25))
        self.assertEqual(sorted(test["id"] for batch in batches for test in batch),
                         sorted(test["id"] for test in tests))

    def test_duration_batches_never_mix_binaries_or_split_filter_groups(self):
        tests = fixture_inventory(["Case", "case", "CASE", "other", "third"])["tests"]
        tests[-1] = dict(tests[-1], binary="tools/other")
        history = {test["id"]: 1 for test in tests}
        batches = SHARDS.make_batches(tests, 2, history, 25)
        pair = next(batch for batch in batches if len(batch) == 3)
        self.assertEqual({test["name"] for test in pair}, {"Case", "case", "CASE"})
        self.assertEqual(sorted(map(len, batches)), [1, 1, 3])
        self.assertTrue(all(len({test["binary"] for test in batch}) == 1 for batch in batches))

    def test_duration_batch_count_and_time_caps_both_apply(self):
        tests = fixture_inventory([str(index) for index in range(10)])["tests"]
        history = {test["id"]: 4 for test in tests}
        for limit, target in ((3, 25), (32, 9)):
            batches = SHARDS.make_batches(tests, limit, history, target)
            self.assertTrue(all(len(batch) <= limit for batch in batches))
            self.assertTrue(all(sum(history[test["id"]] for test in batch) <= target for batch in batches))
            self.assertEqual(sum(map(len, batches)), len(tests))
        self.assertEqual(len(SHARDS.make_batches(tests, 32, {}, 25)), 10)

    def test_duration_batch_invalid_parameters_and_resealed_history_fail(self):
        tests = fixture_inventory(["a", "b"])["tests"]
        for invalid in (0, -1, True, float("nan"), float("inf")):
            with self.subTest(target=invalid), self.assertRaises(SHARDS.ShardError):
                SHARDS.make_batches(tests, 2, {}, invalid)
        for invalid in (0, -1, True, 1.5):
            with self.subTest(size=invalid), self.assertRaises(SHARDS.ShardError):
                SHARDS.make_batches(tests, invalid, {}, 25)
        plan = SHARDS.make_plan(fixture_inventory(["a", "b"]), 2)
        plan["duration_history"] = {tests[0]["id"]: -1}
        with self.assertRaises(SHARDS.ShardError):
            SHARDS.validate_plan(SHARDS.sealed({key: item for key, item in plan.items() if key != "digest"}))

    def test_manifest_tampering_and_empty_or_duplicate_shards_fail(self):
        inventory = fixture_inventory(["a", "b"])
        plan = SHARDS.make_plan(inventory, 2)
        plan["inventory"]["seed"] += 1
        with self.assertRaisesRegex(SHARDS.ShardError, "digest"):
            SHARDS.validate_plan(plan)
        inventory = fixture_inventory(["a", "b"])
        with self.assertRaisesRegex(SHARDS.ShardError, "empty"):
            SHARDS.make_plan(inventory, 3)
        plan = SHARDS.make_plan(inventory, 2)
        plan["shards"][1]["test_ids"] = plan["shards"][0]["test_ids"][:]
        plan = SHARDS.sealed({key: item for key, item in plan.items() if key != "digest"})
        with self.assertRaisesRegex(SHARDS.ShardError, "exactly once"):
            SHARDS.validate_plan(plan)

    def test_invalid_history_and_counts_fail(self):
        inventory = fixture_inventory(["a"])
        for invalid in (0, -1, True, float("nan"), float("inf"), "3"):
            with self.subTest(invalid=invalid), self.assertRaises(SHARDS.ShardError):
                SHARDS.make_plan(inventory, 1, {inventory["tests"][0]["id"]: invalid})
        for invalid in (0, -1, True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(SHARDS.ShardError):
                SHARDS.make_plan(inventory, invalid)

    def test_xml_inventory_rejects_duplicates_entities_and_wrong_reporter(self):
        samples = [
            '<MatchingTests>' + ('<TestCase><Name>a</Name><Tags>[ut]</Tags><SourceInfo>'
                                '<File>fixture.cpp</File><Line>1</Line></SourceInfo></TestCase>') * 2 + '</MatchingTests>',
            '<!DOCTYPE a [<!ENTITY x "value">]><MatchingTests/>',
            '<testsuites/>',
            '<MatchingTests><TestCase><Name> a </Name></TestCase></MatchingTests>',
            '<MatchingTests>',
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.xml"
            for sample in samples:
                path.write_text(sample)
                with self.subTest(sample=sample), self.assertRaises(SHARDS.ShardError):
                    SHARDS.parse_inventory_xml(path)


class ProcessTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.build = self.root / "build"
        self.binary = self.build / "tests" / "mock"
        self.binary.parent.mkdir(parents=True)
        self.binary.write_text("#!" + sys.executable + "\n" + textwrap.dedent(MOCK).lstrip())
        self.binary.chmod(0o755)
        self.config = {"cases": [{"name": "one"}, {"name": "two"}, {"name": "three"},
                                 {"name": "four"}, {"name": "daily", "tags": "[daily]"}]}
        self.write_config()
        self.inventory = self.root / "inventory.json"
        self.plan = self.root / "plan.json"

    def write_config(self):
        self.binary.with_suffix(".json").write_text(json.dumps(self.config))

    def cli(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True,
                              text=True, timeout=20)

    def prepare(self, shards=2):
        result = self.cli("inventory", "--build-root", self.build, "--binary", "tests/mock",
                          "--output", self.inventory)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.cli("plan", "--inventory", self.inventory, "--shards", shards, "--output", self.plan)
        self.assertEqual(result.returncode, 0, result.stderr)

    def run_shard(self, index=0, *extra, output=None):
        output = output or self.root / ("run-" + str(index))
        result = self.cli("run", "--plan", self.plan, "--build-root", self.build,
                          "--shard", index, "--output-dir", output, *extra)
        path = output / "result.json"
        return result, json.loads(path.read_text()) if path.is_file() else None

    def test_four_shards_inventory_exclusion_run_and_merge(self):
        self.prepare(shards=4)
        inventory = json.loads(self.inventory.read_text())
        self.assertEqual(len(inventory["tests"]), 4)
        self.assertNotIn("daily", [test["name"] for test in inventory["tests"]])
        arguments = []
        for index in range(4):
            process, result = self.run_shard(index)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertTrue(result["success"])
            self.assertEqual(result["seed"], 424242)
            arguments.extend(["--result", self.root / f"run-{index}" / "result.json"])
        merged = self.cli("merge", "--plan", self.plan, *arguments, "--output", self.root / "summary.json")
        self.assertEqual(merged.returncode, 0, merged.stderr)
        summary = json.loads((self.root / "summary.json").read_text())
        self.assertTrue(summary["success"])
        self.assertEqual(summary["observed_count"], 4)
        self.assertEqual(len(summary["duration_history"]), 4)
        replanned = self.cli("plan", "--inventory", self.inventory, "--shards", 4,
                            "--durations", self.root / "summary.json", "--output", self.root / "replan.json")
        self.assertEqual(replanned.returncode, 0, replanned.stderr)

    def test_same_name_cases_stay_together_and_merge_with_distinct_identities(self):
        name = "Multi-Hierarchy: Add after Build"
        self.config["cases"] = [
            {"name": name, "tags": "[ft][multi_hierarchy][pipnn][pyramid]", "source_line": 1689},
            {"name": name, "tags": "[ft][multi_hierarchy][pyramid]", "source_line": 1806},
            {"name": "other"},
        ]
        self.write_config()
        self.prepare(shards=2)
        plan = json.loads(self.plan.read_text())
        cases = plan["inventory"]["tests"]
        duplicates = [case for case in cases if case["name"] == name]
        self.assertEqual(len({case["id"] for case in duplicates}), 2)
        history = {case["id"]: weight for case, weight in zip(duplicates, [4, 7])}
        weighted = SHARDS.make_plan(plan["inventory"], 2, history)
        owner = next(shard for shard in weighted["shards"] if duplicates[0]["id"] in shard["test_ids"])
        self.assertEqual(set(owner["test_ids"]), {case["id"] for case in duplicates})
        self.assertEqual(owner["estimated_seconds"], 11)
        results = []
        for index in range(2):
            process, result = self.run_shard(index, "--batch-size", 1, "--workers", 2)
            self.assertEqual(process.returncode, 0, process.stderr)
            results.append(result)
        duplicate_batch = next(batch for result in results for batch in result["batches"] if len(batch["tests"]) == 2)
        self.assertEqual({test["source_line"] for test in duplicate_batch["tests"]}, {1689, 1806})
        summary = SHARDS.merge_results(plan, results)
        self.assertTrue(summary["success"])
        self.assertEqual(summary["observed_count"], 3)
        self.assertEqual(len(summary["duration_history"]), 3)
        duplicate_batch["tests"][0]["tags"] = "[wrong]"
        tampered = [SHARDS.sealed({key: value for key, value in result.items() if key != "digest"}) for result in results]
        self.assertFalse(SHARDS.merge_results(plan, tampered)["success"])

    def test_resealed_plan_cannot_split_same_name_group(self):
        self.config["cases"] = [{"name": "same", "tags": "[a]"},
                                 {"name": "same", "tags": "[b]"}, {"name": "other"}]
        self.write_config()
        self.prepare(shards=2)
        plan = json.loads(self.plan.read_text())
        pair = [case["id"] for case in plan["inventory"]["tests"] if case["name"] == "same"]
        owner = next(shard for shard in plan["shards"] if pair[0] in shard["test_ids"])
        other = plan["shards"][1 - owner["index"]]
        owner["test_ids"].remove(pair[0])
        other["test_ids"].append(pair[0])
        with self.assertRaisesRegex(SHARDS.ShardError, "indivisible"):
            SHARDS.validate_plan(SHARDS.sealed({key: value for key, value in plan.items() if key != "digest"}))

    def test_same_name_wrong_identity_or_missing_case_cannot_pass(self):
        for changed in ({"report_tags": "[wrong]"}, {"report_file": "wrong.cpp"},
                        {"report_line": 99}, {"omit": True}, {"duplicate": True}):
            self.config["cases"] = [{"name": "same", "tags": "[a]", **changed},
                                     {"name": "same", "tags": "[b]"}]
            self.write_config()
            self.prepare(shards=1)
            process, result = self.run_shard(output=self.root / ("identity-" + next(iter(changed))))
            self.assertEqual(process.returncode, 1, process.stderr)
            self.assertFalse(result["success"])

    def test_special_names_exact_selection_and_batch_reuse(self):
        self.config["cases"] = [{"name": name} for name in ('comma, name', 'brackets [a]', 'quote "x"',
                                                            'back\\slash', '中文', 'tilde ~name')]
        self.write_config()
        self.prepare(shards=1)
        process, result = self.run_shard(0, "--batch-size", 3)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(len(result["batches"]), 2)
        self.assertTrue(all(len(batch["tests"]) == 3 for batch in result["batches"]))

    def test_duration_batch_cli_keeps_complete_identity_and_merge(self):
        self.config["cases"] = [{"name": str(index)} for index in range(20)]
        self.config["cases"].extend([{"name": "Int8 SIMD Compute"}, {"name": "INT8 SIMD Compute"}])
        self.write_config()
        self.prepare(shards=2)
        inventory = json.loads(self.inventory.read_text())
        history = {test["id"]: 1 for test in inventory["tests"]}
        plan = SHARDS.make_plan(inventory, 2, history)
        SHARDS.write_json(self.plan, plan)
        results = []
        for index in range(2):
            process, result = self.run_shard(index, "--workers", 3, "--batch-size", 32,
                                             "--batch-target-seconds", 25)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(len(result["batches"]), 1)
            self.assertEqual(result["batch_target_seconds"], 25)
            results.append(result)
        summary = SHARDS.merge_results(plan, results)
        self.assertTrue(summary["success"])
        self.assertEqual(summary["observed_count"], 22)

    def test_parallel_limit_and_workspace_isolation(self):
        events = self.root / "events.jsonl"
        self.config["events"] = str(events)
        self.config["cases"] = [{"name": str(index), "sleep": 0.2} for index in range(6)]
        self.write_config()
        self.prepare(shards=1)
        process, result = self.run_shard(0, "--workers", 2)
        self.assertEqual(process.returncode, 0, process.stderr)
        rows = [json.loads(line) for line in events.read_text().splitlines()]
        active = peak = 0
        for event in sorted(rows, key=lambda row: row["time"]):
            active += 1 if event["phase"] == "start" else -1
            peak = max(peak, active)
        self.assertEqual(peak, 2)
        starts = [event for event in rows if event["phase"] == "start"]
        self.assertEqual(len({event["cwd"] for event in starts}), 6)
        self.assertEqual(len({event["tmp"] for event in starts}), 6)
        self.assertTrue(all(event["seed"] == "424242" for event in starts))
        self.assertEqual(active, 0)

    def test_case_insensitive_names_are_one_complete_group(self):
        self.config["cases"] = [{"name": "Int8 SIMD Compute"}, {"name": "INT8 SIMD Compute"},
                                 {"name": "other"}]
        self.write_config()
        self.prepare(shards=2)
        results = []
        for index in range(2):
            process, result = self.run_shard(index, "--batch-size", 1)
            self.assertEqual(process.returncode, 0, process.stderr)
            results.append(result)
        pair = next(batch for result in results for batch in result["batches"] if len(batch["tests"]) == 2)
        self.assertEqual({test["name"] for test in pair["tests"]}, {"Int8 SIMD Compute", "INT8 SIMD Compute"})
        self.assertEqual(len({test["id"] for test in pair["tests"]}), 2)
        self.assertTrue(SHARDS.merge_results(json.loads(self.plan.read_text()), results)["success"])

    def test_ascii_case_normalization_does_not_fold_unicode(self):
        self.assertEqual(SHARDS.filter_name_key("INT8 SIMD"), "int8 simd")
        self.assertNotEqual(SHARDS.filter_name_key("Ä"), SHARDS.filter_name_key("ä"))
        self.config["cases"] = [{"name": "Ä"}, {"name": "ä"}]
        self.write_config()
        self.prepare(shards=2)
        for index in range(2):
            process, result = self.run_shard(index)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(len(result["batches"][0]["tests"]), 1)

    def test_boundary_wildcard_filter_cannot_silently_select_extra_cases(self):
        self.config["cases"] = [{"name": "a*"}, {"name": "another"}]
        self.write_config()
        self.prepare(shards=2)
        plan = json.loads(self.plan.read_text())
        inventory = {test["id"]: test for test in plan["inventory"]["tests"]}
        index = next(shard["index"] for shard in plan["shards"]
                     if inventory[shard["test_ids"][0]]["name"] == "a*")
        process, result = self.run_shard(index)
        self.assertEqual(process.returncode, 1)
        self.assertIn("unexpected", result["batches"][0]["error"])

    def test_invalid_listing_seed_timeout_and_zero_cases_fail(self):
        for option, value in (("--seed", 0), ("--seed", 2**32), ("--timeout", -1), ("--timeout", "nan")):
            result = self.cli("inventory", "--build-root", self.build, "--binary", "tests/mock",
                              "--output", self.inventory, option, value)
            self.assertEqual(result.returncode, 2)
        self.config["cases"] = [{"name": "daily", "tags": "[daily]"}]
        self.write_config()
        result = self.cli("inventory", "--build-root", self.build, "--binary", "tests/mock", "--output", self.inventory)
        self.assertEqual(result.returncode, 2)
        self.assertIn("No tests", result.stderr)

    def test_inventory_failure_retains_diagnostics_and_refuses_reused_directory(self):
        self.config.update(listing_exit=9, listing_message='mock dynamic-library error')
        self.write_config()
        diagnostics = self.root / 'diagnostics'
        command = ('inventory', '--build-root', self.build, '--binary', 'tests/mock',
                   '--output', self.inventory, '--diagnostics-dir', diagnostics)
        failed = self.cli(*command)
        self.assertEqual(failed.returncode, 2, failed.stderr)
        self.assertIn('exit 9', failed.stderr)
        log = diagnostics / '0/inventory.log'
        self.assertIn('mock dynamic-library error', log.read_text())
        self.assertTrue((diagnostics / '0/inventory.xml').is_file())
        self.assertFalse(self.inventory.exists())
        before = log.read_text()
        repeated = self.cli(*command)
        self.assertEqual(repeated.returncode, 2)
        self.assertEqual(log.read_text(), before)
        invalid = self.cli('inventory', '--build-root', self.build, '--binary', 'tests/mock',
                           '--output', self.inventory, '--diagnostics-dir', self.build / 'logs')
        self.assertEqual(invalid.returncode, 2)
        self.assertIn('outside the build root', invalid.stderr)

    def test_inventory_sigterm_reaps_listing_process(self):
        started = self.root / "listing.pid"
        self.config.update(listing_started=str(started), listing_sleep=30)
        self.write_config()
        process = subprocess.Popen([sys.executable, str(SCRIPT), "inventory", "--build-root", str(self.build),
                                    "--binary", "tests/mock", "--output", str(self.inventory),
                                    "--diagnostics-dir", str(self.root / "cancelled-listing")],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 5
            while not started.exists() and time.monotonic() < deadline:
                time.sleep(0.025)
            self.assertTrue(started.exists())
            process.send_signal(signal.SIGTERM)
            process.communicate(timeout=5)
            self.assertEqual(process.returncode, 143)
            self.assertFalse(self.inventory.exists())
            self.assertTrue((self.root / "cancelled-listing/0/inventory.log").is_file())
            self.assert_process_stopped(int(started.read_text()))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    def test_binary_hash_drift_is_rejected_before_run(self):
        self.prepare()
        self.binary.write_text(self.binary.read_text() + "\n# changed\n")
        process, result = self.run_shard()
        self.assertEqual(process.returncode, 2)
        self.assertIn("digest mismatch", process.stderr)
        self.assertIsNone(result)

    def test_existing_or_in_artifact_output_is_rejected(self):
        self.prepare()
        for output in (self.root, self.build / "output"):
            process, _result = self.run_shard(output=output)
            self.assertEqual(process.returncode, 2, process.stderr)

    def test_partial_duplicate_crash_and_nonzero_results_cannot_pass(self):
        for change in ({"omit": True}, {"duplicate": True}, {"crash": True}, {"fail": True}):
            with self.subTest(change=change):
                self.config["cases"] = [{"name": "single", **change}]
                self.write_config()
                self.prepare(shards=1)
                output = self.root / ("run-" + next(iter(change)))
                process, result = self.run_shard(output=output)
                self.assertEqual(process.returncode, 1, process.stderr)
                self.assertFalse(result["success"])
                if change.get("crash"):
                    self.assertEqual(result["batches"][0]["exit_code"], 17)
                merged = SHARDS.merge_results(json.loads(self.plan.read_text()), [result])
                self.assertFalse(merged["success"])

    def test_zero_exit_with_failed_xml_and_failed_exit_with_good_xml_both_fail(self):
        for failed_xml in (False, True):
            self.config["cases"] = [{"name": "single", "fail": failed_xml}]
            self.config["execution_exit"] = 0 if failed_xml else 23
            self.write_config()
            self.prepare(shards=1)
            process, result = self.run_shard(output=self.root / str(failed_xml))
            self.assertEqual(process.returncode, 1)
            self.assertFalse(result["success"])

    def test_all_skipped_catch2_exit_is_accepted_only_with_complete_skip_evidence(self):
        self.config["cases"] = [{"name": "existing skip", "skip": True}, {"name": "normal"}]
        self.write_config()
        self.prepare(shards=1)
        process, result = self.run_shard(0, "--batch-size", 1)
        self.assertEqual(process.returncode, 0, process.stderr)
        skipped = next(batch for batch in result["batches"] if batch["exit_code"] == 4)
        self.assertTrue(skipped["all_skipped"])
        self.assertEqual(skipped["tests"][0]["skips"], 1)
        plan = json.loads(self.plan.read_text())
        self.assertTrue(SHARDS.merge_results(plan, [result])["success"])
        skipped["tests"][0]["skips"] = 0
        forged = SHARDS.sealed({key: value for key, value in result.items() if key != "digest"})
        self.assertFalse(SHARDS.merge_results(plan, [forged])["success"])
        self.config["cases"] = [{"name": "normal"}]
        self.config["execution_exit"] = 4
        self.write_config()
        self.prepare(shards=1)
        process, result = self.run_shard(output=self.root / "unproven-exit4")
        self.assertEqual(process.returncode, 1)
        self.assertFalse(result["success"])

    def test_timeout_stops_child_group_and_preserves_exit_code(self):
        child_pid = self.root / "child.pid"
        self.config.update(cases=[{"name": "slow", "spawn_child": True, "sleep": 30}], child_pid=str(child_pid))
        self.write_config()
        self.prepare(shards=1)
        process, result = self.run_shard(0, "--timeout", 0.5)
        self.assertEqual(process.returncode, 1, process.stderr)
        self.assertEqual(result["batches"][0]["status"], "timeout")
        self.assertLess(result["batches"][0]["exit_code"], 0)
        self.assert_process_stopped(int(child_pid.read_text()))

    def assert_process_stopped(self, pid):
        for _ in range(40):
            if sys.platform.startswith("linux"):
                try:
                    # Linux CI images need no procps package for this assertion.
                    state = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[0]
                except FileNotFoundError:
                    return
            else:
                probe = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
                state = probe.stdout.strip()
                if probe.returncode:
                    return
            if not state or state.startswith("Z"):
                return
            time.sleep(0.025)
        self.fail(f"Child {pid} still running ({state})")

    def test_sigterm_cancels_running_and_pending_batches(self):
        events = self.root / "events.jsonl"
        self.config.update(cases=[{"name": str(index), "sleep": 30} for index in range(4)], events=str(events))
        self.write_config()
        self.prepare(shards=1)
        output = self.root / "cancelled"
        process = subprocess.Popen([sys.executable, str(SCRIPT), "run", "--plan", str(self.plan),
                                    "--build-root", str(self.build), "--shard", "0", "--workers", "2",
                                    "--output-dir", str(output)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 5
            while not events.exists() and time.monotonic() < deadline:
                time.sleep(0.025)
            self.assertTrue(events.exists())
            process.send_signal(signal.SIGTERM)
            _stdout, stderr = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 143, stderr.decode())
            result = json.loads((output / "result.json").read_text())
            self.assertFalse(result["success"])
            self.assertEqual(result["signal"], signal.SIGTERM)
            self.assertTrue(all(batch["status"] == "cancelled" for batch in result["batches"]))
            for event in map(json.loads, events.read_text().splitlines()):
                self.assert_process_stopped(event["pid"])
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    def test_merge_checks_case_outcome_independently_of_batch_status(self):
        self.prepare(shards=1)
        process, result = self.run_shard(0, "--batch-size", 4)
        self.assertEqual(process.returncode, 0, process.stderr)
        plan = json.loads(self.plan.read_text())
        for field, value in (("success", False), ("success", "true"), ("skips", -1), ("skips", True)):
            changed = json.loads(json.dumps(result))
            changed["batches"][0]["tests"][0][field] = value
            changed = SHARDS.sealed({key: item for key, item in changed.items() if key != "digest"})
            self.assertFalse(SHARDS.merge_results(plan, [changed])["success"])

    def test_merge_rejects_missing_duplicate_and_cross_plan_results(self):
        self.prepare()
        _, first = self.run_shard(0)
        _, second = self.run_shard(1)
        plan = json.loads(self.plan.read_text())
        self.assertFalse(SHARDS.merge_results(plan, [first])["success"])
        self.assertFalse(SHARDS.merge_results(plan, [first, first, second])["success"])
        second["context"] = {"attempt": 999}
        second = SHARDS.sealed({key: item for key, item in second.items() if key != "digest"})
        with self.assertRaisesRegex(SHARDS.ShardError, "different plan"):
            SHARDS.merge_results(plan, [first, second])

    def test_unsafe_binary_paths_fail_without_executing(self):
        for binary in ("../escape", "/bin/sh", "tests/../tests/mock", "tests\\mock"):
            result = self.cli("inventory", "--build-root", self.build, "--binary", binary,
                              "--output", self.inventory)
            self.assertEqual(result.returncode, 2, result.stderr)
        (self.build / "escape").symlink_to(sys.executable)
        result = self.cli("inventory", "--build-root", self.build, "--binary", "escape", "--output", self.inventory)
        self.assertEqual(result.returncode, 2, result.stderr)


if __name__ == "__main__":
    unittest.main()
