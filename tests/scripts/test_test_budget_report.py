"""Budget-report tests consume small manifests only; no C++ tests or network."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("test_budget_report", ROOT / "scripts/testing/test_budget_report.py")
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def seal(value):
    return dict(value, digest=REPORT.digest(value))


def case(name="HGraph Build", binary="tests/functests", line=10):
    value = {"name": name, "binary": binary, "tags": "[ft]", "source_file": "tests/fixture.cpp", "source_line": line}
    return dict(value, id=REPORT.stable_id(binary, value))


def inventory(cases):
    return seal({"kind": "inventory", "schema_version": 3, "context": {"run_id": "123", "run_attempt": "1", "test_sha": "a" * 40},
                 "seed": 424242, "filter": "~[daily]", "binaries": {item["binary"]: "a" * 64 for item in cases}, "tests": cases})


def measured(value, seconds=2, success=True):
    return dict(value, seconds=seconds, success=success, skips=0)


def result(inv, cases, success=True):
    batches = [{"binary": item["binary"], "tests": [item]} for item in cases]
    return seal({"kind": "result", "schema_version": 3, "context": inv["context"], "seed": inv["seed"],
                 "success": success, "batches": batches})


def cost(value, cap=100, points=30, model="logical-v1", unit="logical_points"):
    return {"schema_version": 1, "model_version": model, "unit": unit, "policy": "pr",
            "case": {field: value[field] for field in REPORT.CASE_FIELDS}, "cap_points": cap,
            "total_points": points, "seed": 47, "selected_candidate": 0, "run_count": 1,
            "runs": [{"key": "example", "dim": 3, "num": 10, "cost_points": points, "cap_points": 999999}]}


class BudgetReportTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.a = case()
        self.b = case("HGraph Search", line=20)
        self.c = case("Unclassified contract", line=30)
        self.inv = inventory([self.a, self.b, self.c])

    def write_json(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return path

    def write_listing(self, name, cases):
        root = ET.Element("MatchingTests")
        for value in cases:
            node = ET.SubElement(root, "TestCase")
            ET.SubElement(node, "Name").text = value["name"]
            ET.SubElement(node, "Tags").text = value["tags"]
            source = ET.SubElement(node, "SourceInfo")
            ET.SubElement(source, "File").text = value["source_file"]
            ET.SubElement(source, "Line").text = str(value["source_line"])
        path = self.root / name
        path.write_bytes(ET.tostring(root))
        return path

    def test_list_tests_inventory_retains_template_and_cross_binary_cases(self):
        template = case("HGraph variant - fp32", line=11)
        daily = dict(case("Daily variant", line=12), tags="[daily]")
        listing = self.write_listing("list.xml", [self.a, template, daily])
        inv = REPORT.list_inventory([f"tests/functests={listing}", f"tests/unittests={listing}"])
        self.assertEqual(len(inv["tests"]), 6)
        self.assertEqual(len({item["id"] for item in inv["tests"]}), 6)
        self.assertEqual(inv["tests"][0]["id"], self.a["id"])
        self.assertIsNone(inv["binaries"]["tests/functests"]["binary_sha256"])
        report = REPORT.make_report(REPORT.Inputs(inv))
        self.assertEqual(report["cost_unknown_count"], 6)
        self.assertEqual(report["case_seconds"]["unknown_count"], 6)

    def test_list_tests_absolute_source_normalizes_with_explicit_root(self):
        listing = self.write_listing("list.xml", [dict(self.a, source_file="/ci/tree/tests/fixture.cpp")])
        inv = REPORT.list_inventory([f"tests/functests={listing}"], ["/ci/tree"])
        self.assertEqual(inv["tests"][0]["id"], self.a["id"])
        value = cost(self.a)
        value["case"].update(tags="", source_line=0, source_file="/ci/tree/tests/fixture.cpp")
        inputs = REPORT.Inputs(inv, ["/ci/tree"])
        inputs.cost("tests/functests", value)
        self.assertEqual(REPORT.make_report(inputs)["cost_known_count"], 1)

    def test_list_tests_rejects_duplicates_empty_wrong_reporter_and_entities(self):
        listing = self.write_listing("list.xml", [self.a, self.a])
        with self.assertRaises(REPORT.ReportError):
            REPORT.list_inventory([f"tests/functests={listing}"])
        for text in ("<MatchingTests/>", "<Catch2TestRun/>",
                     '<!DOCTYPE a [<!ENTITY x "v">]><MatchingTests/>',
                     "<MatchingTests><TestCase><Name>x</Name></TestCase></MatchingTests>"):
            listing.write_text(text)
            with self.subTest(text=text), self.assertRaises(REPORT.ReportError):
                REPORT.list_inventory([f"tests/functests={listing}"])
        self.write_listing("list.xml", [self.a])
        with self.assertRaises(REPORT.ReportError):
            REPORT.list_inventory([f"tests/functests={listing}"] * 2)

    def test_cli_accepts_standalone_list_tests_without_team_inventory(self):
        listing = self.write_listing("list.xml", [self.a])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = REPORT.main(["--list-tests", f"tests/functests={listing}", "--format", "json"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["selected_case_count"], 1)

    def test_inventory_keeps_cross_binary_duplicates(self):
        other = case(binary="tests/unittests")
        inv = inventory([self.a, other])
        loaded = REPORT.load_inventory(self.write_json("inventory.json", inv))
        report = REPORT.make_report(REPORT.Inputs(loaded))
        self.assertEqual(report["selected_case_count"], 2)
        self.assertNotEqual(self.a["id"], other["id"])
        self.assertEqual(report["binary_counts"], {"tests/functests": 1, "tests/unittests": 1})

    def test_counts_caps_cost_and_seconds_have_distinct_denominators(self):
        inputs = REPORT.Inputs(self.inv)
        inputs.result(result(self.inv, [measured(self.a, 3), measured(self.b, 1)]))
        inputs.cost(self.a["binary"], cost(self.a, cap=100, points=10))
        inputs.cost(self.b["binary"], cost(self.b, cap=300, points=90))
        report = REPORT.make_report(inputs)
        counts = {item["group"]: item for item in report["case_counts"]}
        timings = {item["group"]: item for item in report["case_seconds"]["groups"]}
        caps = {item["group"]: item for item in report["modeled_costs"][0]["cap"]["groups"]}
        estimates = {item["group"]: item for item in report["modeled_costs"][0]["estimated_cost"]["groups"]}
        self.assertAlmostEqual(counts["build_train"]["share"], 1 / 3)
        self.assertEqual(timings["build_train"]["share_of_known_total"], 0.75)
        self.assertEqual(caps["build_train"]["share_of_known_total"], 0.25)
        self.assertEqual(estimates["build_train"]["share_of_known_total"], 0.1)
        self.assertIsNone(caps["other"]["value"])
        self.assertEqual(caps["other"]["unknown_count"], 1)
        self.assertEqual(report["modeled_costs"][0]["cap"]["total"], 400)
        self.assertEqual(report["cost_unknown_count"], 1)

    def test_no_cost_data_produces_timing_baseline_not_zero_budget(self):
        inputs = REPORT.Inputs(self.inv)
        inputs.result(result(self.inv, [measured(self.a, 2)]))
        report = REPORT.make_report(inputs)
        self.assertEqual(report["cost_unknown_count"], 3)
        self.assertEqual(report["modeled_costs"], [])
        self.assertEqual(report["case_seconds"]["unknown_count"], 2)
        self.assertIn("unavailable", REPORT.markdown(report))
        self.assertIn("not wall time or CPU time", REPORT.markdown(report))

    def test_zero_measurement_is_known_but_zero_total_has_no_share(self):
        inputs = REPORT.Inputs(inventory([self.a]))
        inputs.timing(self.a["binary"], measured(self.a, 0))
        inputs.cost(self.a["binary"], cost(self.a, points=0))
        report = REPORT.make_report(inputs)
        self.assertEqual(report["case_seconds"]["known_count"], 1)
        self.assertEqual(report["case_seconds"]["total"], 0)
        self.assertIsNone(report["case_seconds"]["groups"][0]["share_of_known_total"])
        self.assertIsNone(report["modeled_costs"][0]["estimated_cost"]["groups"][0]["share_of_known_total"])

    def test_model_units_and_policy_are_not_mixed(self):
        inputs = REPORT.Inputs(self.inv)
        inputs.cost(self.a["binary"], cost(self.a))
        inputs.cost(self.b["binary"], cost(self.b, model="logical-v2"))
        inputs.cost(self.c["binary"], dict(cost(self.c), policy="daily"))
        report = REPORT.make_report(inputs)
        self.assertEqual(len(report["modeled_costs"]), 3)
        self.assertTrue(all(model["cap"]["known_count"] == 1 for model in report["modeled_costs"]))

    def test_over_cap_and_failed_cases_remain_visible(self):
        inputs = REPORT.Inputs(self.inv)
        inputs.result(result(self.inv, [measured(self.a, success=False)], success=False))
        inputs.cost(self.a["binary"], cost(self.a, cap=10, points=11))
        report = REPORT.make_report(inputs)
        self.assertEqual(report["over_cap_count"], 1)
        self.assertEqual(report["measured_failed_cases"], 1)
        self.assertEqual(report["failed_input_results"], 1)

    def test_category_priority_is_exclusive_and_other_is_retained(self):
        self.assertEqual(REPORT.classify(case("Tune search and serialization")), "tune")
        self.assertEqual(REPORT.classify(case("recover saved index")), "recovery")
        self.assertEqual(REPORT.classify(case("Concurrent serialize and search")), "serialization")
        self.assertEqual(REPORT.classify(case("Concurrent search")), "mutation_concurrency")
        self.assertEqual(REPORT.classify(case("SIMD quantization")), "data_quant_simd")
        self.assertEqual(REPORT.classify(self.c), "other")

    def test_family_prefers_source_owner_and_supports_binary_filter(self):
        item = dict(self.a, name="compare IVF", source_file="src/algorithm/hgraph/hgraph_test.cpp")
        self.assertEqual(REPORT.family(item), "hgraph")
        inv = inventory([self.a, case(binary="tests/unittests")])
        report = REPORT.make_report(REPORT.Inputs(inv), ["tests/functests"], "family")
        self.assertEqual(report["selected_case_count"], 1)
        self.assertEqual(report["inventory_case_count"], 2)
        self.assertEqual(report["case_counts"][0]["group"], "hgraph")

    def test_json_and_zip_inputs(self):
        archive_path = self.root / "plan.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("inventory.json", json.dumps(self.inv))
        loaded = REPORT.load_inventory(archive_path)
        inputs = REPORT.Inputs(loaded)
        result_zip = self.root / "results.zip"
        with zipfile.ZipFile(result_zip, "w") as archive:
            archive.writestr("shard-0/result.json", json.dumps(result(self.inv, [measured(self.a)])))
        inputs.result_path(result_zip)
        self.assertEqual(REPORT.make_report(inputs)["case_seconds"]["total"], 2)

    def test_cost_jsonl_and_prefixed_log_ignore_unrelated_json(self):
        raw = json.dumps(cost(self.a))
        for prefix in ("", 'unrelated output\n{"not": "a cost record"}\n[test-budget] '):
            with self.subTest(prefix=prefix):
                path = self.root / "cost.log"
                path.write_text(prefix + raw + "\n")
                inputs = REPORT.Inputs(self.inv)
                inputs.cost_path(self.a["binary"], path)
                self.assertEqual(len(inputs.costs), 1)

    def test_xml_exact_identity_and_binary_binding(self):
        root = ET.Element("Catch2TestRun")
        node = ET.SubElement(root, "TestCase", name=self.a["name"], tags=self.a["tags"],
                             filename=self.a["source_file"], line=str(self.a["source_line"]))
        ET.SubElement(node, "OverallResult", success="true", durationInSeconds="2.5", skips="0")
        path = self.root / "report.xml"
        path.write_bytes(ET.tostring(root))
        inputs = REPORT.Inputs(self.inv)
        inputs.xml_path(self.a["binary"], path)
        self.assertEqual(inputs.timings[self.a["id"]]["seconds"], 2.5)
        with self.assertRaises(REPORT.ReportError):
            REPORT.Inputs(self.inv).xml_path("tests/wrong-binary", path)

    def test_cost_xml_decodes_names_multiple_stdout_and_identical_plans(self):
        escaped = case("Build & ContinueAdd <exact>")
        inv = inventory([escaped, self.b])
        root = ET.Element("Catch2TestRun")
        first = ET.SubElement(root, "TestCase", name=escaped["name"], tags=escaped["tags"],
                              filename=escaped["source_file"], line=str(escaped["source_line"]))
        ET.SubElement(first, "OverallResult", success="true", durationInSeconds="1.25", skips="0")
        line = "[test-budget] " + json.dumps(cost(escaped)) + "\n"
        ET.SubElement(first, "StdOut").text = "unrelated text\n" + line
        section = ET.SubElement(first, "Section")
        ET.SubElement(section, "StdOut").text = line
        second = ET.SubElement(root, "TestCase", name=self.b["name"], tags=self.b["tags"],
                               filename=self.b["source_file"], line=str(self.b["source_line"]))
        ET.SubElement(second, "OverallResult", success="true", durationInSeconds="2.75", skips="0")
        ET.SubElement(second, "StdOut").text = "[test-budget] " + json.dumps(cost(self.b))
        path = self.root / "report.xml"
        path.write_bytes(ET.tostring(root))
        self.assertIn(b"&amp;", path.read_bytes())
        inputs = REPORT.Inputs(inv)
        inputs.xml_path(escaped["binary"], path)
        self.assertEqual(inputs.costs, {})
        inputs.cost_path(escaped["binary"], path)
        report = REPORT.make_report(inputs)
        self.assertEqual(report["case_seconds"]["total"], 4)
        self.assertEqual(report["cost_known_count"], 2)
        self.assertEqual(report["modeled_costs"][0]["cap"]["total"], 200)
        self.assertEqual(inputs.costs[escaped["id"]]["identical_plan_repetitions"], 2)

    def test_cost_utf16_same_file_supports_xml_and_cost_cli(self):
        value = case("Build & ContinueAdd <exact>")
        inv_path = self.write_json("inventory.json", inventory([value]))
        root = ET.Element("Catch2TestRun")
        node = ET.SubElement(root, "TestCase", name=value["name"], tags=value["tags"],
                             filename=value["source_file"], line=str(value["source_line"]))
        ET.SubElement(node, "OverallResult", success="true", durationInSeconds="2.5", skips="0")
        ET.SubElement(node, "StdOut").text = "[test-budget] " + json.dumps(cost(value))
        document = ('<?xml version="1.0" encoding="UTF-16"?>'
                    + ET.tostring(root, encoding="unicode"))
        for encoding in ("utf-16", "utf-16-le", "utf-16-be"):
            for suffix in (".xml", ".log"):
                with self.subTest(encoding=encoding, suffix=suffix):
                    path = self.root / ("report" + suffix)
                    path.write_bytes(document.encode(encoding))
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        code = REPORT.main(["--inventory", str(inv_path),
                                            "--xml", f"tests/functests={path}",
                                            "--cost", f"tests/functests={path}", "--format", "json"])
                    self.assertEqual(code, 0)
                    report = json.loads(output.getvalue())
                    self.assertEqual(report["case_seconds"]["total"], 2.5)
                    self.assertEqual(report["cost_known_count"], 1)
                    self.assertEqual(report["modeled_costs"][0]["cap"]["total"], 100)
                    self.assertEqual(report["modeled_costs"][0]["estimated_cost"]["total"], 30)

    def test_cost_utf16_dtds_still_use_encoding_independent_guard(self):
        document = ('<?xml version="1.0" encoding="UTF-16"?>'
                    '<!DOCTYPE Catch2TestRun [<!ENTITY value "expanded">]>'
                    '<Catch2TestRun><StdOut>&value;</StdOut></Catch2TestRun>')
        for encoding in ("utf-16", "utf-16-le", "utf-16-be"):
            for suffix in (".xml", ".log"):
                with self.subTest(encoding=encoding, suffix=suffix):
                    path = self.root / ("report" + suffix)
                    path.write_bytes(document.encode(encoding))
                    inputs = REPORT.Inputs(self.inv)
                    with self.assertRaisesRegex(REPORT.ReportError,
                                                "XML entities and DTDs are not accepted"):
                        inputs.cost_path(self.a["binary"], path)
                    self.assertEqual(inputs.costs, {})

    def test_cost_utf8_bom_jsonl_remains_text(self):
        path = self.root / "cost.log"
        path.write_bytes(json.dumps(cost(self.a)).encode("utf-8-sig"))
        inputs = REPORT.Inputs(self.inv)
        inputs.cost_path(self.a["binary"], path)
        self.assertEqual(len(inputs.costs), 1)

    def test_cost_xml_only_reads_marked_stdout_and_allows_unmodeled_cases(self):
        root = ET.Element("Catch2TestRun")
        ET.SubElement(root, "StdOut").text = '{"not": "a budget"}\nordinary log'
        ET.SubElement(root, "StdErr").text = "[test-budget] " + json.dumps(cost(self.a))
        path = self.root / "report.xml"
        path.write_bytes(ET.tostring(root))
        inputs = REPORT.Inputs(self.inv)
        inputs.cost_path(self.a["binary"], path)
        report = REPORT.make_report(inputs)
        self.assertEqual(report["cost_known_count"], 0)
        self.assertEqual(report["cost_unknown_count"], 3)

    def test_cost_xml_malformed_marked_payload_reporter_or_entities_are_errors(self):
        path = self.root / "report.xml"
        for text in ("<Catch2TestRun><StdOut>[test-budget] broken</StdOut></Catch2TestRun>",
                     "<MatchingTests/>", "broken xml",
                     '<!DOCTYPE a [<!ENTITY x "v">]><Catch2TestRun/>'):
            path.write_text(text)
            with self.subTest(text=text), self.assertRaises(REPORT.ReportError):
                REPORT.Inputs(self.inv).cost_path(self.a["binary"], path)

    def test_duplicate_measurement_and_differing_cost_are_rejected(self):
        inputs = REPORT.Inputs(self.inv)
        inputs.timing(self.a["binary"], measured(self.a))
        with self.assertRaises(REPORT.ReportError):
            inputs.timing(self.a["binary"], measured(self.a))
        inputs.cost(self.a["binary"], cost(self.a))
        with self.assertRaises(REPORT.ReportError):
            inputs.cost(self.a["binary"], cost(self.a, points=31))

    def test_identical_complete_plans_count_once_and_expose_repetitions(self):
        inputs = REPORT.Inputs(self.inv)
        inputs.cost(self.a["binary"], cost(self.a))
        inputs.cost(self.a["binary"], cost(self.a))
        report = REPORT.make_report(inputs)
        self.assertEqual(report["modeled_costs"][0]["cap"]["total"], 100)
        self.assertEqual(report["modeled_costs"][0]["estimated_cost"]["total"], 30)
        self.assertEqual(inputs.costs[self.a["id"]]["identical_plan_repetitions"], 2)
        changed = cost(self.a)
        changed["runs"][0]["dim"] = 4
        with self.assertRaises(REPORT.ReportError):
            inputs.cost(self.a["binary"], changed)

    def test_partial_bridge_identity_resolves_full_path_and_binary(self):
        value = cost(self.a)
        value["case"].update(tags="", source_line=0, source_file="/ci/checkout/tests/fixture.cpp")
        inputs = REPORT.Inputs(inventory([self.a, case(binary="tests/unittests")]), ["/ci/checkout"])
        inputs.cost(self.a["binary"], value)
        self.assertEqual(list(inputs.costs), [self.a["id"]])
        with self.assertRaises(REPORT.ReportError):
            REPORT.Inputs(self.inv).cost(self.a["binary"], value)
        value["case"]["source_file"] = "./tests/fixture.cpp"
        REPORT.Inputs(self.inv).cost(self.a["binary"], value)

    def test_partial_bridge_ambiguity_basename_and_wrong_source_rejected(self):
        value = cost(self.a)
        value["case"].update(tags="", source_line=0)
        with self.assertRaises(REPORT.ReportError):
            REPORT.Inputs(inventory([self.a, case(line=11)])).cost(self.a["binary"], value)
        for source in ("fixture.cpp", "other/fixture.cpp", "../tests/fixture.cpp"):
            value["case"]["source_file"] = source
            with self.subTest(source=source), self.assertRaises(REPORT.ReportError):
                REPORT.Inputs(self.inv).cost(self.a["binary"], value)
        value["case"].update(source_file="tests/fixture.cpp", tags="[ft]")
        with self.assertRaises(REPORT.ReportError):
            REPORT.Inputs(self.inv).cost(self.a["binary"], value)

    def test_partial_cost_matches_absolute_frozen_inventory_without_rewriting_ids(self):
        absolute = dict(self.a, source_file="/__w/vsag/vsag/tests/fixture.cpp")
        absolute["id"] = REPORT.stable_id(absolute["binary"], absolute)
        loaded = REPORT.load_inventory(self.write_json("frozen.json", inventory([absolute])))
        original_digest = loaded["digest"]
        value = cost(absolute)
        value["case"].update(tags="", source_line=0)
        inputs = REPORT.Inputs(loaded, ["/__w/vsag/vsag"])
        inputs.cost(absolute["binary"], value)
        report = REPORT.make_report(inputs)
        self.assertEqual(report["cost_known_count"], 1)
        self.assertEqual(report["cases"][0]["id"], absolute["id"])
        self.assertEqual(report["cases"][0]["source_file"], absolute["source_file"])
        self.assertEqual(loaded["digest"], original_digest)
        REPORT.verify_seal(loaded)

    def test_partial_absolute_inventory_normalization_still_rejects_ambiguity(self):
        candidates = []
        for prefix in ("/build/one", "/build/two"):
            item = dict(self.a, source_file=prefix + "/tests/fixture.cpp")
            item["id"] = REPORT.stable_id(item["binary"], item)
            candidates.append(item)
        value = cost(candidates[0])
        value["case"].update(tags="", source_line=0, source_file="tests/fixture.cpp")
        inputs = REPORT.Inputs(inventory(candidates), ["/build/one", "/build/two"])
        with self.assertRaises(REPORT.ReportError):
            inputs.cost(self.a["binary"], value)

    def test_catch_parent_relative_source_is_identity_not_file_access(self):
        relative = dict(self.a, source_file="../../../__w/vsag/vsag/tests/fixture.cpp")
        relative["id"] = REPORT.stable_id(relative["binary"], relative)
        frozen = REPORT.load_inventory(self.write_json("frozen.json", inventory([relative])))
        listing = self.write_listing("list.xml", [relative])
        self.assertEqual(REPORT.list_inventory([f"tests/functests={listing}"])["tests"][0]["id"],
                         relative["id"])
        root = ET.Element("Catch2TestRun")
        node = ET.SubElement(root, "TestCase", name=relative["name"], tags=relative["tags"],
                             filename=relative["source_file"], line=str(relative["source_line"]))
        ET.SubElement(node, "OverallResult", success="true", durationInSeconds="2.5", skips="0")
        value = cost(relative)
        value["case"].update(tags="", source_line=0)
        ET.SubElement(node, "StdOut").text = "[test-budget] " + json.dumps(value)
        path = self.root / "report.xml"
        path.write_bytes(ET.tostring(root))
        inputs = REPORT.Inputs(frozen, ["/__w/vsag/vsag"])
        inputs.xml_path(relative["binary"], path)
        inputs.cost_path(relative["binary"], path)
        self.assertEqual(inputs.timings[relative["id"]]["seconds"], 2.5)
        self.assertIn(relative["id"], inputs.costs)
        REPORT.verify_seal(frozen)

    def test_changed_source_line_and_case_id_are_rejected(self):
        for mutation in ({"source_line": 11}, {"id": "0" * 64}):
            with self.subTest(mutation=mutation), self.assertRaises(REPORT.ReportError):
                REPORT.Inputs(self.inv).timing(self.a["binary"], dict(measured(self.a), **mutation))

    def test_result_context_and_seed_are_bound(self):
        for mutation in ({"context": {"run_id": "124"}}, {"seed": 1}):
            value = result(self.inv, [measured(self.a)])
            value.pop("digest")
            value.update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(REPORT.ReportError):
                REPORT.Inputs(self.inv).result(seal(value))

    def test_inventory_digest_and_duplicates_are_checked(self):
        value = dict(self.inv, seed=1)
        with self.assertRaises(REPORT.ReportError):
            REPORT.load_inventory(self.write_json("bad.json", value))
        with self.assertRaises(REPORT.ReportError):
            REPORT.load_inventory(self.write_json("bad.json", inventory([self.a, self.a])))

    def test_invalid_numbers_are_rejected(self):
        for value in (-1, True, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(REPORT.ReportError):
                REPORT.Inputs(self.inv).timing(self.a["binary"], measured(self.a, value))
        with self.assertRaises(REPORT.ReportError):
            REPORT.Inputs(self.inv).cost(self.a["binary"], cost(self.a, cap=0))

    def test_xml_entities_are_rejected(self):
        path = self.root / "report.xml"
        path.write_text('<!DOCTYPE test [<!ENTITY value "x">]><Catch2TestRun/>')
        with self.assertRaises(REPORT.ReportError):
            REPORT.Inputs(self.inv).xml_path(self.a["binary"], path)

    def test_utf16_dtds_and_entities_are_rejected_before_expansion(self):
        path = self.root / "report.xml"
        for encoding in ("utf-16", "utf-16-le", "utf-16-be"):
            for root_name in ("MatchingTests", "Catch2TestRun"):
                for subset in ("", ' [<!ENTITY value "expanded">]'):
                    content = "&value;" if subset else "ordinary text"
                    document = (f'<?xml version="1.0" encoding="UTF-16"?>'
                                f'<!DOCTYPE {root_name}{subset}>'
                                f'<{root_name}>{content}</{root_name}>')
                    raw = document.encode(encoding)
                    # This small payload proves the old byte scan misses the
                    # declaration and ordinary ElementTree expands the entity.
                    self.assertNotIn(b"<!DOCTYPE", raw.upper())
                    self.assertEqual(ET.fromstring(raw).text,
                                     "expanded" if subset else "ordinary text")
                    path.write_bytes(raw)
                    with self.subTest(encoding=encoding, root=root_name, subset=subset):
                        with self.assertRaisesRegex(REPORT.ReportError,
                                                    "XML entities and DTDs are not accepted"):
                            REPORT.xml_root(path, root_name)

    def test_utf16_catch_listing_and_timing_without_dtd_are_accepted(self):
        listing = self.write_listing("listing.xml", [self.a])
        listing_text = listing.read_text()
        report = ET.Element("Catch2TestRun")
        node = ET.SubElement(report, "TestCase", name=self.a["name"], tags=self.a["tags"],
                             filename=self.a["source_file"], line=str(self.a["source_line"]))
        ET.SubElement(node, "OverallResult", success="true", durationInSeconds="2.5", skips="0")
        path = self.root / "report.xml"
        for encoding in ("utf-16", "utf-16-le", "utf-16-be"):
            with self.subTest(encoding=encoding):
                declaration = '<?xml version="1.0" encoding="UTF-16"?>'
                listing.write_bytes((declaration + listing_text).encode(encoding))
                inv = REPORT.list_inventory([f"tests/functests={listing}"])
                self.assertEqual(inv["tests"][0]["id"], self.a["id"])
                path.write_bytes((declaration + ET.tostring(report, encoding="unicode")).encode(encoding))
                inputs = REPORT.Inputs(inv)
                inputs.xml_path(self.a["binary"], path)
                self.assertEqual(inputs.timings[self.a["id"]]["seconds"], 2.5)

    def test_cli_outputs_report_without_running_tests(self):
        inv = self.write_json("inventory.json", self.inv)
        res = self.write_json("result.json", result(self.inv, [measured(self.a)]))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = REPORT.main(["--inventory", str(inv), "--result", str(res), "--format", "json"])
        self.assertEqual(code, 0)
        parsed = json.loads(output.getvalue())
        self.assertEqual(parsed["selected_case_count"], 3)
        self.assertEqual(parsed["case_seconds"]["known_count"], 1)

    def test_cli_error_is_nonzero(self):
        inv = self.write_json("inventory.json", self.inv)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(REPORT.main(["--inventory", str(inv), "--xml", "missing-binary"]), 2)


if __name__ == "__main__":
    unittest.main()
