"""M2 regression: exact examples are balanced across fixed per-shard worker pools."""
import copy
import hashlib
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

from tests.scripts import test_team_test_shards as fixture

SHARDS, SCRIPT = fixture.SHARDS, fixture.SCRIPT

class MixedExamplesTest(unittest.TestCase):
    setUp = fixture.ProcessTest.setUp
    write_config = fixture.ProcessTest.write_config
    def cli(self, *args):
        # This fixture launches 56 separate interpreters on macOS as well as Linux.
        return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True,
                              text=True, timeout=90)
    run_shard = fixture.ProcessTest.run_shard

    def add_examples(self, count=3, sleep=.01, failure=None, spawn_marker=None, barrier=False):
        directory = self.build / "examples/cpp"
        directory.mkdir(parents=True, exist_ok=True)
        self.events = self.root / "events.jsonl"
        for i in range(count):
            name = f"{100+i:03d}_example"
            code = textwrap.dedent("""
                import json,os,pathlib,subprocess,sys,time
                name=NAME
                event={'name':name,'cwd':os.getcwd(),'tmp':os.environ['TMPDIR'],'pid':os.getpid()}
                with open(EVENTS,'a') as output:
                    output.write(json.dumps(dict(event,phase='start',time=time.time()))+'\\n')
                pathlib.Path('same-name').write_text(name)
                pathlib.Path(os.environ['TMPDIR'],'same-name').write_text(name)
                CHILD
                BARRIER
                time.sleep(SLEEP)
                with open(EVENTS,'a') as output:
                    output.write(json.dumps(dict(event,phase='end',time=time.time()))+'\\n')
                sys.exit(EXIT)
            """).replace("NAME",repr(name)).replace("EVENTS",repr(str(self.events)))
            child = "pass"
            if spawn_marker is not None:
                child_code = "import pathlib,time; time.sleep(.8); pathlib.Path("+repr(str(spawn_marker))+").write_text('escaped')"
                child = "subprocess.Popen([sys.executable,'-c',"+repr(child_code)+"])"
            wait = "pass"
            if barrier and i < 3:
                prefix = str(self.root / "barrier-")
                wait = "pathlib.Path("+repr(prefix+str(i))+").touch(); deadline=time.monotonic()+10\nwhile not all(pathlib.Path("+repr(prefix)+"+str(j)).exists() for j in range(3)) and time.monotonic()<deadline: time.sleep(.01)"
            code = code.replace("CHILD",child).replace("BARRIER",wait).replace("SLEEP",repr(sleep)).replace("EXIT","7" if i == failure else "0")
            path=directory/name
            path.write_text("#!"+sys.executable+"\n"+code)
            path.chmod(0o755)
        return directory

    def prepare_mixed(self, shards=4, catch_sleep=.01):
        for case in self.config["cases"]:
            case["sleep"]=catch_sleep
        self.config["events"]=str(self.events)
        self.write_config()
        result=self.cli("inventory","--build-root",self.build,"--binary","tests/mock",
                        "--examples-directory","examples/cpp","--output",self.inventory)
        self.assertEqual(result.returncode,0,result.stderr)
        value=json.loads(self.inventory.read_text())
        # Stable public Catch hints above the generic unknown-example weight.
        history=self.root/"history.json"
        history.write_text(json.dumps({t["id"]:400 for t in value["tests"]}))
        result=self.cli("plan","--inventory",self.inventory,"--shards",shards,
                        "--durations",history,"--output",self.plan)
        self.assertEqual(result.returncode,0,result.stderr)
        return json.loads(self.plan.read_text())

    def run_all(self, shards=4, *extra):
        results=[]
        self.processes=[]
        for i in range(shards):
            proc,result=self.run_shard(i,"--workers","4","--batch-target-seconds","25",*extra)
            results.append(result)
            self.processes.append(proc)
        return SHARDS.merge_results(json.loads(self.plan.read_text()),results),results

    def test_all_56_examples_and_four_shards_share_max_four_active_tasks(self):
        self.add_examples(count=56,sleep=.05)
        plan=self.prepare_mixed(catch_sleep=.5)
        summary,results=self.run_all()
        self.assertTrue(summary["success"],summary["errors"])
        self.assertEqual(summary["observed_count"],4)
        self.assertEqual(summary["expected_example_count"],56)
        self.assertEqual(summary["observed_example_count"],56)
        self.assertEqual([len(r["examples"]) for r in results],[14,14,14,14])
        self.assertEqual(len({e["id"] for r in results for e in r["examples"]}),56)
        for result, shard in zip(results, plan["shards"]):
            self.assertEqual(result["example_binaries"], shard["example_binaries"])
            self.assertEqual([e["binary"] for e in result["examples"]], shard["example_binaries"])
        records=[json.loads(line) for line in self.events.read_text().splitlines()]
        active=set(); peak=0; mixed_seen=False
        for row in sorted(records,key=lambda r:r["time"]):
            if row["phase"]=="start": active.add(row["name"])
            else: active.remove(row["name"])
            peak=max(peak,len(active))
            mixed_seen |= any(x.endswith("_example") for x in active) and any(not x.endswith("_example") for x in active)
        self.assertGreaterEqual(peak,2)
        self.assertLessEqual(peak,4)
        self.assertTrue(mixed_seen)
        starts=[r for r in records if r["phase"]=="start" and r["name"].endswith("_example")]
        self.assertEqual(len({r["cwd"] for r in starts}),56)
        self.assertEqual(len({r["tmp"] for r in starts}),56)
        for row in starts:
            self.assertEqual(Path(row["cwd"],"same-name").read_text(),row["name"])
            self.assertEqual(Path(row["tmp"],"same-name").read_text(),row["name"])
        self.assertEqual(plan["shards"],json.loads(self.plan.read_text())["shards"])

    def test_failure_is_preserved_and_all_other_examples_still_run(self):
        self.add_examples(count=7,failure=1)
        self.prepare_mixed(shards=2)
        summary,results=self.run_all(2)
        self.assertFalse(summary["success"])
        examples=sorted((e for r in results for e in r["examples"]), key=lambda e:e["binary"])
        self.assertEqual(len(examples),7)
        self.assertEqual([r["exit_code"] for r in examples],[0,7,0,0,0,0,0])
        self.assertTrue(any("example" in e and "did not pass" in e for e in summary["errors"]))
        for shard, process in zip(results, self.processes):
            lines=[json.loads(line.removeprefix("Example result: "))
                   for line in process.stdout.splitlines() if line.startswith("Example result: ")]
            self.assertEqual(len(lines),len(shard["examples"]))
            by_name={row["example"]:row for row in lines}
            for result in shard["examples"]:
                row=by_name[result["binary"]]
                self.assertEqual(row["status"],result["status"])
                self.assertEqual(row["exit_code"],result["exit_code"])
                prefix="run-"+str(shard["shard"])
                self.assertEqual(row["log"],prefix+"/"+result["log"])
                self.assertTrue((self.root/prefix/result["log"]).is_file())

    def test_example_timeout_does_not_relax_catch_timeout(self):
        self.add_examples(sleep=.4)
        self.prepare_mixed(shards=2)
        proc,result=self.run_shard(1,"--workers","3","--timeout","2","--example-timeout",".08")
        self.assertEqual(proc.returncode,1,proc.stderr)
        self.assertTrue(all(b["status"]=="passed" for b in result["batches"]))
        self.assertTrue(all(e["status"]=="timeout" and e["exit_code"] is not None for e in result["examples"]))

    def test_cancellation_records_all_examples_and_kills_example_children(self):
        marker=self.root/"escaped"
        self.add_examples(count=20,sleep=5,spawn_marker=marker)
        plan=self.prepare_mixed(shards=2,catch_sleep=.05)
        assigned=plan["shards"][1]["example_binaries"]
        output=self.root/"cancelled"
        process=subprocess.Popen([sys.executable,str(SCRIPT),"run","--plan",str(self.plan),
            "--build-root",str(self.build),"--shard","1","--workers","3","--output-dir",str(output)],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        self.addCleanup(lambda: process.kill() if process.poll() is None else None)
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            if self.events.exists() and "_example" in self.events.read_text(): break
            time.sleep(.02)
        else: self.fail("example did not start")
        process.send_signal(signal.SIGTERM)
        stdout,stderr=process.communicate(timeout=5)
        self.assertEqual(process.returncode,143,(stdout,stderr))
        result=json.loads((output/"result.json").read_text())
        self.assertEqual(len(result["examples"]),len(assigned))
        self.assertEqual(result["example_binaries"],assigned)
        self.assertFalse(result["success"])
        self.assertTrue(all(r["status"]=="cancelled" for r in result["examples"]))
        self.assertTrue(any(r["exit_code"] is None for r in result["examples"]))
        rows=[json.loads(line.removeprefix("Example result: "))
              for line in stdout.splitlines() if line.startswith("Example result: ")]
        self.assertEqual(len(rows),len(assigned))
        self.assertTrue(any(row["log"] is None and row["exit_code"] is None for row in rows))
        time.sleep(.9)
        self.assertFalse(marker.exists())

    def test_modified_extra_missing_or_symlink_example_rejected_before_execution(self):
        for mode in ("modified","extra","missing","symlink"):
            with self.subTest(mode=mode):
                self.setUp()
                directory=self.add_examples()
                self.prepare_mixed(shards=2)
                p=directory/"100_example"
                if mode=="modified": p.write_text(p.read_text()+"\n# changed\n")
                elif mode=="extra":
                    extra=directory/"999_extra"; extra.write_text("#!/bin/sh\nexit 0\n"); extra.chmod(0o755)
                elif mode=="missing": p.unlink()
                else: p.unlink(); p.symlink_to(directory/"101_example")
                proc,result=self.run_shard(1,"--workers","3")
                self.assertEqual(proc.returncode,2,proc.stderr)
                self.assertIsNone(result)
                self.assertFalse(self.events.exists())

    def test_inventory_rejects_empty_symlink_and_nonexecutable_numbered_entry(self):
        for mode in ("empty","symlink","nonexecutable"):
            with self.subTest(mode=mode):
                self.setUp()
                directory=self.add_examples(count=0 if mode=="empty" else 1)
                if mode=="symlink":
                    p=directory/"100_example"; p.rename(directory/"target"); p.symlink_to(directory/"target")
                elif mode=="nonexecutable": (directory/"100_example").chmod(0o644)
                result=self.cli("inventory","--build-root",self.build,"--binary","tests/mock",
                    "--examples-directory","examples/cpp","--output",self.inventory)
                self.assertEqual(result.returncode,2,result.stderr)

    def test_merge_rejects_missing_duplicate_wrong_hash_wrong_shard_or_context(self):
        self.add_examples(); plan=self.prepare_mixed(shards=2)
        summary,results=self.run_all(2)
        self.assertTrue(summary["success"])
        for mode in ("missing","duplicate","wrong_hash","wrong_shard","wrong_context","cancelled","timeout"):
            with self.subTest(mode=mode):
                changed=copy.deepcopy(results)
                if mode=="missing": changed[1]["examples"].pop()
                elif mode=="duplicate": changed[1]["examples"].append(changed[1]["examples"][0])
                elif mode=="wrong_hash": changed[1]["examples"][0]["sha256"]="a"*64
                elif mode=="wrong_shard": changed[0]["examples"].append(changed[1]["examples"].pop())
                elif mode=="wrong_context": changed[1]["context"]={"attempt":"stale"}
                else: changed[1]["examples"][0]["status"]=mode
                changed=[SHARDS.sealed({k:v for k,v in r.items() if k!="digest"}) for r in changed]
                if mode=="wrong_context":
                    with self.assertRaises(SHARDS.ShardError): SHARDS.merge_results(plan,changed)
                else: self.assertFalse(SHARDS.merge_results(plan,changed)["success"])

    def test_example_inventory_does_not_change_catch_batch_membership(self):
        self.add_examples(); plan=self.prepare_mixed(shards=2)
        tests={t["id"]:t for t in plan["inventory"]["tests"]}
        selected=[tests[i] for i in plan["shards"][1]["test_ids"]]
        batches=SHARDS.make_batches(selected,32,plan["duration_history"],25)
        tasks=SHARDS.mixed_tasks(batches,plan["inventory"]["examples"],plan["duration_history"],25)
        self.assertEqual(tasks,SHARDS.mixed_tasks(batches,plan["inventory"]["examples"],plan["duration_history"],25))
        restored=[item[4] for item in sorted((t for t in tasks if t[2]=="catch2"),key=lambda t:t[3])]
        self.assertEqual(restored,batches)
        self.assertTrue(all(t[0]==300 for t in tasks if t[2]=="example"))
        self.assertEqual([t[1] for t in tasks if t[2]=="example"],sorted(t[1] for t in tasks if t[2]=="example"))


    def test_empty_owner_still_verifies_full_inventory_and_summary_requires_explicit_assignment(self):
        self.add_examples(count=1)
        plan=self.prepare_mixed(shards=4)
        summary,results=self.run_all(4)
        self.assertTrue(summary["success"],summary["errors"])
        self.assertEqual([len(r["examples"]) for r in results],[1,0,0,0])
        for shard in range(4):
            changed=copy.deepcopy(results)
            del changed[shard]["example_binaries"]
            changed[shard]=SHARDS.sealed({k:v for k,v in changed[shard].items() if k!="digest"})
            self.assertFalse(SHARDS.merge_results(plan,changed)["success"])

    def test_merge_rejects_swapped_owner_manifests_even_when_global_inventory_is_complete(self):
        self.add_examples(count=6)
        plan=self.prepare_mixed(shards=2)
        summary,results=self.run_all(2)
        self.assertTrue(summary["success"])
        changed=copy.deepcopy(results)
        for field in ("example_binaries","examples"):
            changed[0][field],changed[1][field]=changed[1][field],changed[0][field]
        changed=[SHARDS.sealed({k:v for k,v in r.items() if k!="digest"}) for r in changed]
        merged=SHARDS.merge_results(plan,changed)
        self.assertEqual(merged["observed_example_count"],6)
        self.assertFalse(merged["success"])
        self.assertTrue(any("assignment manifest mismatch" in e for e in merged["errors"]))


class DistributedPlannerTest(unittest.TestCase):
    def inventory(self, examples=56):
        value=fixture.fixture_inventory(["a","b","c","d"])
        metadata={}
        for i in range(examples):
            name=f"examples/cpp/{100+i:03d}_fixture"
            item={"sha256":hashlib.sha256(name.encode()).hexdigest(),"size":10}
            metadata[name]=dict(item,id=SHARDS.example_id(name,item))
        return SHARDS.sealed({k:v for k,v in value.items() if k!="digest"} |
                             {"example_directory":"examples/cpp","examples":metadata})

    def reseal(self, value):
        return SHARDS.sealed({k:v for k,v in value.items() if k!="digest"})

    def test_distribution_is_deterministic_with_uniform_unknown_hints(self):
        inventory=self.inventory()
        plan=SHARDS.make_plan(inventory,4)
        self.assertEqual([len(s["example_binaries"]) for s in plan["shards"]],[14]*4)
        self.assertEqual(plan,SHARDS.make_plan(inventory,4))
        reordered=copy.deepcopy(inventory)
        reordered["examples"]=dict(reversed(list(reordered["examples"].items())))
        self.assertEqual(plan,SHARDS.make_plan(self.reseal(reordered),4))
        for shard in plan["shards"]:
            self.assertEqual(shard["estimated_total_seconds"],
                             shard["estimated_seconds"]+300*len(shard["example_binaries"]))
        SHARDS.validate_plan(plan)

    def test_existing_catch_load_controls_assignment_without_rebalancing_cases_or_batches(self):
        inventory=self.inventory(examples=6)
        hints={t["id"]:seconds for t,seconds in zip(inventory["tests"],[1000,1000,1000,1])}
        plan=SHARDS.make_plan(inventory,4,hints)
        plain={k:v for k,v in inventory.items() if k not in ("digest","examples","example_directory")}
        catch_plan=SHARDS.make_plan(SHARDS.sealed(plain),4,hints)
        lookup={t["id"]:t for t in inventory["tests"]}
        for mixed,catch in zip(plan["shards"],catch_plan["shards"]):
            self.assertEqual(mixed["test_ids"],catch["test_ids"])
            self.assertEqual(mixed["estimated_seconds"],catch["estimated_seconds"])
            self.assertEqual(SHARDS.make_batches([lookup[i] for i in mixed["test_ids"]],32,hints,25),
                             SHARDS.make_batches([lookup[i] for i in catch["test_ids"]],32,hints,25))
        lightest=min(plan["shards"],key=lambda s:s["estimated_seconds"])
        self.assertEqual(len(lightest["example_binaries"]),4)
        loads=[s["estimated_total_seconds"] for s in plan["shards"]]
        self.assertLessEqual(max(loads)-min(loads),300)
        self.assertEqual(sum(len(s["example_binaries"]) for s in plan["shards"]),6)

    def test_example_durations_are_not_accepted_from_catch_history(self):
        inventory=self.inventory()
        example_hints={v["id"]:1000000 for v in inventory["examples"].values()}
        self.assertEqual(SHARDS.make_plan(inventory,4,example_hints),SHARDS.make_plan(inventory,4))
        self.assertEqual(SHARDS.EXAMPLE_FALLBACK_SECONDS,300.0)

    def test_unknown_examples_start_before_medium_catch_batches_without_special_names(self):
        inventory=self.inventory(examples=2)
        tests=inventory["tests"]
        history={tests[0]["id"]:600.0,tests[1]["id"]:200.0}
        tasks=SHARDS.mixed_tasks([[tests[0]],[tests[1]],[tests[2]]],
                                 inventory["examples"],history,25)
        self.assertEqual([task[2] for task in tasks],
                         ["catch2","example","example","catch2","catch2"])
        self.assertEqual([task[0] for task in tasks],[600.0,300.0,300.0,200.0,25])
        self.assertEqual([task[3] for task in tasks if task[2]=="example"],
                         sorted(inventory["examples"]))

    def test_schema1_history_envelope_is_consumed_by_schema3_plan_cli(self):
        inventory=self.inventory()
        hints={t["id"]:float(i+1) for i,t in enumerate(inventory["tests"])}
        envelope={"schema":1,"source_main_sha":"a"*40,"inventory_digest":"b"*64,
                  "duration_history":dict(hints,**{"c"*64:9999.0})}
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/"inventory.json").write_text(json.dumps(inventory))
            (root/"history.json").write_text(json.dumps(envelope))
            proc=subprocess.run([sys.executable,str(SCRIPT),"plan",
                "--inventory",str(root/"inventory.json"),"--durations",str(root/"history.json"),
                "--shards","4","--output",str(root/"plan.json")],
                capture_output=True,text=True,timeout=10)
            self.assertEqual(proc.returncode,0,proc.stderr)
            actual=json.loads((root/"plan.json").read_text())
        self.assertEqual(actual["schema_version"],3)
        self.assertEqual(actual["duration_history"],hints)
        self.assertEqual(actual,SHARDS.make_plan(inventory,4,hints))
        SHARDS.validate_plan(actual)

    def test_single_shard_and_empty_example_assignments_are_explicit(self):
        inventory=self.inventory(examples=1)
        one=SHARDS.make_plan(inventory,1);SHARDS.validate_plan(one)
        self.assertEqual(len(one["shards"][0]["example_binaries"]),1)
        four=SHARDS.make_plan(inventory,4);SHARDS.validate_plan(four)
        self.assertEqual([len(s["example_binaries"]) for s in four["shards"]],[1,0,0,0])

    def test_resealed_plan_rejects_missing_duplicate_moved_unknown_or_legacy_examples(self):
        plan=SHARDS.make_plan(self.inventory(),4)
        for mode in ("missing","duplicate","moved","unknown","absent_field","legacy","bad_total"):
            with self.subTest(mode=mode):
                bad=copy.deepcopy(plan);a,b=bad["shards"][:2]
                if mode=="missing":a["example_binaries"].pop()
                elif mode=="duplicate":a["example_binaries"].append(a["example_binaries"][0])
                elif mode=="moved":a["example_binaries"][0],b["example_binaries"][0]=b["example_binaries"][0],a["example_binaries"][0]
                elif mode=="unknown":a["example_binaries"][0]="examples/cpp/999_unknown"
                elif mode=="absent_field":del a["example_binaries"]
                elif mode=="legacy":bad["example_shard"]=1
                else:a["estimated_total_seconds"]+=1
                with self.assertRaises(SHARDS.ShardError):SHARDS.validate_plan(self.reseal(bad))

    def test_schema3_rejects_old_inventory_plan_result_but_retains_catch_ids(self):
        inventory=self.inventory()
        plain=fixture.fixture_inventory(["a","b","c","d"])
        self.assertEqual([t["id"] for t in inventory["tests"]],[t["id"] for t in plain["tests"]])
        self.assertEqual(SHARDS.SCHEMA,3)
        for kind,value in (("inventory",inventory),("plan",SHARDS.make_plan(inventory,4)),
                           ("result",{"kind":"result","schema_version":3})):
            with self.subTest(kind=kind):
                old=self.reseal(dict(value,schema_version=2))
                with self.assertRaisesRegex(SHARDS.ShardError,"schema 3"):SHARDS.verify_seal(old,kind)

if __name__=="__main__":
    unittest.main()
