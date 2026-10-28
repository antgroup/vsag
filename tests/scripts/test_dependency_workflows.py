#!/usr/bin/env python3
"""Regression checks for dependency workflow trust and source guards."""

from pathlib import Path
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[2]


def job_text(workflow: Path, job_name: str) -> str:
    text = workflow.read_text(encoding="utf-8")
    match = re.search(
        rf"(?ms)^  {re.escape(job_name)}:\n(?P<body>.*?)(?=^  [a-zA-Z0-9_-]+:\n|\Z)", text
    )
    if match is None:
        raise AssertionError(f"job {job_name} not found in {workflow}")
    return match.group("body")


class DependencyWorkflowTest(unittest.TestCase):
    def test_metrics_helpers_preserve_dedicated_coverage_checks(self) -> None:
        workflow = ROOT / ".github/workflows/pr-ci-community.yml"
        coverage = job_text(workflow, "coverage-config")
        asan = job_text(workflow, "build-asan-x86")

        self.assertIn("tests/scripts/coverage_diagnostics_test.py", coverage)
        self.assertIn("scripts/coverage/collect_cpp_coverage.sh", coverage)
        self.assertNotIn("tests/scripts/coverage_diagnostics_test.py", asan)
        self.assertIn("tests/scripts/test_collect_build_metrics.py", asan)
        self.assertIn("tests/scripts/test_dependency_workflows.py", asan)

    def test_daily_x86_build_explicitly_disables_system_openblas(self) -> None:
        daily_x86 = job_text(
            ROOT / ".github/workflows/daily_test.yml", "daily_build_asan_x86"
        )

        self.assertIn("-DVSAG_USE_SYSTEM_OPENBLAS=OFF", daily_x86)
        self.assertNotIn("-DVSAG_USE_SYSTEM_OPENBLAS=AUTO", daily_x86)

    def test_pull_request_archive_cache_is_restore_only(self) -> None:
        pr_ci = (ROOT / ".github/workflows/pr-ci-community.yml").read_text(encoding="utf-8")

        self.assertIn("uses: actions/cache/restore@v4", pr_ci)
        self.assertNotIn("uses: actions/cache@v4", pr_ci)
        self.assertNotIn("uses: actions/cache/save@v4", pr_ci)
        benchmark = (ROOT / ".github/workflows/build_performance.yml").read_text(encoding="utf-8")
        self.assertIn("dependency-cache-write-policy restore-only", benchmark)

    def test_archive_directory_is_exported_for_the_entire_metrics_job(self) -> None:
        asan = job_text(ROOT / ".github/workflows/pr-ci-community.yml", "build-asan-x86")
        job_configuration, steps = asan.split("    steps:\n", 1)
        self.assertIn(
            "      VSAG_THIRDPARTY_DOWNLOAD_DIR: ${{ github.workspace }}/.ci-downloads\n",
            job_configuration,
        )
        self.assertIn("          path: .ci-downloads\n", steps)

    def test_team_compiler_cache_is_private_and_never_published_by_pr(self) -> None:
        team = job_text(ROOT / ".github/workflows/pr-ci-team.yml", "build")
        self.assertIn('cache="$RUNNER_TEMP/team-ccache"', team)
        self.assertIn('team_cache.py restore', team)
        self.assertIn('--source /__ci/cache/snapshot --manifest /__ci/cache/manifest.json', team)
        self.assertIn('--namespace-file ci-logs/cache-namespace.json', team)
        self.assertIn('CCACHE_BASEDIR=$GITHUB_WORKSPACE', team)
        self.assertIn('CCACHE_COMPILERCHECK=content', team)
        self.assertNotIn('team_cache.py manifest', team)
        self.assertNotIn('actions/cache/save', team)
        self.assertNotIn('actions/cache@', team)

    def test_external_archives_keep_hash_validation_and_pin_overrides(self) -> None:
        for dependency in ("antlr4", "hdf5", "openblas"):
            recipe = (ROOT / f"extern/{dependency}/{dependency}.cmake").read_text(
                encoding="utf-8"
            )
            with self.subTest(dependency=dependency):
                self.assertRegex(recipe, r"URL_HASH\s+(?:MD5|SHA256)=")
                self.assertIn("vsag_resolve_thirdparty_override", recipe)


class TsanCleanupTest(unittest.TestCase):
    def cleanup_script(self, sandbox: Path) -> str:
        workflow = ROOT / ".github/workflows/tsan_build_and_test.yml"
        build = job_text(workflow, "build_tsan")
        self.assertIn("        - /opt:/useless\n", build)
        match = re.search(
            r"(?ms)^      - name: Free Disk Space\n        run: \|\n"
            r"(?P<script>.*?)(?=^      - |\Z)", build
        )
        self.assertIsNotNone(match)
        # Execute the workflow block itself. Only map its destructive target to
        # our temporary directory; diagnostic commands never touch the host.
        script = textwrap.dedent(match.group("script"))
        self.assertIn("node24_diagnostics before\n", script)
        self.assertIn("node24_diagnostics after\n", script)
        diagnostics = "\n".join(
            f"{command}() {{ return 1; }}"
            for command in ("date", "ls", "readlink", "stat", "awk")
        )
        diagnostics += '\ntimeout() { return "${PROBE_STATUS:-1}"; }'
        return diagnostics + "\n" + script.replace("/useless", str(sandbox))

    def run_cleanup(self, sandbox: Path, env=None) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["sh", "-e", "-c", self.cleanup_script(sandbox)],
            env={**os.environ, **(env or {})},
            text=True, capture_output=True, timeout=10,
        )

    def test_preserves_runner_runtime_and_hidden_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sandbox = Path(temporary)
            node = sandbox / "runner-cache/externals/node24/bin/node"
            node.parent.mkdir(parents=True)
            node.write_text("runtime\n")
            for name in ("tools", "directory with spaces", "runner-cache-old"):
                directory = sandbox / name
                directory.mkdir()
                (directory / "payload").write_text("delete me\n")
            for name in ("file", "file with spaces", "-dash", ".hidden"):
                (sandbox / name).write_text("payload\n")
            hidden = sandbox / ".hidden-directory"
            hidden.mkdir()
            (hidden / "payload").write_text("keep me\n")

            result = self.run_cleanup(sandbox)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(node.read_text(), "runtime\n")
            self.assertEqual(
                {path.name for path in sandbox.iterdir()},
                {"runner-cache", ".hidden", ".hidden-directory"},
            )
            self.assertEqual((hidden / "payload").read_text(), "keep me\n")
            self.assertIn("(before cleanup)", result.stdout)
            self.assertIn("(after cleanup)", result.stdout)

    def test_empty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sandbox = Path(temporary)
            result = self.run_cleanup(sandbox)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(list(sandbox.iterdir()), [])
            self.assertIn("(after cleanup)", result.stdout)

    def test_cleanup_failure_attempts_later_entries_and_prints_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sandbox = root / "cleanup"
            sandbox.mkdir()
            node = sandbox / "runner-cache/externals/node24/bin/node"
            node.parent.mkdir(parents=True)
            node.write_text("runtime\n")
            for name in ("a", "b", "c"):
                (sandbox / name).mkdir()
                (sandbox / name / "payload").write_text("delete me\n")
            stub_dir = root / "bin"
            stub_dir.mkdir()
            stub = stub_dir / "rm"
            real_rm = shlex.quote(shutil.which("rm"))
            failed_path = shlex.quote(str(sandbox / "b"))
            stub.write_text(
                "#!/bin/sh\n"
                f'if [ "$3" = {failed_path} ]; then\n'
                '  echo "cleanup failed: $3" >&2\n'
                "  exit 23\n"
                "fi\n"
                f'exec {real_rm} "$@"\n'
            )
            stub.chmod(0o755)

            result = self.run_cleanup(
                sandbox, {"PATH": str(stub_dir) + os.pathsep + os.environ["PATH"]}
            )

            self.assertNotEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stderr.count("cleanup failed:"), 1)
            self.assertIn(str(sandbox / "b"), result.stderr)
            self.assertIn("(before cleanup)", result.stdout)
            self.assertIn("(after cleanup)", result.stdout)
            self.assertEqual({path.name for path in sandbox.iterdir()}, {"b", "runner-cache"})
            self.assertEqual((sandbox / "b/payload").read_text(), "delete me\n")
            self.assertEqual(node.read_text(), "runtime\n")

    def test_failed_node_probes_report_status_and_allow_cleanup(self) -> None:
        for status in (124, 127):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temporary:
                sandbox = Path(temporary)
                (sandbox / "tools").mkdir()

                result = self.run_cleanup(sandbox, {"PROBE_STATUS": str(status)})

                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(list(sandbox.iterdir()), [])
                before, after = result.stdout.split("(after cleanup)")
                self.assertIn("(before cleanup)", before)
                self.assertEqual(before.count(f"node --version exit={status}"), 1)
                self.assertEqual(after.count(f"node --version exit={status}"), 1)


class FailurePropagationTest(unittest.TestCase):
    def test_pytest_exit_status_wins_over_success_looking_log(self):
        import os
        import subprocess
        import tempfile
        import textwrap
        from tests.scripts.test_build_benchmark_workflow import step
        jobs = [('pr-ci-community.yml', 'test-python-x86'),
                ('python_build_and_test.yml', 'test_python_x86'),
                ('python_build_and_test.yml', 'test_python_aarch64')]
        for filename, name in jobs:
            body = job_text(ROOT / '.github/workflows' / filename, name)
            run = step(body, 'Run Python Test')
            self.assertNotIn('continue-on-error', run)
            self.assertIn('if: always()', step(body, 'Upload Test Logs'))
            script = textwrap.dedent(run.split('run: |\n', 1)[1])
            for code in (0, 1, 2, 5, 137):
                with self.subTest(workflow=filename, job=name, code=code), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp); (root / 'tests/python').mkdir(parents=True)
                    fake = root / 'python3'
                    marker = f'pytest-result-exit-{code}'
                    fake.write_text('#!/bin/sh\necho "1 passed"\necho ' + marker + '\nexit ' + str(code) + '\n')
                    fake.chmod(0o755)
                    result = subprocess.run(['bash', '-c', script], cwd=root,
                        env=dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH']),
                        capture_output=True, text=True)
                    self.assertEqual(result.returncode, code)
                    self.assertIn(marker, result.stdout)
                    self.assertIn(marker, (root / 'tests/python/test-output.log').read_text())

    def test_lint_pattern_is_data_and_dispatch_has_a_base(self):
        import os
        import subprocess
        import tempfile
        import textwrap
        from tests.scripts.test_build_benchmark_workflow import step
        body = job_text(ROOT / '.github/workflows/pr-ci-community.yml', 'build-asan-x86')
        self.assertIn("BASE_BRANCH: ${{ github.base_ref || 'main' }}", body)
        lint = step(body, 'Run Lint')
        script = textwrap.dedent(lint.split('run: |\n', 1)[1])
        self.assertNotIn('${{', script)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'scripts/linters').mkdir(parents=True)
            runner=root/'scripts/linters/run-clang-tidy-15.sh'
            runner.write_text('#!/bin/bash\nprintf "%s\\n" "${@: -1}"\n');runner.chmod(0o755)
            pattern="src/it's $(touch injected).cpp$"
            result=subprocess.run(['bash','-c',script],cwd=root,env=dict(os.environ,LINT_PATTERN=pattern),capture_output=True,text=True)
            self.assertEqual(result.returncode,0)
            self.assertEqual(result.stdout.strip(),pattern)
            self.assertFalse((root/'injected').exists())

    def test_compatibility_workflows_fail_closed_and_use_private_input_directory(self):
        from tests.scripts.test_build_benchmark_workflow import step
        for filename in ['pr-ci-community.yml','check_compatibility.yml']:
            body=job_text(ROOT/'.github/workflows'/filename,'compatibility')
            download=step(body,'Download Compatibility Indexes from Release')
            self.assertIn('download_compatibility_inputs.py',download)
            self.assertNotIn('||',download)
            self.assertIn('COMPATIBILITY_INDEX_DIR: ${{ github.workspace }}/.ci-compatibility-inputs',body)
            self.assertNotIn('runner.temp', body.split('    steps:', 1)[0])
            self.assertLess(body.index('actions/checkout@v4'),body.index('download_compatibility_inputs.py'))


class CompatibilityDownloadTest(unittest.TestCase):
    def setUp(self):
        import importlib.util
        import tempfile
        spec=importlib.util.spec_from_file_location('compat_download',ROOT/'scripts/ci/download_compatibility_inputs.py')
        self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.directory=Path(self.temp.name)/'inputs'
        import hashlib
        self.expected={'release_id':123,'assets':{'v1_test.index':{'id':1,'size':4,'digest':'sha256:'+hashlib.sha256(b'data').hexdigest(),'updated_at':'now'}}}

    def test_download_failure_partial_size_digest_and_release_changes_reject(self):
        import copy
        import subprocess
        from unittest.mock import patch
        for case in ['success','download-error','missing','truncated','corrupt','extra','changed-release']:
            with self.subTest(case=case):
                directory=self.directory/case
                def transfer(*args,**kwargs):
                    if case=='download-error':raise subprocess.CalledProcessError(1,args[0])
                    if case!='missing':(directory/'v1_test.index').write_bytes(b'bad' if case=='truncated' else b'xxxx' if case=='corrupt' else b'data')
                    if case=='extra':(directory/'extra.index').write_bytes(b'data')
                after=copy.deepcopy(self.expected)
                if case=='changed-release':after['assets']['v1_test.index']['id']=2
                with patch.object(self.module,'inventory',side_effect=[self.expected,after]), patch.object(self.module.subprocess,'run',side_effect=transfer):
                    if case=='success':self.module.download(directory)
                    elif case=='changed-release':
                        with self.assertRaisesRegex(ValueError, 'changed during download'):
                            self.module.download(directory)
                    else:
                        with self.assertRaises((ValueError,subprocess.CalledProcessError)):self.module.download(directory)

    def test_inventory_uses_all_pages_and_optional_api_digests(self):
        from unittest.mock import patch
        assets=[{'id':1,'name':'v1_test.index','size':4,'state':'uploaded','digest':None},
                {'id':2,'name':'vectors.bin','size':4,'state':'uploaded','digest':None}]
        with patch.object(self.module,'api',side_effect=[{'id':123,'tag_name':'compatibility-indexes'},[[assets[0]],[assets[1]]]]) as api:
            actual=self.module.inventory()
            self.assertEqual(set(actual['assets']),{'v1_test.index','vectors.bin'})
            self.assertTrue(api.call_args.kwargs['paginated'])
        self.directory.mkdir()
        for name in actual['assets']:(self.directory/name).write_bytes(b'data')
        self.module.verify(self.directory,actual)


    def test_missing_and_null_digest_through_api_json(self):
        import json
        from unittest.mock import patch
        for missing in (True, False):
            with self.subTest(missing=missing):
                asset = {'id': 1, 'name': 'v1_test.index', 'size': 4, 'state': 'uploaded'}
                if not missing:
                    asset['digest'] = None
                responses = [json.dumps({'id': 123, 'tag_name': 'compatibility-indexes'}),
                             json.dumps([[asset]])]
                with patch.object(self.module.subprocess, 'check_output', side_effect=responses) as command:
                    actual = self.module.inventory()
                self.assertEqual(command.call_args.args[0][-2:], ['--paginate', '--slurp'])
                self.assertIsNone(actual['assets']['v1_test.index']['digest'])
                self.directory.mkdir(exist_ok=True)
                path = self.directory / 'v1_test.index'
                path.write_bytes(b'data')
                self.module.verify(self.directory, actual)
                path.write_bytes(b'bad')
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    self.module.verify(self.directory, actual)


    def test_republished_release_through_api_json(self):
        import copy
        import json
        from unittest.mock import patch
        asset = dict(self.expected['assets']['v1_test.index'],
                     name='v1_test.index', state='uploaded')
        release = {'id': 123, 'tag_name': 'compatibility-indexes'}
        for mutation in ('replaced-id', 'added-asset'):
            with self.subTest(mutation=mutation):
                after = [copy.deepcopy(asset)]
                if mutation == 'replaced-id':
                    after[0]['id'] = 2
                else:
                    after.append(dict(asset, id=2, name='new.index'))
                responses = [json.dumps(release), json.dumps([[asset]]),
                             json.dumps(release), json.dumps([after])]
                directory = self.directory / mutation
                def transfer(*args, **kwargs):
                    (directory / 'v1_test.index').write_bytes(b'data')
                with patch.object(self.module.subprocess, 'check_output', side_effect=responses) as api, \
                        patch.object(self.module.subprocess, 'run', side_effect=transfer):
                    with self.assertRaisesRegex(ValueError, 'changed during download'):
                        self.module.download(directory)
                self.assertEqual(api.call_count, 4)
                self.assertEqual((directory / 'v1_test.index').read_bytes(), b'data')

    def test_correct_size_wrong_digest_rejected(self):
        self.directory.mkdir()
        (self.directory / 'v1_test.index').write_bytes(b'xxxx')
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            self.module.verify(self.directory, self.expected)


if __name__ == "__main__":
    unittest.main()
