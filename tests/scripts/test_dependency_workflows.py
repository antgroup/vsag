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
    body = match.group("body")
    # Inspect the actual shared execution steps as well as the entry job envelope.
    local = re.search(r"(?m)^      - uses: (\./\.github/actions/[^\n]+)\n(?P<call>(?:        .*\n|\n)*)", body)
    if local:
        action = (ROOT / local[1][2:] / 'action.yml').read_text()
        steps = action.split('  steps:\n', 1)[1]
        parameters = {}
        for key, value in re.findall(r"(?m)^          ([\w-]+): (.*)$", local['call']):
            value = value.strip()
            if value.startswith('${{') and value.endswith('}}'):
                parameters[key] = value[3:-2].strip()
            else:
                parameters[key] = value
        for key, value in parameters.items():
            steps = steps.replace('inputs.' + key, '(' + value + ')')
        # Normalize trivial matrix substitutions for existing executable-shell tests.
        steps = re.sub(r'\$\{\{ \((matrix\.[\w-]+)\) \}\}', r'${{ \1 }}', steps)
        # Tests execute extracted commands under an explicit shell; omit redundant
        # composite shell declarations from the normalized command view.
        steps = re.sub(r'(?m)^      shell: (?:bash|sh|bash -e \{0\})\n', '', steps)
        expanded = ''.join('  ' + line if line.strip() else line for line in steps.splitlines(True))
        body = body[:local.start()] + expanded + body[local.end():]
    return body


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
        benchmark = (ROOT / ".github/workflows/daily_test.yml").read_text(encoding="utf-8")
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
    workflow_name = "main-ci.yml"
    job_name = "tsan-build-tsan"

    def cleanup_script(self, sandbox: Path) -> str:
        workflow = ROOT / ".github/workflows" / self.workflow_name
        build = job_text(workflow, self.job_name)
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


class SimdBuildCleanupTest(TsanCleanupTest):
    workflow_name = "main-ci.yml"
    job_name = "simd-build-asan-simd"


class SimdTestCleanupTest(TsanCleanupTest):
    workflow_name = "main-ci.yml"
    job_name = "simd-test-asan-simd"


class FailurePropagationTest(unittest.TestCase):
    def test_tsan_keeps_vsag_concurrency_and_propagates_sanitizer_failures(self):
        from tests.scripts.test_build_benchmark_workflow import step
        body = job_text(ROOT / '.github/workflows/main-ci.yml', 'tsan-test-tsan')
        run = step(body, 'Run TSAN tests')
        self.assertIn('OPENBLAS_NUM_THREADS: "1"', run)
        script = textwrap.dedent(run.split('run: |\n', 1)[1])
        self.assertIn('"[concurrent]~[daily]"', script)
        self.assertNotIn('race:libopenblas', script)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / 'build/tests/functests'
            binary.parent.mkdir(parents=True)
            binary.write_text(
                '#!/bin/sh\n'
                'test "$OPENBLAS_NUM_THREADS" = 1 || exit 99\n'
                'test "$1" = "[concurrent]~[daily]" || exit 98\n'
                'exit "$TEST_TSAN_EXIT"\n')
            for code in [0, 66]:
                result = subprocess.run(
                    ['bash', '-e', '-c', script], cwd=root,
                    env=dict(os.environ, OPENBLAS_NUM_THREADS='1', TEST_TSAN_EXIT=str(code)),
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, code, result.stderr)

    def test_pytest_exit_status_wins_over_success_looking_log(self):
        import os
        import subprocess
        import tempfile
        import textwrap
        from tests.scripts.test_build_benchmark_workflow import step
        jobs = [('pr-ci-community.yml', 'test-python-x86'),
                ('main-ci.yml', 'python-test-python-x86'),
                ('main-ci.yml', 'python-test-python-aarch64')]
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
        self.assertIn("LINT_BASE_SHA: ${{ github.event.pull_request.base.sha }}", body)
        lint = step(body, 'Run Lint')
        script = textwrap.dedent(lint.split('run: |\n', 1)[1])
        self.assertNotIn('${{', script)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'scripts/linters').mkdir(parents=True)
            runner=root/'scripts/linters/run-clang-tidy-20.sh'
            runner.write_text('#!/bin/bash\nprintf "%s\\n" "${@: -1}"\n');runner.chmod(0o755)
            pattern="src/it's $(touch injected).cpp$"
            result=subprocess.run(['bash','-c',script],cwd=root,env=dict(os.environ,LINT_PATTERN=pattern),capture_output=True,text=True)
            self.assertEqual(result.returncode,0)
            self.assertEqual(result.stdout.strip(),pattern)
            self.assertFalse((root/'injected').exists())

    def test_compatibility_workflows_fail_closed_and_use_private_input_directory(self):
        from tests.scripts.test_build_benchmark_workflow import step
        for filename in ['pr-ci-community.yml','main-ci.yml']:
            body=job_text(ROOT/'.github/workflows'/filename,'compatibility-compatibility' if filename == 'main-ci.yml' else 'compatibility')
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



class LeanPrChecksTest(unittest.TestCase):
    def test_main_ci_policy_suite(self):
        subprocess.run(["python3", "-m", "unittest", "tests/scripts/test_main_ci.py", "tests/scripts/test_workflow_entries.py"], cwd=ROOT, check=True)

    def selected(self, name, *, base="main", event="pull_request", team=False,
                 cpp=False, python=False, cmake=False):
        body = job_text(ROOT / ".github/workflows/pr-ci-community.yml", name)
        expression = re.search(r"^    if: (.*)$", body, re.M).group(1)
        values = {"github.base_ref": base, "github.event_name": event,
                  "inputs.team-linux": team, "needs.changes.outputs.cpp": str(cpp).lower(),
                  "needs.changes.outputs.python": str(python).lower(),
                  "needs.changes.outputs.cmake": str(cmake).lower()}
        for key in sorted(values, key=len, reverse=True):
            expression = expression.replace(key, repr(values[key]))
        expression = expression.replace("&&", " and ").replace("||", " or ")
        expression = re.sub(r"!(?!=)", " not ", expression)
        return eval(expression, {"__builtins__": {}}, {})

    def test_main_pr_selection_matrix(self):
        for team in (False, True):
            self.assertFalse(self.selected("build-asan-macos-arm64", cpp=True, team=team))
            self.assertFalse(self.selected("build-python-x86", cpp=True, python=True, team=team))
            for name in ("build-python-x86", "test-python-x86"):
                for python, cmake in ((True, False), (False, True), (True, True)):
                    self.assertFalse(self.selected(name, python=python, cmake=cmake, team=team))
                self.assertFalse(self.selected(name, team=team))
            self.assertEqual(self.selected("format", cpp=True, team=team), not team)
            self.assertTrue(self.selected("format", python=True, team=team))
        body = job_text(ROOT / ".github/workflows/pr-ci-community.yml", "build-python-x86")
        self.assertIn("needs: [changes, format]", body)

    def test_version_and_dispatch_preserve_legacy_selection(self):
        for context in ({"base": "0.18"}, {"event": "workflow_dispatch", "base": ""}):
            self.assertTrue(self.selected("build-asan-macos-arm64", cpp=True, **context))
            self.assertTrue(self.selected("format", cpp=True, team=True, **context))
            self.assertTrue(self.selected("build-python-x86", python=True, **context))
            self.assertTrue(self.selected("test-python-x86", cmake=True, **context))
            self.assertFalse(self.selected("build-python-x86", **context))

    def test_team_format_coverage(self):
        community = (ROOT / ".github/workflows/pr-ci-community.yml").read_text()
        team = (ROOT / ".github/workflows/pr-ci-team.yml").read_text()
        def paths(text, label):
            section = re.search(r"(?m)^            " + label + r":\n((?:              .*\n)+)", text).group(1)
            return re.findall(r"- '([^']+)'", section)
        self.assertEqual(paths(community, "cpp"), paths(team, "cpp"))
        self.assertIn("./scripts/format/check_format.sh", job_text(ROOT / ".github/workflows/pr-ci-team.yml", "checks"))

    def test_tsan_push_and_version_triggers_retained(self):
        branch = (ROOT / ".github/workflows/main-ci.yml").read_text()
        manual = (ROOT / ".github/workflows/manual-checks.yml").read_text()
        community = (ROOT / ".github/workflows/pr-ci-community.yml").read_text()
        self.assertIn("tsan-build-tsan:", branch)
        self.assertIn("needs.entry-scope.outputs.version == 'true'", branch)
        self.assertIn("startsWith(github.base_ref, '0.')", community)
        self.assertIn("version-tsan-build-tsan:", community)
        self.assertIn('tsan', manual)

    def test_circleci_only_skips_verified_current_main_prs(self):
        import importlib.util
        from unittest.mock import Mock, patch
        spec = importlib.util.spec_from_file_location("circle_scope", ROOT / "scripts/ci/circleci_pr_scope.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        env = {"CIRCLE_PULL_REQUEST": "https://github.com/antgroup/vsag/pull/123", "CIRCLE_SHA1": "abc"}
        def pr(base="main", sha="abc", state="open"):
            return {"state": state, "base": {"ref": base, "repo": {"full_name": "antgroup/vsag"}}, "head": {"sha": sha}}
        self.assertTrue(module.should_skip(env, lambda _: pr()))
        for metadata in (pr("0.18"), pr(sha="old"), pr(state="closed"), {}):
            self.assertFalse(module.should_skip(env, lambda _: metadata))
        for invalid in ({}, dict(env, CIRCLE_PULL_REQUEST="https://evil.test/pull/123"),
                        dict(env, CIRCLE_PULL_REQUESTS="https://github.com/antgroup/vsag/pull/124")):
            api = Mock(side_effect=[pr(), pr("0.18")])
            self.assertFalse(module.should_skip(invalid, api))
        with patch.object(module, "fetch_pr", side_effect=TimeoutError), patch.dict(os.environ, env, clear=True), patch.object(module.subprocess, "run") as run:
            module.main()
            run.assert_not_called()
        with patch.object(module, "fetch_pr", return_value=pr()), patch.dict(os.environ, env, clear=True), patch.object(module.subprocess, "run") as run:
            module.main()
            run.assert_called_once_with(["circleci-agent", "step", "halt"], check=True)
        config = (ROOT / ".circleci/continue_config.yml").read_text()
        pr_job = config.split("  pull-request-check:", 1)[1].split("  main-branch-check:", 1)[0]
        self.assertLess(pr_job.index("circleci_pr_scope.py"), pr_job.index("restore_cache:"))
        self.assertNotIn("circleci_pr_scope.py", config.split("  main-branch-check:", 1)[1])

    def test_circleci_transport_halt_and_merge_preview(self):
        import importlib.util
        from http.client import IncompleteRead
        from unittest.mock import Mock, patch
        spec = importlib.util.spec_from_file_location("circle_scope", ROOT / "scripts/ci/circleci_pr_scope.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        env = {"CIRCLE_PULL_REQUEST": "https://github.com/antgroup/vsag/pull/123", "CIRCLE_SHA1": "merge-current"}
        pr = {"state": "open", "base": {"ref": "main", "repo": {"full_name": "antgroup/vsag"}},
              "head": {"sha": "head-current"}, "merge_commit_sha": "merge-current"}
        self.assertTrue(module.should_skip(env, lambda _: pr))
        self.assertTrue(module.should_skip(dict(env, CIRCLE_SHA1="head-current"), lambda _: pr))
        self.assertFalse(module.should_skip(dict(env, CIRCLE_SHA1="merge-stale"), lambda _: pr))
        self.assertFalse(module.should_skip(dict(env, CIRCLE_SHA1=""), lambda _: pr))
        with patch.object(module.urllib.request, "urlopen", side_effect=IncompleteRead(b"partial")) as request, patch.object(module.time, "sleep") as sleep, patch.dict(os.environ, env, clear=True), patch.object(module.subprocess, "run") as run:
            module.main()
            self.assertEqual(request.call_count, 2)
            sleep.assert_called_once_with(1)
            run.assert_not_called()
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = __import__("json").dumps(pr)
        with patch.object(module.urllib.request, "urlopen", side_effect=[IncompleteRead(b"partial"), response]), patch.object(module.time, "sleep"):
            self.assertEqual(module.fetch_pr("https://api.github.com/repos/antgroup/vsag/pulls/123"), pr)
        for error in (subprocess.CalledProcessError(1, "circleci-agent"), FileNotFoundError()):
            with patch.object(module, "fetch_pr", return_value=pr), patch.dict(os.environ, env, clear=True), patch.object(module.subprocess, "run", side_effect=error) as run:
                module.main()
                run.assert_called_once()

if __name__ == "__main__":
    unittest.main()
