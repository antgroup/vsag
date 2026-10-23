"""Run Team workflow identity and result gates without GitHub or Kubernetes."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

from tests.scripts.test_build_benchmark_workflow import job, python_step, step

ROOT = Path(__file__).resolve().parents[2]
TEAM = (ROOT / '.github/workflows/pr-ci-team.yml').read_text()
CALLER = (ROOT / '.github/workflows/pr-ci.yml').read_text()
SPEC = importlib.util.spec_from_file_location('team_test_shards', ROOT / 'scripts/testing/team_test_shards.py')
SHARDS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SHARDS)


class TeamWorkflowTest(unittest.TestCase):
    def environment(self, root):
        return dict(os.environ, TEAM_ARTIFACTS=str(root / 'artifacts'), TEAM_REPOSITORY_ID='123',
                    GITHUB_RUN_ID='456', GITHUB_RUN_ATTEMPT='1', TEAM_HEAD_SHA='a' * 40,
                    TEAM_TEST_SHA='b' * 40, TEAM_IMAGE='example.invalid/image@sha256:' + 'c' * 64,
                    GITHUB_STEP_SUMMARY=str(root / 'step-summary.md'))

    def context(self, env):
        return {key: env[name] for key, name in (
            ('repository_id', 'TEAM_REPOSITORY_ID'), ('run_id', 'GITHUB_RUN_ID'),
            ('run_attempt', 'GITHUB_RUN_ATTEMPT'), ('head_sha', 'TEAM_HEAD_SHA'),
            ('test_sha', 'TEAM_TEST_SHA'), ('image', 'TEAM_IMAGE'))}

    def run_identity_step(self, name, cwd, env, mode):
        source = python_step(name, TEAM)
        env = dict(env, GITHUB_WORKSPACE=str(cwd.resolve()))
        wrapper = textwrap.dedent('''\
            import os, pathlib, subprocess
            from unittest.mock import patch
            original = pathlib.Path.read_text
            def read_text(path, *args, **kwargs):
                if str(path) == '/proc/self/mountinfo':
                    return '1 2 3 / ' + os.environ['TEAM_ARTIFACTS'] + ' MOUNT_MODE - ext4 /dev/disk MOUNT_MODE\\n'
                return original(path, *args, **kwargs)
            with patch('os.path.ismount', return_value=True), patch.object(pathlib.Path, 'read_text', read_text), patch.object(subprocess, 'check_output', return_value=os.environ['TEAM_TEST_SHA'] + '\\n'):
                exec(SOURCE)
        ''').replace('MOUNT_MODE', mode).replace('SOURCE', repr(source))
        return subprocess.run([sys.executable, '-c', wrapper], cwd=cwd, env=env,
                              capture_output=True, text=True)

    def artifact_metadata(self, root, env):
        meta = root / 'artifacts/meta'
        meta.mkdir(parents=True)
        (root / 'artifacts/build').mkdir()
        source = root / 'artifacts/source'
        source.mkdir()
        (source / 'fixture.txt').write_bytes(b'tracked fixture')
        (source / 'fixture.txt').chmod(0o644)
        context = self.context(env)
        inventory = {'context': context}
        source_manifest = {'schema': 1, 'context': context, 'files': {'fixture.txt': {
            'mode': 0o644, 'size': 15, 'sha256': hashlib.sha256(b'tracked fixture').hexdigest()}}}
        values = {'context.json': context, 'inventory.json': inventory, 'plan.json': {'inventory': inventory},
                  'source.json': source_manifest}
        for name, value in values.items():
            (meta / name).write_text(json.dumps(value))
        self.complete(meta)
        return meta

    def complete(self, meta):
        values = {name: hashlib.sha256((meta / name).read_bytes()).hexdigest()
                  for name in ('context.json', 'inventory.json', 'plan.json', 'source.json')}
        (meta / 'complete.json').write_text(json.dumps(values))

    def test_changed_source_filter_failure_stops_before_publishing_output(self):
        build = job(TEAM, 'build')
        self.assertIn('    defaults:\n      run:\n        shell: bash\n', build)
        body = step(TEAM, 'Find changed source files')
        script = textwrap.dedent(body.split('        run: |\n', 1)[1])
        cases = (
            ('fetch failure', 23, "print('unexpected')", 23, None),
            ('filter failure with stdout', 0, "print('partial'); raise SystemExit(17)", 17, None),
            ('filter syntax error', 0, "def broken(", 1, None),
            ('no changed source', 0, "print('')", 0, 'pattern=\n'),
            ('changed source', 0, "print('^src/example[.]cpp$')", 0, 'pattern=^src/example[.]cpp$\n'),
        )
        for name, fetch_exit, source, expected_exit, expected_output in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                helper = root / 'scripts/linters/changed_source_filter.py'
                helper.parent.mkdir(parents=True)
                helper.write_text(source + '\n')
                output = root / 'step-output'
                shell = root / 'step.sh'
                shell.write_text('git() { return "$FETCH_EXIT"; }\n' + script)
                result = subprocess.run(
                    ['bash', '--noprofile', '--norc', '-e', '-o', 'pipefail', str(shell)],
                    cwd=root, env=dict(os.environ, BASE_BRANCH='main', FETCH_EXIT=str(fetch_exit),
                                       GITHUB_OUTPUT=str(output)),
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, expected_exit, result.stdout + result.stderr)
                if expected_output is None:
                    self.assertFalse(output.exists(), 'A failed filter must not publish an empty pattern')
                else:
                    self.assertEqual(output.read_text(), expected_output)

    def test_history_validation_fallback_keeps_workflow_running(self):
        cache_spec = importlib.util.spec_from_file_location('team_cache_hints', ROOT / 'scripts/ci/team_cache.py')
        cache = importlib.util.module_from_spec(cache_spec)
        cache_spec.loader.exec_module(cache)
        start = TEAM.index('          duration_args=()')
        end = TEAM.index('          python3 scripts/testing/team_test_shards.py plan', start)
        source = textwrap.dedent(TEAM[start:end])
        for mode, count in (('valid', 2), ('missing', 0), ('corrupt', 0), ('bad-schema', 0)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                (root / 'scripts/ci').mkdir(parents=True)
                shutil.copy(ROOT / 'scripts/ci/team_cache.py', root / 'scripts/ci/team_cache.py')
                (root / 'ci-logs').mkdir()
                (root / 'cache/snapshot').mkdir(parents=True)
                (root / 'cache/snapshot/object').write_bytes(b'cache entry')
                namespace = dict(image='example.invalid/image@sha256:' + 'a' * 64,
                                 compiler='gcc', arch='x86_64', sanitizer='asan',
                                 build_options='no-pch', path_scheme='canonical')
                (root / 'ci-logs/cache-namespace.json').write_text(json.dumps(namespace))
                history = root / 'cache/history.json'
                history.write_text(json.dumps(dict(schema=1, source_main_sha='1' * 40,
                    inventory_digest='2' * 64, duration_history={'a' * 64: 5.0})))
                cache.generate(root / 'cache/snapshot', root / 'cache/manifest.json', namespace,
                               '1' * 40, history)
                if mode == 'missing':
                    history.unlink()
                elif mode == 'corrupt':
                    history.write_text('{}')
                elif mode == 'bad-schema':
                    history.write_text('{"schema":true}')
                script = source.replace('/__ci/cache', str(root / 'cache'))
                script += '\nprintf "HINT_COUNT=%s\\n" "${#duration_args[@]}"\n'
                result = subprocess.run(['bash', '-e', '-o', 'pipefail', '-c', script], cwd=root,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(f'HINT_COUNT={count}', result.stdout)
                if mode != 'valid':
                    self.assertIn('complete inventory without hints', result.stdout)

    def test_every_inline_python_block_compiles(self):
        blocks = list(re.finditer(r"^([ ]+)python3 (?:-I )?- <<'PY'\n(.*?)^\1PY$", TEAM, re.MULTILINE | re.DOTALL))
        self.assertGreaterEqual(len(blocks), 8)
        for index, block in enumerate(blocks):
            with self.subTest(index=index):
                compile(textwrap.dedent(block[2]), f'pr-ci-team-inline-{index}', 'exec')

    def test_caller_queue_wraps_whole_workflow_and_does_not_cancel_other_prs(self):
        team = job(CALLER, 'team')
        self.assertIn('queue: max', team)
        self.assertIn('cancel-in-progress: false', team)
        self.assertIn('uses: ./.github/workflows/pr-ci-team.yml', team)
        self.assertNotIn('concurrency:', TEAM)
        self.assertIn("group: pr-ci-${{ github.event.pull_request.number || github.sha }}", CALLER)
        self.assertIn('      runner:\n        type: string\n        default: vsag-linux-x64', TEAM)

    def test_build_publishes_fresh_identity_and_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / 'artifacts').mkdir()
            cwd = root / 'checkout'
            cwd.mkdir()
            env = self.environment(root)
            result = self.run_identity_step('Prepare shared build and execution identity', cwd, env, 'rw')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((root / 'artifacts/meta/context.json').read_text()), self.context(env))
            self.assertEqual((cwd / 'build').resolve(), (root / 'artifacts/build').resolve())
            repeated = self.run_identity_step('Prepare shared build and execution identity', cwd, env, 'rw')
            self.assertNotEqual(repeated.returncode, 0)

    def test_read_only_mount_and_exact_run_identity_are_required(self):
        for changed in (None, 'TEAM_REPOSITORY_ID', 'GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT', 'TEAM_TEST_SHA', 'TEAM_HEAD_SHA', 'TEAM_IMAGE'):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                env = self.environment(root)
                self.artifact_metadata(root, env)
                cwd = root / 'checkout'
                cwd.mkdir()
                if changed:
                    env[changed] += 'different'
                result = self.run_identity_step('Verify read-only artifacts and execution identity', cwd, env, 'ro')
                self.assertEqual(result.returncode == 0, changed is None, result.stderr)
                if changed:
                    self.assertIn('mismatch', result.stderr)
                else:
                    self.assertTrue((cwd / 'build').is_symlink())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            env = self.environment(root)
            self.artifact_metadata(root, env)
            result = self.run_identity_step('Verify read-only artifacts and execution identity', root, env, 'rw')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('read-only', result.stderr)

    def test_missing_and_modified_completion_metadata_fail(self):
        for changed in ('complete.json', 'context.json', 'inventory.json', 'source.json'):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                env = self.environment(root)
                meta = self.artifact_metadata(root, env)
                if changed == 'complete.json':
                    (meta / changed).unlink()
                else:
                    (meta / changed).write_text('{}')
                result = self.run_identity_step('Verify read-only artifacts and execution identity', root, env, 'ro')
                self.assertNotEqual(result.returncode, 0)

    def test_single_build_and_four_bounded_test_jobs_keep_diagnostics(self):
        self.assertEqual(TEAM.count('make asan DEBUG_BUILD_DIR="$TEAM_ARTIFACTS/build" COMPILE_JOBS="$COMPILE_JOBS"'), 1)
        self.assertIn("COMPILE_JOBS: '24'", job(TEAM, 'build'))
        tests = job(TEAM, 'tests')
        self.assertIn('shard: [0, 1, 2, 3]', tests)
        self.assertIn('max-parallel: 4', tests)
        self.assertIn('fail-fast: false', tests)
        self.assertIn('--workers 3 --batch-size 32 --batch-target-seconds 25', tests)
        self.assertNotIn('continue-on-error', TEAM)
        for name in ('Upload build diagnostics', 'Upload test diagnostics', 'Upload small test results'):
            self.assertIn('if: always()', step(TEAM, name))
        self.assertLess(TEAM.index('./scripts/format/check_format.sh'), TEAM.index('mkdir -p ci-logs'))
        self.assertIn('if: needs.changes.outputs.cpp == \'true\'', job(TEAM, 'build'))

    def test_consumers_bootstrap_exact_run_source_without_checkout_or_importing_it_first(self):
        build, tests = job(TEAM, 'build'), job(TEAM, 'tests')
        self.assertIn('actions/checkout@v4', build)
        self.assertIn('persist-credentials: false', build)
        self.assertIn('python3 scripts/ci/team_source.py', build)
        self.assertIn('--destination "$TEAM_ARTIFACTS/source"', build)
        self.assertNotIn('actions/checkout', tests)
        bootstrap = step(TEAM, 'Verify read-only artifacts and execution identity')
        self.assertIn("python3 -I - <<'PY'", bootstrap)
        self.assertNotIn('importlib', bootstrap)
        self.assertNotIn('subprocess', bootstrap)
        self.assertNotIn('configure-git-safe-directory', tests)
        self.assertIn("'source.json'", bootstrap)
        self.assertIn('The controller authenticates the run/PVC', bootstrap)
        self.assertIn('/__ci/artifacts/meta/source.json', step(TEAM, 'Upload small test plan'))

    def test_no_pch_build_flag_reaches_cmake_without_changing_sanitizer_scope(self):
        build = job(TEAM, 'build')
        extra = re.search(r"EXTRA_DEFINED: '([^']+)'", build)[1]
        env = dict(os.environ, EXTRA_DEFINED=extra, CMAKE_GENERATOR='Ninja', COMPILE_JOBS='24',
                   VSAG_ENABLE_TOOLS='ON', VSAG_ENABLE_EXAMPLES='ON')
        result = subprocess.run(['make', '--no-print-directory', '-n', 'configure-asan', 'UNAME_S=Linux'],
                                cwd=ROOT, env=env, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in ('-DCMAKE_DISABLE_PRECOMPILE_HEADERS=ON', '-DVSAG_USE_SYSTEM_OPENBLAS=ON',
                       '-DCMAKE_BUILD_TYPE=Sanitize', '-DENABLE_ASAN=ON', '-DENABLE_TSAN=OFF',
                       '-DENABLE_CCACHE=ON', '-DENABLE_TESTS=ON', '-DENABLE_TOOLS=ON',
                       '-DENABLE_EXAMPLES=ON', '-DNUM_BUILDING_JOBS=24'):
            self.assertIn(option, result.stdout)

    def test_no_pch_namespace_rejects_the_previous_pch_snapshot(self):
        source = python_step('Restore a private compiler cache', TEAM)
        cache_spec = importlib.util.spec_from_file_location('team_cache', ROOT / 'scripts/ci/team_cache.py')
        cache = importlib.util.module_from_spec(cache_spec)
        cache_spec.loader.exec_module(cache)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / 'ci-logs').mkdir()
            script = "from unittest.mock import patch\nwith patch('subprocess.check_output', return_value='g++ trusted-test\\n'):\n    exec(" + repr(source) + ")"
            result = subprocess.run([sys.executable, '-c', script], cwd=root, env=self.environment(root),
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            namespace = json.loads((root / 'ci-logs/cache-namespace.json').read_text())
            self.assertIn(';CMAKE_DISABLE_PRECOMPILE_HEADERS=ON;', namespace['build_options'])
            old = dict(namespace, build_options=namespace['build_options'].replace(';CMAKE_DISABLE_PRECOMPILE_HEADERS=ON', ''))
            self.assertNotEqual(cache.namespace_id(namespace), cache.namespace_id(old))
            snapshot = root / 'snapshot'
            snapshot.mkdir()
            (snapshot / 'entry').write_bytes(b'trusted compiler cache')
            manifest = root / 'manifest.json'
            cache.generate(snapshot, manifest, old, '1' * 40)
            destination = root / 'private-cache'
            with self.assertRaisesRegex(ValueError, 'namespace mismatch'):
                cache.restore(snapshot, manifest, destination, namespace)
            self.assertFalse(destination.exists())

    @unittest.skipUnless(shutil.which('cmake') and shutil.which('ninja'), 'CMake and Ninja required')
    def test_real_cmake_ninja_build_uses_physical_shared_directory(self):
        command = re.search(r'make asan [^\n]*?(?= 2>&1|\n|$)', step(TEAM, 'Build ASan with 24 compile jobs'))[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            checkout = root / 'checkout'
            artifacts = root / 'artifacts'
            physical = artifacts / 'build'
            checkout.mkdir()
            physical.mkdir(parents=True)
            (checkout / 'build').symlink_to(physical, target_is_directory=True)
            # Exercise the actual repository's configure-asan/build-asan recipes.
            (checkout / 'Makefile').write_text((ROOT / 'Makefile').read_text())
            (checkout / 'CMakeLists.txt').write_text(textwrap.dedent("""\
                cmake_minimum_required(VERSION 3.20)
                project(SharedBuildPath NONE)
                add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/smoke.txt"
                    COMMAND "${CMAKE_COMMAND}" -E touch "${CMAKE_CURRENT_BINARY_DIR}/smoke.txt"
                    DEPENDS "${CMAKE_SOURCE_DIR}/input.txt")
                add_custom_target(smoke ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/smoke.txt")
            """))
            (checkout / 'input.txt').write_text('source dependency\n')
            env = dict(os.environ, TEAM_ARTIFACTS=str(artifacts), COMPILE_JOBS='2', CMAKE_GENERATOR='Ninja')
            for _ in range(2):
                result = subprocess.run(['bash', '-eu', '-c', command], cwd=checkout, env=env,
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertTrue((physical / 'smoke.txt').is_file())
                self.assertTrue((checkout / 'build/smoke.txt').is_file())
                # Reconfiguration must also resolve the source from the physical build cwd.
                with (checkout / 'CMakeLists.txt').open('a') as output:
                    output.write('# reconfigure\n')

    def test_all_original_catch2_binaries_and_extra_checks_are_required(self):
        inventory = step(TEAM, 'Publish test inventory and shard plan')
        for binary in ('tests/unittests', 'tests/functests', 'tools/eval/eval_dataset_test',
                       'tools/autotune/autotune_test', 'tests/eval_monitor_test'):
            self.assertIn('--binary ' + binary, inventory)
        self.assertIn("assert default == filtered", inventory)
        self.assertIn('--context "$TEAM_ARTIFACTS/meta/context.json"', inventory)
        self.assertIn('--diagnostics-dir "$PWD/ci-logs/inventory"', inventory)
        self.assertIn("pathlib.Path('ci-logs/tool-inventory')", inventory)
        self.assertNotIn('TemporaryDirectory', inventory)
        compatibility = step(TEAM, 'Check index compatibility')
        self.assertIn('matrix.shard == 0', compatibility)
        self.assertLess(compatibility.index('verify_compatibility_cache.py'), compatibility.index('check_compatibility.sh'))
        self.assertNotIn('||', compatibility)
        examples = step(TEAM, 'Run examples')
        self.assertIn('matrix.shard == 1', examples)
        self.assertIn('module.execute', examples)
        self.assertIn("assert all(result['status'] == 'passed'", examples)
        self.assertIn('Lint changed sources', TEAM)
        self.assertIn('clang-tidy-15', step(TEAM, 'Lint changed sources'))

    def test_summary_downloads_only_small_result_artifacts(self):
        summary = job(TEAM, 'summary')
        self.assertIn('runs-on: ubuntu-22.04', summary)
        self.assertIn('team-plan-', summary)
        self.assertIn('pattern: team-results-', summary)
        self.assertNotIn('team-test-diagnostics-', summary)
        self.assertNotIn('/__ci/artifacts/build', summary)
        uploads = (step(TEAM, 'Upload small test plan') + step(TEAM, 'Upload small test results'))
        self.assertNotIn('build/', uploads)
        self.assertNotIn('publish', uploads)
        merged = step(TEAM, 'Upload merged test summary')
        self.assertIn("!cancelled() && hashFiles('ci-summary/summary.json') != ''", merged)
        self.assertIn('name: team-summary-${{ github.run_id }}-${{ github.run_attempt }}', merged)
        self.assertIn('path: ci-summary/summary.json', merged)
        self.assertIn('if-no-files-found: error', merged)

    def test_team_gate_fails_closed_and_allows_only_intentional_skip(self):
        body = step(TEAM, 'Require all Team jobs to pass or be intentionally skipped')
        script = textwrap.dedent(body.split('run: |\n', 1)[1])
        for change_result, changed, build, tests, expected in (
            ('success', 'false', 'skipped', 'skipped', True),
            ('success', 'true', 'success', 'success', True),
            ('success', 'false', 'success', 'skipped', False),
            ('failure', 'false', 'skipped', 'skipped', False),
            ('success', '', 'skipped', 'skipped', False),
            ('success', 'true', 'failure', 'skipped', False),
            ('success', 'true', 'success', 'failure', False),
            ('success', 'true', 'success', 'cancelled', False),
        ):
            with self.subTest(changed=changed, build=build, tests=tests):
                result = subprocess.run(['bash', '-e', '-c', script], env=dict(os.environ,
                    CHANGE_RESULT=change_result, CPP_CHANGED=changed, BUILD_RESULT=build, TEST_RESULT=tests))
                self.assertEqual(result.returncode == 0, expected)

    def summary_fixture(self, root, env):
        binary = 'tests/mock'
        cases = [{'binary': binary, 'name': str(index), 'tags': '',
                  'source_file': 'fixture.cpp', 'source_line': index + 1} for index in range(4)]
        for case in cases:
            case['id'] = SHARDS.stable_id(binary, case)
        inventory = SHARDS.sealed({'schema_version': SHARDS.SCHEMA, 'kind': 'inventory', 'seed': 424242,
            'filter': '~[daily]', 'context': self.context(env), 'binaries': {binary: 'd' * 64}, 'tests': cases})
        plan = SHARDS.make_plan(inventory, 4)
        SHARDS.write_json(root / 'ci-summary/plan/plan.json', plan)
        script_dir = root / 'scripts/testing'
        script_dir.mkdir(parents=True)
        (script_dir / 'team_test_shards.py').symlink_to(ROOT / 'scripts/testing/team_test_shards.py')
        for shard in plan['shards']:
            index = shard['index']
            directory = root / f'ci-summary/results/team-results-{index}-{env["GITHUB_RUN_ID"]}-{env["GITHUB_RUN_ATTEMPT"]}'
            result = SHARDS.sealed({'schema_version': SHARDS.SCHEMA, 'kind': 'result', 'plan_digest': plan['digest'],
                'context': self.context(env), 'shard': index, 'test_ids': shard['test_ids'], 'seed': 424242,
                'success': True, 'batches': [{'index': 0, 'status': 'passed', 'exit_code': 0, 'binary': binary,
                    'test_ids': shard['test_ids'],
                    'tests': [dict(next(case for case in cases if case['id'] == shard['test_ids'][0]), seconds=1, success=True, skips=0)]}]})
            SHARDS.write_json(directory / f'shard-{index}/result.json', result)
            extras = {'shard': index, 'run_id': env['GITHUB_RUN_ID'], 'run_attempt': env['GITHUB_RUN_ATTEMPT'],
                      'test_sha': env['TEAM_TEST_SHA'], 'catch2': 'success',
                      'compatibility': 'success' if index == 0 else 'skipped',
                      'examples': 'success' if index == 1 else 'skipped'}
            SHARDS.write_json(directory / 'extras.json', extras)

    def test_summary_executes_real_merge_and_rejects_extra_check_failure(self):
        for failure in (None, 'examples', 'case'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                env = self.environment(root)
                self.summary_fixture(root, env)
                if failure == 'examples':
                    path = root / 'ci-summary/results/team-results-1-456-1/extras.json'
                    value = json.loads(path.read_text())
                    value['examples'] = 'failure'
                    path.write_text(json.dumps(value))
                elif failure == 'case':
                    path = root / 'ci-summary/results/team-results-0-456-1/shard-0/result.json'
                    value = json.loads(path.read_text())
                    value['batches'][0]['tests'][0]['success'] = False
                    SHARDS.write_json(path, SHARDS.sealed({key: item for key, item in value.items() if key != 'digest'}))
                process = subprocess.run([sys.executable, '-c', python_step('Verify complete test results', TEAM)],
                                         cwd=root, env=env, text=True, capture_output=True)
                self.assertEqual(process.returncode == 0, failure is None, process.stderr)
                if failure is None:
                    self.assertTrue(json.loads((root / 'ci-summary/summary.json').read_text())['success'])


if __name__ == '__main__':
    unittest.main()
