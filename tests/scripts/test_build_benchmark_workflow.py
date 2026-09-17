"""Focused workflow contracts; stdlib only, no C++ build or network required."""

import copy
import json
import os
import re
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[2]
PR = (ROOT / '.github/workflows/pr-ci.yml').read_text(encoding='utf-8')
BENCH = (ROOT / '.github/workflows/build_performance.yml').read_text(encoding='utf-8')


def block(workflow, header, description):
    """Extract our block-style YAML subset, stopping at the next peer/ancestor."""
    match = re.search(r'^( *)' + header + r'[ \t]*(?:#.*)?$', workflow, re.MULTILINE)
    workflow_name = 'pr-ci.yml' if workflow == PR else 'build_performance.yml' if workflow == BENCH else 'provided workflow'
    if match is None:
        raise AssertionError(f'Missing {description} in {workflow_name}; check its name and block formatting')
    lines = []
    for line in workflow[match.end():].splitlines()[1:]:
        stripped = line.lstrip(' ')
        if stripped and not stripped.startswith('#') and len(line) - len(stripped) <= len(match[1]):
            break
        lines.append(line)
    return '\n'.join(lines) + '\n'


def step(workflow, name):
    return block(workflow, r'-[ \t]+name:[ \t]+' + re.escape(name), f'step {name!r}')


def job(workflow, name):
    return block(workflow, re.escape(name) + ':', f'job {name!r}')


def python_step(name, workflow=None):
    body = textwrap.dedent(step(BENCH if workflow is None else workflow, name))
    opening = re.search(r"^([ \t]*)python3 - <<'PY'[ \t]*$", body, re.MULTILINE)
    if opening is None:
        raise AssertionError(f"Step {name!r} missing expected inline Python marker python3 - <<'PY'")
    source = body[opening.end() + 1:]
    closing = re.search(r'^' + re.escape(opening[1]) + r'PY[ \t]*$', source, re.MULTILINE)
    if closing is None:
        raise AssertionError(f"Step {name!r} missing closing inline Python delimiter PY")
    return textwrap.dedent(source[:closing.start()])


class WorkflowContractTest(unittest.TestCase):
    def test_block_extraction_accepts_spacing_comments_and_unnamed_steps(self):
        for indent in (2, 6, 9):
            prefix = ' ' * indent
            workflow = '\n'.join(prefix + line for line in (
                '-  name: Target # comment', '   run: |', '     echo selected',
                '# between steps', '- uses: actions/checkout@v4', '  with:',
                '    ref: main', '- name: Later', '  run: echo later'))
            body = step(workflow, 'Target')
            self.assertIn('echo selected', body)
            self.assertNotIn('actions/checkout', body)
            self.assertNotIn('echo later', body)
        workflow = 'jobs:\n    other:\n      runs-on: ubuntu-latest\n    target: # job\n      name: selected\n    final:\n      name: excluded\n'
        self.assertIn('name: selected', job(workflow, 'target'))
        self.assertNotIn('excluded', job(workflow, 'target'))
        with self.assertRaisesRegex(AssertionError, "Missing job 'missing' in provided workflow"):
            job(workflow, 'missing')

    def test_python_extraction_reports_missing_delimiters(self):
        workflow = "    - name: Inline\n      run: |\n        python3 - <<'PY'\n        print('ok')\n        PY\n    - uses: example/action@v1\n"
        self.assertEqual(python_step('Inline', workflow), "print('ok')\n")
        for candidate, message in (
                (workflow.replace("python3 - <<'PY'", "python - <<'EOF'"), 'missing expected inline Python marker'),
                (workflow.replace('\n        PY\n', '\n        EOF\n'), 'missing closing inline Python delimiter')):
            with self.subTest(message=message):
                with self.assertRaisesRegex(AssertionError, "Step 'Inline' " + message):
                    python_step('Inline', candidate)

    def test_toolchain_metadata_uses_runtime_environment(self):
        prepare = step(BENCH, 'Prepare trusted Ubuntu host')
        script = textwrap.dedent(prepare.split('run: |\n', 1)[1])
        # Run only the metadata block; no apt installation or real toolchain required.
        script = script[script.index('{\n'):]
        values = dict(BENCHMARK_JOBS='7', CMAKE_GENERATOR='Custom Generator',
                      CMAKE_EXPORT_COMPILE_COMMANDS='OFF', EXTRA_DEFINED='-DCUSTOM=ON',
                      VSAG_ENABLE_EXAMPLES='OFF', VSAG_ENABLE_TOOLS='OFF')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'build-metrics').mkdir()
            stubs = 'g++() { :; }; cmake() { :; }; ninja() { :; }; ccache() { :; };\n'
            result = subprocess.run(['bash', '-eu', '-c', stubs + script], cwd=root,
                                    env=dict(os.environ, **values), text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            metadata = (root / 'build-metrics/toolchain.txt').read_text()
            for key, value in values.items():
                self.assertIn(f'{key}={value}\n', metadata)
            self.assertNotIn('Sanitize ASan/UBSan', metadata)

    def test_recorded_collector_command_matches_executed_arguments(self):
        script = textwrap.dedent(step(BENCH, 'Collect isolated cold warm and no-op metrics').split('run: |\n', 1)[1])
        values = dict(BENCHMARK_JOBS='7', BUILD_BASE_SHA='a' * 40, BUILD_COMMIT_SHA='b' * 40,
                      DEPENDENCY_CACHE_KEY='key with spaces', DEPENDENCY_CACHE_MATCHED_KEY='',
                      DEPENDENCY_CACHE_STATE='miss', DEPENDENCY_SOURCE_PREPARATION_SECONDS='1',
                      DEPENDENCY_CACHE_RESTORE_SECONDS='2', DEPENDENCY_PREPARATION_SECONDS='3')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'build-metrics').mkdir()
            stubs = "g++() { printf '11.4.0\\n'; }; python3() { printf '%s\\0' \"$@\" > executed; };\n"
            result = subprocess.run(['bash', '-eu', '-c', stubs + script], cwd=root,
                                    env=dict(os.environ, **values), text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            executed = (root / 'executed').read_bytes()
            self.assertIn(b'--jobs\x007\x00', executed)
            self.assertIn(b'--dependency-cache-key\x00key with spaces\x00', executed)
            recorded = (root / 'build-metrics/collector-command.txt').read_text()
            replay = subprocess.run(['bash', '-eu', '-c', stubs + recorded], cwd=root,
                                    text=True, capture_output=True)
            self.assertEqual(replay.returncode, 0, replay.stderr)
            self.assertEqual((root / 'executed').read_bytes(), executed)

    def test_single_pr_build_and_preserved_options(self):
        with self.assertRaisesRegex(AssertionError, "Missing step 'Missing step' in pr-ci.yml"):
            step(PR, 'Missing step')
        self.assertIn('uses: actions/cache/restore@v4', step(PR, 'Restore ExternalProject Archives'))
        x86 = job(PR, 'build-asan-x86')
        self.assertEqual(x86.count('make asan COMPILE_JOBS=3'), 1)
        # Collector unit tests remain valid PR checks; only executing the benchmark is forbidden.
        for forbidden in ('python3 scripts/collect_build_metrics.py', '--clear-ccache', 'build-performance-metrics', 'COMPILER_CACHE_KEY'):
            self.assertNotIn(forbidden, x86)
        for option in ('CMAKE_GENERATOR="Ninja"', 'CMAKE_EXPORT_COMPILE_COMMANDS=ON',
                       '-DVSAG_USE_SYSTEM_OPENBLAS=ON', 'VSAG_ENABLE_EXAMPLES=ON',
                       'VSAG_ENABLE_TOOLS=ON', 'ccache --zero-stats', '/usr/bin/time -p'):
            self.assertIn(option, x86)
        self.assertIn('if: always()', step(PR, 'Show compiler cache diagnostics'))
        self.assertIn('ccache --show-stats --verbose || ccache --show-stats || true', x86)
        self.assertIn('name: ASan Build X86\n    needs: [changes, format]', x86)
        for name in ('Get changed source files', 'Run Lint', 'Verify changed-source lint execution', 'Clean', 'Save Build'):
            self.assertIn('      - name: ' + name + '\n', x86)
        self.assertIn('name: build-x86-${{ github.run_id }}', x86)
        self.assertIn('path: ./build\n', x86)
        self.assertIn('retention-days: 1', step(PR, 'Save Build'))
        makefile = (ROOT / 'Makefile').read_text(encoding='utf-8')
        self.assertIn('-DCMAKE_BUILD_TYPE=Sanitize -DENABLE_ASAN=ON -DENABLE_TSAN=OFF -DENABLE_CCACHE=ON -DENABLE_TESTS=ON', makefile)

    def test_security_and_isolation(self):
        self.assertIn('permissions:\n  contents: read\n', BENCH)
        self.assertNotIn('secrets.', BENCH)
        self.assertNotIn('pull_request:', BENCH)
        self.assertNotIn('pull_request_target', BENCH)
        self.assertIn("cron: '23 3 * * 1'", BENCH)
        self.assertIn('runs-on: ubuntu-22.04', BENCH)
        self.assertIn('timeout-minutes: 180', BENCH)
        self.assertIn('group: build-performance-', BENCH)
        self.assertIn('persist-credentials: false', BENCH)
        self.assertIn('CCACHE_DIR: ${{ github.workspace }}/.benchmark-ccache', BENCH)
        self.assertNotIn('uses: actions/cache@', BENCH)
        self.assertNotIn('actions/cache/save', BENCH)
        self.assertNotIn('ccache-action', BENCH)
        self.assertIn('uses: actions/cache/restore@v4', BENCH)
        self.assertNotIn('install_deps', step(BENCH, 'Prepare trusted Ubuntu host'))
        self.assertNotIn('/opt/hostedtoolcache', BENCH)
        self.assertIn("REQUESTED_REF: ${{ github.event.inputs.ref || 'main' }}", BENCH)
        self.assertIn('if: always()', step(BENCH, 'Upload build metrics'))
        self.assertIn('retention-days: 14', step(BENCH, 'Upload build metrics'))
        collect = step(BENCH, 'Collect isolated cold warm and no-op metrics')
        self.assertIn('--commit-sha "$BUILD_COMMIT_SHA"', collect)
        self.assertNotIn('--commit-sha "$GITHUB_SHA"', collect)
        self.assertIn("BENCHMARK_JOBS: '3'", BENCH)
        self.assertIn('--jobs "$BENCHMARK_JOBS"', collect)
        self.assertIn('"${collector_command[@]}"', collect)
        self.assertIn('build-metrics/collector-command.txt', collect)
        self.assertIn('--build-dir build', collect)
        self.assertLess(BENCH.index('Require verified-noop'), BENCH.index('Prepare trusted Ubuntu host'))

    def run_python(self, name, directory):
        env = dict(os.environ, GITHUB_STEP_SUMMARY=str(directory / 'summary.md'), BUILD_COMMIT_SHA='a' * 40)
        return subprocess.run(['python3', '-c', python_step(name)], cwd=directory,
                              env=env, text=True, capture_output=True)

    def test_preflight_rejects_legacy_without_executing_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'scripts').mkdir()
            (root / 'build-metrics').mkdir()
            collector = root / 'scripts/collect_build_metrics.py'
            valid = textwrap.dedent('''\
                raise RuntimeError('must not execute')
                class Collector:
                    def write_reports(self):
                        report = {'schema_version': 3}
                    def capture_build(self):
                        phase['true_noop'] = True
                ''')
            cases = [
                (valid, True),
                (valid + "other = {'schema_version': 1}\n", True),
                (valid.replace("{'schema_version': 3}", "{'schema_version': 1}"), False),
                (valid.replace("report =", "unrelated ="), False),
                (valid.replace('write_reports', 'unrelated_method'), False),
                (valid.replace('capture_build', 'unrelated_method'), False),
                (valid.replace("phase['true_noop'] = True", "value = phase['true_noop']"), False),
                (valid.replace("phase['true_noop']", "other['true_noop']"), False),
                (valid.replace("report = {'schema_version': 3}",
                               "report = {'schema_version': 3}\n        report = {'schema_version': 1}"), False),
                ("report = {'schema_version': 3}\nphase['true_noop'] = True\n", False),
                ('invalid Python!', False),
            ]
            for source, should_pass in cases:
                with self.subTest(source=source):
                    collector.write_text(source, encoding='utf-8')
                    result = self.run_python('Require verified-noop collector support', root)
                    self.assertEqual(result.returncode == 0, should_pass, result.stderr)
                    if not should_pass:
                        self.assertIn('NOT trustworthy', (root / 'summary.md').read_text())
                        self.assertIn('#2899', (root / 'build-metrics/preflight.txt').read_text())

    def test_report_gate_requires_available_verified_noop(self):
        # Schema 3 phase/field names from #2899 head 9073806, not legacy schema 1.
        report = {'schema_version': 3, 'status': 'success', 'commit_sha': 'a' * 40, 'phases': [
            {'name': 'noop_build', 'exit_code': 0, 'true_noop': True,
             'ninja': {'available': True}, 'build_edges': 0}]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'build-metrics').mkdir()
            variants = []
            for key, value in [('true_noop', False), ('true_noop', None), ('exit_code', 1),
                               ('ninja', {'available': False}), ('build_edges', 1),
                               ('build_edges', False), ('build_edges', '0'),
                               ('name', 'noop_incremental_build')]:
                bad = copy.deepcopy(report)
                bad['phases'][0][key] = value
                variants.append(bad)
            variants.extend([dict(report, schema_version=1), dict(report, status='failed'),
                             dict(report, phases=[]), dict(report, commit_sha='b' * 40)])
            cases = [(report, True), (copy.deepcopy(report), True)]
            cases.extend((candidate, False) for candidate in variants)
            for candidate, should_pass in cases:
                (root / 'build-metrics/build-metrics.json').write_text(json.dumps(candidate), encoding='utf-8')
                result = self.run_python('Verify benchmark report', root)
                self.assertEqual(result.returncode == 0, should_pass, result.stderr)
            path = root / 'build-metrics/build-metrics.json'
            path.write_text('invalid JSON', encoding='utf-8')
            self.assertNotEqual(self.run_python('Verify benchmark report', root).returncode, 0)
            path.unlink()
            self.assertNotEqual(self.run_python('Verify benchmark report', root).returncode, 0)
            self.assertIn('FAILED', (root / 'build-metrics/verification.txt').read_text())

    def test_ref_gate_with_real_git_history(self):
        script = textwrap.dedent(step(BENCH, 'Resolve approved benchmark commit').split('run: |\n', 1)[1])
        self.assertNotIn('${{', script)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def git(*args):
                return subprocess.check_output(['git', *args], cwd=root, text=True).strip()

            git('init', '-q')
            git('config', 'user.name', 'Workflow Test')
            git('config', 'user.email', 'workflow-test@example.invalid')
            git('commit', '--allow-empty', '-qm', 'ancestor')
            ancestor = git('rev-parse', 'HEAD')
            git('commit', '--allow-empty', '-qm', 'approved')
            approved = git('rev-parse', 'HEAD')
            tree = git('rev-parse', 'HEAD^{tree}')
            git('tag', '-a', 'annotated', '-m', 'tag is not a commit')
            tag = git('rev-parse', 'annotated')
            git('commit', '--allow-empty', '-qm', 'not in approved history')
            unapproved = git('rev-parse', 'HEAD')
            for requested, allowed in [(ancestor, True), ('main', True), (approved, True),
                                       (tree, False), (tag, False), (unapproved, False), ('0' * 40, False),
                                       ('refs/pull/1/head', False), ('-h', False),
                                       ('$(touch injected)', False), ('main\ninjected', False)]:
                git('checkout', '--detach', '-q', approved)
                env = dict(os.environ, REQUESTED_REF=requested,
                           GITHUB_OUTPUT=str(root / 'output'), GITHUB_STEP_SUMMARY=str(root / 'summary'))
                result = subprocess.run(['bash', '-c', script], cwd=root, env=env,
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode == 0, allowed, result.stderr)
                if requested == '0' * 40:
                    self.assertIn('::error::Unknown commit SHA', result.stdout)
                    self.assertNotIn('fatal:', result.stderr)
            self.assertFalse((root / 'injected').exists())
            self.assertIn('commit_sha=' + approved, (root / 'build-metrics/target.txt').read_text())


if __name__ == '__main__':
    unittest.main()
