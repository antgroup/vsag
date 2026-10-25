"""Budget arithmetic, partial-run reporting and API pagination regressions."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('timing', ROOT / 'scripts/ci/team_timing.py')
TIMING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TIMING)
BASELINE = TIMING.load(TIMING.BASELINE)


def job(name, start, end, conclusion='success'):
    def date(seconds):
        return TIMING.datetime.datetime.fromtimestamp(seconds, TIMING.datetime.timezone.utc).isoformat()
    return dict(name='Team / ' + name, started_at=date(start), completed_at=date(end),
                status='completed', conclusion=conclusion, steps=[])


def fixture():
    return [job(TIMING.BUILD, 1000, 1343), job(TIMING.CHECKS, 990, 1060),
            job(TIMING.COMPAT, 1413, 1551)] + [job(name, 1360 + offset, 1690 - offset)
                                                for offset, name in enumerate(TIMING.SHARDS)]


class TimingTest(unittest.TestCase):
    def test_parallel_wall_clock_not_sum_or_case_cpu_time(self):
        self.assertEqual(TIMING.stages(fixture()), BASELINE['stages'])
        self.assertEqual(sum(BASELINE['budgets'][k] for k in ('build', 'parallel', 'handoff')), 900)
        report = TIMING.aggregate(fixture(), BASELINE)
        self.assertIn('11.50 min', report)
        self.assertIn('Not the original 753.991 s', report)
        self.assertIn('all succeeded', report)

    def test_missing_and_skipped_are_not_zero_or_success(self):
        for jobs in (fixture()[:-1], fixture() + []):
            if len(jobs) == 7:
                jobs[-1]['conclusion'] = 'skipped'
            report = TIMING.aggregate(jobs, BASELINE)
            self.assertIn('| Missing |', report)
            self.assertNotIn('all succeeded', report)

    def test_failed_job_retains_duration_and_failure(self):
        jobs = fixture()
        jobs[-1]['conclusion'] = 'failure'
        report = TIMING.aggregate(jobs, BASELINE)
        self.assertIn('[failure]', report)
        self.assertNotIn('all succeeded', report)
        self.assertIn('11.50 min', report)

    def test_overlap_and_duplicate_job_names_rejected(self):
        jobs = fixture()
        jobs[-1]['started_at'] = jobs[0]['started_at']
        self.assertIsNone(TIMING.stages(jobs))
        with self.assertRaises(ValueError):
            TIMING.stages(fixture() + [fixture()[0]])

    def test_over_budget_warns_but_exits_normally(self):
        jobs = fixture()
        jobs[-1]['completed_at'] = '1970-01-01T01:00:00+00:00'
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'jobs.json'
            path.write_text(json.dumps({'jobs': jobs}))
            summary = Path(temp) / 'summary.md'
            with patch('sys.argv', ['timing', '--kind', 'aggregate', '--jobs', str(path), '--output', str(summary)]), contextlib.redirect_stdout(io.StringIO()) as out:
                TIMING.main()
            self.assertIn('::warning::', out.getvalue())
            self.assertIn('WARN: over budget', summary.read_text())

    def test_api_uses_exact_attempt_and_paginates(self):
        payloads = [{'jobs': [{}] * 100}, {'jobs': [{'name': 'last'}]}]
        requests = []
        def open_url(request, timeout):
            requests.append(request.full_url)
            return io.BytesIO(json.dumps(payloads.pop(0)).encode())
        env = dict(GITHUB_REPOSITORY='owner/repo', GITHUB_RUN_ID='123', GITHUB_RUN_ATTEMPT='2', GH_TOKEN='test')
        with patch.dict(os.environ, env), patch.object(TIMING.urllib.request, 'urlopen', side_effect=open_url):
            self.assertEqual(len(TIMING.fetch_jobs()), 101)
        self.assertIn('/runs/123/attempts/2/jobs?', requests[0])
        self.assertIn('page=2', requests[1])

    def test_api_failure_does_not_print_private_error_or_fail_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'summary.md'
            with patch('sys.argv', ['timing', '--kind', 'aggregate', '--output', str(output)]), patch.object(TIMING, 'fetch_jobs', side_effect=RuntimeError('secret address')):
                TIMING.main()
            self.assertIn('unavailable', output.read_text())
            self.assertNotIn('secret address', output.read_text())

    def test_build_cache_and_missing_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertIn('N/A', TIMING.local('build', root, BASELINE))
            (root / 'build.log').write_text('real 221.19\n')
            (root / 'ccache-stats.tsv').write_text('direct_cache_hit\t790\npreprocessed_cache_hit\t8\ncache_miss\t5\n')
            report = TIMING.local('build', root, BASELINE)
            self.assertIn('798 hits / 5 misses (99.38%)', report)
            self.assertIn('3.69 min', report)

    def test_case_escaping_identity_and_slowest_first(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'shard-0').mkdir()
            data = dict(shard=0, seconds=240.676, context={'run_id': '123', 'run_attempt': '2', 'test_sha': 'abc'}, success=True,
                        batches=[{'tests': [{'name': '<script>|x\ny', 'seconds': 50}, {'name': 'fast', 'seconds': 1}]}], examples=[])
            path = root / 'shard-0/result.json'
            path.write_text(json.dumps(data))
            with patch.dict(os.environ, {'GITHUB_RUN_ID': '123', 'GITHUB_RUN_ATTEMPT': '2', 'TEAM_TEST_SHA': 'abc'}):
                report = TIMING.local('test', root, BASELINE)
                self.assertIn('&lt;script&gt;&#124;x y', report)
                self.assertLess(report.index('&lt;script&gt;'), report.index('| fast |'))
                data['context']['run_attempt'] = '1'
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    TIMING.local('test', root, BASELINE)

    def test_workflow_reporting_is_advisory_and_permissions_are_scoped(self):
        text = (ROOT / '.github/workflows/pr-ci-team.yml').read_text()
        for name in ('build', 'test', 'compatibility'):
            self.assertIn('name: Report {} timing budget\n        if: always()\n        continue-on-error: true'.format(name), text)
        self.assertIn("if: always() && needs.changes.outputs.cpp == 'true'\n        continue-on-error: true", text)
        self.assertEqual(text.count('actions: read'), 1)
        self.assertIn('Require all Team jobs to pass or be intentionally skipped', text)


if __name__ == '__main__':
    unittest.main()
