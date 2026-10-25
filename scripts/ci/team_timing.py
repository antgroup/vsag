#!/usr/bin/env python3
"""Advisory Team CI timing summaries; never a correctness or authorization gate."""
import argparse
import datetime
import html
import json
import math
import os
from pathlib import Path
import re
import urllib.request

BASELINE = Path(__file__).with_name('team_timing_baseline.json')
BUILD = 'Team ASan Build'
CHECKS = 'Team Format and Helper Checks'
COMPAT = 'Team Index Compatibility'
SHARDS = ['Team Test shard {}'.format(i) for i in range(4)]
REQUIRED = [BUILD, CHECKS, COMPAT] + SHARDS


def number(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value >= 0)


def escape(value):
    return html.escape(str(value)).replace('|', '&#124;').replace('\n', ' ').replace('\r', ' ')[:240]


def minutes(value):
    return '{:.2f} min'.format(value / 60) if number(value) else 'N/A'


def row(label, actual, baseline, budget):
    used = '{:.0%}'.format(actual / budget) if number(actual) else 'N/A'
    delta = '{:+.1f} s'.format(actual - baseline) if number(actual) and number(baseline) else 'N/A'
    state = 'WARN: over budget' if number(actual) and actual > budget else ('Within budget' if number(actual) else 'Missing')
    return '| {} | {} | {} | {} | {} | {} | {} |'.format(
        escape(label), minutes(budget), minutes(baseline), minutes(actual), used, delta, state)


def table():
    return ['| Stage | Budget | Baseline | Actual | Budget used | vs baseline | Timing status |',
            '| --- | ---: | ---: | ---: | ---: | ---: | --- |']


def load(path):
    return json.loads(Path(path).read_text())


def stamp(value):
    try:
        return datetime.datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
    except (AttributeError, ValueError, TypeError):
        return None


def interval(job):
    if job.get('status') != 'completed' or job.get('conclusion') == 'skipped':
        return None
    start, end = stamp(job.get('started_at')), stamp(job.get('completed_at'))
    return (start, end) if start is not None and end is not None and end >= start else None


def select_jobs(jobs):
    selected = {}
    for job in jobs:
        name = job.get('name', '').rsplit(' / ', 1)[-1]
        if name in REQUIRED:
            if name in selected:
                raise ValueError('Ambiguous Team job names')
            selected[name] = job
    return selected


def stages(jobs):
    selected = select_jobs(jobs)
    spans = {name: interval(job) for name, job in selected.items()}
    if any(not spans.get(name) for name in REQUIRED):
        return None
    start, end = spans[BUILD]
    consumers = [spans[name] for name in SHARDS + [COMPAT]]
    if min(s[0] for s in consumers) < end:
        return None
    parallel = max(s[1] for s in consumers) - min(s[0] for s in consumers)
    total = max(s[1] for s in spans.values()) - start
    return {'build': end - start, 'parallel': parallel,
            'handoff': total - (end - start) - parallel, 'total': total}


def fetch_jobs():
    repo = os.environ['GITHUB_REPOSITORY']
    run, attempt = os.environ['GITHUB_RUN_ID'], os.environ['GITHUB_RUN_ATTEMPT']
    api = os.environ.get('GITHUB_API_URL', 'https://api.github.com').rstrip('/')
    jobs = []
    for page in range(1, 21):
        url = '{}/repos/{}/actions/runs/{}/attempts/{}/jobs?per_page=100&page={}'.format(api, repo, run, attempt, page)
        request = urllib.request.Request(url, headers={
            'Authorization': 'Bearer ' + os.environ['GH_TOKEN'],
            'Accept': 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28'})
        with urllib.request.urlopen(request, timeout=15) as response:
            data = json.load(response)
        jobs.extend(data['jobs'])
        if len(data['jobs']) < 100:
            return jobs
    raise ValueError('Job pagination limit reached')


def header(baseline):
    return ['## Team CI timing budget', '',
            '[Baseline: #3143, attempt 2]({}) — fixed warm-cache reference.'.format(baseline['url']),
            'Advisory only: timing warnings do not change test conclusions or the PR gate.', '']


def aggregate(jobs, baseline):
    lines = header(baseline)
    lines += ['### Execution envelope (15-minute budget)',
              'Build job start → last required build/check/test/compatibility job completion. '
              'Includes internal handoff gaps; excludes initial admission queue, this Summary job, '
              'final resource cleanup and Community jobs. Not the original 753.991 s acceptance metric.', '']
    lines += table()
    values = stages(jobs)
    for key, label in [('build', 'Build (46.7%)'), ('parallel', 'Parallel validation envelope (46.7%)'),
                       ('handoff', 'Handoff / checks tail (6.7%)'), ('total', 'Execution envelope (100%)')]:
        lines.append(row(label, values.get(key) if values else None,
                         baseline['stages'][key], baseline['budgets'][key]))
    selected = select_jobs(jobs)
    complete_success = len(selected) == len(REQUIRED) and all(j.get('conclusion') == 'success' for j in selected.values())
    lines += ['', 'Required execution jobs: **{}**. Timing status is not test success.'.format(
        'all succeeded' if complete_success else 'failed, skipped, missing or incomplete'), '',
        '### Individual jobs (parallel rows must not be added)', ''] + table()
    for name in REQUIRED:
        job = selected.get(name, {})
        span = interval(job)
        actual = span[1] - span[0] if span else None
        budget = baseline['job_budgets'][name]
        lines.append(row(name + ' [' + str(job.get('conclusion') or 'unknown') + ']',
                         actual, baseline['jobs'][name], budget))
    # A separate step table helps distinguish container startup, compile and upload regressions.
    lines += ['', '### Slowest completed steps', '', '| Job / step | Actual | Baseline | Change |',
              '| --- | ---: | ---: | ---: |']
    steps = []
    for name, job in selected.items():
        for step in job.get('steps', []):
            span = interval(step)
            if span:
                steps.append((span[1] - span[0], name, step.get('name', 'unknown')))
    for seconds, name, step in sorted(steps, reverse=True)[:12]:
        old = baseline['steps'].get(name, {}).get(step)
        delta = '{:+.1f} s'.format(seconds - old) if number(old) else 'N/A'
        lines.append('| {} / {} | {} | {} | {} |'.format(escape(name), escape(step), minutes(seconds), minutes(old), delta))
    durations = [interval(selected.get(name, {})) for name in SHARDS]
    if all(durations):
        seconds = [end - start for start, end in durations]
        lines += ['', 'Shard job spread: **{:.1f} s**, slowest/fastest **{:.2f}×**.'.format(
            max(seconds) - min(seconds), max(seconds) / max(min(seconds), .001))]
    return '\n'.join(lines) + '\n'


def local(kind, directory, baseline):
    root = Path(directory)
    lines = header(baseline) + table()
    if kind == 'build':
        log = root / 'build.log'
        matches = re.findall(r'^real\s+(\d+(?:\.\d+)?)\s*$', log.read_text(errors='replace'), re.M) if log.exists() else []
        seconds = float(matches[-1]) if matches else None
        lines.append(row('Compile + link command', seconds, 221.19, 270))
        cache = root / 'ccache-stats.tsv'
        if cache.exists():
            stats = {}
            for line in cache.read_text().splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1].isdigit():
                    stats[parts[0]] = int(parts[1])
            keys = ['direct_cache_hit', 'preprocessed_cache_hit', 'cache_miss']
            if all(key in stats for key in keys):
                hits = stats[keys[0]] + stats[keys[1]]
                misses = stats[keys[2]]
                rate = '{:.2%}'.format(hits / (hits + misses)) if hits + misses else 'N/A'
                lines += ['', 'ccache: **{} hits / {} misses ({})**; baseline: 798 hits / 5 misses (99.38%).'.format(hits, misses, rate)]
            else:
                lines += ['', 'ccache counters: N/A (missing or unsupported format).']
        else:
            lines += ['', 'ccache counters: N/A (no statistics file).']
    elif kind == 'test':
        files = list(root.glob('shard-*/result.json'))
        if len(files) != 1:
            return '\n'.join(lines + ['No unique shard result: interrupted, failed before reporting, or unavailable.']) + '\n'
        result = load(files[0])
        context = result.get('context', {})
        for key, env in [('run_id', 'GITHUB_RUN_ID'), ('run_attempt', 'GITHUB_RUN_ATTEMPT'), ('test_sha', 'TEAM_TEST_SHA')]:
            if os.environ.get(env) and str(context.get(key)) != os.environ[env]:
                raise ValueError('Timing result identity mismatch')
        index = result['shard']
        if type(index) is not int or index not in range(4):
            raise ValueError('Invalid shard index')
        lines.append(row('Shard {} test execution'.format(index), result.get('seconds'), baseline['shard_execution'][index], 315))
        lines += ['', 'Test result: **{}** (independent of timing budget).'.format('passed' if result.get('success') is True else 'failed/incomplete')]
        cases = [(t.get('seconds'), t.get('name', t.get('id', '?'))) for batch in result.get('batches', []) for t in batch.get('tests', [])]
        cases += [(e.get('seconds'), e.get('binary', '?')) for e in result.get('examples', [])]
        lines += ['', '### Slowest cases / examples', '', '| Case / example | Actual |', '| --- | ---: |']
        for seconds, name in sorted((c for c in cases if number(c[0])), reverse=True)[:10]:
            lines.append('| {} | {} |'.format(escape(name), minutes(seconds)))
        lines += ['', 'Case times overlap across workers; do not sum them into wall-clock time.']
    else:
        result = root / 'compatibility-result.json'
        outcome = load(result).get('compatibility', 'unknown') if result.exists() else 'unknown'
        lines.append(row('Compatibility job (see final Summary)', None, baseline['jobs'][COMPAT], baseline['job_budgets'][COMPAT]))
        lines += ['', 'Compatibility outcome: **{}**. Full job and check-step timing appear in Team Summary.'.format(escape(outcome))]
    lines += ['', 'Local command timing excludes job setup and artifact uploads. Full job timing appears in Team Summary.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['aggregate', 'build', 'test', 'compatibility'], required=True)
    parser.add_argument('--directory', default='ci-logs')
    parser.add_argument('--jobs', help='Offline exact-attempt Jobs API JSON fixture')
    parser.add_argument('--baseline', default=str(BASELINE))
    parser.add_argument('--output', default=os.environ.get('GITHUB_STEP_SUMMARY'))
    args = parser.parse_args()
    try:
        baseline = load(args.baseline)
        if args.kind == 'aggregate':
            jobs = load(args.jobs)['jobs'] if args.jobs else fetch_jobs()
            report = aggregate(jobs, baseline)
        else:
            report = local(args.kind, args.directory, baseline)
    except Exception as error:
        # Do not print exception text: HTTP or file errors can contain private URLs/paths.
        report = '## Team CI timing budget\n\nTiming data unavailable ({}). Existing test gates remain authoritative.\n'.format(type(error).__name__)
    if args.output:
        with open(args.output, 'a') as output:
            output.write(report)
    else:
        print(report)
    if 'WARN: over budget' in report:
        print('::warning::Team CI timing budget exceeded; see Job Summary. Advisory only.')
    if 'unavailable' in report or '| Missing |' in report:
        print('::warning::Team CI timing data incomplete; see Job Summary.')


if __name__ == '__main__':
    main()
