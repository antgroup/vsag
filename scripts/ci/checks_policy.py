#!/usr/bin/env python3
"""Select the original five check scopes without running PR code with write tokens."""
import json
import os
from pathlib import Path
import re
import subprocess

PATHS = json.loads(Path(__file__).with_name('checks_paths.json').read_text())
DEFAULT_ACTIONS = {'opened', 'synchronize', 'reopened'}


def matches(path, pattern):
    expression = re.escape(pattern).replace(r'\*\*/', '(?:.*/)?')
    expression = expression.replace(r'\*\*', '.*').replace(r'\*', '[^/]*')
    return re.fullmatch(expression, path) is not None


def code_event(event):
    return event['action'] in DEFAULT_ACTIONS or (
        event['action'] == 'edited' and 'base' in event.get('changes', {}))


def select(event_name, event, paths):
    selected = dict.fromkeys(('format', 'docs', 'issue', 'simd', 'lite'), False)
    if event_name == 'push' and event.get('deleted') is True:
        return selected
    if event_name == 'workflow_dispatch':
        choice = event.get('inputs', {}).get('check', 'full')
        if choice in ('full', 'full-pr', 'full-branch'):
            return selected
        if choice not in ('format', 'docs', 'lite'):
            raise ValueError('Unknown manual check')
        for key in ('format', 'docs', 'lite'):
            selected[key] = choice == key
        return selected
    if event_name == 'pull_request':
        branch = event['pull_request']['base']['ref']
        maintained = branch == 'main' or bool(re.fullmatch(r'0\.[^/]*', branch))
        action = event['action']
        selected['issue'] = maintained and action in DEFAULT_ACTIONS | {'edited', 'labeled', 'unlabeled'}
        if code_event(event):
            for key in ('docs', 'simd', 'lite'):
                eligible = key == 'docs' or (key == 'simd' and branch == 'main') or (key == 'lite' and maintained)
                selected[key] = eligible and any(matches(p, pat) for p in paths for pat in PATHS[key])
    elif event_name == 'push':
        branch = event['ref'].removeprefix('refs/heads/')
        maintained = branch == 'main' or bool(re.fullmatch(r'0\.[^/]*', branch))
        selected['format'] = maintained and any(not p.startswith('docs/') for p in paths)
        selected['docs'] = branch == 'main' and any(matches(p, pat) for p in paths for pat in PATHS['docs'])
    return selected


def git(*args):
    return subprocess.check_output(['git', *args])


def changed_paths(event_name, event):
    if event_name == 'push' and event.get('deleted') is True:
        return []
    if event_name == 'workflow_dispatch' or (event_name == 'pull_request' and not code_event(event)):
        return []
    if event_name == 'pull_request':
        base, head = event['pull_request']['base']['sha'], event['pull_request']['head']['sha']
        for sha in (base, head):
            if not re.fullmatch('[0-9a-fA-F]{40}', sha):
                raise ValueError('Invalid PR SHA')
        base = git('merge-base', base, head).decode().strip()
    else:
        base, head = event['before'], event['after']
        for sha in (base, head):
            if not re.fullmatch('[0-9a-fA-F]{40}', sha):
                raise ValueError('Invalid push SHA')
        if base == '0' * 40:
            # GitHub new-branch path filtering compares against the parent of
            # the oldest pushed commit. Without that payload, fail visibly.
            commits = event.get('commits', [])
            if not commits or len(commits) >= 2048:
                raise ValueError('New branch commit range unavailable')
            oldest = commits[0]['id']
            if not re.fullmatch('[0-9a-fA-F]{40}', oldest):
                raise ValueError('Invalid oldest push SHA')
            parents = git('rev-list', '--parents', '-n', '1', oldest).decode().split()
            base = parents[1] if len(parents) > 1 else git('hash-object', '-t', 'tree', '/dev/null').decode().strip()
    # No rename detection: both old and new paths must participate in selection.
    return [p.decode('utf-8', errors='surrogateescape') for p in git('diff', '--name-only', '--no-renames', '-z', base, head, '--').split(b'\0') if p]


def main():
    event = json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text())
    name = os.environ['GITHUB_EVENT_NAME']
    result = select(name, event, changed_paths(name, event))
    with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
        for key, value in result.items():
            output.write(f'{key}={str(value).lower()}\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
