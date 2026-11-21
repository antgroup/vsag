#!/usr/bin/env python3
"""Docs synchronization retains its original docs/** push filter."""
import json
import os
from pathlib import Path
from checks_policy import changed_paths


def main():
    event = json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text())
    paths = changed_paths(os.environ['GITHUB_EVENT_NAME'], event)
    selected = any(path.startswith('docs/') for path in paths)
    with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
        output.write(f'docs={str(selected).lower()}\n')


if __name__ == '__main__':
    main()
