#!/usr/bin/env python3
"""Refresh only a retry's execution clock; original source/date plan stays frozen."""
import argparse
import datetime as dt
import math
import sys
import os
from pathlib import Path

import daily_plan as dp


def refresh(context, attempt, expected_attempt):
    dp.validate_context(context)
    if attempt['source_sha'] != context['source_sha'] or attempt['attempt'] != expected_attempt:
        raise dp.PlanError('Retry attempt/source mismatch')
    window = dp.sealed({'schema_version': 1, 'kind': 'daily-window', **attempt})
    dp.execution_deadline(context, window)
    return window


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--context', required=True)
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    ctx = dp.load(args.context)
    window = refresh(ctx, dp.load(args.attempt), int(os.environ['GITHUB_RUN_ATTEMPT']))
    dp.write(Path(args.context).with_name('window.json'), window)
    remaining = (dp.execution_deadline(ctx, window) - dt.datetime.now(dt.timezone.utc)).total_seconds()
    minutes = math.floor(remaining / 60)
    if minutes < 1:
        print('Job/attempt execution window exhausted before stage start', file=sys.stderr)
        return 1
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
            print('remaining_minutes=' + str(minutes), file=output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
