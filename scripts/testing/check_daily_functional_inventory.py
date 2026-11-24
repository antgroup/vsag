"""Check Nightly ownership of the compiled functional test inventory."""

import argparse
from pathlib import Path
import signal
import sys
import threading

import daily_plan as dp
import daily_run
import team_test_shards as shards


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("diagnostics", type=Path)
    args = parser.parse_args()

    cancelled = threading.Event()
    received_signal = None

    def cancel(signum, _frame):
        nonlocal received_signal
        if received_signal is None:
            received_signal = signum
        cancelled.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, cancel)
    try:
        tests, _ = daily_run.discover_inventory(
            args.binary.resolve(), args.diagnostics.resolve(), 424242, cancelled
        )
        if not tests:
            raise dp.PlanError("Empty functional test inventory")
        for group in shards.case_groups(tests):
            dp.group_owner(group, "functional")
    except (ValueError, OSError) as error:
        if received_signal is not None:
            return 128 + received_signal
        print(f"Nightly functional inventory failed: {error}", file=sys.stderr)
        return 1

    if received_signal is not None:
        return 128 + received_signal
    print(f"Nightly functional ownership verified for {len(tests)} tests.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
