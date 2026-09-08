#!/usr/bin/env python3
"""Run the FP32 five-stage cycle with a saved initial index and durable status."""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import sweep_mci_thresholds as common


def read_rows(root):
    path = root / "curve.csv"
    if not path.exists():
        return []
    with path.open(newline="") as source:
        return [row for row in csv.DictReader(source) if all(v is not None for v in row.values())]


def worker(root):
    manifest = json.loads((root / "manifest.json").read_text())
    environment = dict(os.environ, LD_LIBRARY_PATH=str(root / "bin"))
    start = time.monotonic()
    process = None
    try:
        if common.fingerprint(Path(manifest["dataset"]["path"])) != manifest["dataset"]:
            raise ValueError("dataset changed after launch preparation")
        for file in manifest["artifacts"]:
            if common.fingerprint(Path(file["path"]), True) != file:
                raise ValueError("benchmark snapshot changed")
        with (root / "benchmark.log").open("w") as log:
            process = subprocess.Popen(manifest["command"], stdout=log,
                                       stderr=subprocess.STDOUT, env=environment)
            while process.poll() is None:
                rows = read_rows(root)
                completed = [stage for stage in common.STAGES
                             if sum(row["stage"] == stage for row in rows) == 4]
                common.write_json(root / "status.json", {
                    "state": "running", "pid": process.pid,
                    "completed_stages": completed, "curve_points": len(rows),
                    "initial_index_saved": (root / "initial.index.json").exists(),
                    "elapsed_seconds": time.monotonic() - start})
                time.sleep(5)
        if process.returncode:
            raise RuntimeError(f"benchmark exited with {process.returncode}")
        rows = read_rows(root)
        common.validate_rows(rows, [40, 80, 160, 320], "force", common.BASELINE)
        subprocess.run([sys.executable, str(common.REPO / "scripts/perf_reports/"
                                            "plot_mci_mutation_curve.py"),
                        str(root / "curve.csv"), "--output", str(root / "qps-recall.png")],
                       check=True, env=dict(os.environ, MPLBACKEND="Agg"))
        common.write_json(root / "status.json", {
            "state": "complete", "completed_stages": common.STAGES,
            "curve_points": len(rows), "elapsed_seconds": time.monotonic() - start})
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=30)
        common.write_json(root / "status.json", {"state": "failed", "error": str(error)})
        raise


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--worker-root":
        worker(Path(sys.argv[2]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=Path("/root/data/codefilter-3m-384-angular-f32.hdf5"))
    parser.add_argument("--binary", type=Path,
                        default=common.REPO / "build-release/tools/eval/mci_mutation_benchmark")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    import h5py
    with h5py.File(args.dataset, "r") as data:
        count = len(data["train"])
    root = args.output_dir.resolve()
    if root.exists() and any(root.iterdir()):
        parser.error("output directory is nonempty; refusing to overwrite")
    (root / "bin").mkdir(parents=True)
    binary = root / "bin/mci_mutation_benchmark"
    shutil.copy2(args.binary, binary)
    artifacts = [common.fingerprint(binary, True)]
    linkage = subprocess.run(["ldd", str(args.binary.resolve())], capture_output=True,
                             text=True, check=True)
    for soname, path in re.findall(r"(libvsag\S*)\s+=>\s+(\S+)", linkage.stdout):
        destination = root / "bin" / soname
        shutil.copy2(path, destination)
        artifacts.append(common.fingerprint(destination, True))
    cmd = [str(binary), "--dataset", str(args.dataset.resolve()),
           "--output", str(root / "curve.csv"), "--force-remove",
           "--build-threads", str(args.threads), "--search-threads", str(args.threads),
           "--mutation-batch-size", str(int(count * 0.1 + 0.5)),
           "--ef-search-values", "40,80,160,320", "--query-count", "200",
           "--search-count", "10000", "--random-seed", "20260907"]
    cmd += ["--save-initial-index", str(root / "initial.index")]
    common.write_json(root / "manifest.json", {
        "command": cmd, "artifacts": artifacts, "dataset": common.fingerprint(args.dataset),
        "base_count": count, "parameters": common.BASELINE,
        "truth_policy": "protect evaluation top-k neighbors, same as historical five-stage test",
        "mutation_parallelism": "sequential batches; HGraph build/search use requested threads"})
    common.write_json(root / "status.json", {"state": "starting"})
    with (root / "worker.log").open("w") as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                    "--worker-root", str(root)], stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({"worker_pid": process.pid, "output_dir": str(root)}))


if __name__ == "__main__":
    main()
