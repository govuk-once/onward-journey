#!/usr/bin/env python3
"""
Benchmark harness to compare test runtime with and without session memory.

Default commands mirror the repo's suggested test invocation. You can tweak
paths, region, memory store, run count, and extra args via flags.
"""

from __future__ import annotations

import argparse
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Sequence

import pandas as pd


def run_once(cmd: Sequence[str], passthrough: bool) -> float:
    """Run a single command and return wall-clock seconds; raises on failure."""
    start = time.perf_counter()
    if passthrough:
        # Inherit stdout/stderr so child logs stream to console
        result = subprocess.run(cmd)
    else:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    duration = time.perf_counter() - start
    if result.returncode != 0:
        sys.stderr.write(
            f"\nCommand failed (exit {result.returncode}): {' '.join(shlex.quote(c) for c in cmd)}\n"
        )
        if not passthrough:
            sys.stderr.write(result.stdout)
            sys.stderr.write(result.stderr)
        raise RuntimeError("Command failed; aborting benchmark run")
    return duration


def compute_accuracy(output_dir: Path) -> float | None:
    """
    Read confusion_matrix_uid.csv in the output_dir and compute accuracy
    (trace / total). Returns None if the file is missing or empty.
    """
    cm_path = output_dir / "confusion_matrix_uid.csv"
    if not cm_path.exists():
        return None
    df = pd.read_csv(cm_path, index_col=0)
    if df.empty:
        return None
    total = df.to_numpy().sum()
    if total == 0:
        return None
    diag = df.to_numpy().trace()
    return diag / total


def fmt_summary(values: List[float], unit: str) -> str:
    """Return a formatted summary string for mean/stdev/min/max."""
    mean = statistics.mean(values)
    stdev = statistics.pstdev(values) if len(values) > 1 else 0.0
    return (
        f"runs={len(values):2d}  mean={mean:6.2f}{unit}  stdev={stdev:5.2f}{unit}  "
        f"min={min(values):5.2f}{unit}  max={max(values):5.2f}{unit}"
    )


def build_commands(args: argparse.Namespace, fast_answer_threshold: float, output_dir: Path) -> tuple[List[str], List[str]]:
    base = [
        "uv",
        "run",
        "main.py",
        "test",
        "--kb_path",
        args.kb_path,
        "--test_data_path",
        args.test_data_path,
        "--region",
        args.region,
    ]
    if args.session_id:
        base += ["--session_id", args.session_id]
    if args.extra_args:
        base += args.extra_args

    no_mem = base + ["--memory_store", "none", "--output_dir", str(output_dir)]

    with_mem = base + [
        "--memory_store",
        args.memory_store,
        "--fast_answer_threshold",
        str(fast_answer_threshold),
        "--output_dir",
        str(output_dir),
    ]
    if args.memory_path:
        with_mem += ["--memory_path", args.memory_path]
    if args.memory_k:
        with_mem += ["--memory_k", str(args.memory_k)]
    if args.memory_max_items:
        with_mem += ["--memory_max_items", str(args.memory_max_items)]
    if args.verbose:
        with_mem.append("--verbose")

    return no_mem, with_mem


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=1, help="iterations per config (baseline and each threshold)")
    parser.add_argument("--kb_path", default="../data/processed/kb/oj_knowledge.csv")
    parser.add_argument(
        "--test_data_path",
        default="../data/test/prototype1/user_prompts_small.csv",
    )
    parser.add_argument("--region", default="eu-west-2")
    parser.add_argument("--session_id", help="optional session id to reuse across runs")

    parser.add_argument(
        "--memory_store",
        default="in_memory",
        choices=["in_memory", "json", "none"],
        help="store to use for the memory-enabled variant",
    )
    parser.add_argument("--memory_path", help="path for json memory store")
    parser.add_argument("--memory_k", type=int, help="override memory_k")
    parser.add_argument("--memory_max_items", type=int, help="override memory_max_items")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.85], help="fast-answer thresholds to test (e.g., 0.5 0.6 0.7)")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="pass --verbose to the memory-enabled runs to log fast-answer hits",
    )
    parser.add_argument(
        "--extra_args",
        nargs=argparse.REMAINDER,
        help="additional args appended to BOTH commands (place after --)",
    )
    parser.add_argument(
        "--passthrough",
        action="store_true",
        help="stream child stdout/stderr instead of capturing (useful with --verbose in child)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    base_output_root = Path("./output/bench_thresholds")
    base_output_root.mkdir(parents=True, exist_ok=True)

    baseline_out = base_output_root / "baseline_no_memory"
    baseline_cmd, _ = build_commands(args, fast_answer_threshold=args.thresholds[0], output_dir=baseline_out)

    print("Running benchmark from app/ directory")
    print("Baseline (no memory) command:", " ".join(shlex.quote(c) for c in baseline_cmd))
    print(f"Thresholds to test (with memory): {', '.join(str(t) for t in args.thresholds)}")
    print(f"Iterations per config: {args.runs}")
    print()

    # Baseline: no memory
    baseline_times: List[float] = []
    baseline_acc: List[float] = []
    for i in range(args.runs):
        print(f"[baseline {i+1}/{args.runs}] no-memory run...")
        baseline_times.append(run_once(baseline_cmd, passthrough=args.passthrough))
        acc = compute_accuracy(baseline_out)
        if acc is not None:
            baseline_acc.append(acc)

    # Memory runs per threshold
    threshold_summaries: list[tuple[float, List[float], List[float]]] = []
    for thr in args.thresholds:
        mem_out = base_output_root / f"mem_{thr}"
        mem_cmd = build_commands(args, fast_answer_threshold=thr, output_dir=mem_out)[1]
        runs: List[float] = []
        accs: List[float] = []
        for i in range(args.runs):
            print(f"[threshold {thr} run {i+1}/{args.runs}] memory-enabled run...")
            runs.append(run_once(mem_cmd, passthrough=args.passthrough))
            acc = compute_accuracy(mem_out)
            if acc is not None:
                accs.append(acc)
        threshold_summaries.append((thr, runs, accs))

    print("\nSummary (wall-clock seconds + accuracy):")
    base_acc_pct = statistics.mean(baseline_acc) * 100 if baseline_acc else None
    base_line = f"no memory  {fmt_summary(baseline_times, 's')}  acc={base_acc_pct:5.1f}%" if base_acc_pct is not None else f"no memory  {fmt_summary(baseline_times, 's')}  acc= n/a"
    print(base_line)
    for thr, times, accs in threshold_summaries:
        acc_pct = statistics.mean(accs) * 100 if accs else None
        acc_part = f"  acc={acc_pct:5.1f}%" if acc_pct is not None else "  acc= n/a"
        print(f"mem@{thr:<4} {fmt_summary(times, 's')}{acc_part}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
