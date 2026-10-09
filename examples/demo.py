"""Show what monoclone sees and what it does about it.

Run:
    python examples/demo.py             all workloads
    python examples/demo.py record_score one workload
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monoclone import analyze, calibrate, workloads
from monoclone.timing import best_of_interleaved, normalize, runner


def main():
    wanted = sys.argv[1:]
    chosen = [w for w in workloads.all_workloads() if not wanted or w.name in wanted]
    if not chosen:
        print("no such workload. available: %s" % ", ".join(
            w.name for w in workloads.all_workloads()
        ))
        return 1

    print("calibrating this machine")
    machine = calibrate()
    print("  " + machine.describe())
    print()

    for workload in chosen:
        print("=" * 78)
        print("%s: %s" % (workload.name, workload.note))
        print("=" * 78)
        report = analyze(workload.fn, workload.stream_factory, machine=machine)
        print(report.summary())

        if report.clone_set is not None:
            print()
            print("generated dispatcher")
            for line in report.clone_set.source.rstrip().splitlines():
                print("    " + line)

        items = normalize(workload.stream_factory())
        timings = best_of_interleaved(
            {
                "baseline": runner(workload.fn, items),
                "monoclone": runner(report.callable or workload.fn, items),
            },
            reps=11,
        )
        speedup = timings["baseline"] / timings["monoclone"]
        print()
        print(
            "measured            %.2f ms baseline, %.2f ms after, %.3fx"
            % (timings["baseline"] * 1e3, timings["monoclone"] * 1e3, speedup)
        )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
