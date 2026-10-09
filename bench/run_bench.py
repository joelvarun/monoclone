"""Measure monoclone against three alternatives and write the results.

Four callables are timed over the identical argument list, round robin, and the
minimum of each is reported:

baseline
    the original function, left alone.
ungated
    the same pipeline with the cost model switched off and the group floor
    dropped to zero, which is what cloning every polymorphic function looks
    like. Grouping, clone budget and verification are unchanged, so the only
    variable against monoclone is the decision rule.
ablation
    the same number of dispatches into a single shared clone. Per-call cost is
    identical to a real clone set and monomorphism is the one thing removed, so
    any speedup that survives here did not come from specialization.
monoclone
    the cost model's decision.

A fifth timing, a second independently built clone set using the identical
configuration, is the noise floor: it should match monoclone exactly, and
whatever it differs by is the resolution of everything else on the chart.

Usage:
    python bench/run_bench.py [--reps 13] [--quick]
"""

import argparse
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monoclone import cloner, costmodel, workloads
from monoclone.api import analyze
from monoclone.timing import best_of_interleaved, normalize, runner

DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")


def _ablation_for(report, workload):
    """A cost matched dispatcher whose clones are all the same code object."""
    if not report.decision.cloned or not report.group_keys:
        return None
    return cloner.build_single(
        workload.fn, report.group_keys, report.decision.positions
    )


def measure(workload, machine, reps):
    gated = analyze(workload.fn, workload.stream_factory, machine=machine)
    control = analyze(workload.fn, workload.stream_factory, machine=machine)
    ungated = analyze(
        workload.fn,
        workload.stream_factory,
        machine=machine,
        force=True,
        min_share=0.0,
    )
    ablation = _ablation_for(ungated, workload)

    items = normalize(workload.stream_factory())
    thunks = {
        "baseline": runner(workload.fn, items),
        "monoclone": runner(gated.callable or workload.fn, items),
        "control": runner(control.callable or workload.fn, items),
        "ungated": runner(ungated.callable or workload.fn, items),
        "ablation": runner(
            ablation.dispatcher if ablation else workload.fn, items
        ),
    }
    timings = best_of_interleaved(thunks, reps=reps)
    base = timings["baseline"]
    return {
        "name": workload.name,
        "note": workload.note,
        "calls": gated.profile.calls,
        "signatures": gated.profile.distinct,
        "sites": len(gated.sites),
        "recoverable": len(gated.recoverable),
        "statuses": sorted({s.status for s in gated.recoverable}),
        "ops_by_bucket": {
            "%s/%s" % b: v for b, v in gated.decision.ops_by_bucket.items()
        },
        "ops_per_call": sum(gated.decision.ops_by_bucket.values()),
        "cloned": gated.decision.cloned,
        "clones": gated.decision.clones,
        "ungated_clones": ungated.decision.clones,
        "positions": list(gated.decision.positions),
        "groups": gated.group_labels,
        "covered_share": gated.decision.covered_share,
        "reason": gated.decision.reason,
        "verified_calls": gated.decision.verified_calls,
        "predicted_speedup": gated.prediction.predicted_speedup,
        "predicted_net_ms": gated.prediction.net_s * 1e3,
        "ms": {k: v * 1e3 for k, v in timings.items()},
        "speedup": {k: base / v for k, v in timings.items()},
        "expect_clone": workload.expect_clone,
    }


def aggregate(records):
    keys = ("baseline", "monoclone", "control", "ungated", "ablation")
    totals = {k: sum(r["ms"][k] for r in records) for k in keys}
    out = {
        "total_ms": totals,
        "speedup": {k: totals["baseline"] / totals[k] for k in keys},
    }
    # For a workload the cost model left alone, the "after" callable is the
    # original function object itself, so its measured ratio against the
    # baseline is a direct read of the harness noise and nothing else. That is
    # a stricter floor than comparing two independently built clone sets.
    identical = [
        abs(r["speedup"][key] - 1.0) * 100.0
        for r in records
        if not r["cloned"]
        for key in ("monoclone", "control")
    ]
    rebuilt = [
        abs(r["speedup"]["control"] - r["speedup"]["monoclone"]) * 100.0
        for r in records
        if r["cloned"]
    ]
    out["noise_floor_pct"] = max(identical) if identical else max(rebuilt)
    out["rebuild_spread_pct"] = max(rebuilt) if rebuilt else 0.0
    out["decisions_correct"] = sum(
        1 for r in records if r["cloned"] == (r["speedup"]["ungated"] > 1.0)
    )
    out["worst_ungated_speedup"] = min(r["speedup"]["ungated"] for r in records)
    out["best_monoclone_speedup"] = max(r["speedup"]["monoclone"] for r in records)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=13)
    parser.add_argument("--quick", action="store_true", help="fewer repeats")
    args = parser.parse_args()
    reps = 5 if args.quick else args.reps

    print("python %s on %s" % (platform.python_version(), platform.machine()))
    print("calibrating the machine")
    started = time.time()
    machine = costmodel.calibrate(reps=reps)
    print("  " + machine.describe())

    records = []
    for workload in workloads.all_workloads():
        record = measure(workload, machine, reps)
        records.append(record)
        print(
            "  %-14s %-9s clones %d  baseline %6.2f ms  monoclone %.3fx  "
            "ungated %.3fx  ablation %.3fx"
            % (
                record["name"],
                "cloned" if record["cloned"] else "left alone",
                record["clones"],
                record["ms"]["baseline"],
                record["speedup"]["monoclone"],
                record["speedup"]["ungated"],
                record["speedup"]["ablation"],
            )
        )

    summary = aggregate(records)
    metrics = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "reps": reps,
        "elapsed_s": round(time.time() - started, 1),
        "calibration": {
            "dispatch_ns": machine.dispatch_s * 1e9,
            "saving_ns_per_op": {
                "%s/%s" % k: v * 1e9 for k, v in machine.saving_s.items()
            },
            "breakeven_ops": {
                "%s/%s" % k: machine.breakeven_ops(k) for k in machine.saving_s
            },
            "probe_ops_per_call": {"%s/%s" % k: v for k, v in machine.probe_ops.items()},
        },
        "workloads": records,
        "summary": summary,
    }
    os.makedirs(DOCS, exist_ok=True)
    path = os.path.join(DOCS, "metrics.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, default=str)
    print("wrote %s" % path)

    print(
        "total %.2f ms baseline, %.2f ms monoclone (%.3fx), %.2f ms ungated (%.3fx)"
        % (
            summary["total_ms"]["baseline"],
            summary["total_ms"]["monoclone"],
            summary["speedup"]["monoclone"],
            summary["total_ms"]["ungated"],
            summary["speedup"]["ungated"],
        )
    )
    print("noise floor %.1f percent" % summary["noise_floor_pct"])

    from chart import render

    render(metrics, os.path.join(DOCS, "before_after.png"))
    print("wrote %s" % os.path.join(DOCS, "before_after.png"))


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
