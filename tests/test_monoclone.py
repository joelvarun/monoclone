"""Correctness checks. Run with pytest, or directly with python.

The interesting property is that a dispatcher is indistinguishable from the
function it replaces, including for argument types the profile never saw, for
keyword calls, and for calls that raise.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fractions import Fraction

from monoclone import calibrate, cloner, grouping, sites, thrash, typeprof, workloads
from monoclone.api import analyze


def scaled(a, b, factor=2):
    total = a * b
    for _ in range(4):
        total = total + a * factor
    return total


def test_decoder_matches_dis():
    import dis

    for _ in range(80):
        scaled(2, 3)
    mine = [(o, n) for o, n, _ in sites.decode(scaled.__code__)]
    reference = [
        (i.offset, i.opname)
        for i in dis.get_instructions(scaled.__code__, adaptive=True)
    ]
    assert mine == reference


def test_monitor_separates_warmup_from_thrash():
    namespace = {}
    exec(compile("def f(a, b):\n    return a * b + a\n", "<t>", "exec"), namespace)
    fn = namespace["f"]
    monitor = thrash.ThrashMonitor(0.0).watch(fn.__code__)
    with thrash.fine_grained_switching():
        with monitor:
            for _ in range(40000):
                fn(2, 3)
                fn(2.0, 3.0)
    observed = monitor.sites_for(fn.__code__)
    assert observed, "no specializable sites found"
    assert any(s.status == "form_thrash" for s in observed)


def test_dispatcher_matches_original_everywhere():
    keys = [(int,), (float,)]
    built = cloner.build(scaled, keys, positions=(0,))
    cases = [
        ((2, 3), {}),
        ((2.0, 3.0), {}),
        ((Fraction(1, 3), 2), {}),           # type the profile never saw
        ((2, 3), {"factor": 5}),             # keyword argument
        ((2, 3, 7), {}),                     # explicit third positional
        (("ab", 2), {}),                     # str times int still works
    ]
    for args, kwargs in cases:
        assert built.dispatcher(*args, **kwargs) == scaled(*args, **kwargs)


def test_dispatcher_propagates_exceptions():
    built = cloner.build(scaled, [(int,), (float,)], positions=(0,))
    for bad in (None, object()):
        try:
            scaled(bad, 2)
        except TypeError:
            pass
        else:
            raise AssertionError("expected TypeError from the original")
        try:
            built.dispatcher(bad, 2)
        except TypeError:
            pass
        else:
            raise AssertionError("expected TypeError from the dispatcher")


def test_defaults_are_preserved():
    built = cloner.build(scaled, [(int,)], positions=(0,))
    assert built.dispatcher.__defaults__ == scaled.__defaults__
    assert built.dispatcher(2, 3) == scaled(2, 3)


def test_can_clone_rejects_closures():
    def outer(n):
        def inner(x):
            return x * n

        return inner

    ok, reason = cloner.can_clone(outer(2))
    assert not ok and "free variables" in reason


def test_redundant_positions_are_dropped():
    profile = typeprof.TypeProfile()
    profile.calls = 100
    profile.signatures[(int, int, int)] = 60
    profile.signatures[(float, float, float)] = 40
    kept = grouping.minimal_positions(profile, frozenset({0, 1, 2}))
    assert len(kept) == 1


def test_monomorphic_stream_is_left_alone():
    machine = calibrate(reps=3, calls=300)
    stream = [(2, 3)] * 400
    report = analyze(scaled, lambda: stream, machine=machine, count_limit=100)
    assert not report.decision.cloned


def test_every_workload_agrees_with_its_clones():
    machine = calibrate(reps=3, calls=300)
    for workload in workloads.all_workloads():
        report = analyze(
            workload.fn,
            workload.stream_factory,
            machine=machine,
            count_limit=150,
            verify_limit=200,
        )
        if report.decision.cloned:
            checked, mismatches = cloner.verify(
                workload.fn, report.callable, workload.stream_factory(), limit=200
            )
            assert checked > 0
            assert not mismatches, "%s disagreed" % workload.name


def main():
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print("pass  %s" % name)
        except AssertionError as exc:
            failures += 1
            print("FAIL  %s: %s" % (name, exc))
    print("%d failures" % failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
