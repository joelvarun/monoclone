"""Minimum-of-repeats timing.

The mean of a wall clock sample is dominated by whatever else the machine was
doing; the minimum is the closest available estimate of the cost of the work
itself. Every number this project reports is a minimum over repeats.
"""

import gc
import time


def best_of(thunk, reps=9, warmup=1):
    """Smallest wall clock time of ``reps`` runs of ``thunk``, in seconds."""
    for _ in range(warmup):
        thunk()
    best = float("inf")
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(reps):
            start = time.perf_counter()
            thunk()
            elapsed = time.perf_counter() - start
            if elapsed < best:
                best = elapsed
    finally:
        if gc_was_enabled:
            gc.enable()
    return best


def run_stream(fn, items):
    """Return a thunk that calls ``fn`` once per item of a materialised list."""

    def thunk():
        for args in items:
            fn(*args)

    return thunk


def best_of_interleaved(thunks, reps=5, warmup=1):
    """Time several alternatives round robin and return the minimum of each.

    Running one alternative to completion before starting the next lets a slow
    drift in machine load show up as a difference between alternatives.
    Cycling through them spreads any drift across all of them equally.
    """
    names = list(thunks)
    for name in names:
        for _ in range(warmup):
            thunks[name]()
    best = {name: float("inf") for name in names}
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(reps):
            for name in names:
                start = time.perf_counter()
                thunks[name]()
                elapsed = time.perf_counter() - start
                if elapsed < best[name]:
                    best[name] = elapsed
    finally:
        if gc_was_enabled:
            gc.enable()
    return best


def normalize(stream):
    """Materialise a call stream as a list of plain positional argument tuples.

    Done once, before any timing, so that the measured loop contains nothing
    but the calls themselves. Argument normalisation inside the timed loop
    would add a few hundred nanoseconds per call and dilute every ratio
    reported here toward one.
    """
    from monoclone.typeprof import _split

    items = []
    for item in stream:
        args, kwargs = _split(item)
        if kwargs:
            raise ValueError("timing streams must be positional only")
        items.append(args)
    return items


def runner(fn, items):
    """A thunk calling ``fn`` once per normalised argument tuple."""

    def thunk():
        for args in items:
            fn(*args)

    return thunk
