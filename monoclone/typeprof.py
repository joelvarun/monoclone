"""Type profiling and exact per-instruction execution counts.

Two passes, each measuring only what it can measure correctly.

The type pass wraps the target and records the tuple of argument types seen on
every call. It perturbs timing badly and its timings are discarded; the call
counts and type signatures it produces are exact.

The count pass uses ``sys.monitoring`` INSTRUCTION events to get exact
execution counts per bytecode offset. Instrumentation replaces specialized
instructions with their instrumented forms, so no specialization state can be
read during this pass, but control flow is unchanged and therefore the counts
are right. The thrash pass runs separately, uninstrumented.
"""

import collections
import functools
import sys
from dataclasses import dataclass, field

#: sys.monitoring tool id. 0 to 5 are reserved for well known tools; 5 is the
#: documented slot for ad hoc profilers.
TOOL_ID = 5


@dataclass
class TypeProfile:
    """Argument type signatures observed for one function."""

    calls: int = 0
    signatures: collections.Counter = field(default_factory=collections.Counter)
    kwarg_calls: int = 0
    arity: int = 0

    @property
    def distinct(self):
        return len(self.signatures)

    @property
    def polymorphic_positions(self):
        """Positions whose observed type varies across calls."""
        if not self.signatures:
            return frozenset()
        width = max(len(s) for s in self.signatures)
        varying = set()
        for i in range(width):
            seen = {s[i] for s in self.signatures if len(s) > i}
            if len(seen) > 1:
                varying.add(i)
        return frozenset(varying)

    def share(self, signature):
        return self.signatures[signature] / self.calls if self.calls else 0.0


def profile_calls(fn, stream):
    """Run ``fn`` over ``stream`` and record argument types.

    ``stream`` yields argument tuples, or ``(args, kwargs)`` pairs.
    """
    profile = TypeProfile()

    @functools.wraps(fn)
    def recorder(*args, **kwargs):
        profile.calls += 1
        if kwargs:
            profile.kwarg_calls += 1
        profile.signatures[tuple(type(a) for a in args)] += 1
        return fn(*args, **kwargs)

    for item in stream:
        args, kwargs = _split(item)
        recorder(*args, **kwargs)
    if profile.signatures:
        profile.arity = max(len(s) for s in profile.signatures)
    return profile


def _split(item):
    if (
        isinstance(item, tuple)
        and len(item) == 2
        and isinstance(item[0], tuple)
        and isinstance(item[1], dict)
    ):
        return item
    if isinstance(item, tuple):
        return item, {}
    return (item,), {}


def count_instructions(fn, stream, limit=None):
    """Exact execution count per bytecode offset of ``fn``.

    Returns ``(counts, calls)``. Only ``fn.__code__`` is instrumented, so the
    rest of the program runs at full speed.
    """
    mon = sys.monitoring
    counts = collections.Counter()
    code = fn.__code__

    def on_instruction(_code, offset):
        counts[offset] += 1

    calls = 0
    mon.use_tool_id(TOOL_ID, "monoclone")
    try:
        mon.register_callback(TOOL_ID, mon.events.INSTRUCTION, on_instruction)
        mon.set_local_events(TOOL_ID, code, mon.events.INSTRUCTION)
        try:
            for item in stream:
                if limit is not None and calls >= limit:
                    break
                args, kwargs = _split(item)
                fn(*args, **kwargs)
                calls += 1
        finally:
            mon.set_local_events(TOOL_ID, code, 0)
            mon.register_callback(TOOL_ID, mon.events.INSTRUCTION, None)
    finally:
        mon.free_tool_id(TOOL_ID)
    return counts, calls
