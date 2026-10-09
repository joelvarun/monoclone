"""When cloning pays for itself.

Cloning buys monomorphic specialization at every recoverable site and costs one
type dispatch per call. Both sides are machine constants, so both are measured
here rather than assumed.

One constant is not enough. A failed operand guard on an int add, a moved
inline cache on an attribute load and a subscript switching between dict and
list are three different costs, and a model that averages them over-predicts
container code while under-predicting attribute code. So three probes run, one
per class of site, and each site's predicted saving uses the constant for its
own class.

The ratio of dispatch cost to saving is the only number the decision rule
needs: the break-even count of recoverable instruction executions per call.
Below it a polymorphic function should be left alone. This is not a small
correction. A three-operation body called with two types runs substantially
slower after cloning, because the dispatch is most of the call.
"""

import collections
from dataclasses import dataclass, field

from monoclone import thrash, typeprof
from monoclone.timing import best_of_interleaved

#: Instruction families grouped by what their inline cache guards.
SCALAR = "scalar"
ATTRIBUTE = "attribute"
CONTAINER = "container"

FAMILY_CLASS = {
    "BINARY_OP": SCALAR,
    "COMPARE_OP": SCALAR,
    "TO_BOOL": SCALAR,
    "FOR_ITER": SCALAR,
    "UNPACK_SEQUENCE": SCALAR,
    "CONTAINS_OP": SCALAR,
    "SEND": SCALAR,
    "LOAD_GLOBAL": SCALAR,
    "LOAD_ATTR": ATTRIBUTE,
    "STORE_ATTR": ATTRIBUTE,
    "LOAD_SUPER_ATTR": ATTRIBUTE,
    "CALL": ATTRIBUTE,
    "CALL_KW": ATTRIBUTE,
    "BINARY_SUBSCR": CONTAINER,
    "STORE_SUBSCR": CONTAINER,
    "BINARY_SLICE": CONTAINER,
    "STORE_SLICE": CONTAINER,
}


#: How a site fails, which costs differently from how much it costs to reach.
FORM = "form"
CACHE = "cache"


def site_class(family):
    return FAMILY_CLASS.get(family, SCALAR)


def site_kind(status):
    """Thrash kind of a site, from its observed specialization history.

    A site flipping between two specialized opcodes pays a full respecialization;
    a site holding one opcode whose inline cache guard keeps moving pays only a
    failed guard and a cache refill. The two differ by several nanoseconds per
    execution, enough that one constant for both over-predicts the cheap case by
    half. A megamorphic site stuck in its generic form is charged at the form
    rate, which is an upper bound on what cloning can recover there.
    """
    return CACHE if status == "cache_thrash" else FORM


def bucket_of(site):
    return (site_class(site.family), site_kind(site.status))


_SCALAR_PROBE = """
def probe(a, b):
    t = a * b
    for i in range(24):
        t = t + a
        t = t * b
        if t > b:
            t = t - a
    return t
"""

_ATTRIBUTE_PROBE = """
def probe(r, w):
    t = r.base * w
    for i in range(12):
        t = t + r.alpha
        t = t - r.beta
        t = t + r.gamma
    return t
"""

_CONTAINER_PROBE = """
def probe(bucket, keys):
    t = 0
    for k in keys:
        t = t + bucket[k]
    return t
"""

_TRIVIAL = """
def trivial(a, b):
    return a
"""


def _compile(source, name):
    namespace = {}
    exec(compile(source, "<monoclone calibration>", "exec"), namespace)
    return namespace[name]


@dataclass(frozen=True)
class Machine:
    """Measured cost of dispatch and value of monomorphic specialization."""

    dispatch_s: float
    saving_s: dict = field(default_factory=dict)
    probe_ops: dict = field(default_factory=dict)
    reps: int = 0

    def saving_for(self, bucket):
        """Saving per execution for a (site class, thrash kind) bucket."""
        if bucket in self.saving_s:
            return self.saving_s[bucket]
        return self.saving_s.get((bucket[0], FORM), 0.0)

    def breakeven_ops(self, bucket=(SCALAR, FORM)):
        """Recoverable executions per call at which cloning breaks even."""
        saving = self.saving_for(bucket)
        if saving <= 0:
            return float("inf")
        return self.dispatch_s / saving

    def describe(self):
        parts = ["dispatch %.1f ns/call" % (self.dispatch_s * 1e9)]
        for bucket in sorted(self.saving_s):
            parts.append(
                "%s/%s %.2f ns/op (break-even %.0f)"
                % (
                    bucket[0],
                    bucket[1],
                    self.saving_s[bucket] * 1e9,
                    self.breakeven_ops(bucket),
                )
            )
        return ", ".join(parts)


def _measure_dispatch(reps, calls):
    """Cost of one generated type dispatch, in seconds per call."""
    trivial = _compile(_TRIVIAL, "trivial")
    table = {int: _compile(_TRIVIAL, "trivial"), float: _compile(_TRIVIAL, "trivial")}
    get = table.get

    def dispatcher(a, b):
        fn = get(type(a))
        if fn is None:
            return trivial(a, b)
        return fn(a, b)

    items = []
    for _ in range(calls):
        items.append((3, 2))
        items.append((3.0, 2.0))

    def direct():
        for a, b in items:
            trivial(a, b)

    def through():
        for a, b in items:
            dispatcher(a, b)

    timings = best_of_interleaved({"direct": direct, "through": through}, reps)
    return (timings["through"] - timings["direct"]) / len(items)


def _recoverable_offsets(fn, items, bucket):
    """Offsets of ``fn`` sites in ``bucket`` that fail to stay specialized."""
    monitor = thrash.ThrashMonitor(0.0).watch(fn.__code__)

    def run():
        for args in items:
            fn(*args)

    with thrash.fine_grained_switching():
        with monitor:
            for _ in range(3):
                run()
    return {
        site.offset
        for site in monitor.sites_for(fn.__code__)
        if site.recoverable and bucket_of(site) == bucket
    }


def _measure_saving(source, groups, bucket, reps):
    """Saving per recoverable execution for one class of site.

    ``groups`` is a list of same-typed argument lists. The obvious measurement,
    one interleaved run against one run per type group, is wrong: segregating
    the calls by type also segregates their data, and the resulting cache
    locality gets credited to monomorphism. Here both runs walk the *same*
    interleaved argument list in the same order, over a list of
    ``(callable, args)`` pairs, so the loop shape and the memory access pattern
    are identical and the only difference is whether the callable is one shared
    code object or one code object per type.

    Routing through a precomputed pair list rather than a dispatcher also keeps
    the dispatch cost out of this number, which is measured separately.
    """
    mixed_fn = _compile(source, "probe")
    clones = [_compile(source, "probe") for _ in groups]

    interleaved = []
    routed = []
    for index in range(max(len(g) for g in groups)):
        for group, clone in zip(groups, clones):
            if index < len(group):
                interleaved.append(group[index])
                routed.append((clone, group[index]))
    shared = [(mixed_fn, args) for args in interleaved]

    def mixed():
        for fn, args in shared:
            fn(*args)

    def split():
        for fn, args in routed:
            fn(*args)

    offsets = _recoverable_offsets(mixed_fn, interleaved, bucket)
    counts, counted = typeprof.count_instructions(
        mixed_fn, interleaved, limit=min(300, len(interleaved))
    )
    ops_per_call = sum(counts[o] for o in offsets) / max(counted, 1)

    timings = best_of_interleaved({"mixed": mixed, "split": split}, reps)
    total_ops = len(interleaved) * ops_per_call
    if total_ops <= 0:
        return 0.0, 0.0
    return (timings["mixed"] - timings["split"]) / total_ops, ops_per_call


class _Plain:
    def __init__(self, base, alpha, beta, gamma):
        self.base = base
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma


class _Slotted:
    __slots__ = ("base", "alpha", "beta", "gamma")

    def __init__(self, base, alpha, beta, gamma):
        self.base = base
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma


class _Reordered:
    def __init__(self, gamma, beta, alpha, base):
        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha
        self.base = base


def calibrate(reps=15, calls=1500):
    """Measure dispatch cost and the value of monomorphism, per bucket."""
    dispatch_s = _measure_dispatch(reps, calls)

    saving = {}
    ops = {}

    def probe(bucket, source, groups):
        saving[bucket], ops[bucket] = _measure_saving(source, groups, bucket, reps)

    probe(
        (SCALAR, FORM),
        _SCALAR_PROBE,
        [[(3, 2)] * calls, [(3.0, 2.0)] * calls],
    )

    # Plain against slotted: the opcode itself changes, which is the expensive
    # way an attribute site can fail.
    probe(
        (ATTRIBUTE, FORM),
        _ATTRIBUTE_PROBE,
        [
            [(_Plain(3, 2, 1, 4), 2)] * calls,
            [(_Slotted(5, 1, 2, 3), 2)] * calls,
        ],
    )

    # Two ordinary classes: same opcode throughout, only the cached type
    # version moves, which is the cheap way and by far the more common one.
    probe(
        (ATTRIBUTE, CACHE),
        _ATTRIBUTE_PROBE,
        [
            [(_Plain(3, 2, 1, 4), 2)] * calls,
            [(_Reordered(1, 2, 3, 4), 2)] * calls,
        ],
    )

    keys = list(range(32))
    as_list = [k % 7 for k in keys]
    as_dict = {k: k % 7 for k in keys}
    probe(
        (CONTAINER, FORM),
        _CONTAINER_PROBE,
        [[(as_dict, keys)] * calls, [(as_list, keys)] * calls],
    )

    return Machine(dispatch_s=dispatch_s, saving_s=saving, probe_ops=ops, reps=reps)


@dataclass(frozen=True)
class Prediction:
    """Predicted effect of cloning one function over an observed call stream."""

    calls: int
    ops_by_bucket: dict
    covered_share: float
    gain_s: float
    cost_s: float
    baseline_s: float

    @property
    def total_ops_per_call(self):
        return sum(self.ops_by_bucket.values())

    @property
    def net_s(self):
        return self.gain_s - self.cost_s

    @property
    def worthwhile(self):
        return self.net_s > 0

    @property
    def predicted_speedup(self):
        if self.baseline_s <= 0:
            return 1.0
        after = self.baseline_s - self.net_s
        if after <= 0:
            return float("inf")
        return self.baseline_s / after


def ops_by_bucket(counts, calls, sites):
    """Recoverable executions per call, split by site class and thrash kind."""
    totals = collections.Counter()
    if not calls:
        return dict(totals)
    for site in sites:
        totals[bucket_of(site)] += counts[site.offset]
    return {bucket: total / calls for bucket, total in totals.items()}


def predict(machine, calls, ops, covered_share, baseline_s=0.0):
    """Expected saving and cost of cloning, in seconds over the whole stream."""
    gain = 0.0
    for bucket, per_call in ops.items():
        gain += machine.saving_for(bucket) * per_call * calls * covered_share
    cost = machine.dispatch_s * calls
    return Prediction(
        calls=calls,
        ops_by_bucket=dict(ops),
        covered_share=covered_share,
        gain_s=gain,
        cost_s=cost,
        baseline_s=baseline_s,
    )
