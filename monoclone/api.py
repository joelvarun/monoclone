"""End to end analysis: profile, observe thrash, count, decide, clone, verify.

Three measurement passes run over the same call stream, in this order, because
each one destroys what the next would need:

1. type profile, wrapped, exact call counts and argument type signatures;
2. thrash poll, uninstrumented, the only pass in which specialization state
   exists to be read;
3. instruction counts, instrumented, exact per offset but with no
   specialization state at all.

The cost model then decides whether cloning is worth the dispatch, and a
differential check confirms the clones agree with the original before the
dispatcher is handed back.
"""

from dataclasses import dataclass, field

from monoclone import cloner, costmodel, dataflow, grouping, thrash, typeprof
from monoclone.timing import best_of


@dataclass
class Decision:
    """Whether to clone, and the numbers behind the answer."""

    cloned: bool
    reason: str
    clones: int = 0
    positions: tuple = ()
    covered_share: float = 0.0
    ops_by_bucket: dict = field(default_factory=dict)
    breakeven_ops: float = 0.0
    verified_calls: int = 0


@dataclass
class Report:
    """Everything observed about one function."""

    name: str
    profile: typeprof.TypeProfile
    sites: list
    counts: object
    counted_calls: int
    decision: Decision
    prediction: costmodel.Prediction
    machine: costmodel.Machine
    clone_set: object = None
    baseline_s: float = 0.0
    group_labels: list = field(default_factory=list)
    group_keys: tuple = ()

    @property
    def thrashing(self):
        return [s for s in self.sites if s.thrashing]

    @property
    def recoverable(self):
        return [s for s in self.sites if s.recoverable]

    @property
    def callable(self):
        if self.clone_set is None:
            return None
        return self.clone_set.dispatcher

    def summary(self):
        lines = []
        lines.append("function            %s" % self.name)
        lines.append("calls profiled      %d" % self.profile.calls)
        signatures = ", ".join(
            "/".join(getattr(t, "__name__", str(t)) for t in sig)
            for sig, _ in self.profile.signatures.most_common(4)
        )
        lines.append("type signatures     %d (%s)" % (self.profile.distinct, signatures))
        statuses = ", ".join(sorted({s.status for s in self.recoverable})) or "none"
        lines.append(
            "specializable sites %d, recoverable %d (%s)"
            % (len(self.sites), len(self.recoverable), statuses)
        )
        for site in self.recoverable:
            lines.append("    " + site.describe())
        lines.append(
            "recoverable ops/call %s"
            % (
                ", ".join(
                    "%s/%s %.1f (break-even %.0f)"
                    % (b[0], b[1], v, self.machine.breakeven_ops(b))
                    for b, v in sorted(self.decision.ops_by_bucket.items())
                )
                or "none"
            )
        )
        if self.decision.cloned:
            lines.append(
                "decision            clone into %d groups on positions %s"
                % (self.decision.clones, list(self.decision.positions))
            )
            lines.append("groups              %s" % ", ".join(self.group_labels))
            lines.append(
                "coverage            %.1f percent of calls, %d differential checks passed"
                % (self.decision.covered_share * 100.0, self.decision.verified_calls)
            )
        else:
            lines.append("decision            leave alone (%s)" % self.decision.reason)
        lines.append(
            "predicted           gain %.1f ms, dispatch cost %.1f ms, net %.1f ms"
            % (
                self.prediction.gain_s * 1e3,
                self.prediction.cost_s * 1e3,
                self.prediction.net_s * 1e3,
            )
        )
        return "\n".join(lines)


def _consume(fn, stream):
    for item in stream:
        args, kwargs = typeprof._split(item)
        fn(*args, **kwargs)


def analyze(
    fn,
    stream_factory,
    machine=None,
    max_clones=4,
    min_share=0.02,
    count_limit=400,
    poll_interval=0.0,
    thrash_reps=3,
    verify_limit=300,
    measure_baseline=True,
    baseline_reps=5,
    force=False,
    min_gain_share=0.03,
):
    """Analyse ``fn`` over the call stream and clone it when that pays."""
    if machine is None:
        machine = costmodel.calibrate()

    profile = typeprof.profile_calls(fn, stream_factory())

    monitor = thrash.ThrashMonitor(poll_interval).watch(fn.__code__)
    with thrash.fine_grained_switching():
        with monitor:
            for _ in range(thrash_reps):
                _consume(fn, stream_factory())
    sites = monitor.sites_for(fn.__code__)
    thrashing = [s for s in sites if s.recoverable]

    counts, counted_calls = typeprof.count_instructions(
        fn, stream_factory(), limit=count_limit
    )
    ops = costmodel.ops_by_bucket(counts, counted_calls, thrashing)
    total_ops = sum(ops.values())

    baseline_s = 0.0
    if measure_baseline:
        items = list(stream_factory())
        baseline_s = best_of(lambda: _consume(fn, items), reps=baseline_reps)

    def reject(reason, covered=0.0):
        return Report(
            name=getattr(fn, "__qualname__", repr(fn)),
            profile=profile,
            sites=sites,
            counts=counts,
            counted_calls=counted_calls,
            decision=Decision(
                cloned=False,
                reason=reason,
                ops_by_bucket=ops,
                breakeven_ops=machine.breakeven_ops(),
                covered_share=covered,
            ),
            prediction=costmodel.predict(
                machine, profile.calls, ops, covered, baseline_s
            ),
            machine=machine,
            baseline_s=baseline_s,
        )

    ok, why = cloner.can_clone(fn)
    if not ok:
        return reject(why)
    if not thrashing:
        return reject("no recoverable sites")
    if profile.distinct < 2:
        return reject("call stream is monomorphic")

    lines = {s.line for s in thrashing}
    positions, _params = dataflow.relevant_positions(fn, lines)
    narrowed = grouping.narrow(profile, positions)
    if not narrowed:
        return reject("no polymorphic parameter feeds a thrashing site")
    narrowed = grouping.minimal_positions(profile, narrowed)

    group = grouping.build(profile, narrowed, max_clones=max_clones, min_share=min_share)
    if group.clones < 2:
        return reject(
            "no two type groups clear the %.0f percent floor" % (min_share * 100)
        )

    prediction = costmodel.predict(
        machine, profile.calls, ops, group.covered_share, baseline_s
    )
    floor = min_gain_share * baseline_s
    if not force and prediction.net_s <= floor:
        return reject(
            "predicted net %.2f ms from %.1f recoverable ops per call over "
            "%.0f percent of calls does not clear the %.2f ms floor"
            % (
                prediction.net_s * 1e3,
                total_ops,
                group.covered_share * 100,
                floor * 1e3,
            ),
            group.covered_share,
        )

    clone_set = cloner.build(fn, group.table_keys(), group.positions)
    checked, mismatches = cloner.verify(
        fn, clone_set.dispatcher, stream_factory(), limit=verify_limit
    )
    if mismatches:
        return reject("differential check failed on %d calls" % len(mismatches))

    return Report(
        name=getattr(fn, "__qualname__", repr(fn)),
        profile=profile,
        sites=sites,
        counts=counts,
        counted_calls=counted_calls,
        decision=Decision(
            cloned=True,
            reason="predicted net %.2f ms clears the %.2f ms floor"
            % (prediction.net_s * 1e3, floor * 1e3)
            if prediction.net_s > floor
            else "forced",
            clones=group.clones,
            positions=group.positions,
            covered_share=group.covered_share,
            ops_by_bucket=ops,
            breakeven_ops=machine.breakeven_ops(),
            verified_calls=checked,
        ),
        prediction=prediction,
        machine=machine,
        clone_set=clone_set,
        baseline_s=baseline_s,
        group_labels=[g.label() + " x%d" % g.calls for g in group.groups],
        group_keys=tuple(group.table_keys()),
    )


def optimize(fn, stream_factory, **kwargs):
    """Return ``(callable, report)``, the callable being ``fn`` when cloning loses."""
    report = analyze(fn, stream_factory, **kwargs)
    return (report.callable or fn), report
