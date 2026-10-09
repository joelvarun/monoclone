"""Detecting call-site specialization thrash by polling the adaptive bytecode.

A release build of CPython exposes no deoptimization counters, so thrash has to
be observed rather than queried. This module samples the live instruction
stream of a running code object and watches two things per site:

form flips
    The opcode itself changes, for example ``BINARY_OP_ADD_INT`` becoming
    ``BINARY_OP_ADD_FLOAT``. The site is being respecialized to a different
    operand type.

cache flips
    The opcode stays put but its inline cache guard changes, for example a
    ``LOAD_ATTR_INSTANCE_VALUE`` whose cached type version moves between two
    classes. Attribute and call sites thrash this way and the opcode name alone
    never shows it.

Sampling gives a lower bound on the number of flips, not an exact count: two
flips between consecutive polls look like one. That is enough to classify a
site, which is all the cost model needs.
"""

import contextlib
import sys
import threading
import time
from dataclasses import dataclass, field

from monoclone import sites

#: A site needs at least this many observed flips to count as thrashing. One
#: flip is the ordinary generic-to-specialized warmup transition.
FLIP_THRESHOLD = 2

#: Thread switch interval used while sampling. The default 5 ms is far too
#: coarse: a workload that runs for 20 ms would be sampled four times and most
#: flips would fall between polls.
SAMPLING_SWITCH_INTERVAL = 5e-5


@dataclass
class Site:
    """Observed specialization history of one instruction."""

    offset: int
    family: str
    line: int
    forms: list = field(default_factory=list)
    form_flips: int = 0
    cache_flips: int = 0
    samples: int = 0

    @property
    def specialized_forms(self):
        return [f for f in self.forms if sites.is_specialized(f)]

    @property
    def status(self):
        if not self.specialized_forms:
            return "generic"
        if len(self.specialized_forms) > 1 and self.form_flips >= FLIP_THRESHOLD:
            return "form_thrash"
        if self.cache_flips >= FLIP_THRESHOLD:
            return "cache_thrash"
        return "stable"

    @property
    def thrashing(self):
        return self.status in ("form_thrash", "cache_thrash")

    @property
    def recoverable(self):
        """Sites whose operand types cloning could make uniform.

        Thrashing sites are the obvious case. A site stuck in its generic form
        is the other one: past a handful of distinct types the interpreter stops
        trying, marks the site megamorphic and never specializes it again, so it
        never flips and a flip detector alone would call it healthy. Generic is
        a weaker signal than thrash, because a site can also be generic for
        reasons no amount of cloning fixes, such as operand types that have no
        specialized form at all.
        """
        return self.thrashing or self.status == "generic"

    def describe(self):
        return "line %d offset %-4d %-12s %-13s %s" % (
            self.line,
            self.offset,
            self.family,
            self.status,
            ",".join(self.specialized_forms) or "-",
        )


class ThrashMonitor:
    """Polls one or more code objects while a workload runs."""

    def __init__(self, interval=0.0005):
        self.interval = interval
        self._codes = []
        self._lines = {}
        self._state = {}
        self._last = {}
        self._thread = None
        self._stop = threading.Event()
        self.polls = 0

    def watch(self, code):
        self._codes.append(code)
        self._lines[id(code)] = sites.line_map(code)
        self._state[id(code)] = {}
        self._last[id(code)] = {}
        return self

    def sample(self):
        """Take one snapshot of every watched code object."""
        for code in self._codes:
            key = id(code)
            state = self._state[key]
            last = self._last[key]
            lines = self._lines[key]
            for offset, name, cache in sites.decode(code):
                family = sites.FAMILY.get(name)
                if family is None or family in sites.IGNORED_FAMILIES:
                    continue
                site = state.get(offset)
                if site is None:
                    site = Site(offset=offset, family=family, line=lines.get(offset, -1))
                    state[offset] = site
                site.samples += 1
                if name not in site.forms:
                    site.forms.append(name)
                prev = last.get(offset)
                if prev is not None:
                    prev_name, prev_cache = prev
                    if prev_name != name:
                        site.form_flips += 1
                    elif cache and prev_cache != cache:
                        site.cache_flips += 1
                last[offset] = (name, cache)
        self.polls += 1

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            self.sample()
            time.sleep(self.interval)


    def stop(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)
            self._thread = None
        self.sample()
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    def sites_for(self, code):
        state = self._state.get(id(code), {})
        return [state[off] for off in sorted(state)]

    def thrashing_sites(self, code):
        return [s for s in self.sites_for(code) if s.thrashing]


@contextlib.contextmanager
def fine_grained_switching(interval=SAMPLING_SWITCH_INTERVAL):
    """Hand the sampling thread the GIL often enough to see flips."""
    previous = sys.getswitchinterval()
    sys.setswitchinterval(interval)
    try:
        yield
    finally:
        sys.setswitchinterval(previous)
