"""Grouping observed type signatures into clone groups.

Every clone is a separate code object with its own specialization state, so the
goal is to partition the observed calls such that each group is homogeneous at
the thrashing sites. Grouping on the full signature always achieves that but
produces more clones than necessary: a signature position that no thrashing
site reads can be ignored, and two signatures that differ only there can share
one clone.

So the key is the projection of a signature onto the positions the dataflow
analysis says reach the thrashing lines. Groups are then taken in order of call
count, which is optimal for a fixed clone budget because the value of a group
is its call count and the groups are disjoint.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Group:
    """One clone: a projected type key and the calls it serves."""

    key: tuple
    calls: int

    def label(self):
        return "(" + ", ".join(getattr(t, "__name__", str(t)) for t in self.key) + ")"


@dataclass(frozen=True)
class Grouping:
    positions: tuple
    groups: tuple
    total_calls: int
    uncovered_calls: int

    @property
    def clones(self):
        return len(self.groups)

    @property
    def covered_share(self):
        if not self.total_calls:
            return 0.0
        return (self.total_calls - self.uncovered_calls) / self.total_calls

    def table_keys(self):
        return [g.key for g in self.groups]


def project(signature, positions):
    return tuple(signature[i] for i in positions if i < len(signature))


def build(profile, positions, max_clones=4, min_share=0.02):
    """Partition ``profile`` signatures into at most ``max_clones`` groups."""
    positions = tuple(sorted(positions))
    buckets = {}
    skipped = 0
    for signature, count in profile.signatures.items():
        if any(i >= len(signature) for i in positions):
            # Call arity differs from the signature the clones are built for.
            skipped += count
            continue
        key = project(signature, positions)
        buckets[key] = buckets.get(key, 0) + count
    ranked = sorted(buckets.items(), key=lambda kv: (-kv[1], str(kv[0])))
    floor = min_share * profile.calls
    chosen = []
    for key, count in ranked:
        if len(chosen) >= max_clones:
            break
        if count < floor:
            break
        chosen.append(Group(key=key, calls=count))
    covered = sum(g.calls for g in chosen)
    return Grouping(
        positions=positions,
        groups=tuple(chosen),
        total_calls=profile.calls,
        uncovered_calls=profile.calls - covered + skipped,
    )


def narrow(profile, positions):
    """Drop positions that are monomorphic across every observed call.

    A position whose type never varies cannot be the cause of any thrash and
    keying on it would only create redundant clones.
    """
    varying = profile.polymorphic_positions
    narrowed = frozenset(positions) & varying
    return narrowed if narrowed else frozenset()


def minimal_positions(profile, positions):
    """Drop positions that do not refine the partition the others induce.

    Arguments are often perfectly correlated: a function taking the two
    endpoints of an interval never sees an int paired with a float. Keying the
    dispatch on all of them builds a tuple on every call for no extra
    discrimination, so redundant positions are removed greedily, widest index
    first, as long as the number of distinct groups is unchanged.
    """
    positions = tuple(sorted(positions))
    if len(positions) <= 1:
        return frozenset(positions)

    def group_count(subset):
        keys = set()
        for signature in profile.signatures:
            if any(i >= len(signature) for i in subset):
                return -1
            keys.add(project(signature, subset))
        return len(keys)

    target = group_count(positions)
    kept = list(positions)
    for index in reversed(positions):
        if len(kept) == 1:
            break
        candidate = tuple(i for i in kept if i != index)
        if group_count(candidate) == target:
            kept = list(candidate)
    return frozenset(kept)
