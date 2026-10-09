"""Polymorphic workloads used to measure the tool.

Each workload is a plain function plus a factory that produces a deterministic
call stream mixing argument types. They cover the three ways a site thrashes
(arithmetic operand types, attribute layouts, container kinds) and two cases
where cloning should be refused: a body too small to amortise the dispatch,
and a call stream so megamorphic that no clone budget covers it.
"""

import random
from dataclasses import dataclass


def interval_poly(lo, hi, coeff):
    """Interval arithmetic through a degree-8 polynomial.

    Generic over int and float endpoints, which is exactly the shape that makes
    every multiply and compare in the body thrash between the two.
    """
    a = lo
    b = hi
    for _ in range(8):
        p1 = a * lo
        p2 = a * hi
        p3 = b * lo
        p4 = b * hi
        low = p1
        if p2 < low:
            low = p2
        if p3 < low:
            low = p3
        if p4 < low:
            low = p4
        high = p1
        if p2 > high:
            high = p2
        if p3 > high:
            high = p3
        if p4 > high:
            high = p4
        a = low + coeff
        b = high + coeff
    return a, b


def horner(coeffs, x):
    """Polynomial evaluation at an int, float or complex point."""
    acc = coeffs[0]
    for c in coeffs:
        acc = acc * x + c
    return acc


def moments(values, scale):
    """First and second moment of a scaled sample."""
    total = values[0] * scale
    sq = total * total
    for v in values:
        w = v * scale
        total = total + w
        sq = sq + w * w
    return total, sq


def record_score(record, weight):
    """Weighted score over an attribute bundle.

    Every class reaching this function has the same attribute names, so the
    opcode stays LOAD_ATTR_INSTANCE_VALUE or LOAD_ATTR_SLOT and only the inline
    cache guard moves. Opcode names alone cannot see this.
    """
    score = record.base * weight
    for _ in range(10):
        score = score + record.alpha
        score = score - record.beta
        score = score + record.gamma * weight
    return score


def shape_area(shape, factor):
    """Weighted area over two classes with identical attribute names.

    Both classes are ordinary, both lay their attributes out the same way, so
    every load here stays LOAD_ATTR_INSTANCE_VALUE whichever class arrives. The
    opcode never changes and only the cached type version moves, which is the
    case a detector watching opcode names alone cannot see at all.
    """
    total = shape.w * shape.h
    for _ in range(8):
        total = total + shape.w * factor
        total = total + shape.h * factor
        total = total - shape.pad
    return total


def bucket_total(bucket, keys):
    """Sum selected entries of a dict or of a list."""
    total = 0
    for k in keys:
        total = total + bucket[k]
    return total


def blend(a, b):
    """A body far too small to pay for a dispatch."""
    return a + b - a * b


def tag_weight(tag, base):
    """Attribute arithmetic over a megamorphic population of tag classes."""
    total = base
    for _ in range(10):
        total = total + tag.weight
        total = total * tag.factor - tag.offset
    return total


class Box:
    def __init__(self, w, h, pad):
        self.w = w
        self.h = h
        self.pad = pad


class Panel:
    def __init__(self, w, h, pad):
        self.w = w
        self.h = h
        self.pad = pad


class PlainRecord:
    def __init__(self, base, alpha, beta, gamma):
        self.base = base
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma


class SlottedRecord:
    __slots__ = ("base", "alpha", "beta", "gamma")

    def __init__(self, base, alpha, beta, gamma):
        self.base = base
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma


class ReorderedRecord:
    def __init__(self, gamma, beta, alpha, base):
        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha
        self.base = base


def _tag_classes(count):
    classes = []
    for i in range(count):
        namespace = {
            "__init__": _make_tag_init(i),
            "weight": 0,
            "factor": 1,
            "offset": 0,
        }
        classes.append(type("Tag%02d" % i, (object,), namespace))
    return classes


def _make_tag_init(index):
    def __init__(self):
        self.weight = 1 + index % 3
        self.factor = 1
        self.offset = index % 2

    return __init__


TAG_CLASSES = _tag_classes(60)


@dataclass(frozen=True)
class Workload:
    """A function, a call stream and what it is meant to demonstrate."""

    name: str
    fn: object
    stream_factory: object
    note: str
    expect_clone: bool

    def stream(self):
        return self.stream_factory()


def _interval_stream(calls=5600, seed=11):
    rng = random.Random(seed)
    items = []
    for _ in range(calls):
        if rng.random() < 0.5:
            items.append((-1, 1, rng.choice((-1, 0, 1))))
        else:
            items.append((-1.0, 1.0, rng.choice((-0.5, 0.0, 0.5))))
    return items


def _horner_stream(calls=19200, seed=12):
    rng = random.Random(seed)
    int_coeffs = [rng.choice((-1, 0, 1)) for _ in range(16)]
    float_coeffs = [rng.uniform(-1.0, 1.0) for _ in range(16)]
    complex_coeffs = [complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(16)]
    items = []
    for _ in range(calls):
        draw = rng.random()
        if draw < 0.45:
            items.append((int_coeffs, rng.choice((-1, 0, 1))))
        elif draw < 0.9:
            items.append((float_coeffs, rng.uniform(-0.9, 0.9)))
        else:
            items.append((complex_coeffs, complex(0.3, 0.4)))
    return items


def _moments_stream(calls=6000, seed=13):
    rng = random.Random(seed)
    ints = [rng.randint(-4, 4) for _ in range(32)]
    floats = [rng.uniform(-1.0, 1.0) for _ in range(32)]
    items = []
    for _ in range(calls):
        if rng.random() < 0.5:
            items.append((ints, rng.choice((1, 2, 3))))
        else:
            items.append((floats, rng.uniform(0.5, 1.5)))
    return items


def _record_stream(calls=11600, seed=14):
    rng = random.Random(seed)
    pool = [
        PlainRecord(3, 2, 1, 4),
        SlottedRecord(5, 1, 2, 3),
        ReorderedRecord(1, 2, 3, 4),
    ]
    items = []
    for _ in range(calls):
        items.append((pool[rng.randrange(3)], rng.choice((1, 2))))
    return items


def _shape_stream(calls=12000, seed=18):
    rng = random.Random(seed)
    pool = [Box(3, 4, 1), Panel(5, 2, 2)]
    items = []
    for _ in range(calls):
        items.append((pool[rng.randrange(2)], rng.choice((1, 2))))
    return items


def _bucket_stream(calls=12800, seed=15):
    rng = random.Random(seed)
    keys = list(range(32))
    as_list = [rng.randint(0, 9) for _ in range(32)]
    as_dict = {k: rng.randint(0, 9) for k in keys}
    items = []
    for _ in range(calls):
        items.append(((as_dict if rng.random() < 0.5 else as_list), keys))
    return items


def _blend_stream(calls=120000, seed=16):
    rng = random.Random(seed)
    items = []
    for _ in range(calls):
        if rng.random() < 0.5:
            items.append((rng.randint(1, 5), rng.randint(1, 5)))
        else:
            items.append((rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)))
    return items


def _tag_stream(calls=9600, seed=17):
    rng = random.Random(seed)
    pool = [cls() for cls in TAG_CLASSES]
    items = []
    for _ in range(calls):
        items.append((pool[rng.randrange(len(pool))], rng.choice((1, 2))))
    return items


def _frozen(builder):
    """Materialise a stream once so every timing pass sees identical data."""
    items = builder()

    def factory():
        return items

    return factory


def all_workloads():
    """Every workload, in reporting order."""
    return [
        Workload(
            "interval_poly",
            interval_poly,
            _frozen(_interval_stream),
            "interval arithmetic over int and float endpoints",
            True,
        ),
        Workload(
            "horner",
            horner,
            _frozen(_horner_stream),
            "polynomial evaluation at int, float and complex points",
            True,
        ),
        Workload(
            "moments",
            moments,
            _frozen(_moments_stream),
            "scaled first and second moment over int and float samples",
            True,
        ),
        Workload(
            "record_score",
            record_score,
            _frozen(_record_stream),
            "attribute bundle with three different class layouts",
            True,
        ),
        Workload(
            "shape_area",
            shape_area,
            _frozen(_shape_stream),
            "two classes with identical attribute names, inline cache thrash only",
            True,
        ),
        Workload(
            "bucket_total",
            bucket_total,
            _frozen(_bucket_stream),
            "subscript over a dict and over a list",
            True,
        ),
        Workload(
            "blend",
            blend,
            _frozen(_blend_stream),
            "three-operation body, below the dispatch break-even",
            False,
        ),
        Workload(
            "tag_weight",
            tag_weight,
            _frozen(_tag_stream),
            "60 tag classes, no clone budget covers the population",
            False,
        ),
    ]
