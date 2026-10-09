"""Reading the live tier-1 adaptive bytecode of a CPython code object.

CPython 3.11 and later rewrite hot bytecode in place: a generic ``BINARY_OP``
becomes ``BINARY_OP_ADD_INT`` once the interpreter has seen enough int
operands, and each specialized instruction carries an inline cache holding the
guard it checks (a type version, a function pointer, a dict keys version).

None of that state is reported by any profiler, but it is readable from pure
Python: ``code._co_code_adaptive`` returns the live quickened instruction
stream, caches included. This module decodes that stream.
"""

import opcode

_INLINE_CACHE = opcode._inline_cache_entries
_SPECIALIZATIONS = opcode._specializations
_SPECIALIZED_OPMAP = opcode._specialized_opmap

# Opcode number to name, including the specialized forms, which do not appear
# in the public ``opcode.opname`` table.
OPNAME = dict(enumerate(opcode.opname))
for _name, _code in _SPECIALIZED_OPMAP.items():
    OPNAME[_code] = _name

# Specialized form to the family it deoptimizes back to.
FAMILY = {}
for _fam, _forms in _SPECIALIZATIONS.items():
    FAMILY[_fam] = _fam
    for _form in _forms:
        FAMILY[_form] = _fam

#: Instruction families the adaptive interpreter can specialize.
SPECIALIZABLE = frozenset(_SPECIALIZATIONS)

# RESUME specializes to a single form and can never thrash, so it carries no
# information about operand types.
IGNORED_FAMILIES = frozenset({"RESUME"})


def cache_words(opname):
    """Number of inline cache words that follow ``opname``."""
    family = FAMILY.get(opname, opname)
    if isinstance(_INLINE_CACHE, dict):
        return _INLINE_CACHE.get(family, 0)
    code = opcode.opmap.get(family)
    return _INLINE_CACHE[code] if code is not None else 0


def decode(code):
    """Decode the live adaptive stream of ``code``.

    Returns a list of ``(offset, opname, cache_bytes)``. ``cache_bytes``
    excludes the first cache word, which is the adaptive counter: it ticks on
    every execution and would swamp any real change in the guard.
    """
    raw = code._co_code_adaptive
    out = []
    i = 0
    n = len(raw)
    while i < n:
        name = OPNAME.get(raw[i])
        if name is None:
            name = "<unknown %d>" % raw[i]
            words = 0
        else:
            words = cache_words(name)
        if words:
            cache = bytes(raw[i + 4 : i + 2 + 2 * words])
        else:
            cache = b""
        out.append((i, name, cache))
        i += 2 + 2 * words
    return out


def line_map(code):
    """Map instruction offset to source line for every offset in ``code``."""
    table = {}
    for start, end, line in code.co_lines():
        if line is None:
            continue
        for off in range(start, end, 2):
            table[off] = line
    return table


def is_specialized(opname):
    """True if ``opname`` is a specialized form rather than a family head."""
    return opname in _SPECIALIZED_OPMAP


def tracked_sites(code):
    """Offsets of instructions that belong to a specializable family."""
    sites = {}
    for offset, name, _cache in decode(code):
        family = FAMILY.get(name)
        if family is None or family in IGNORED_FAMILIES:
            continue
        sites[offset] = family
    return sites
