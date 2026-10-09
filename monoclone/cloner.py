"""Building monomorphic clones and the dispatcher in front of them.

Specialization state lives in the code object, so two functions compiled from
the same source do not share it. That is the whole mechanism: recompile the
body once per type group, send each call to the clone that matches its argument
types, and every clone then sees one type per site.

The dispatcher is generated as source rather than written as a closure over
``*args``, because the overhead being traded against is a handful of
nanoseconds and an exact parameter list avoids building an argument tuple on
every call. Any type combination that was never profiled falls through to the
original function, so behaviour is unchanged for inputs the profile never saw.
"""

import ast
import inspect
from dataclasses import dataclass

from monoclone import dataflow


@dataclass(frozen=True)
class CloneSet:
    """A dispatcher, its clones, and the source that was generated for it."""

    dispatcher: object
    clones: tuple
    original: object
    source: str
    positions: tuple

    @property
    def count(self):
        return len(self.clones)


def can_clone(fn):
    """Whether ``fn`` can be cloned, and why not when it cannot."""
    if not inspect.isfunction(fn):
        return False, "not a plain Python function"
    if fn.__closure__:
        return False, "closes over free variables"
    try:
        node = dataflow.function_ast(fn)
    except (OSError, TypeError, SyntaxError, IndentationError) as exc:
        return False, "source unavailable (%s)" % type(exc).__name__
    if isinstance(node, ast.AsyncFunctionDef):
        return False, "async def"
    if not dataflow.parameter_names(node):
        return False, "no positional parameters to dispatch on"
    return True, ""


def _rebuild(fn, index):
    """Compile a fresh code object from the source of ``fn``."""
    node = dataflow.function_ast(fn)
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    filename = "<monoclone clone %d of %s>" % (index, fn.__name__)
    namespace = {}
    exec(compile(module, filename, "exec"), fn.__globals__, namespace)
    clone = namespace[node.name]
    clone.__defaults__ = fn.__defaults__
    clone.__kwdefaults__ = fn.__kwdefaults__
    clone.__module__ = fn.__module__
    clone.__qualname__ = "%s[clone %d]" % (fn.__qualname__, index)
    clone.__doc__ = fn.__doc__
    return clone


def _signature_text(fn):
    """Parameter list and forwarding call text for ``fn``."""
    signature = inspect.signature(fn)
    params = list(signature.parameters.values())
    declare = []
    forward = []
    positional = []
    saw_star = False
    for param in params:
        kind = param.kind
        if kind is param.POSITIONAL_ONLY:
            declare.append(param.name)
            forward.append(param.name)
            positional.append(param.name)
        elif kind is param.POSITIONAL_OR_KEYWORD:
            if positional and params[len(positional) - 1].kind is param.POSITIONAL_ONLY:
                declare.append("/")
            declare.append(param.name)
            forward.append(param.name)
            positional.append(param.name)
        elif kind is param.VAR_POSITIONAL:
            declare.append("*" + param.name)
            forward.append("*" + param.name)
            saw_star = True
        elif kind is param.KEYWORD_ONLY:
            if not saw_star:
                declare.append("*")
                saw_star = True
            declare.append(param.name)
            forward.append("%s=%s" % (param.name, param.name))
        else:
            declare.append("**" + param.name)
            forward.append("**" + param.name)
    if positional and not any(
        p.kind is p.POSITIONAL_OR_KEYWORD for p in params
    ) and params and params[0].kind is params[0].POSITIONAL_ONLY:
        index = len(positional)
        declare.insert(index, "/")
    return ", ".join(declare), ", ".join(forward), positional


def build(fn, group_keys, positions, max_clones=None):
    """Clone ``fn`` once per key in ``group_keys`` and return a ``CloneSet``."""
    ok, reason = can_clone(fn)
    if not ok:
        raise ValueError("cannot clone %r: %s" % (fn, reason))
    positions = tuple(sorted(positions))
    declare, forward, positional = _signature_text(fn)
    for index in positions:
        if index >= len(positional):
            raise ValueError("dispatch position %d is not a fixed parameter" % index)
    keys = list(group_keys)
    if max_clones is not None:
        keys = keys[:max_clones]
    clones = tuple(_rebuild(fn, i) for i in range(len(keys)))
    table = {key: clone for key, clone in zip(keys, clones)}

    names = [positional[i] for i in positions]
    if len(names) == 1:
        key_expr = "type(%s)" % names[0]
        lookup = {k[0] if isinstance(k, tuple) else k: v for k, v in table.items()}
    else:
        key_expr = "(" + ", ".join("type(%s)" % n for n in names) + ")"
        lookup = table

    source = (
        "def %(name)s(%(declare)s):\n"
        "    _fn = _get(%(key)s)\n"
        "    if _fn is None:\n"
        "        return _original(%(forward)s)\n"
        "    return _fn(%(forward)s)\n"
    ) % {
        "name": fn.__name__,
        "declare": declare,
        "key": key_expr,
        "forward": forward,
    }
    namespace = {"_get": lookup.get, "_original": fn}
    exec(compile(source, "<monoclone dispatcher for %s>" % fn.__name__, "exec"), namespace)
    dispatcher = namespace[fn.__name__]
    dispatcher.__defaults__ = fn.__defaults__
    dispatcher.__kwdefaults__ = fn.__kwdefaults__
    dispatcher.__module__ = fn.__module__
    dispatcher.__qualname__ = fn.__qualname__
    dispatcher.__doc__ = fn.__doc__
    dispatcher.__wrapped__ = fn
    dispatcher.monoclone_clones = clones
    return CloneSet(
        dispatcher=dispatcher,
        clones=clones,
        original=fn,
        source=source,
        positions=positions,
    )


def verify(original, candidate, stream, limit=400):
    """Differential check that ``candidate`` agrees with ``original``.

    Compares return values, and treats two calls that raise the same exception
    type as agreeing. Returns ``(checked, mismatches)``.
    """
    from monoclone.typeprof import _split

    checked = 0
    mismatches = []
    for item in stream:
        if checked >= limit:
            break
        args, kwargs = _split(item)
        try:
            expected = original(*args, **kwargs)
            expected_exc = None
        except Exception as exc:  # noqa: BLE001 - mirrored below on purpose
            expected, expected_exc = None, type(exc)
        try:
            actual = candidate(*args, **kwargs)
            actual_exc = None
        except Exception as exc:  # noqa: BLE001
            actual, actual_exc = None, type(exc)
        checked += 1
        if expected_exc is not None or actual_exc is not None:
            if expected_exc is not actual_exc:
                mismatches.append((args, expected_exc, actual_exc))
            continue
        if repr(expected) != repr(actual):
            mismatches.append((args, expected, actual))
    return checked, mismatches


def build_single(fn, group_keys, positions):
    """A cost matched ablation: one clone serving every observed type.

    The dispatcher is generated the same way and does the same dictionary
    lookup and call, so the per-call overhead is identical to a real clone set.
    The one thing removed is monomorphism, since the single clone sees every
    type. Any speedup that survives this ablation did not come from
    specialization.
    """
    ok, reason = can_clone(fn)
    if not ok:
        raise ValueError("cannot clone %r: %s" % (fn, reason))
    positions = tuple(sorted(positions))
    declare, forward, positional = _signature_text(fn)
    shared = _rebuild(fn, 0)
    keys = list(group_keys)
    names = [positional[i] for i in positions]
    if len(names) == 1:
        key_expr = "type(%s)" % names[0]
        lookup = {(k[0] if isinstance(k, tuple) else k): shared for k in keys}
    else:
        key_expr = "(" + ", ".join("type(%s)" % n for n in names) + ")"
        lookup = {k: shared for k in keys}
    source = (
        "def %(name)s(%(declare)s):\n"
        "    _fn = _get(%(key)s)\n"
        "    if _fn is None:\n"
        "        return _original(%(forward)s)\n"
        "    return _fn(%(forward)s)\n"
    ) % {
        "name": fn.__name__,
        "declare": declare,
        "key": key_expr,
        "forward": forward,
    }
    namespace = {"_get": lookup.get, "_original": fn}
    exec(compile(source, "<monoclone ablation for %s>" % fn.__name__, "exec"), namespace)
    dispatcher = namespace[fn.__name__]
    dispatcher.__defaults__ = fn.__defaults__
    dispatcher.__kwdefaults__ = fn.__kwdefaults__
    return CloneSet(
        dispatcher=dispatcher,
        clones=(shared,),
        original=fn,
        source=source,
        positions=positions,
    )
