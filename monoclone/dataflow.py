"""Which parameters feed which source lines.

A thrashing site is a bytecode offset, and the offset maps to a source line.
To decide how to group calls into clones, the question is which *parameters*
control the operand types at that line. A local may be many assignments away
from the parameter that gave it its type, so the answer needs a flow analysis.

The analysis is deliberately coarse: flow insensitive, name based, and it
ignores control flow entirely. It over-approximates, which is the safe
direction here, since including an irrelevant parameter in the grouping key
only splits calls into more clones than necessary. It never drops a parameter
that matters.
"""

import ast
import inspect
import textwrap

#: Nodes whose result type is fixed by the operator, not by the operands, so
#: they do not propagate an operand's type to their target.
_TYPE_OPAQUE = (ast.Compare, ast.BoolOp, ast.In, ast.NotIn, ast.Is, ast.IsNot)


def function_ast(fn):
    """Parse ``fn`` and return its ``FunctionDef`` with decorators stripped."""
    source = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(source)
    node = tree.body[0]
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise TypeError("%s is not a plain function definition" % fn)
    node.decorator_list = []
    return node


def parameter_names(node):
    """Positional parameter names of ``node``, in order."""
    args = node.args
    return [a.arg for a in list(args.posonlyargs) + list(args.args)]


def _loaded_names(node):
    out = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
            out.add(sub.id)
    return out


def _targets(node):
    out = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)):
            out.add(sub.id)
    return out


def type_origins(node, params):
    """Map every local name to the set of parameters that can shape its type.

    Iterated to a fixed point so that values carried around a loop, which is
    the usual shape of an accumulator, reach the parameters that seed them.
    """
    origins = {p: {p} for p in params}
    for _ in range(len(params) + 4):
        changed = False
        for sub in ast.walk(node):
            if isinstance(sub, ast.Assign):
                sources = _loaded_names(sub.value)
                targets = _targets(ast.Tuple(elts=list(sub.targets), ctx=ast.Store()))
            elif isinstance(sub, ast.AugAssign):
                sources = _loaded_names(sub.value) | _targets(sub.target)
                targets = _targets(sub.target)
            elif isinstance(sub, (ast.For, ast.AsyncFor)):
                sources = _loaded_names(sub.iter)
                targets = _targets(sub.target)
            elif isinstance(sub, ast.NamedExpr):
                sources = _loaded_names(sub.value)
                targets = _targets(sub.target)
            elif isinstance(sub, ast.comprehension):
                sources = _loaded_names(sub.iter)
                targets = _targets(sub.target)
            else:
                continue
            incoming = set()
            for name in sources:
                incoming |= origins.get(name, set())
            for target in targets:
                before = origins.setdefault(target, set())
                after = before | incoming
                if after != before:
                    origins[target] = after
                    changed = True
        if not changed:
            break
    return origins


def params_by_line(fn):
    """Map source line to the set of parameter names reaching that line."""
    node = function_ast(fn)
    params = parameter_names(node)
    origins = type_origins(node, params)
    offset = node.lineno - fn.__code__.co_firstlineno
    by_line = {}
    for sub in ast.walk(node):
        line = getattr(sub, "lineno", None)
        if line is None or not isinstance(sub, ast.Name):
            continue
        if not isinstance(sub.ctx, ast.Load):
            continue
        bucket = by_line.setdefault(line - offset, set())
        bucket |= origins.get(sub.id, set())
    return by_line, params


def relevant_positions(fn, lines):
    """Positional indices of the parameters that feed the given source lines.

    Falls back to every position when the analysis finds nothing, which happens
    for sites whose operands come from globals or from attribute chains.
    """
    by_line, params = params_by_line(fn)
    index = {name: i for i, name in enumerate(params)}
    hit = set()
    for line in lines:
        for name in by_line.get(line, ()):
            if name in index:
                hit.add(index[name])
    if not hit:
        return frozenset(range(len(params))), params
    return frozenset(hit), params
