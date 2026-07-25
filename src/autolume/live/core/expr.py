"""Safe evaluator for user mapping expressions.

User text is never handed to `eval`. A source string is parsed to an AST and
compiled node by node into nested closures. Only the node types, operators and
names handled below can be compiled, so anything outside the whitelist is
rejected before it can run. All failures, at compile time and at run time, are
raised as `ExpressionError`.
"""

import ast
import functools
import math
import operator
from collections.abc import Callable

_CACHE_SIZE = 128

# Bounds both parser recursion depth and the memory the cache can hold. A
# mapping expression is a one-liner, so this is far above any real use.
_MAX_SOURCE_LENGTH = 512

# Integer exponentiation is arbitrary precision, so `9**9**9` runs for minutes
# and can exhaust memory before there is any result left to reject. Both bounds
# are checked before the operation runs, since afterwards is too late. The bit
# bound is what stops nesting (`((2**64)**64)**64`) from compounding past the
# exponent bound; 1024 bits is already outside the float range, so no result
# that could survive the float conversion is lost.
_MAX_POW_EXPONENT = 64
_MAX_POW_BITS = 1024


_SYNTAX_PREFIX = "invalid syntax"


class ExpressionError(ValueError):
    """A mapping expression is not valid or failed to produce a usable value."""


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


_NAMES: dict[str, object] = {
    "pi": math.pi,
    "e": math.e,
}

_FUNCTIONS: dict[str, Callable[..., float]] = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "log": math.log,
    "floor": math.floor,
    "ceil": math.ceil,
    "clamp": clamp,
}

def _guarded_pow(left: float, right: float) -> float:
    if isinstance(left, int) and isinstance(right, int):
        if right > _MAX_POW_EXPONENT:
            raise ExpressionError(
                f"exponent is too large: {right}, the limit is {_MAX_POW_EXPONENT}"
            )
        if left.bit_length() * max(right, 0) > _MAX_POW_BITS:
            raise ExpressionError(
                f"power result is too large, the limit is {_MAX_POW_BITS} bits"
            )
    return left**right


_BINARY_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: _guarded_pow,
}

_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], object]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
}

_COMPARE_OPS: dict[type[ast.cmpop], Callable[[object, object], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

_Node = Callable[[float], object]


def _syntax_message(exc: SyntaxError, source: str) -> str:
    """Say where the syntax error is, and keep the parser's hint when it has one.

    `offset` is one based and is 0 when the parser ran out of input, so an
    offset outside the source is dropped rather than pointed at. Python's own
    `msg` is already "invalid syntax" for the plain cases and starts with those
    same words for the helpful ones, so the prefix is taken off what is left
    and the message never says it twice.
    """
    detail = (exc.msg or "").strip()
    if detail.lower().startswith(_SYNTAX_PREFIX):
        detail = detail[len(_SYNTAX_PREFIX) :].lstrip(". ")
    offset = exc.offset
    where = ""
    if isinstance(offset, int) and 1 <= offset <= len(source):
        where = f" at character {offset}"
    if detail:
        return f"{_SYNTAX_PREFIX}{where}. {detail}"
    return f"{_SYNTAX_PREFIX}{where}"


def _reject(node: ast.AST) -> ExpressionError:
    return ExpressionError(f"unsupported expression: {type(node).__name__}")


def _compile_node(node: ast.AST) -> _Node:
    if isinstance(node, ast.Constant):
        value = node.value
        if not isinstance(value, (bool, int, float)):
            raise ExpressionError(f"unsupported constant: {value!r}")
        return lambda x: value

    if isinstance(node, ast.Name):
        if not isinstance(node.ctx, ast.Load):
            raise _reject(node)
        if node.id == "x":
            return lambda x: x
        if node.id in _NAMES:
            constant = _NAMES[node.id]
            return lambda x: constant
        raise ExpressionError(f"unknown name: {node.id}")

    if isinstance(node, ast.BinOp):
        op = _BINARY_OPS.get(type(node.op))
        if op is None:
            raise _reject(node.op)
        left = _compile_node(node.left)
        right = _compile_node(node.right)
        return lambda x: op(left(x), right(x))

    if isinstance(node, ast.UnaryOp):
        unary = _UNARY_OPS.get(type(node.op))
        if unary is None:
            raise _reject(node.op)
        operand = _compile_node(node.operand)
        return lambda x: unary(operand(x))

    if isinstance(node, ast.BoolOp):
        values = [_compile_node(value) for value in node.values]
        if isinstance(node.op, ast.And):
            return lambda x: _eval_and(values, x)
        if isinstance(node.op, ast.Or):
            return lambda x: _eval_or(values, x)
        raise _reject(node.op)

    if isinstance(node, ast.Compare):
        ops = []
        for op_node in node.ops:
            compare = _COMPARE_OPS.get(type(op_node))
            if compare is None:
                raise _reject(op_node)
            ops.append(compare)
        left = _compile_node(node.left)
        comparators = [_compile_node(value) for value in node.comparators]
        return lambda x: _eval_compare(left, ops, comparators, x)

    if isinstance(node, ast.IfExp):
        test = _compile_node(node.test)
        body = _compile_node(node.body)
        orelse = _compile_node(node.orelse)
        return lambda x: body(x) if test(x) else orelse(x)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise _reject(node.func)
        function = _FUNCTIONS.get(node.func.id)
        if function is None:
            raise ExpressionError(f"unknown function: {node.func.id}")
        if node.keywords:
            raise ExpressionError(f"keyword arguments are not allowed: {node.func.id}")
        args = [_compile_node(arg) for arg in node.args]
        return lambda x: function(*[arg(x) for arg in args])

    raise _reject(node)


def _eval_and(values: list[_Node], x: float) -> object:
    result: object = True
    for value in values:
        result = value(x)
        if not result:
            return result
    return result


def _eval_or(values: list[_Node], x: float) -> object:
    result: object = False
    for value in values:
        result = value(x)
        if result:
            return result
    return result


def _eval_compare(
    left: _Node,
    ops: list[Callable[[object, object], bool]],
    comparators: list[_Node],
    x: float,
) -> bool:
    current = left(x)
    for compare, comparator in zip(ops, comparators):
        right = comparator(x)
        if not compare(current, right):
            return False
        current = right
    return True


@functools.lru_cache(maxsize=_CACHE_SIZE)
def compile_expression(source: str) -> Callable[[float], float]:
    """Compile a mapping expression into a float to float callable.

    Raises `ExpressionError` if the source is not a valid single expression
    built from the whitelisted nodes, operators and names.
    """
    if len(source) > _MAX_SOURCE_LENGTH:
        raise ExpressionError(
            f"expression is too long: {len(source)} characters, "
            f"the limit is {_MAX_SOURCE_LENGTH}"
        )
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(_syntax_message(exc, source)) from exc
    except ValueError as exc:
        raise ExpressionError(f"invalid expression: {exc}") from exc

    body = _compile_node(tree.body)

    def run(x: float) -> float:
        try:
            result = float(body(float(x)))
        except ExpressionError:
            raise
        except (ArithmeticError, ValueError, TypeError) as exc:
            raise ExpressionError(f"evaluation failed: {exc}") from exc
        if not math.isfinite(result):
            raise ExpressionError(f"result is not finite: {result}")
        return result

    return run


def evaluate(source: str, x: float) -> float:
    """Compile (cached) and evaluate `source` for the input value `x`."""
    return compile_expression(source)(x)
