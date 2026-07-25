import math
import time

import pytest

from autolume.live.core.expr import ExpressionError, compile_expression, evaluate


def test_identity_returns_input():
    assert evaluate("x", 0.42) == 0.42


@pytest.mark.parametrize(
    "source, x, expected",
    [
        ("x*2+1", 0.25, 1.5),
        ("-x", 0.25, -0.25),
        ("x**2", 3.0, 9.0),
        ("(x-0.5)*2", 1.0, 1.0),
    ],
)
def test_arithmetic(source, x, expected):
    assert evaluate(source, x) == pytest.approx(expected)


@pytest.mark.parametrize(
    "source, x, expected",
    [
        ("clamp(x, 0, 1)", 2.5, 1.0),
        ("clamp(x, 0, 1)", -2.5, 0.0),
        ("clamp(x, 0, 1)", 0.25, 0.25),
        ("sin(x*pi)", 0.5, 1.0),
        ("max(x, 0.2)", 0.1, 0.2),
        ("abs(x)", -3.0, 3.0),
        ("min(x, 0.2)", 0.1, 0.1),
        ("round(x)", 1.6, 2.0),
        ("sqrt(x)", 9.0, 3.0),
        ("floor(x)", 1.7, 1.0),
        ("ceil(x)", 1.2, 2.0),
        ("log(exp(x))", 2.0, 2.0),
        ("cos(x)", 0.0, 1.0),
        ("tan(x)", 0.0, 0.0),
        ("e**x", 1.0, math.e),
    ],
)
def test_functions_and_constants(source, x, expected):
    assert evaluate(source, x) == pytest.approx(expected)


@pytest.mark.parametrize("source", ["1 if x > 0.5 else 0", "x > 0.5"])
def test_gate_expressions_yield_floats(source):
    high = evaluate(source, 0.9)
    low = evaluate(source, 0.1)
    assert isinstance(high, float) and isinstance(low, float)
    assert (high, low) == (1.0, 0.0)


def test_boolean_and_chained_comparison():
    assert evaluate("x > 0.2 and x < 0.8", 0.5) == 1.0
    assert evaluate("0.2 < x < 0.8", 0.9) == 0.0
    assert evaluate("not x", 0.0) == 1.0


@pytest.mark.parametrize(
    "source",
    [
        "__import__('os').system('ls')",
        "x.__class__",
        "open('f')",
        "[i for i in range(3)]",
        "lambda: 1",
        "y",
        "x; y",
        "(y := 2)",
        "x[0]",
        "f'{x}'",
        "'text'",
        "min(x, key=None)",
        "max(*[x, 1])",
        "x is None",
        "x in [1, 2]",
        "",
        "1j",
    ],
)
def test_rejected_at_compile_time(source):
    with pytest.raises(ExpressionError):
        compile_expression(source)


def test_compile_error_names_the_offending_construct():
    with pytest.raises(ExpressionError, match="Attribute"):
        compile_expression("x.__class__")
    with pytest.raises(ExpressionError, match="open"):
        compile_expression("open('f')")
    with pytest.raises(ExpressionError, match="unknown name: y"):
        compile_expression("y")


@pytest.mark.parametrize(
    "source", ["1/0", "log(0-1)", "x*1e400", "1 % 0", "(0-1)**0.5", "1e308**2"]
)
def test_rejected_at_runtime(source):
    fn = compile_expression(source)
    with pytest.raises(ExpressionError):
        fn(1.0)


_PROMPT_SECONDS = 1.0


@pytest.mark.parametrize(
    "source",
    [
        "9**9**9",
        "2**2**26",
        "(((2**64)**64)**64)**64",
        "1**10**20",
    ],
)
def test_oversized_integer_power_is_rejected_promptly(source):
    start = time.perf_counter()
    with pytest.raises(ExpressionError, match="too large"):
        evaluate(source, 1.0)
    assert time.perf_counter() - start < _PROMPT_SECONDS


@pytest.mark.parametrize(
    "source, x, expected",
    [
        ("x**2", 3.0, 9.0),
        ("x**0.5", 9.0, 3.0),
        ("2**10", 1.0, 1024.0),
        ("2**-2", 1.0, 0.25),
        ("x**64", 1.5, 1.5**64),
        ("(1+x)**2", 1.0, 4.0),
    ],
)
def test_bounded_powers_still_evaluate(source, x, expected):
    assert evaluate(source, x) == pytest.approx(expected)


def test_runtime_errors_do_not_leak_native_exception_types():
    for source in ("1/0", "log(0-1)", "x*1e400"):
        try:
            evaluate(source, 1.0)
        except ExpressionError:
            pass
        else:
            pytest.fail(f"{source} should have raised ExpressionError")


def test_oversized_source_is_rejected_without_leaking_recursion_error():
    for source in ("+".join(["x"] * 5000), "-" * 5000 + "x"):
        with pytest.raises(ExpressionError, match="too long"):
            compile_expression(source)


def test_deeply_nested_but_allowed_source_still_evaluates():
    source = "+".join(["x"] * 200)
    assert evaluate(source, 1.0) == 200.0


def test_expression_error_is_a_value_error():
    assert issubclass(ExpressionError, ValueError)


def test_cache_returns_the_same_callable():
    assert compile_expression("x*2+1") is compile_expression("x*2+1")


def test_cache_is_bounded():
    for i in range(1000):
        compile_expression(f"x*{i}")
    info = compile_expression.cache_info()
    assert info.currsize <= info.maxsize
    assert info.maxsize is not None
