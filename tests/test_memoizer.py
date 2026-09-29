import math
import random

import numpy as np
import pytest
from scipy import stats

from src.statisticaly_pure_memoizer import StatisticalyPureMemoizer


def counting(func):
    """Wrap func so the test can see how many times it really ran."""
    def counted(*args, **kwargs):
        counted.calls += 1
        return func(*args, **kwargs)
    counted.calls = 0
    return counted


def test_deterministic_function_is_cached_after_n_executions():
    memo = StatisticalyPureMemoizer(number_of_executions=5, max_error=0.01)
    f = counting(lambda: 42.0)
    g = memo(f)

    results = [g() for _ in range(20)]

    assert f.calls == 5
    assert results == [42.0] * 20


def test_runs_normally_before_n_executions():
    memo = StatisticalyPureMemoizer(number_of_executions=5, max_error=10.0)
    values = iter([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    f = counting(lambda: next(values))
    g = memo(f)

    first_four = [g() for _ in range(4)]

    assert first_four == [1.0, 2.0, 3.0, 4.0]
    assert f.calls == 4
    # 5th call reaches n: interval half-width for 1..5 is ~1.96 <= 10 -> stable
    assert g() == pytest.approx(3.0)
    assert g() == pytest.approx(3.0)
    assert f.calls == 5


def test_noisy_function_is_never_cached():
    rng = random.Random(1)
    memo = StatisticalyPureMemoizer(number_of_executions=5, max_error=0.1)
    f = counting(lambda: rng.gauss(25.0, 10.0))
    g = memo(f)

    for _ in range(200):
        g()

    assert f.calls == 200


def test_low_noise_function_converges_to_the_mean():
    rng = random.Random(2)
    memo = StatisticalyPureMemoizer(number_of_executions=10, max_error=0.01)
    f = counting(lambda: 3.0 + rng.gauss(0.0, 0.01))
    g = memo(f)

    for _ in range(50):
        last = g()

    assert f.calls < 50
    assert last == pytest.approx(3.0, abs=0.01)


@pytest.mark.parametrize("data", [[1.0, 2.0, 3.0, 4.0, 5.0], [0.1, 0.4, 0.2, 0.9, 0.3, 0.3, 0.7]])
@pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
def test_margin_matches_scipy_t_interval(data, confidence):
    memo = StatisticalyPureMemoizer(confidence_level=confidence)
    low, high = stats.t.interval(confidence, len(data) - 1,
                                 loc=np.mean(data), scale=stats.sem(data))
    assert memo._scalar_margin(data) == pytest.approx((high - low) / 2)


def test_margin_needs_two_points():
    assert StatisticalyPureMemoizer()._scalar_margin([1.0]) == math.inf


def test_numpy_array_results():
    memo = StatisticalyPureMemoizer(number_of_executions=3, max_error=0.01)
    f = counting(lambda: np.array([1.0, 2.0, 3.0]))
    g = memo(f)

    for _ in range(10):
        out = g()

    assert f.calls == 3
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0])


def test_numpy_array_needs_every_element_stable():
    rng = np.random.default_rng(3)
    memo = StatisticalyPureMemoizer(number_of_executions=3, max_error=0.01)
    # First element constant, second very noisy.
    f = counting(lambda: np.array([1.0, rng.normal(0.0, 5.0)]))
    g = memo(f)

    for _ in range(30):
        g()

    assert f.calls == 30


def test_vector_margin_matches_scalar_margin():
    memo = StatisticalyPureMemoizer()
    rows = np.array([[1.0, 10.0], [2.0, 10.5], [4.0, 9.0], [3.0, 11.0]])
    mean, margin = memo._vector_stats(rows)
    np.testing.assert_allclose(mean, rows.mean(axis=0))
    expected = max(memo._scalar_margin(list(rows[:, i])) for i in range(2))
    assert margin == pytest.approx(expected)


def test_cache_is_per_arguments():
    memo = StatisticalyPureMemoizer(number_of_executions=2, max_error=0.01)
    f = counting(lambda x, scale=1: x * scale)
    g = memo(f)

    for _ in range(5):
        assert g(1) == 1
        assert g(2) == 2
        assert g(2, scale=10) == 20

    assert f.calls == 6  # 2 executions for each of the 3 distinct calls


def test_keyword_order_does_not_matter():
    memo = StatisticalyPureMemoizer(number_of_executions=2, max_error=0.01)
    f = counting(lambda a=0, b=0: a + b)
    g = memo(f)

    for _ in range(3):
        g(a=1, b=2)
        g(b=2, a=1)

    assert f.calls == 2


def test_functions_with_the_same_name_do_not_share_results():
    memo = StatisticalyPureMemoizer(number_of_executions=2, max_error=0.01)

    def make(value):
        def f():
            return value
        return f

    a, b = memo(make(1.0)), memo(make(100.0))
    assert a.__qualname__ == b.__qualname__

    assert [a() for _ in range(3)] == [1.0, 1.0, 1.0]
    assert [b() for _ in range(3)] == [100.0, 100.0, 100.0]


def test_wrapper_keeps_function_metadata():
    memo = StatisticalyPureMemoizer()

    @memo
    def measure():
        """Docstring."""
        return 1.0

    assert measure.__name__ == "measure"
    assert measure.__doc__ == "Docstring."


@pytest.mark.parametrize("kwargs", [{"number_of_executions": 1},
                                    {"confidence_level": 0},
                                    {"confidence_level": 1.5}])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        StatisticalyPureMemoizer(**kwargs)
