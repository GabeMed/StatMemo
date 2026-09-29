# statisticaly_pure_memoizer.py
import functools
import math
import statistics

import numpy as np
from scipy import stats


class StatisticalyPureMemoizer:
    """Memoize a function whose output is noisy but statistically stable.

    Every distinct call (same function, positional and keyword arguments)
    runs normally until ``number_of_executions`` results have been collected.
    From then on, after each run, the mean of all results so far and the
    half-width of its two-sided ``confidence_level`` Student-t interval are
    computed. Once that half-width is at most ``max_error`` (an absolute
    tolerance, in the function's own units), the call is treated as
    statistically pure: the mean is stored and returned from then on without
    running the function again.

    Return values must be numbers, or NumPy arrays that always have the same
    shape. For arrays, the widest per-element interval is compared with
    ``max_error``. Arguments must be hashable.
    """

    def __init__(self, number_of_executions=10, max_error=0.05, confidence_level=0.95):
        if number_of_executions < 2:
            raise ValueError("number_of_executions must be at least 2")
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")
        self.n_exec   = number_of_executions
        self.max_err  = max_error
        self.conf_lvl = confidence_level
        self._cache   = {}

    def _t_critical(self, n):
        """Two-sided critical value of Student's t with n - 1 degrees of freedom."""
        return stats.t.ppf((1 + self.conf_lvl) / 2, n - 1)

    def _scalar_margin(self, data):
        """Margin of error (CI half-width) of the mean of a list of numbers.

        Uses the sample standard deviation (n - 1), which is what the
        t-interval is defined with.
        """
        n = len(data)
        if n < 2:
            return math.inf
        stdev = statistics.stdev(data)
        t     = self._t_critical(n)
        return t * stdev / math.sqrt(n)

    def _vector_stats(self, stack):
        """Element-wise mean and the widest element-wise margin of error.

        ``stack`` has one result per row (axis 0).
        """
        n    = stack.shape[0]
        mean = stack.mean(axis=0)
        if n < 2:
            return mean, math.inf
        stdev   = stack.std(axis=0, ddof=1)
        margins = self._t_critical(n) * stdev / math.sqrt(n)
        return mean, float(np.max(margins))

    def __call__(self, f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Keyed on the function object, not its name: two different
            # functions with the same __qualname__ must not share results.
            key = (f, args, frozenset(kwargs.items()))
            entry = self._cache.setdefault(
                key, {"vals": [], "stable": False, "mean": None}
            )

            if entry["stable"]:
                return entry["mean"]

            # Roda a função normalmente
            val = f(*args, **kwargs)
            entry["vals"].append(val)

            # Se tiver o número mínimo de execuções, testa se é SPF
            if len(entry["vals"]) >= self.n_exec:
                if isinstance(val, np.ndarray):
                    stack      = np.stack(entry["vals"])
                    mean_val, margin = self._vector_stats(stack)
                else:  # scalar
                    mean_val = statistics.mean(entry["vals"])
                    margin   = self._scalar_margin(entry["vals"])

                entry["mean"] = mean_val
                if margin <= self.max_err:
                    entry["stable"] = True
                    return mean_val

            return val

        return wrapper
