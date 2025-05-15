# statisticaly_pure_memoizer.py
import statistics, math
import numpy as np
from scipy import stats

class StatisticalyPureMemoizer:

    def __init__(self, number_of_executions=10, max_error=0.05, confidence_level=0.95):
        self.n_exec   = number_of_executions
        self.max_err  = max_error
        self.conf_lvl = confidence_level
        self._cache   = {}   


    def _scalar_margin(self, data):
        """Margin-of-error for a list of numbers (std pop!)."""
        n = len(data)
        if n < 2:
            return math.inf
        stdev = statistics.pstdev(data)
        z     = stats.t.ppf(self.conf_lvl, n - 1)
        return z * stdev / math.sqrt(n)

    def __call__(self, f):
        def wrapper(*args, **kwargs):
            key = (f.__qualname__, args, frozenset(kwargs.items()))
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
