import numpy as np
import functools
import statistics
import math
from scipy import stats

class StatisticalyPureMemoizer:
    def __init__(self, number_of_executions, max_error=0.5, confidence_level=0.95):
        self.cache = {}
        self.number_of_executions = number_of_executions
        self.max_error = max_error
        self.confidence_level = confidence_level
        self.mean_outputs = {}
        self.confidence_errors = {}
        self.revalidated = {}
    
    def _get_number_of_executions(self, f, args):
        key = (f.__name__, args)
        if key not in self.cache:
            return 0
        return len(self.cache[key]["values"])
    
    def _get_mean_output(self, f, args):
        # TODO HERE
        return 

    def _compute_margin_of_error(self, data):

        n = len(data)
        if n < 2:
            return float('inf') 
        stdev = statistics.pstdev(data)
        z = stats.t.ppf(self.confidence_level, n-1)
        return z * (stdev / math.sqrt(n))
    
    def _execute_and_store(self, f, key, *args):
        result = f(*args)
        self.cache[key]["values"].append(result)
        return result
    
    def __call__(self, f):
        def wrapper(*args):
            key = (f.__name__, args)
            if key not in self.cache:
                self.cache[key] = {
                    "values": [],
                    "mean": None,
                    "stable": False,
                    "calls": 0
                }

            entry = self.cache[key]

            if entry["stable"]:
                return entry["mean"]

            val = self._execute_and_store(f, key, *args)
            entry["calls"] += 1

            if entry["calls"] >= self.number_of_executions:
                data = entry["values"]
                mean_val = statistics.mean(data)
                margin = self._compute_margin_of_error(data)
                entry["mean"] = mean_val

                if margin <= self.max_error:
                    entry["stable"] = True
                    return mean_val
                else:
                    return val
            else:
                return val

        return wrapper

