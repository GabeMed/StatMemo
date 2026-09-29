"""Quick demo of StatisticalyPureMemoizer. Run from the repository root:

    python -m src

Three of the example experiments are called 30 times each through the
memoizer. For each one it prints how many of those calls really executed
the function, the value returned by the last call (the cached mean once the
result is stable, otherwise the last raw result), and the total time.
"""
import argparse
import random
import time

import numpy as np

from experiments.genetic_algorithm_traveling_sales_person import ga_tsp
from experiments.math_calculation_with_noise import compute_integral
from experiments.termometer_measures import read_temperature
from src.statisticaly_pure_memoizer import StatisticalyPureMemoizer

DEMOS = [
    # label, function, max_error
    ("compute_integral  (noise sd 0.001)", compute_integral, 0.001),
    ("read_temperature  (noise sd 10)", read_temperature, 0.5),
    ("ga_tsp            (random 14 cities)", ga_tsp, 0.5),
]


def run(label, func, max_error, calls, n_exec):
    executions = 0

    def counted():
        nonlocal executions
        executions += 1
        return func()

    memoized = StatisticalyPureMemoizer(
        number_of_executions=n_exec, max_error=max_error, confidence_level=0.95
    )(counted)

    start = time.perf_counter()
    for _ in range(calls):
        last = memoized()
    elapsed = time.perf_counter() - start

    if executions < calls:
        status = "cached"
    elif calls < n_exec:
        status = "untested"  # fewer calls than the minimum sample size
    else:
        status = "not stable"
    print(f"{label:<38} {max_error:>9} {executions:>4}/{calls:<3} "
          f"{status:<11} {float(last):>9.4f} {elapsed:>8.2f}")


def main():
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Run three example experiments through StatisticalyPureMemoizer.",
    )
    parser.add_argument("--calls", type=int, default=30, help="calls per experiment (default 30)")
    parser.add_argument("--n-exec", type=int, default=10,
                        help="executions before the stability test (default 10)")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"{'experiment':<38} {'max_error':>9} {'ran':>8} {'status':<11} "
          f"{'last value':>9} {'time (s)':>8}")
    for label, func, max_error in DEMOS:
        run(label, func, max_error, args.calls, args.n_exec)


if __name__ == "__main__":
    main()
