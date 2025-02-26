# main.py

from StatMemo.src.statisticaly_pure_memoizer import StatisticalyPureMemoizer  # from step 1
from StatMemo.experiments.math_calculation_with_noise import compute_integral  # from step 2

# Create an instance of the memoizer
memo = StatisticalyPureMemoizer(number_of_executions=5, max_error=1.5)

# Wrap your functions with the memoizer decorator
@memo
def compute_integral_wrapped():
    return compute_integral()


if __name__ == "__main__":
    # We'll call noisy_func_wrapped(10) multiple times
    print("=== Calls to compute_integral_wrapped(10) ===")
    for i in range(1, 10):
        val = compute_integral_wrapped()
        print(f"Call {i} => {val:.3f}")