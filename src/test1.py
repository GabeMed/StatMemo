
from StatMemo.src.statisticaly_pure_memoizer import StatisticalyPureMemoizer  
from StatMemo.experiments.math_calculation_with_noise import compute_integral  
from StatMemo.experiments.termometer_measures import read_temperature

memo = StatisticalyPureMemoizer(number_of_executions=5, max_error=1.5)

@memo
def compute_integral_wrapped():
    return compute_integral()

@memo 
def termometer_measures_wrapped():
    return read_temperature()


if __name__ == "__main__":
    print("=== Calls to compute_integral_wrapped(10) ===")
    for i in range(1, 10):
        val = compute_integral_wrapped()
        print(f"Call {i} => {val:.3f}")

    print("     ")

    print("=== Calls to read_temperature_wrapped(10) ===")
    for i in range(1, 10):
        val = termometer_measures_wrapped()
        print(f"Call {i} => {val:.3f}")