"""Smoke tests for the example experiments that run offline and fast."""
import math

import numpy as np

from experiments.file_read_timer import time_to_read_file
from experiments.genetic_algorithm_traveling_sales_person import ga_tsp
from experiments.math_calculation_with_noise import compute_integral
from experiments.termometer_measures import read_temperature
from src.statisticaly_pure_memoizer import StatisticalyPureMemoizer


def test_compute_integral_is_close_to_one_minus_cos_one():
    assert abs(compute_integral() - (1 - math.cos(1))) < 0.01


def test_read_temperature_returns_a_number():
    assert isinstance(float(read_temperature()), float)


def test_file_read_timer(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("hello")
    assert time_to_read_file(str(path)) >= 0


def test_ga_tsp_returns_a_valid_tour_length():
    # 14 random cities in the unit square: any closed tour is longer than 0
    # and shorter than 14 * sqrt(2).
    length = ga_tsp(num_cities=14, pop_size=120, generations=50)
    assert 0 < length < 14 * math.sqrt(2)


def test_memoized_integral_stops_executing():
    np.random.seed(0)  # compute_integral adds N(0, 0.001) noise
    memo = StatisticalyPureMemoizer(number_of_executions=10, max_error=0.001)
    calls = 0

    def integral():
        nonlocal calls
        calls += 1
        return compute_integral()

    g = memo(integral)
    for _ in range(30):
        value = g()

    assert calls == 10
    assert abs(value - (1 - math.cos(1))) < 0.001
    assert isinstance(value, (float, np.floating))
