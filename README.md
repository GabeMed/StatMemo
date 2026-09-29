# StatMemo

Memoization for functions whose results are noisy.

Ordinary memoization only works for pure functions: same input, same output.
Many expensive functions are not pure in that sense. Randomized searches,
simulations and measurements return a slightly different number every time,
but what the caller usually wants is their *expected* value. StatMemo's
`StatisticalyPureMemoizer` is a decorator that keeps running such a function
until the mean of its results is known within a tolerance, at a chosen
confidence level. After that it returns the mean without running the
function again.

## How it works

For each distinct call (same function, same arguments):

1. The first `number_of_executions` calls (*n*, default 10) run the function
   and return its real result. Every result is recorded.
2. From the *n*-th call on, after each run the memoizer computes the mean of
   all recorded results and the half-width of its two-sided Student-t
   confidence interval:

   `margin = t((1 + confidence_level) / 2, k - 1) * s / sqrt(k)`

   where *k* is the number of results so far and *s* their sample standard
   deviation.
3. If `margin <= max_error`, the call is marked stable. That call and every
   later one return the stored mean, and the function is not run again.
   Otherwise the function keeps running, and the check is repeated after each
   new result.

A function that returns a NumPy array is handled element-wise: the mean is
element-wise, and every element's margin must be within `max_error`.

```python
from src.statisticaly_pure_memoizer import StatisticalyPureMemoizer
from experiments.genetic_algorithm_traveling_sales_person import ga_tsp

memo = StatisticalyPureMemoizer(number_of_executions=10, max_error=0.5, confidence_level=0.95)

@memo
def best_tour_length():
    return ga_tsp()

for _ in range(30):
    best_tour_length()   # runs ga_tsp 10 times, then returns the cached mean
```

## Quick start

Python 3.11 to 3.13.

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m src            # demo, run from the repository root
python -m pytest         # tests
```

Demo output (seed 0, from an actual run):

```
experiment                             max_error      ran status      last value time (s)
compute_integral  (noise sd 0.001)         0.001   10/30  cached         0.4604     0.00
read_temperature  (noise sd 10)              0.5   30/30  not stable    21.9770     0.00
ga_tsp            (random 14 cities)         0.5   10/30  cached         3.2786     3.19
```

`ran` is how many of the 30 calls actually executed the function. The
integral (true value 1 - cos 1 = 0.4597, plus tiny noise) and the genetic
algorithm settle after the minimum 10 runs. The thermometer, with a noise
standard deviation of 10, needs about 1,500 samples to reach a margin of 0.5
((1.96 * 10 / 0.5)^2), so it is not cached within 30 calls.

## Experiment runner

`src/test_experiments_25mar.py` runs heavier experiments through the
memoizer and prints the time of every call:

```sh
python src/test_experiments_25mar.py                          # all
python src/test_experiments_25mar.py --only genetic_algorithm
```

With `belief_propagation` (a NumPy array result, deterministic), the first
10 calls took about 1.6 s each on an Apple Silicon laptop, and the other 50
calls returned from the cache in under 0.1 ms. For `genetic_algorithm`,
calls 1 to 10 took about 0.32 s each and calls 11 and 12 were cached.

## Experiments

| File | Function | Returns |
|---|---|---|
| `experiments/file_read_timer.py` | `time_to_read_file()` | Seconds taken to read `data/file_read_timer_data.txt` (run from the repository root). |
| `experiments/api_data_fetch_timer.py` | `time_to_fetch_data()` | Seconds taken to fetch a public JSON API. Needs network access. |
| `experiments/math_calculation_with_noise.py` | `compute_integral()` | Trapezoid-rule integral of sin(x) on [0, 1], plus N(0, 0.001) noise. |
| `experiments/termometer_measures.py` | `read_temperature()` | 25 plus N(0, 10) noise: a simulated noisy sensor. |
| `experiments/genetic_algorithm_traveling_sales_person.py` | `ga_tsp()` | Best tour length found by a genetic algorithm on 14 random cities in the unit square. |
| `experiments/belief_propagation/belief_propagation.py` | `belief_propagation(N)` | A 5000-element array after N normalized log-domain iterations with a fixed random matrix. Deterministic (seed 0); allocates a 5000 x 5000 matrix. |
| `experiments/galactic_microwave_sky.py` | command-line script | Mean intensity of a simulated microwave sky map. Needs `pysm3`, which is not in `requirements.txt`. |
| `experiments/tiny/` | | Third-party reference code by Alberto Moraglio (geometric semantic GP with classic memoization). See its README. |

## Tests

`python -m pytest` runs `tests/`:

- `tests/test_memoizer.py`: caching after *n* runs, raw results before
  that, noisy functions never cached, the margin equal to the half-width of
  `scipy.stats.t.interval` at 90/95/99%, NumPy array results, cache keys
  per argument and independent of keyword order, different functions with
  the same name kept apart, and parameter validation.
- `tests/test_experiments.py`: fast, offline smoke tests of the experiments.

CI runs the tests and the demo on Python 3.11, 3.12 and 3.13.

## Limitations

- `max_error` is absolute, in the function's own units, not relative.
- Arguments must be hashable. The cache lives in memory, per memoizer
  instance, and is never evicted.
- Once a call is stable, the function is never run again for those
  arguments. Side effects stop, and later drift is not detected.
- The interval assumes the results are independent draws from one
  distribution. Timings (file reads, HTTP requests) are affected by caches
  and warm-up, so the early samples can differ from the later ones.
- The stability check runs after every new result and stops at the first
  pass. Repeated checking like this makes the real coverage lower than the
  nominal `confidence_level`, so treat it as a tuning knob, not a
  guarantee.
- `src/test1.py` is an old scratch file and is entirely commented out.
