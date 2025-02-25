import numpy as np

def compute_integral():
    x = np.linspace(0, 1, 1000)
    y = np.sin(x)
    integral = np.trapezoid(y, x) + np.random.normal(0, 0.001)

    return integral

# print(compute_integral())