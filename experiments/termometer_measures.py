import numpy as np

def read_temperature():
    base_value = 25
    noise = np.random.normal(0, 2)
    temperature = base_value + noise
    
    return temperature

# print(read_temperature())
