import time

def time_to_read_file(filename="data/file_read_timer_data.txt"):
    start_time = time.time()
    with open(filename, "r") as file:
        data = file.read()
    end_time = time.time()

    return end_time - start_time     

# print(time_to_read_file())