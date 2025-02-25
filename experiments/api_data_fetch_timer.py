import requests 
import time

def time_to_fetch_data(url='https://narutodb.xyz/api/character'):
    start_time = time.time()
    response = requests.get(url)
    data = response.json()
    end_time = time.time()
    
    return end_time - start_time

# print(time_to_fetch_data())