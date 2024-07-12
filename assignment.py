import asyncio
import aiohttp
import logging
from collections import deque
import time
import sys
import datetime as dt
import requests
from scipy.stats import gmean
import hashlib
import math
import statistics
import itertools
import queue
import async_timeout



file_sources= ["http://10.137.132.2/ubuntu-23.10.1-desktop-amd64.iso", 
                "http://10.135.163.2/ubuntu-23.10.1-desktop-amd64.iso",
                "http://10.134.48.2/ubuntu-23.10.1-desktop-amd64.iso"]

response = requests.head(file_sources[0])
file_size = int(response.headers.get('Content-Length', 0))
print("File size:", file_size)
# logging.info(f"File size determined as {file_size} bytes.")

start_byte=0
end_byte=0
currentRequestedByte=0
throughput_gm= 0
number_of_chunk=0
count =0 
remaining_chunk_queue = []
time_download = []
combination_time= []
wait_flag = False
small_distribution = ()
big_distribution = ()



def find_chunk_combinations(total_size, size_large, size_small):
    combinations = []
    print("1111111111111")
    # Iterate over the possible number of large chunks
    for large_chunks in range(total_size // size_large + 1):
        remaining_size = total_size - large_chunks * size_large
        print("22222222222")
        logging.info("remaining_size:",remaining_size)
        # Check if the remaining size can be covered by small chunks
        small_chunks = math.ceil(remaining_size / size_small)
        print("33333333333")
        logging.info("small_chunks is:",small_chunks)
        combinations.append((large_chunks, small_chunks))

    return combinations
    
def find_best_server_combination(remaining_size, throughput, large_chunk, small_chunk, urls):
    
    chunk_combinations = find_chunk_combinations(remaining_size, large_chunk, small_chunk)
    download_times ={server: {'large': large_chunk / speed, 'small': small_chunk / speed} 
                      for server, speed in throughput.items()}
    results = []
    # Iterate through each chunk combination
    for big_chunks, small_chunks in chunk_combinations:
        # Distribute the big and small chunks among the three servers
        for big_distribution in itertools.combinations_with_replacement(range(len(urls)), big_chunks):
            for small_distribution in itertools.combinations_with_replacement(range(len(urls)), small_chunks):
                times = {url: 0 for url in urls} 
                # Accumulate times for big chunks
                for i in big_distribution:
                    server = urls[i]  # Correctly reference the server based on its index
                    times[server] += download_times[server]['large']
                # Accumulate times for small chunks
                for i in small_distribution:
                    server = urls[i]  # Correctly reference the server based on its index
                    times[server] += download_times[server]['small']
                
                # Calculate max time as the bottleneck time for this distribution
                max_time = max(times.values())
                results.append((big_distribution, small_distribution, times, max_time))
                logging.info("results",results)
                print("4444444444")

    return results
    

async def fetch_range(session, url, s_b, e_b, throughput, range_dict, file_path):
    global start_byte, end_byte, number_of_chunk
    local_start_byte= s_b
    local_end_byte = e_b
    print("updating start byte")
    start_byte += e_b - s_b +1
    end_byte = e_b
    print("fetching")
    print(f"Fetching range::::: {local_start_byte}-{local_end_byte} from {url}")
    logging.info(f"Fetching range::::: {local_start_byte}-{local_end_byte} from {url}")
    headers = {'Range': f"bytes={local_start_byte}-{local_end_byte}"}
    start_time = time.time()
    async with async_timeout.timeout(200):
        async with session.get(url, headers=headers) as response:
            data = await response.read()
            end_time= time.time()
            delay = end_time - start_time
            print("delay:", delay)
            logging.info(f"Delay {delay}")
            # await compare_files(url, data,local_start_byte, local_end_byte, file_path)
            throughput[url]= (local_end_byte - local_start_byte)/delay
            range_dict[local_start_byte]= data
            print("throughput",throughput)
            logging.info(f"Successfully fetched range {local_start_byte}-{local_end_byte} from {url}")
            logging.info(f"Throughput: {throughput}")
            number_of_chunk= number_of_chunk + 1

            print("number_of_chunk:",number_of_chunk)
            logging.info(f"number_of_chunk: {number_of_chunk}")
            return data, throughput

async def download_file(session, url, throughput, range_dict, file_path, urls, idx):
    big_chunk= 5* 1024 * 1024
    global throughput_gm
    global end_byte, start_byte, wait_flag, small_distribution, big_distribution
    end_byte= start_byte + big_chunk -1
    print (start_byte, end_byte)
    data, throughput = await fetch_range(session, url, start_byte, end_byte, throughput, range_dict, file_path)
    fast_dic={}
    slow_dic={}
    
    while end_byte < int(0.97 * file_size):
        if len(throughput)>1:
            values= list(throughput.values())
            throughput_gm = gmean(values)
            print("throughput_gm", throughput_gm)

        if throughput[url]>= throughput_gm:
            fast_dic[url]=throughput
            big_start_byte, big_end_byte = await download_big_chunk()
            if big_end_byte>file_size:
                big_end_byte=file_size
            data, throughput = await fetch_range(session, url, big_start_byte, big_end_byte, throughput, range_dict, file_path)
        else:
            slow_dic[url]=throughput
            small_start_byte , small_end_byte = await download_small_chunk()
            if small_end_byte>file_size:
                small_end_byte=file_size
            data, throughput = await fetch_range(session, url, small_start_byte , small_end_byte, throughput, range_dict, file_path)

    if (wait_flag==False and not small_distribution and not big_distribution):
        wait_flag = True
        remaining_size = file_size - end_byte
        print(remaining_size)
        best_server_combinations = find_best_server_combination(remaining_size, throughput, 40*1024*1024, 10*1024*1024, urls)
        logging.info("The results is {best_server_combinations}")
        print("55555555555555")
        min_max_time_element = min(best_server_combinations, key=lambda x: x[3])
        print("The element with the minimum max_time is:", min_max_time_element)
        print("6666666666666")
        small_distribution = min_max_time_element[1]
        big_distribution = min_max_time_element[0]
        wait_flag = False

    while wait_flag is True:
        pass
    print(big_distribution, small_distribution)
    for big_distr in big_distribution:
        if idx == big_distr:
            big_start_byte, big_end_byte = await download_big_chunk()
            if big_end_byte>file_size:
                big_end_byte=file_size
            data, throughput = await fetch_range(session, url, big_start_byte, big_end_byte, throughput, range_dict, file_path)
    for small_distr in small_distribution:
        if idx == small_distr:
            small_start_byte , small_end_byte = await download_small_chunk()
            if small_end_byte>file_size:
                small_end_byte=file_size
            data, throughput = await fetch_range(session, url, small_start_byte , small_end_byte, throughput, range_dict, file_path)

async def download_small_chunk():
    global start_byte
    small_chunk = 10* 1024 * 1024
    small_start_byte = start_byte
    small_end_byte = small_start_byte + small_chunk -1
    return small_start_byte , small_end_byte

async def download_big_chunk():
    big_chunk= 40* 1024 * 1024
    global start_byte
    big_start_byte = start_byte
    big_end_byte = big_start_byte + big_chunk -1
    return big_start_byte , big_end_byte


async def main():
    start_time = time.time()
    # Initialize logging
    now = dt.datetime.now()
    formatted_now = now.strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"Dynamic_Chunking_Async_{formatted_now}.log"
    logging.basicConfig(filename=log_file,level=logging.INFO,format='%(asctime)s:%(levelname)s:%(message)s')
    session_dict = {url: aiohttp.ClientSession() for url in file_sources}
    filename = "output_file"
    file_path= "/Users/sabdollah42/Desktop/output1.txt"
    throughput= {}
    
    range_dict = {}
    await asyncio.gather(
        *(download_file(session_dict[url], url, throughput, range_dict, file_path, file_sources, idx) for idx, url in enumerate(file_sources))
    )

    logging.info("All chunks downloaded, writing to disk.")
    with open(filename, 'wb') as f:
        for start_byte in sorted(range_dict.keys()):
            data = range_dict[start_byte]
            f.write(data)
    end_time= time.time()
    print("they printed all")
    delay= end_time - start_time
    print("Delay:", delay)
    logging.info(f"Delay {delay}")
    total_throughput= file_size/delay
    print("total_throughput:", total_throughput)
    logging.info(f"Total download throughput: {total_throughput}")


if __name__ == "__main__":
    asyncio.run(main())


