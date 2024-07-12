import asyncio
import aiohttp
import logging
from collections import deque
import datetime as dt
import requests
from concurrent.futures import ThreadPoolExecutor
import time
import async_timeout

file_sources= ["http://10.137.132.2/ubuntu-23.10.1-desktop-amd64.iso", 
               "http://10.135.163.2/ubuntu-23.10.1-desktop-amd64.iso",
               "http://10.134.48.2/ubuntu-23.10.1-desktop-amd64.iso"]

response = requests.head(file_sources[0])
file_size = int(response.headers.get('Content-Length', 0))
print("File size:", file_size)

start_byte = 0
end_byte = 0
number_of_chunk = 0


async def measure_throughput(session, url, throughput, range_dict, file_path, idx, semaphore):
    async with semaphore:
        first_chunk = 5 * 1024 * 1024  # Smaller initial chunk to get more accurate throughput
        global end_byte, start_byte
        end_byte = start_byte + first_chunk - 1
        data, throughput = await fetch_range(session, url, start_byte, end_byte, throughput, range_dict, file_path)
        return throughput


def best_fit(file_size, remaining_file, servers, throughput, small_chunk, large_chunk):
    global end_byte
    server_loads = [(0, i) for i in range(len(servers))]
    server_assignments = {i: [] for i in range(len(servers))}
    total_throughput = sum(throughput.values())
    t = remaining_file / total_throughput
    next_byte = end_byte + 1
    value_list = list(throughput.values())

    while remaining_file > 0:
        best_server = None
        min_excess = float('inf')
        selected_chunk = None

        for chunk in [large_chunk, small_chunk]:
            for i in range(len(value_list)):
                current_load = server_loads[i][0]
                new_load = current_load + (chunk / value_list[i])
                excess = t - new_load

                if excess >= 0 and excess < min_excess:
                    best_server = (new_load, i)
                    min_excess = excess
                    selected_chunk = chunk

        if best_server is not None:
            current_load, server_index = best_server
            end_byte = next_byte + selected_chunk - 1
            if end_byte >= file_size:
                end_byte = file_size
            server_assignments[server_index].append((next_byte, end_byte))
            server_loads[server_index] = (current_load, server_index)
            next_byte = end_byte + 1
            remaining_file -= selected_chunk
        else:
            break

    return server_assignments


async def fetch_range(session, url, s_b, e_b, throughput, range_dict, file_path):
    global start_byte, end_byte, number_of_chunk
    local_start_byte = s_b
    local_end_byte = e_b
    headers = {'Range': f"bytes={local_start_byte}-{local_end_byte}"}
    start_time = time.time()
    async with async_timeout.timeout(1000):
        async with session.get(url, headers=headers) as response:
            data = await response.read()
            end_time = time.time()
            delay = end_time - start_time
            throughput[url] = (local_end_byte - local_start_byte) / delay
            range_dict[local_start_byte] = data
            number_of_chunk += 1
            print("number_of_chunk:", number_of_chunk)
            return data, throughput


async def download_file(session, url, throughput, range_dict, file_path, server_queue, semaphore):
    tasks = []
    while server_queue:
        s_b, e_b = server_queue.popleft()
        async with semaphore:
            tasks.append(fetch_range(session, url, s_b, e_b, throughput, range_dict, file_path))
    await asyncio.gather(*tasks)
    print(f"Download completed for {url}")


async def main():
    start_time = time.time()
    now = dt.datetime.now()
    formatted_now = now.strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"Dynamic_Chunking_Async_{formatted_now}.log"
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
    session_dict = {url: aiohttp.ClientSession() for url in file_sources}
    filename = "output_file"
    file_path = "output1.txt"
    throughput = {}
    range_dict = {}
    small_chunk = 10 * 1024 * 1024
    large_chunk = 40 * 1024 * 1024
    max_concurrent_downloads = 10
    semaphore = asyncio.Semaphore(max_concurrent_downloads)

    await asyncio.gather(
        *(measure_throughput(session_dict[url], url, throughput, range_dict, file_path, idx, semaphore) for idx, url in enumerate(file_sources))
    )
    remaining_file = file_size - end_byte

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        server_assignment = await loop.run_in_executor(
            pool, best_fit, file_size, remaining_file, file_sources, throughput, small_chunk, large_chunk
        )
    print("server_assignment:", server_assignment)
    server_queues = {i: deque() for i in range(len(file_sources))}
    for server_idx, assignments in server_assignment.items():
        for start, end in assignments:
            server_queues[server_idx].append((start, end))
            print("server_idx, assignments:", server_idx, assignments)
            print("start,end:", server_queues[server_idx])
        print("server_queues:", server_queues)

    await asyncio.gather(
        *(download_file(session_dict[url], url, throughput, range_dict, file_path, server_queues[idx], semaphore) for idx, url in enumerate(file_sources))
    )
    fast_url = max(zip(throughput.values(), throughput.keys()))[1]
    remain = file_size - end_byte
    await fetch_range(session_dict[fast_url], fast_url, end_byte + 1, file_size, throughput, range_dict, file_path)

    for session in session_dict.values():
        await session.close()

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


if __name__ == "__main__":
    
    asyncio.run(main())
    

