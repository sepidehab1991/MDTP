import asyncio
import aiohttp
import logging
from collections import deque
import time
import datetime as dt
import requests
from scipy.stats import gmean
import math
import async_timeout
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# =======================
# 1. A3C Model Definition
# =======================
class A3CChunkSelector(nn.Module):
    def __init__(self, input_dim=3, action_dim=4):  # <<< Changed input_dim to 3
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.policy = nn.Linear(128, action_dim)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        probs = F.softmax(self.policy(x), dim=-1)
        value = self.value(x)
        return probs, value

a3c_model = A3CChunkSelector()
optimizer = torch.optim.Adam(a3c_model.parameters(), lr=1e-4)

# ================================
# 2. A3C Chunk Selection + Update
# ================================
def choose_chunk_size_and_learn(avg_speed, num_servers, download_time):
    # <<< Now include download_time in the input tensor
    state = torch.tensor([[avg_speed, num_servers, download_time]], dtype=torch.float32)
    probs, value = a3c_model(state)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    selected_chunk_size = [4, 8, 16, 32][action.item()] * 1024 * 1024

    reward = -download_time  # minimize download time
    R = torch.tensor([reward], dtype=torch.float32)
    advantage = R - value

    policy_loss = -dist.log_prob(action) * advantage.detach()
    value_loss = advantage.pow(2)
    entropy_loss = -(probs * probs.log()).sum()

    total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return selected_chunk_size

# =====================
# 3. Downloader Script
# =====================
file_size = sys.argv[1]
iteration = sys.argv[2]
num_servers = int(sys.argv[3])
folder_name = sys.argv[4]

urls = [
 "http://10.129.144.2/",
 "http://10.147.11.2/",
 "http://10.132.136.2/"
]
def get_file_sources(urls, num_sources):
    if num_sources > len(urls):
        raise ValueError("Number of sources exceeds the number of available URLs.")
    return urls[:num_sources]

file_ips = get_file_sources(urls, num_servers)
file_sources = [i + file_size for i in file_ips]

response = requests.head(file_sources[0])
file_size = int(response.headers.get('Content-Length', 0))

start_byte = 0
end_byte = 0
throughput_gm = 0
number_of_chunk = 0
fast_downloadTime = 0

async def fetch_range(session, url, s_b, e_b, throughput, range_dict):
    global start_byte, end_byte, number_of_chunk
    local_start_byte = s_b
    local_end_byte = e_b
    start_byte += e_b - s_b + 1
    end_byte = e_b
    headers = {'Range': f"bytes={local_start_byte}-{local_end_byte}"}
    start_time = time.time()
    try:
        async with async_timeout.timeout(500):
            async with session.get(url, headers=headers) as response:
                data = await response.read()
                end_time = time.time()
                delay = end_time - start_time
                logging.info(f"Delay {delay}")
                throughput[url] = (local_end_byte - local_start_byte) / delay
                range_dict[local_start_byte] = data
                logging.info(f"Successfully fetched range {local_start_byte}-{local_end_byte} from {url}")
                logging.info(f"Throughput: {throughput}")
                number_of_chunk += 1
                logging.info(f"number_of_chunk: {number_of_chunk}")
                logging.info(f"Fetched range: start_byte={local_start_byte}, end_byte={local_end_byte}, delay={delay:.4f}, throughput={throughput[url]}")
            return data, throughput
    except Exception as e:
        logging.info(f"Error fetching data from {url}: {e}")

async def download_file(session, url, throughput, range_dict, large_chunk):
    logging.info(f"Downloading file from URL: {url}")
    global fast_downloadTime
    initial_chunk = 5 * 1024 * 1024
    global throughput_gm, end_byte, start_byte
    end_byte = start_byte + initial_chunk - 1
    data, throughput = await fetch_range(session, url, start_byte, end_byte, throughput, range_dict)

    fast_dic = {}
    slow_dic = {}

    while end_byte < file_size and start_byte < file_size:
        values = list(throughput.values())
        throughput_gm = gmean(values)
        logging.info(f"Geometric mean of throughput: {throughput_gm}")
        logging.info(f"start and end byte are: {start_byte} and {end_byte}")
        logging.info(f"Current throughput for {url}: {throughput[url]}")
        if math.ceil(throughput[url]) >= math.ceil(throughput_gm):
            fast_dic[url] = throughput[url]
            fastest_server = max(fast_dic.values())
            logging.info(f"Fast server throughput: {fastest_server}")
            fast_downloadTime = large_chunk / fastest_server
            logging.info(f"Fast download time: {fast_downloadTime}")
            large_start_byte = start_byte
            logging.info(f"Large chunk start byte: {large_start_byte}")
            large_end_byte = large_start_byte + large_chunk + 1
            if large_end_byte > file_size:
                large_end_byte = file_size
            if large_start_byte > file_size:
                break
            logging.info(f"Large_start_byte, Large_end_byte: {large_start_byte} and {large_end_byte}")
            data, throughput = await fetch_range(session, url, large_start_byte, large_end_byte, throughput, range_dict)
        else:
            slow_dic[url] = throughput[url]
            logging.info(f"Slow server throughput: {slow_dic[url]}")
            small_chunk = math.floor(fast_downloadTime * (slow_dic[url]))
            logging.info(f"Small chunk size: {small_chunk}")
            small_start_byte = start_byte
            small_end_byte = small_start_byte + small_chunk + 1
            if small_end_byte > file_size:
                small_end_byte = file_size
            if small_start_byte > file_size:
                break
            logging.info(f"small_start_byte, small_end_byte: {small_start_byte} and {small_end_byte}")
            data, throughput = await fetch_range(session, url, small_start_byte, small_end_byte, throughput, range_dict)

#async def main():
async def main(file_size, iteration, num_servers, folder_name):

    start_time = time.time()
    now = dt.datetime.now()
    formatted_now = now.strftime("%Y-%m-%d_%H-%M-%S")

    log_file = f"{folder_name}/Dynamic_Chunking_Async_{formatted_now}_{num_servers}_{iteration}_{file_size}.log"
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

    logging.info(f"Starting download with {num_servers} servers, file size {file_size} at {formatted_now}")

    session_dict = {url: aiohttp.ClientSession() for url in file_sources}
    filename = f"{folder_name}/output_file"
    throughput = {}
    range_dict = {}

    avg_speed = sum(random.uniform(5.0, 15.0) for _ in file_sources) / len(file_sources)
    num_servers = len(file_sources)
    initial_download_time = 10.0  # placeholder, can update later

    large_chunk = choose_chunk_size_and_learn(avg_speed, num_servers, initial_download_time)

    await asyncio.gather(
        *(download_file(session_dict[url], url, throughput, range_dict, large_chunk) for url in file_sources)
    )

    for session in session_dict.values():
        await session.close()
    
    logging.info("All chunks downloaded, writing to disk.")
    start_Disk_time = time.time()
    with open(filename, 'wb') as f:
        for start_byte in sorted(range_dict.keys()):
            f.write(range_dict[start_byte])
    end_Disk_time = time.time()

    Disk_delay = end_Disk_time - start_Disk_time
    end_time = time.time()
    delay = end_time - start_time

    print("Disk_delay:", Disk_delay)
    print("File size,{},Delay,{}".format(file_size, delay))
    logging.info(f"Delay {delay}")
    logging.info(f"Total download delay: {delay}")
    logging.info(f"Disk_delay: {Disk_delay}")

# =========
# MAIN CALL
# =========
if __name__ == "__main__":
    asyncio.run(main(file_size, iteration, num_servers, folder_name))

