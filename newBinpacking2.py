import asyncio
import aiohttp
import logging
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
    def __init__(self, input_dim=3, action_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.policy = nn.Linear(128, action_dim)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        probs = F.softmax(self.policy(x), dim=-1)
        value = self.value(x)
        return probs, value

# Global shared model and optimizer
a3c_model = A3CChunkSelector()
optimizer = torch.optim.Adam(a3c_model.parameters(), lr=1e-4)
lock = asyncio.Lock()  # For safe updates

# ================================
# 2. A3C Chunk Selection + Update
# ================================
async def choose_chunk_size_and_learn(avg_speed, num_servers, download_time):
    state = torch.tensor([[avg_speed, num_servers, download_time]], dtype=torch.float32)
    async with lock:  # Safe update
        probs, value = a3c_model(state)
        dist = torch.distributions.Categorical(probs)

        epsilon = 0.1
        if random.random() < epsilon:
            action = torch.randint(0, 4, (1,))
        else:
            action = dist.sample()

        selected_chunk_size = [32, 64, 128, 256][action.item()] * 1024 * 1024

        reward = -download_time
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

start_byte_global = 0
end_byte_global = 0
throughput_gm = 0
number_of_chunk = 0

async def fetch_range(session, url, s_b, e_b, throughput, range_dict):
    global start_byte_global, end_byte_global, number_of_chunk
    local_start_byte = s_b
    local_end_byte = e_b
    start_byte_global += e_b - s_b + 1
    end_byte_global = e_b
    headers = {'Range': f"bytes={local_start_byte}-{local_end_byte}", 'Connection': 'keep-alive'}
    start_time = time.time()
    try:
        async with async_timeout.timeout(60):
            async with session.get(url, headers=headers) as response:
                data = await response.read()
                end_time = time.time()
                delay = end_time - start_time
                throughput[url] = (local_end_byte - local_start_byte) / delay
                range_dict[local_start_byte] = data
                number_of_chunk += 1
                logging.info(f"Fetched range {local_start_byte}-{local_end_byte} from {url} (delay {delay:.4f}s, throughput {throughput[url]:.2f}B/s)")
            return data, throughput
    except Exception as e:
        logging.info(f"Error fetching data from {url}: {e}")

async def download_file(session, url, throughput, range_dict):
    global throughput_gm
    initial_chunk = 32 * 1024 * 1024  # Start bigger
    global start_byte_global, end_byte_global

    # Download initial chunk
    end_byte_global = start_byte_global + initial_chunk - 1
    data, throughput = await fetch_range(session, url, start_byte_global, end_byte_global, throughput, range_dict)

    fast_dic = {}
    slow_dic = {}

    while end_byte_global < file_size and start_byte_global < file_size:
        values = list(throughput.values())
        throughput_gm = gmean(values)

        avg_speed = throughput[url] / 1e6  # MB/s
        current_download_time = (end_byte_global - start_byte_global) / throughput[url]
        large_chunk = await choose_chunk_size_and_learn(avg_speed, num_servers, current_download_time)

        logging.info(f"Worker {url}: Selected large chunk size: {large_chunk/(1024*1024)} MB")

        if math.ceil(throughput[url]) >= math.ceil(throughput_gm):
            fast_dic[url] = throughput[url]
            fastest_server = max(fast_dic.values())
            fast_downloadTime = large_chunk / fastest_server
            large_start_byte = start_byte_global
            large_end_byte = large_start_byte + large_chunk + 1
            if large_end_byte > file_size:
                large_end_byte = file_size
            if large_start_byte > file_size:
                break
            data, throughput = await fetch_range(session, url, large_start_byte, large_end_byte, throughput, range_dict)
        else:
            slow_dic[url] = throughput[url]
            fast_downloadTime = large_chunk / max(fast_dic.values()) if fast_dic else 1
            small_chunk = math.floor(fast_downloadTime * (slow_dic[url]))
            small_start_byte = start_byte_global
            small_end_byte = small_start_byte + small_chunk + 1
            if small_end_byte > file_size:
                small_end_byte = file_size
            if small_start_byte > file_size:
                break
            data, throughput = await fetch_range(session, url, small_start_byte, small_end_byte, throughput, range_dict)

# ==================
# 4. MAIN Downloader
# ==================
async def main(file_size, iteration, num_servers, folder_name):
    start_time = time.time()
    now = dt.datetime.now()
    formatted_now = now.strftime("%Y-%m-%d_%H-%M-%S")

    log_file = f"{folder_name}/Dynamic_Chunking_Async_A3C_Workers_{formatted_now}_{num_servers}_{iteration}_{file_size}.log"
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

    logging.info(f"Starting download with {num_servers} servers, file size {file_size} at {formatted_now}")

    session_dict = {url: aiohttp.ClientSession() for url in file_sources}
    filename = f"{folder_name}/output_file"
    throughput = {}
    range_dict = {}

    await asyncio.gather(
        *(download_file(session_dict[url], url, throughput, range_dict) for url in file_sources)
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
    logging.info(f"Total download delay: {delay}")
    logging.info(f"Disk writing delay: {Disk_delay}")

# =========
# 5. MAIN CALL
# =========
if __name__ == "__main__":
    asyncio.run(main(file_size, iteration, num_servers, folder_name))

