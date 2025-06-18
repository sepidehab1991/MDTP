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

a3c_model = A3CChunkSelector()
optimizer = torch.optim.Adam(a3c_model.parameters(), lr=1e-4)
lock = asyncio.Lock()

# ================================
# 2. A3C Chunk Selection + Update
# ================================
async def choose_chunk_size_and_learn(avg_speed, download_time):
    state = torch.tensor([[avg_speed, download_time, download_time]], dtype=torch.float32)
    async with lock:
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
folder_name = sys.argv[3]

urls = [
    "http://10.129.138.2/",
    "http://10.141.8.2/",
    "http://10.135.141.2/"
]

file_sources = [i + file_size for i in urls]

response = requests.head(file_sources[0])
file_size = int(response.headers.get('Content-Length', 0))

global_position = 0
position_lock = asyncio.Lock()

async def fetch_range(session, url, s_b, e_b, throughput, range_dict):
    headers = {'Range': f"bytes={s_b}-{e_b}", 'Connection': 'keep-alive'}
    start_time = time.time()
    try:
        async with async_timeout.timeout(60):
            async with session.get(url, headers=headers) as response:
                data = await response.read()
                end_time = time.time()
                delay = end_time - start_time
                throughput[url] = (e_b - s_b + 1) / delay
                range_dict[s_b] = data
                logging.info(f"Fetched range {s_b}-{e_b} from {url} (delay {delay:.4f}s, throughput {throughput[url]:.2f}B/s)")
            return data, throughput
    except Exception as e:
        logging.info(f"Error fetching data from {url}: {e}")

async def download_file(session, url, throughput, range_dict):
    global global_position

    # Initial download for warm-up
    initial_chunk = 32 * 1024 * 1024
    async with position_lock:
        s_b = global_position
        e_b = min(global_position + initial_chunk - 1, file_size - 1)
        global_position = e_b + 1

    await fetch_range(session, url, s_b, e_b, throughput, range_dict)

    while True:
        speed = throughput.get(url, 1e6)
        avg_speed = speed / 1e6
        download_time = (e_b - s_b + 1) / speed

        next_chunk_size = await choose_chunk_size_and_learn(avg_speed, download_time)
        
        async with position_lock:
            s_b = global_position
            e_b = min(s_b + next_chunk_size - 1, file_size - 1)
            if s_b > file_size - 1:
                break
            global_position = e_b + 1

        await fetch_range(session, url, s_b, e_b, throughput, range_dict)

# ==================
# 4. MAIN Downloader
# ==================
async def main(file_size, iteration, folder_name):
    start_time = time.time()
    now = dt.datetime.now()
    formatted_now = now.strftime("%Y-%m-%d_%H-%M-%S")

    log_file = f"{folder_name}/RL_DynamicChunking_{formatted_now}_{iteration}_{file_size}.log"
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
    logging.info(f"Starting download at {formatted_now} for file size {file_size}")

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
    asyncio.run(main(file_size, iteration, folder_name))

