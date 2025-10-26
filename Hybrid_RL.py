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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

# ----------------------------
# CLI args
# ----------------------------
file_size = sys.argv[1]
iteration = sys.argv[2]
num_servers = int(sys.argv[3])
folder_name = sys.argv[4]

# ----------------------------
# Model / scaling config
# ----------------------------
MODEL_PATH = "hybrid_model.pth"  # Match the saved model name
C_MIN_MB = 10.0
C_MAX_MB = 128.0

# ----------------------------
# Define ActorCritic architecture (must match training)
# ----------------------------
class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99):
        super().__init__()
        self.gamma = gamma

        # MLP
        self.fc1 = nn.Linear(input_dims, 128)
        self.fc2 = nn.Linear(128, 128)

        # Actor heads (Beta params) and Critic
        self.actor_alpha = nn.Linear(128, n_actions)
        self.actor_beta = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        alpha = F.softplus(self.actor_alpha(x)) + 1.0
        beta = F.softplus(self.actor_beta(x)) + 1.0
        value = self.critic(x)
        return (alpha, beta), value

# ----------------------------
# Load trained model
# ----------------------------
def load_trained_model(path: str, input_dims=3, n_actions=1):
    """Load the trained A3C model"""
    try:
        model = ActorCritic(input_dims, n_actions)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        print(f"Successfully loaded model from {path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# ----------------------------
# Predict action using trained model
# ----------------------------
def predict_large_chunk(model, server_speeds: np.ndarray) -> float:
    """
    Predict large chunk size in MB using the trained model.
    
    Args:
        model: Trained ActorCritic model
        server_speeds: np.ndarray of shape [n_servers] with current speeds
        
    Returns:
        large_chunk_mb: float, chunk size in MB
    """
    if model is None:
        # Fallback to default
        return 40.0
    
    try:
        with torch.no_grad():
            # Prepare observation (server speeds)
            obs = torch.tensor(server_speeds, dtype=torch.float32).unsqueeze(0)  # [1, 3]
            
            # Get Beta distribution parameters
            (alpha, beta), _ = model(obs)  # [1, 1] each
            
            # Sample from Beta distribution
            alpha, beta = alpha.squeeze(0), beta.squeeze(0)  # [1]
            dist = Beta(alpha, beta)
            y = dist.sample().clamp(1e-6, 1 - 1e-6)  # Sample in (0, 1)
            
            # Map to MB range
            large_chunk_mb = C_MIN_MB + y.item() * (C_MAX_MB - C_MIN_MB)
            
            return float(large_chunk_mb)
    except Exception as e:
        print(f"Error predicting action: {e}")
        return 40.0  # Fallback

# ----------------------------
# URLs / sources
# ----------------------------
def get_file_sources(urls, num_sources):
    if num_sources > len(urls):
        raise ValueError("Number of sources exceeds the number of available URLs.")
    return urls[:num_sources]

urls = ["http://10.129.130.2/", "http://10.141.1.2/", "http://10.135.132.2/"]
file_ips = get_file_sources(urls, num_servers)
file_sources = [i + file_size for i in file_ips]
print(file_sources)

response = requests.head(file_sources[0])
file_size = int(response.headers.get('Content-Length', 0))
print("File size:", file_size)

# ----------------------------
# Shared globals
# ----------------------------
start_byte = 0
end_byte = 0
currentRequestedByte = 0
throughput_gm = 0
number_of_chunk = 0
fast_downloadTime = 0

INV_MIB = 1.0 / (1024.0**2)

# ----------------------------
# Networking helpers
# ----------------------------
async def fetch_range(session, url, s_b, e_b, throughput, range_dict):
    logging.info(f"Fetching range for URL: {url}, start_byte={s_b}, end_byte={e_b}")
    global start_byte, end_byte, number_of_chunk
    local_start_byte = s_b
    local_end_byte = e_b
    start_byte += e_b - s_b + 1
    end_byte = e_b
    logging.info(f"Fetching range::::: {local_start_byte}-{local_end_byte} from {url}")
    headers = {'Range': f"bytes={local_start_byte}-{local_end_byte}"}
    start_time = time.time()
    try:
        async with async_timeout.timeout(500):
            async with session.get(url, headers=headers) as response:
                data = await response.read()
                end_time = time.time()
                delay = end_time - start_time
                logging.info(f"Delay {delay}")
                throughput[url] = (local_end_byte - local_start_byte) / max(delay, 1e-9)
                range_dict[local_start_byte] = data
                logging.info(f"Successfully fetched range {local_start_byte}-{local_end_byte} from {url}")
                logging.info(f"Throughput: {throughput}")
                number_of_chunk += 1
                logging.info(f"number_of_chunk: {number_of_chunk}")
            return data, throughput
    except Exception as e:
        logging.error(f"Error fetching data from {url}: {e}")

async def download_file(session, url, throughput, range_dict, model):
    logging.info(f"Downloading file from URL: {url}")
    global fast_downloadTime
    initial_chunk = 5 * 1024 * 1024  # 5MB probe
    global throughput_gm
    global end_byte, start_byte

    # Initial probe
    end_byte = min(file_size - 1, start_byte + initial_chunk - 1)
    data, throughput = await fetch_range(session, url, start_byte, end_byte, throughput, range_dict)

    fast_dic = {}
    slow_dic = {}

    while end_byte < file_size and start_byte < file_size:
        values = list(throughput.values()) if throughput else []
        if not values:
            end_byte = min(file_size - 1, start_byte + initial_chunk - 1)
            await fetch_range(session, url, start_byte, end_byte, throughput, range_dict)
            continue

        fastest_server = max(values)

        # ----------------------------
        # Fast server path
        # ----------------------------
        if throughput.get(url, 0.0) == fastest_server:
            fast_dic[url] = throughput[url]
            fastest_server = max(fast_dic.values())
            logging.info(f"Fast server throughput: {fastest_server}")

            # Build server speed observation for the model
            # Create a vector of current throughputs for all servers
            server_speeds = np.array([
                throughput.get(file_sources[0], 0.0),
                throughput.get(file_sources[1], 0.0) if len(file_sources) > 1 else 0.0,
                throughput.get(file_sources[2], 0.0) if len(file_sources) > 2 else 0.0
            ], dtype=np.float32)

            server_speeds_MBps = server_speeds * INV_MIB 
            logging.info(f"Speeds in MBps: {server_speeds_MBps}")
            # Predict large chunk size using the model
            large_chunk_mb = predict_large_chunk(model, server_speeds_MBps)
            large_chunk = int(large_chunk_mb * 1024 * 1024)  # Convert to bytes

            fast_downloadTime = large_chunk / max(fastest_server, 1e-9)
            logging.info(f"Model predicted large_chunk: {large_chunk_mb:.1f} MB ({large_chunk} bytes)")
            logging.info(f"Fast download time: {fast_downloadTime}")

            large_start_byte = start_byte
            large_end_byte = large_start_byte + large_chunk
            if large_end_byte > file_size:
                large_end_byte = file_size
            if large_start_byte >= file_size:
                break
            logging.info(f"Large_start_byte, Large_end_byte: {large_start_byte} and {large_end_byte}")
            data, throughput = await fetch_range(session, url, large_start_byte, large_end_byte, throughput, range_dict)

        # ----------------------------
        # Slow server path
        # ----------------------------
        if throughput.get(url, 0.0) < fastest_server:
            slow_dic[url] = throughput[url]
            logging.info(f"Slow server throughput: {slow_dic[url]}")
            small_chunk = math.floor(fast_downloadTime * slow_dic[url])
            logging.info(f"Small chunk size: {small_chunk}")
            small_start_byte = start_byte
            small_end_byte = small_start_byte + small_chunk
            if small_end_byte > file_size:
                small_end_byte = file_size
            if small_start_byte >= file_size:
                break
            logging.info(f"small_start_byte, small_end_byte: {small_start_byte} and {small_end_byte}")
            data, throughput = await fetch_range(session, url, small_start_byte, small_end_byte, throughput, range_dict)

async def main():
    # Load trained model
    model = load_trained_model(MODEL_PATH, input_dims=3, n_actions=1)

    start_time = time.time()
    now = dt.datetime.now()
    formatted_now = now.strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"{folder_name}/RL_Dynamic_Chunking_{formatted_now}_{num_servers}_{iteration}_{file_size}.log"
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

    session_dict = {url: aiohttp.ClientSession() for url in file_sources}
    filename = f"{folder_name}/output_file"
    throughput = {}
    range_dict = {}

    await asyncio.gather(
        *(download_file(session_dict[url], url, throughput, range_dict, model) for url in file_sources)
    )
    
    for session in session_dict.values():
        await session.close()

    logging.info("All chunks downloaded, writing to disk.")
    #start_Disk_time = time.time()
    #with open(filename, 'wb') as f:
    #    for s_b in sorted(range_dict.keys()):
    #        data = range_dict[s_b]
    #        f.write(data)
    end_time = time.time()
    
    delay = end_time - start_time
    #end_Disk_time = time.time()
    #Disk_delay = end_Disk_time - start_Disk_time
    
    print(f"Total delay: {delay:.2f}s")
    #print(f"Disk delay: {Disk_delay:.2f}s")
    logging.info(f"Total download delay: {delay}")
    #logging.info(f"Disk_delay: {Disk_delay}")

if __name__ == "__main__":
    asyncio.run(main())
