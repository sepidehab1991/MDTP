import asyncio
import aiohttp
import logging
import time
import datetime as dt
import requests
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

# ----------------------------
# CLI args
# ----------------------------
file_size_arg = sys.argv[1]
iteration = sys.argv[2]
num_servers = int(sys.argv[3])
folder_name = sys.argv[4]

# ----------------------------
# Model config - MUST MATCH TRAINING
# ----------------------------
MODEL_PATH = "/home/ubuntu/Test_Case/a3c_download_model_3.pth"  # From new_all_rl.py
ACTION_MIN_MB = 10.0
ACTION_MAX_MB = 128.0

# ----------------------------
# ActorCritic (3-action version)
# ----------------------------
class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99):
        super().__init__()
        self.gamma = gamma
        self.n_actions = n_actions

        # MLP
        self.fc1 = nn.Linear(input_dims, 128)
        self.fc2 = nn.Linear(128, 128)

        # Actor heads (Beta params)
        self.actor_alpha = nn.Linear(128, n_actions)
        self.actor_beta = nn.Linear(128, n_actions)
        
        # Critic
        self.critic = nn.Linear(128, 1)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        alpha = F.softplus(self.actor_alpha(x)) + 1.0
        beta = F.softplus(self.actor_beta(x)) + 1.0
        value = self.critic(x)
        return (alpha, beta), value

# ----------------------------
# Load model
# ----------------------------
def load_model(path: str, input_dims=3, n_actions=3):
    """Load trained 3-action A3C model"""
    try:
        model = ActorCritic(input_dims, n_actions)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        print(f"✓ Loaded model from {path}")
        return model
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None

# ----------------------------
# Predict 3 chunk sizes using trained model
# ----------------------------
def predict_chunk_sizes(model, server_speeds_mb: np.ndarray) -> np.ndarray:
    """
    Predict chunk sizes for all 3 servers.
    
    Args:
        model: Trained ActorCritic model
        server_speeds_mb: np.ndarray [3] of server speeds in MB/s
        
    Returns:
        chunk_sizes_mb: np.ndarray [3] of chunk sizes in MB
    """
    if model is None:
        # Fallback: proportional to speed
        total = server_speeds_mb.sum()
        if total > 0:
            return 40.0 * (server_speeds_mb / total) + 10.0
        return np.array([40.0, 40.0, 40.0])
    
    try:
        with torch.no_grad():
            # Normalize observation (important!)
            obs = torch.tensor(server_speeds_mb, dtype=torch.float32).unsqueeze(0)  # [1, 3]
            
            # Get Beta parameters for each server
            (alpha, beta), _ = model(obs)  # [1, 3]
            alpha, beta = alpha.squeeze(0), beta.squeeze(0)  # [3]
            
            # Sample from Beta distributions
            dist = Beta(alpha, beta)
            y = dist.sample().clamp(1e-6, 1 - 1e-6)  # [3] in (0,1)
            
            # Map to MB range
            chunk_sizes = ACTION_MIN_MB + y * (ACTION_MAX_MB - ACTION_MIN_MB)
            
            return chunk_sizes.cpu().numpy().astype(np.float32)
    except Exception as e:
        print(f"✗ Prediction error: {e}")
        return np.array([40.0, 40.0, 40.0])

# ----------------------------
# URLs / sources
# ----------------------------
urls = ["http://10.129.130.2/", "http://10.141.1.2/", "http://10.135.132.2/"]
file_sources = [url + file_size_arg for url in urls[:num_servers]]
print(f"File sources: {file_sources}")

response = requests.head(file_sources[0])
total_file_size = int(response.headers.get('Content-Length', 0))
print(f"Total file size: {total_file_size} bytes ({total_file_size/(1024**2):.1f} MB)")

# ----------------------------
# Shared state
# ----------------------------
download_state = {
    'remaining': total_file_size,
    'throughputs': {},  # url -> bytes/sec
    'start_byte': 0
}

# ----------------------------
# Fetch a chunk
# ----------------------------
async def fetch_chunk(session, url, start, end, state):
    """Download a single chunk from a server"""
    headers = {'Range': f'bytes={start}-{end}'}
    logging.info(f"Fetching {start}-{end} from {url}")
    
    start_time = time.time()
    try:
        async with session.get(url, headers=headers, timeout=500) as response:
            data = await response.read()
            elapsed = time.time() - start_time
            
            # Update throughput (bytes/sec)
            chunk_size = end - start + 1
            state['throughputs'][url] = chunk_size / max(elapsed, 1e-9)
            
            logging.info(f"✓ Downloaded {chunk_size} bytes in {elapsed:.2f}s "
                        f"({state['throughputs'][url]/(1024**2):.1f} MB/s)")
            return start, data
    except Exception as e:
        logging.error(f"✗ Error fetching from {url}: {e}")
        return start, b''

# ----------------------------
# Download file using RL policy
# ----------------------------
async def download_with_rl(sessions, model):
    """Main download loop using 3-action RL policy"""
    state = download_state
    range_dict = {}
    
    # Initial probe to measure speeds
    probe_size = 5 * 1024 * 1024  # 5 MB
    tasks = []
    for url in file_sources:
        end = min(state['start_byte'] + probe_size - 1, total_file_size - 1)
        tasks.append(fetch_chunk(sessions[url], url, state['start_byte'], end, state))
        state['start_byte'] += probe_size
    
    results = await asyncio.gather(*tasks)
    for start, data in results:
        if data:
            range_dict[start] = data
    
    # Main download loop
    while state['remaining'] > 0 and state['start_byte'] < total_file_size:
        # Get current throughputs in MB/s
        server_speeds = np.array([
            state['throughputs'].get(url, 1.0) / (1024**2)  # Convert to MB/s
            for url in file_sources
        ], dtype=np.float32)
        
        # Get RL policy's chunk size predictions (in MB)
        chunk_sizes_mb = predict_chunk_sizes(model, server_speeds)
        chunk_sizes_bytes = (chunk_sizes_mb * 1024 * 1024).astype(int)
        
        logging.info(f"Server speeds (MB/s): {server_speeds}")
        logging.info(f"RL policy chunks (MB): {chunk_sizes_mb}")
        
        # Cap by remaining file
        total_requested = chunk_sizes_bytes.sum()
        if total_requested > state['remaining']:
            scale = state['remaining'] / total_requested
            chunk_sizes_bytes = (chunk_sizes_bytes * scale).astype(int)
        
        # Fetch chunks in parallel
        tasks = []
        for i, url in enumerate(file_sources):
            if chunk_sizes_bytes[i] > 0:
                start = state['start_byte']
                end = min(start + chunk_sizes_bytes[i] - 1, total_file_size - 1)
                tasks.append(fetch_chunk(sessions[url], url, start, end, state))
                state['start_byte'] += chunk_sizes_bytes[i]
        
        results = await asyncio.gather(*tasks)
        for start, data in results:
            if data:
                range_dict[start] = data
        
        state['remaining'] -= chunk_sizes_bytes.sum()
        logging.info(f"Remaining: {state['remaining']/(1024**2):.1f} MB\n")
    
    return range_dict

# ----------------------------
# Main
# ----------------------------
async def main():
    # Setup logging
    now = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"{folder_name}/RL_3Action_{now}_{num_servers}_{iteration}.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )
    
    # Load RL model
    model = load_model(MODEL_PATH, input_dims=3, n_actions=3)
    
    # Create sessions
    sessions = {url: aiohttp.ClientSession() for url in file_sources}
    
    try:
        start_time = time.time()
        
        # Download using RL policy
        range_dict = await download_with_rl(sessions, model)
        
        # Write to disk
        filename = f"{folder_name}/output_file"
        logging.info("Writing to disk...")
        disk_start = time.time()
        
        with open(filename, 'wb') as f:
            for start in sorted(range_dict.keys()):
                f.write(range_dict[start])
        
        disk_time = time.time() - disk_start
        total_time = time.time() - start_time
        
        print(f"\n✓ Download complete!")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Disk write: {disk_time:.2f}s")
        print(f"  Throughput: {total_file_size/(total_time*1024**2):.2f} MB/s")
        
        logging.info(f"Total time: {total_time:.2f}s")
        logging.info(f"Disk time: {disk_time:.2f}s")
        
    finally:
        for session in sessions.values():
            await session.close()

if __name__ == "__main__":
    asyncio.run(main())
