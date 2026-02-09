# ============================================================================
# FULL RL APPROACH: RL predicts chunks for all servers
# ============================================================================

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
import math

# ----------------------------
# CLI args
# ----------------------------
file_size_arg = sys.argv[1]
iteration = sys.argv[2]
num_servers = int(sys.argv[3])
folder_name = sys.argv[4]

# ----------------------------
# INITIALIZE LOGGING IMMEDIATELY
# ----------------------------
timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = f"{folder_name}/FullRL_{timestamp}_{num_servers}_{iteration}.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logging.info("=== LOGGING INITIALIZED ===")

# ----------------------------
# Model config
# ----------------------------
ACTION_MIN_MB = 10.0
ACTION_MAX_MB = 128.0
PROBE_SIZE_MB = 5.0

# ----------------------------
# ActorCritic - IDENTICAL to Hybrid
# ----------------------------
class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99):
        super().__init__()
        self.gamma = gamma
        self.n_actions = n_actions
        self.fc1 = nn.Linear(input_dims, 128)
        self.fc2 = nn.Linear(128, 128)
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
# Load model
# ----------------------------
def load_model(path: str, input_dims=3, n_actions=3):
    try:
        model = ActorCritic(input_dims, n_actions)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        logging.info(f"✓ Loaded model: {path}")
        return model
    except Exception as e:
        logging.error(f"✗ Model load failed: {e}")
        return None

# ----------------------------
# URLs / File setup
# ----------------------------
#urls = ["http://10.135.132.2/", "http://10.141.1.2/", "http://10.132.132.2/"]
urls = [
    "http://10.135.132.2/",
    "http://10.141.1.2/",
    "http://10.132.132.2/",
    "http://10.147.130.2/",
    "http://10.133.2.2/",
    "http://10.137.3.2/"
]
file_sources = [url + file_size_arg for url in urls[:num_servers]]

response = requests.head(file_sources[0])
total_file_size = int(response.headers.get('Content-Length', 0))
logging.info(f"File: {file_size_arg}, Size: {total_file_size} bytes ({total_file_size/(1024**2):.1f} MB)")
print(f"File sources: {file_sources}")
print(f"Total file size: {total_file_size} bytes ({total_file_size/(1024**2):.1f} MB)")

# ----------------------------
# Shared state structure
# ----------------------------
class DownloadState:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.next_byte = 0
        self.throughputs = {url: 0.0 for url in file_sources}
        self.chunks_completed = 0
        
    async def allocate_range(self, size_bytes):
        """Thread-safe byte range allocation"""
        async with self.lock:
            if self.next_byte >= total_file_size:
                return None, None
            start = self.next_byte
            end = min(start + size_bytes - 1, total_file_size - 1)
            self.next_byte = end + 1
            return start, end
    
    def get_speeds_mbps(self):
        """Get current throughputs in MB/s"""
        return np.array([
            self.throughputs[url] / (1024**2)
            for url in file_sources
        ], dtype=np.float32)
    
    def get_speeds_bps(self):
        """Get current throughputs in bytes/s"""
        return np.array([
            self.throughputs[url]
            for url in file_sources
        ], dtype=np.float32)

# ----------------------------
# Fetch chunk (identical to Hybrid)
# ----------------------------
async def fetch_chunk(session, url, start, end, state: DownloadState):
    headers = {'Range': f'bytes={start}-{end}'}
    chunk_size = end - start + 1
    
    logging.info(f"[FETCH] {url} | bytes {start}-{end} ({chunk_size/(1024**2):.2f} MB)")
    
    start_time = time.time()
    try:
        async with session.get(url, headers=headers, timeout=500) as response:
            data = await response.read()
            elapsed = time.time() - start_time
            throughput_bps = len(data) / max(elapsed, 1e-9)
            
            async with state.lock:
                state.throughputs[url] = throughput_bps
                state.chunks_completed += 1
            
            logging.info(f"[DONE] {url} | {len(data)} bytes | {elapsed:.2f}s | "
                        f"{throughput_bps/(1024**2):.2f} MB/s")
            return start, data
            
    except Exception as e:
        logging.error(f"[ERROR] {url} | {start}-{end} | {e}")
        return start, b''

# ============================================================================
# FULL RL APPROACH: RL predicts chunks for ALL servers
# ============================================================================

MODEL_PATH = "/home/ubuntu/MyClient/Models/New_Reward_function_Full_RL.pth"

def predict_all_chunks_mb(model, server_speeds_mb: np.ndarray) -> np.ndarray:
    """Predict chunk sizes for all servers using RL"""
    if model is None:
        # Fallback: proportional allocation
        total = server_speeds_mb.sum()
        if total > 0:
            return 40.0 * (server_speeds_mb / total) + 10.0
        return np.full(len(server_speeds_mb), 40.0)
    
    try:
        with torch.no_grad():
            obs = torch.tensor(server_speeds_mb, dtype=torch.float32).unsqueeze(0)
            (alpha, beta), _ = model(obs)
            alpha, beta = alpha.squeeze(0), beta.squeeze(0)
            dist = Beta(alpha, beta)
            y = dist.sample().clamp(1e-6, 1 - 1e-6)
            chunks_mb = ACTION_MIN_MB + y * (ACTION_MAX_MB - ACTION_MIN_MB)
            return chunks_mb.cpu().numpy()
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return np.full(len(server_speeds_mb), 40.0)

async def download_file_full_rl(session, url, state: DownloadState, model, server_idx):
    """Each server runs independently"""
    
    logging.info(f"[SERVER {server_idx}] Starting download from {url}")
    
    results = []
    
    # Wait briefly for ALL servers to complete their initial probe
    await asyncio.sleep(0.5)
    
    # Continuous loop
    while True:
        speeds_mbps = state.get_speeds_mbps()
        speeds_bps = state.get_speeds_bps()
        
        # Check if we have valid speeds from ALL servers
        if np.all(speeds_bps > 0):
            # RL predicts for ALL servers
            chunks_mb = predict_all_chunks_mb(model, speeds_mbps)
            chunk_mb = chunks_mb[server_idx]
            
            # Convert to bytes
            chunk_bytes = int(chunk_mb * 1024 * 1024)
            
            logging.info(f"[SERVER {server_idx}] RL predicted: {chunk_mb:.1f} MB "
                       f"(all predictions: {chunks_mb})")
        else:
            # Still waiting for initial throughput data - use probe
            chunk_bytes = int(PROBE_SIZE_MB * 1024 * 1024)
            logging.info(f"[SERVER {server_idx}] Waiting for speed data, using probe: {PROBE_SIZE_MB:.1f} MB")
        
        # NO MINIMUM ENFORCEMENT (to match old code exactly)
        if chunk_bytes <= 0:
            logging.warning(f"[SERVER {server_idx}] Calculated chunk ≤ 0, using 1 MB minimum")
            chunk_bytes = 1024 * 1024  # Absolute minimum: 1 MB
        
        # Allocate next chunk
        start, end = await state.allocate_range(chunk_bytes)
        if start is None:
            logging.info(f"[SERVER {server_idx}] No more bytes to allocate, exiting")
            break
        
        result = await fetch_chunk(session, url, start, end, state)
        results.append(result)
    
    logging.info(f"[SERVER {server_idx}] Completed {len(results)} chunks")
    return results


async def main():
    logging.info("=== FULL RL APPROACH START ===")
    
    model = load_model(MODEL_PATH, input_dims=3, n_actions=3)  # 3 actions
    sessions = {url: aiohttp.ClientSession() for url in file_sources}
    state = DownloadState()
    
    try:
        start_time = time.time()
        
        # ===== INITIAL PROBE PHASE =====
        probe_bytes = int(PROBE_SIZE_MB * 1024 * 1024)
        probe_tasks = []
        
        logging.info(f"=== PROBE PHASE: {PROBE_SIZE_MB} MB per server ===")
        
        for i, url in enumerate(file_sources):
            start, end = await state.allocate_range(probe_bytes)
            if start is not None:
                logging.info(f"[PROBE] Server {i} ({url}): bytes {start}-{end}")
                task = asyncio.create_task(
                    fetch_chunk(sessions[url], url, start, end, state)
                )
                probe_tasks.append((i, task))
        
        # Wait for all probes to complete
        probe_results = []
        for i, task in probe_tasks:
            try:
                result = await task
                probe_results.append(result)
                logging.info(f"[PROBE] Server {i} completed")
            except Exception as e:
                logging.error(f"[PROBE] Server {i} failed: {e}")
        
        speeds = state.get_speeds_mbps()
        logging.info(f"Probe complete | Speeds: {speeds} MB/s")
        
        # ===== MAIN DOWNLOAD PHASE =====
        logging.info(f"\n=== MAIN DOWNLOAD PHASE ===")
        logging.info(f"Launching {len(file_sources)} download workers")
        
        worker_results = await asyncio.gather(
            *(download_file_full_rl(sessions[url], url, state, model, i) 
              for i, url in enumerate(file_sources)),
            return_exceptions=True
        )
        
        logging.info(f"All workers completed, processing results")
        
        # Flatten results from probe + workers
        range_dict = {}
        total_chunks = 0
        
        # Add probe results
        for result in probe_results:
            if result and len(result) == 2:
                start, data = result
                if data and len(data) > 0:
                    range_dict[start] = data
                    total_chunks += 1
        
        # Add worker results
        for i, server_results in enumerate(worker_results):
            if isinstance(server_results, Exception):
                logging.error(f"Server {i} failed with exception: {server_results}")
                continue
            
            logging.info(f"Server {i} returned {len(server_results)} chunks")
            
            for start, data in server_results:
                if data and len(data) > 0:
                    range_dict[start] = data
                    total_chunks += 1
        
        logging.info(f"Total chunks collected: {total_chunks}")
        
        # Calculate average chunk size
        total_bytes = sum(len(data) for data in range_dict.values())
        avg_chunk_size = total_bytes / max(total_chunks, 1) / (1024**2)
        
        # Write file
        filename = f"{folder_name}/output_fullrl_{iteration}"
        logging.info(f"Writing to {filename}")
        
        total_bytes_written = 0
        with open(filename, 'wb') as f:
            for start in sorted(range_dict.keys()):
                data = range_dict[start]
                f.write(data)
                total_bytes_written += len(data)
        
        total_time = time.time() - start_time
        
        logging.info(f"\n=== RESULTS ===")
        logging.info(f"Total chunks: {total_chunks}")
        logging.info(f"Average chunk size: {avg_chunk_size:.2f} MB")
        logging.info(f"Total bytes: {total_bytes} ({total_bytes/(1024**2):.1f} MB)")
        logging.info(f"Expected bytes: {total_file_size} ({total_file_size/(1024**2):.1f} MB)")
        logging.info(f"Total time: {total_time:.2f}s")
        logging.info(f"Throughput: {total_bytes/(total_time*1024**2):.2f} MB/s")
        logging.info("=== FULL RL APPROACH END ===")
        
        print(f"✓ Full RL complete: {total_time:.2f}s, {total_chunks} chunks, "
              f"avg {avg_chunk_size:.1f} MB/chunk")
        
    except Exception as e:
        logging.error(f"FATAL ERROR: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise
        
    finally:
        for session in sessions.values():
            await session.close()

if __name__ == "__main__":
    asyncio.run(main())
