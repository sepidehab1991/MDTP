# --- CONFIG (edit these for your run) ---
file_size_mb   = 4096.0     # 4 GB
iteration      = "1"
num_servers    = 3          # how many servers to simulate
folder_name    = "./sim_logs"
large_chunk_mb = 40.0       # large chunk for the fastest server each round (MB)

# Baseline speeds (MB/s) — MUST have exactly num_servers entries.
BASE_MBPS = [20.0, 1.5, 11.0]  # <-- edit this list to match num_servers
assert len(BASE_MBPS) == num_servers, (
    f"len(BASE_MBPS)={len(BASE_MBPS)} must equal num_servers={num_servers}"
)

# ============== SIMULATION ==============
import asyncio
import logging
import time
import datetime as dt
import math
import os
import random
import numpy as np

FILE_SIZE_BYTES   = int(file_size_mb * 1024 * 1024)
INITIAL_CHUNK     = 5 * 1024 * 1024
LARGE_CHUNK_BYTES = int(large_chunk_mb * 1024 * 1024)

# Exactly N servers with IDs S1..SN
servers = [f"S{i+1}" for i in range(num_servers)]
print("Sources:", servers)

# Convert MB/s → bytes/s
BASE_BPS = {srv: BASE_MBPS[i] * 1024 * 1024 for i, srv in enumerate(servers)}

def sample_speed_bps(srv: str) -> float:
    """Instantaneous speed with ~10% multiplicative noise + rare slowdowns."""
    base = BASE_BPS[srv]
    speed = base * random.lognormvariate(0.0, 0.1)
    if random.random() < 0.05:  # occasional contention
        speed *= 0.5
    return max(0.5 * 1024 * 1024, min(speed, 500 * 1024 * 1024))  # 0.5–500 MB/s

def extra_latency_s() -> float:
    """Small per-request latency/jitter (seconds)."""
    return max(0.0, random.uniform(0.005, 0.025) + random.gauss(0, 0.003))

def gmean(vals):
    vals = np.asarray([v for v in vals if v > 0], dtype=float)
    if len(vals) == 0:
        return 0.0
    return float(np.exp(np.mean(np.log(vals))))

# ---- Global mutable state for this run ----
next_start   = 0                        # next unassigned byte
throughput   = {srv: 0.0 for srv in servers}   # last measured B/s

async def simulate_fetch(srv: str, size_bytes: int):
    """Sleep for size/speed + latency; return (measured_throughput_Bps, delay_s)."""
    speed = sample_speed_bps(srv)
    delay = size_bytes / speed + extra_latency_s()
    await asyncio.sleep(delay)
    measured_tp = size_bytes / delay if delay > 0 else speed
    return measured_tp, delay

def plan_chunks_strict(throughput_now: dict, remaining_bytes: int):
    """
    STRICT policy you requested:
      - fast if tp >= GM  (no ceil, no fallback)
      - pick the single fastest among those fast servers
      - fastest gets LARGE_CHUNK_BYTES
      - everyone else gets floor(fast_time * tp_i),
        where fast_time = LARGE_CHUNK_BYTES / tp_fastest
      - if sum > remaining, scale down and give leftover to the fastest
    """
    srvs = list(throughput_now.keys())
    tps  = np.array([max(throughput_now[s], 1.0) for s in srvs], dtype=float)  # avoid zeros
    gm   = gmean(tps)

    fast_mask   = (tps >= gm)              # strict GM comparison
    masked_tps  = np.where(fast_mask, tps, -np.inf)
    fastest_idx = int(np.argmax(masked_tps))   # single fastest among the fast
    tp_fastest  = float(tps[fastest_idx])

    fast_time = LARGE_CHUNK_BYTES / tp_fastest if tp_fastest > 0 else LARGE_CHUNK_BYTES # fix  else LARGE_CHUNK_BYTES

    sizes = np.zeros(len(srvs), dtype=np.int64)
    for i, tp in enumerate(tps):
        if i == fastest_idx:
            sizes[i] = LARGE_CHUNK_BYTES
        else:
            sizes[i] = max(1, math.floor(fast_time * tp)) # fix this also

    total = int(sizes.sum())
    if total > remaining_bytes:
        scale = remaining_bytes / total
        sizes = np.floor(sizes * scale).astype(np.int64)
        # hand any leftover bytes to the fastest so we exactly fill the file
        deficit = remaining_bytes - int(sizes.sum())
        if deficit > 0:
            sizes[fastest_idx] += deficit
        sizes = np.maximum(sizes, 0)

    plan = {srvs[i]: int(sizes[i]) for i in range(len(srvs)) if sizes[i] > 0}
    return plan, srvs[fastest_idx], gm, tp_fastest

def allocate_ranges(plan: dict):
    """Assign sequential, non-overlapping byte ranges for each (srv -> size)."""
    global next_start
    assignments = {}
    for srv, size in plan.items():
        s_b = next_start
        e_b = min(FILE_SIZE_BYTES - 1, s_b + size - 1)
        size_eff = e_b - s_b + 1
        assignments[srv] = (s_b, e_b, size_eff)
        next_start = e_b + 1
    return assignments

async def fetch_round(assignments: dict):
    """Run all assigned fetches concurrently; update per-server throughput."""
    tasks, srvs = [], []
    for srv, (_s, _e, size) in assignments.items():
        srvs.append(srv)
        tasks.append(simulate_fetch(srv, size))
    results = await asyncio.gather(*tasks)
    delays = []
    for srv, (measured_tp, delay) in zip(srvs, results):
        throughput[srv] = measured_tp
        delays.append(delay)
    print("  per-server delays (s):", {srv: round(d, 3) for srv, d in zip(srvs, delays)},
      "| round_time≈", round(max(delays), 3))
    return max(delays) if delays else 0.0



async def main():
    """Run the full simulation once (no disk writing)."""
    # simple per-run logging to a file (optional)
    os.makedirs(folder_name, exist_ok=True)
    now = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(folder_name, f"Dynamic_Chunking_Async_SIM_{now}_{num_servers}_{iteration}_{int(file_size_mb)}MB.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

    global next_start
    next_start = 0  # reset if you rerun this cell
    for s in throughput: throughput[s] = 0.0

    total_start = time.time()
    remaining   = FILE_SIZE_BYTES

    # ---- initial 5MB probe (concurrent) ----
    probe_plan  = {srv: min(INITIAL_CHUNK, remaining // max(1, len(servers))) for srv in servers}
    probe_assign = allocate_ranges(probe_plan)
    probe_time   = await fetch_round(probe_assign)
    remaining    = FILE_SIZE_BYTES - next_start
    print(f"[Probe] next_start={next_start}, remaining={remaining/1_048_576:.2f} MB")

    # ---- main rounds ----
    step = 0
    while remaining > 0:
        step += 1
        plan, fastest_srv, gm, tp_fastest = plan_chunks_strict(throughput, remaining)
        assign     = allocate_ranges(plan)
        round_time = await fetch_round(assign)
        remaining  = FILE_SIZE_BYTES - next_start

        sizes_mb = {srv: round((assign[srv][2]) / 1_048_576, 2) for srv in assign}
        print(
            f"[Step {step:04}] fastest={fastest_srv}  "
            f"tp_fastest={tp_fastest/1e6:.2f} MB/s  GM={gm/1e6:.2f} MB/s  "
            f"assigned(MB)={sizes_mb}  round_time≈{round_time:.3f}s  "
            f"remaining={remaining/1_048_576:.2f} MB"
        )

    total_wall = time.time() - total_start
    print("they printed all")
    print(f"File size,{FILE_SIZE_BYTES},Delay,{total_wall}")
