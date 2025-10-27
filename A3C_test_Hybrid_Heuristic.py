# ===== ac_test_module_async.py =====
# Pure test code: load saved actor (TorchScript or state_dict); asynchronous event-driven scheduler.
# No training. No rewards. No per-round barrier. Stop immediately when FILE_SIZE_MB is reached.

import os
import heapq
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- CONFIG ----------------
N_SERVERS = 3
c_min, c_max = 10.0, 128.0             # Min/max chunk size (MB)
FILE_SIZE_MB = 4096.0                  # total file size (MB)
MODEL_PATH = "./hybrid_model.pth"  # Accepts TorchScript .pt/.pth or state_dict .pth
rel_std = 0.12
# ------------- ENVIRONMENT -------------
def simulate_server_speeds():
    rng = np.random.default_rng()  # new unseeded RNG each call
    base = np.asarray([50.0, 11.5, 28.0], dtype=float)
    sigma = np.sqrt(np.log1p(rel_std**2))
    factors = rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma, size=base.shape)
    speeds = base * factors
    return speeds

def measure_throughput(speeds, chunk_sizes):
    """
    Given speeds (MB/s) and chunk_sizes (MB), return:
    - times (s) to finish those chunks
    - measured throughputs (MB/s) with extra noise & clipping
    """
    speeds = np.maximum(np.asarray(speeds, dtype=float), 1e-9)
    chunk_sizes = np.asarray(chunk_sizes, dtype=float)
    times = chunk_sizes / speeds
    throughputs = chunk_sizes / np.maximum(times, 1e-9)
    throughputs += np.random.normal(0, 0.1 * throughputs, size=3)
    throughputs = np.clip(throughputs, 0.1, None)
    return times, throughputs

def scale_action(x):
    """Map scalar action in [-1, 1] to chunk size in [c_min, c_max]."""
    x = max(-1.0, min(1.0, float(x)))
    return c_min + 0.5 * (x + 1.0) * (c_max - c_min)

# --- geometric-mean based fastest selection ---
def fastest_idx(cur_throughputs: np.ndarray) -> int:
    """
    Classify fast/slow via global geometric mean; pick fastest among 'fast'.
    Fallback to overall max if none exceed the geometric mean.
    """
    x = np.maximum(np.asarray(cur_throughputs, dtype=float), 1e-9)
    return int(np.argmax(x))

# ---------------- MODEL (for state_dict loads; supports variable #heads) ----------------
class ACModel(nn.Module):
    """
    Training-time head:
      - Input obs: [last_time (3), last_thr (3)] => 6-D
      - Actor outputs Beta params with 'n_heads' heads (1 or 3)
      - Critic scalar value
    """
    def __init__(self, n_heads: int = N_SERVERS):
        super().__init__()
        self.n_heads = int(n_heads)
        obs_dim = 2 * N_SERVERS
        hid = 128
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
        )
        self.actor_alpha = nn.Linear(hid, self.n_heads)
        self.actor_beta  = nn.Linear(hid, self.n_heads)
        self.v = nn.Linear(hid, 1)

    def forward(self, obs: torch.Tensor):
        x = self.net(obs)
        alpha = F.softplus(self.actor_alpha(x)) + 2.0
        beta  = F.softplus(self.actor_beta(x)) + 2.0
        v = self.v(x)
        return (alpha, beta), v

# ---------------- LOADER (TorchScript -> pickled nn.Module -> state_dict) ----------------
def _infer_heads_from_state(state_dict) -> int:
    for k in ("actor_alpha.bias", "actor_beta.bias", "actor_alpha.weight", "actor_beta.weight"):
        if k in state_dict and isinstance(state_dict[k], torch.Tensor):
            return int(state_dict[k].shape[0])
    return N_SERVERS

def load_actor(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file '{path}' not found.")

    # 1) Try TorchScript
    try:
        actor = torch.jit.load(path, map_location="cpu")
        actor.eval()
        # Attach n_heads hint if present; otherwise leave it to plan_sizes logic
        actor.n_heads = getattr(actor, "n_heads", N_SERVERS)
        return actor
    except Exception:
        pass

    # 2) Try pickled nn.Module or state_dict
    try:
        obj = torch.load(path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Could not load model from '{path}': {e}")

    if isinstance(obj, nn.Module):
        obj.eval()
        obj.n_heads = getattr(obj, "n_heads", N_SERVERS)
        return obj

    # 3) Assume it's a plain state_dict
    state = obj
    n_heads = _infer_heads_from_state(state)
    actor = ACModel(n_heads=n_heads)
    missing, unexpected = actor.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARN] load_state_dict: missing={missing}, unexpected={unexpected}")
    actor.eval()
    return actor

# ----------------- TEST RUN (ASYNC) -----------------
def run_test():
    actor = load_actor(MODEL_PATH)

    remaining = float(FILE_SIZE_MB)
    current_time = 0.0
    event_count = 0

    # Stats
    chunks_per_server = [0, 0, 0]
    mb_per_server = [0.0, 0.0, 0.0]

    # Initial probe to get first throughputs (and initial last_times)
    speeds = simulate_server_speeds()
    probe_chunks = np.array([10.0, 10.0, 10.0], dtype=float)
    probe_times, throughputs = measure_throughput(speeds, probe_chunks)
    last_times = probe_times.copy()  # (s) last observed durations per server

    # ---------- Helpers ----------
    def plan_sizes(cur_speeds, cur_throughputs, last_times_arr):
        """
        Actor -> propose c1 for fastest server; match others to same finish time.
        Input to actor matches training: obs = [last_time (3), last_thr (3)].
        Returns per-server chunk sizes (MB).
        """
        # Build 6-D observation (float32): [last_times, cur_throughputs]
        obs_vec = np.concatenate([last_times_arr, cur_throughputs], axis=0).astype(np.float32)
        inp = torch.from_numpy(obs_vec).unsqueeze(0)  # shape (1, 6)

        with torch.no_grad():
            out = actor(inp)

        # Pick fastest by geometric mean rule
        fastest = fastest_idx(cur_throughputs)

        # Interpret outputs robustly
        try:
            # Preferred ACModel-like output: ((alpha, beta), v)
            (alpha, beta), _ = out
            alpha = torch.as_tensor(alpha)
            beta  = torch.as_tensor(beta)
            n_heads = int(alpha.shape[1]) if alpha.dim() == 2 else 1

            if n_heads == 1:
                a = alpha[0, 0].clamp(1e-6, 1e6)
                b = beta [0, 0].clamp(1e-6, 1e6)
            else:
                a = alpha[0, fastest].clamp(1e-6, 1e6)
                b = beta [0, fastest].clamp(1e-6, 1e6)

            y = float((a / (a + b)).item())           # Beta mean ∈ (0,1)
            c1 = float(c_min + y * (c_max - c_min))   # map (0,1) -> [c_min, c_max]
        except Exception:
            # Fallback: scalar/vector in [-1,1]
            a_scalar = float(torch.as_tensor(out).view(-1)[0])
            c1 = float(np.clip(scale_action(a_scalar), c_min, c_max))

        # Derive other servers' chunks so they aim to finish with the fastest
        v_fast = float(cur_speeds[fastest])
        t_fast = c1 / max(v_fast, 1e-9)
        sizes = np.array(
            [c1 if j == fastest else max(t_fast, 1e-9) * cur_throughputs[j]
             for j in range(3)],
            dtype=float,
        )
        return sizes

    def assign_and_push(i, size, now, cur_speeds):
        """Record the new in-flight size for server i and push its finish event."""
        inflight_sizes[i] = float(size)
        last_chunk_sizes[i] = float(size)
        dur = float(size) / max(cur_speeds[i], 1e-9)
        last_times[i] = dur  # keep last observed time (s)
        heapq.heappush(pq, (now + dur, i))
        # stats
        chunks_per_server[i] += 1
        mb_per_server[i] += float(size)

    # ---------- Initial assignment ----------
    init_sizes = plan_sizes(speeds, throughputs, last_times)
    total_init = float(init_sizes.sum())
    if total_init > remaining:
        init_sizes *= remaining / max(total_init, 1e-9)

    pq = []
    inflight_sizes = init_sizes.copy()
    last_chunk_sizes = init_sizes.copy()
    for i in range(3):
        assign_and_push(i, init_sizes[i], current_time, speeds)

    print("=== TEST RUN (ASYNC) START ===")
    print(f"Init | chunks={np.round(inflight_sizes, 2)} MB | speeds={np.round(speeds, 2)} MB/s")

    # ---------- Event-driven loop ----------
    while remaining > 1e-9 and pq:
        finish_at, i = heapq.heappop(pq)
        finished_size = inflight_sizes[i]
        current_time = float(finish_at)
        event_count += 1

        remaining -= finished_size
        print(
            f"Evt {event_count:3d} @ {current_time:9.3f}s | S{i+1} finished {finished_size:.2f} MB "
            f"| remaining={max(remaining,0):.2f} MB"
        )
        if remaining <= 1e-9:
            break

        # Update env + measured throughputs using *last* sizes
        speeds = simulate_server_speeds()
        _, throughputs = measure_throughput(speeds, last_chunk_sizes)

        # Recompute last_times from last_chunk_sizes and new speeds
        last_times[:] = last_chunk_sizes / np.maximum(speeds, 1e-9)

        # ONE call: compute all sizes, but schedule only server i (others remain in-flight)
        sizes = plan_sizes(speeds, throughputs, last_times)
        next_size = float(sizes[i])
        # Clamp final piece so remaining never goes negative:
        next_size = min(next_size, remaining)

        assign_and_push(i, next_size, current_time, speeds)

    print("\n=== TEST COMPLETED ===")
    print(f"File size: {FILE_SIZE_MB} MB")
    print(f"Total events (completions): {event_count}")
    print(f"Total time:  {current_time:.3f} s")   # true async completion time (no barriers)
    for i in range(3):
        print(f"Server {i+1}: {chunks_per_server[i]} chunks, {mb_per_server[i]:.2f} MB")

if __name__ == "__main__":
    run_test()
