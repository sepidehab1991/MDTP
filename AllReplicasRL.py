# =========================
# A3C with per-server state = [last_time, last_thr]
# Per-chunk reward + mid-episode updates AFTER each set of N_SERVERS completions
# (No autograd graph kept across optimizer steps)
# =========================
import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output

# ----------------------------
# Config
# ----------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

N_SERVERS = 3
CHUNK_CHOICES_MB = torch.tensor([8, 16, 32, 64])  # discrete actions per server
K = len(CHUNK_CHOICES_MB)

BASE_SPEEDS = np.array([50.0, 29.0, 11.2])   # MB/s
SPEED_NOISE_STD = 0.15

EPISODES = 3000           # total episodes across all workers
LR = 1e-3
ALPHA = 0.0               # (legacy) not used in the new reward, kept for compatibility
SLEEP_SCALE = 0.01
PRINT_EVERY = 100
LIVE_PLOT_EVERY = 5
SMOOTH_WINDOW = 20

# Target file size per episode (e.g., 4 GB)
FILE_SIZE_MB = 4096.0

# A3C knobs
WORKERS = 3               # you can tie to N_SERVERS if you like
ENTROPY_BETA = 0.01
GRAD_CLIP = 5.0

# -------- NEW: Reward shaping weights --------
THR_W = 0.05      # weight for instantaneous throughput (MB/s)
SYNC_W = 1.0      # weight for reducing std of server busy-times
FINAL_W = 1.0     # extra bonus near the end to align finishes
NEAR_END_MB = float(CHUNK_CHOICES_MB.max() * 4)  # "near end" threshold

# ----------------------------
# Async utils (script + notebook safe)
# ----------------------------
def run_coro(coro):
    """Run a coroutine in scripts and notebooks safely."""
    try:
        loop = asyncio.get_running_loop()
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except Exception:
            pass
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)

# ----------------------------
# Async fetch simulation
# ----------------------------
async def fetch_one(name: str, speed_mb_s: float, chunk_mb: float):
    # duration = size / speed (scaled for fast sim)
    duration_s = max(1e-6, chunk_mb / max(1e-6, speed_mb_s))
    await asyncio.sleep(duration_s * SLEEP_SCALE)
    return {"server": name, "speed": speed_mb_s, "chunk": chunk_mb, "sim_time": duration_s}

# ----------------------------
# Environment helpers
# ----------------------------
def sample_speeds():
    noise = np.random.normal(0.0, SPEED_NOISE_STD, size=N_SERVERS)
    return np.clip(BASE_SPEEDS + noise, 0.3, None)

def reward_from_totals(total_mb, server_elapsed):
    """Kept for reporting/eval; training uses per-chunk shaped reward now."""
    #server_elapsed = array of how long each server has been busy.
    #if len(server_elapsed) else 1.0 = if the array is empty (edge case), use 1.0 instead so later math won’t divide by zero.
    round_time = float(np.max(server_elapsed)) if len(server_elapsed) else 1.0
    throughput = float(total_mb) / max(round_time, 1e-6)
    imbalance = float(np.std(server_elapsed)) if len(server_elapsed) > 1 else 0.0
    reward = throughput - ALPHA * imbalance
    return reward, throughput, imbalance, round_time

# ----------------------------
# Build observation = [last_time_norm..., last_thr_norm...]
# ----------------------------
def build_observation(last_time, last_thr):
    """
    last_time: np.array [N_SERVERS] seconds of last finished chunk per server (0 if none yet)
    last_thr : np.array [N_SERVERS] MB/s of last finished chunk per server (0 if none yet)
    Returns: torch.FloatTensor [2*N_SERVERS] normalized per-block by its own max.
    """
    t = last_time.astype(float)
    r = last_thr.astype(float)
    t_norm = t / (np.max(t) + 1e-6) if np.any(t > 0.0) else t
    r_norm = r / (np.max(r) + 1e-6) if np.any(r > 0.0) else r
    obs = np.concatenate([t_norm, r_norm], axis=0).astype(np.float32)
    return torch.from_numpy(obs)

# ----------------------------
# Actor-Critic network (A3C) — input is 2*N_SERVERS
# ----------------------------
class ActorCritic(nn.Module):
    def __init__(self, n_servers=3, k=4):
        super().__init__()
        in_dim = 2 * n_servers  # last_time + last_thr
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.actor = nn.Linear(64, n_servers * k)  # logits per server/action
        self.critic = nn.Linear(64, 1)             # V(s)
        self.n_servers = n_servers
        self.k = k

    def forward(self, obs_tensor):
        # obs_tensor shape: [2*n_servers]
        x = self.shared(obs_tensor)
        logits = self.actor(x).view(self.n_servers, self.k)  # (servers, actions)
        value = self.critic(x).squeeze(-1)                   # scalar
        return logits, value

# ----------------------------
# Global training state (for live plots)
# ----------------------------
reward_history = []
variance_history = []
chunk_history = []
throughput_history = []
episodes_done = 0

# ----------------------------
# Per-chunk shaped reward helper  (NEW)
# ----------------------------
def step_reward(prev_elapsed, new_elapsed, chunk_mb, sim_time, remaining_after):
    """
    prev_elapsed/new_elapsed: np.array of server busy-times before/after this completion
    chunk_mb: size of the completed chunk
    sim_time: duration of this chunk
    remaining_after: MB remaining after scheduling this completion
    """
    # 1) throughput term
    r_thr = float(chunk_mb) / max(sim_time, 1e-6)

    # 2) synchrony improvement: reward if std decreased
    prev_std = float(np.std(prev_elapsed)) if prev_elapsed.size > 1 else 0.0
    new_std  = float(np.std(new_elapsed))  if new_elapsed.size  > 1 else 0.0
    r_sync = (prev_std - new_std)

    # 3) near-end alignment bonus (encourage tight finishes)
    align_bonus = 0.0
    if remaining_after <= NEAR_END_MB and new_elapsed.size > 0:
        mean_t = float(np.mean(new_elapsed)) + 1e-6
        align = max(0.0, 1.0 - (new_std / mean_t))
        align_bonus = align

    return THR_W * r_thr + SYNC_W * r_sync + FINAL_W * align_bonus

# ----------------------------
# Pipelined episode with per-block updates
# After each set of N_SERVERS completions:
#   - recompute log-probs from stored (obs, server_idx, action_idx)
#   - compute block loss
#   - optimizer.step()
#   - sync local from global
# ----------------------------
async def run_pipelined_episode_by_size(
    speeds, local_model, global_model, optimizer, opt_lock, file_size_mb: float
):
    remaining = float(file_size_mb)
    server_elapsed = np.zeros(N_SERVERS, dtype=float)
    total_mb = 0.0

    per_server_chunks = [[] for _ in range(N_SERVERS)]

    # per-server last metrics (start at 0 = no history yet)
    last_time = np.zeros(N_SERVERS, dtype=float)
    last_thr  = np.zeros(N_SERVERS, dtype=float)

    # episode stats
    step_rewards = []

    # async bookkeeping
    active = {}            # server_idx -> task
    # task -> (server_idx, actual_chunk_MB, obs_at_choice_detached, action_idx_int)
    task_to_info = {}

    # ---- current block buffers ----
    block_steps = []     # list of (obs_detached, server_i, action_idx)
    block_rewards = []   # scalars (floats)
    since_update = 0
    obs_block_start = build_observation(last_time, last_thr).detach()

    def pick_action_for_server(i):
        """
        Build current observation from last_time/last_thr and get logits from the model.
        Sample an action for server i. Return (action_idx, obs_detached, proposed_chunk_MB)
        """
        obs = build_observation(last_time, last_thr)
        logits, _ = local_model(obs)        # forward build graph, but we won't keep it
        dist = Categorical(logits=logits[i])
        a = dist.sample()
        proposed_chunk = float(CHUNK_CHOICES_MB[a])
        return int(a.item()), obs.detach(), proposed_chunk  # store obs detached (no graph kept)

    def schedule_for_server(i):
        nonlocal remaining
        if remaining <= 1e-9:
            return False

        a, obs_det, proposed_chunk = pick_action_for_server(i)

        actual_chunk = min(proposed_chunk, remaining)
        if actual_chunk <= 0.0:
            return False
        remaining -= actual_chunk
        per_server_chunks[i].append(actual_chunk)

        # launch async job and remember obs & action (not log-probs with graph!)
        t = asyncio.create_task(fetch_one(f"S{i+1}", float(speeds[i]), actual_chunk))
        active[i] = t
        task_to_info[t] = (i, actual_chunk, obs_det, a)
        return True

    # helper: do one block update (uses current local_model parameters)
    def do_block_update():
        nonlocal block_steps, block_rewards, obs_block_start
        if since_update != N_SERVERS or len(block_steps) != N_SERVERS:
            return  # safety

        # Recompute log-probs and entropies fresh from stored (obs, action)
        sum_log_prob_blk = 0.0
        sum_entropy_blk = 0.0
        for (obs_det, srv_i, a_idx) in block_steps:
            logits, _ = local_model(obs_det)
            dist = Categorical(logits=logits[srv_i])
            # convert a_idx to tensor on same device
            a_t = torch.tensor(a_idx, dtype=torch.long, device=logits.device)
            sum_log_prob_blk = (sum_log_prob_blk + dist.log_prob(a_t))
            sum_entropy_blk   = (sum_entropy_blk + dist.entropy())

        shaped_R_blk = float(np.sum(block_rewards))

        # critic baseline at the block start
        _, value_blk = local_model(obs_block_start)
        advantage_blk = torch.tensor(shaped_R_blk, dtype=torch.float32, device=value_blk.device) - value_blk

        policy_loss = -(sum_log_prob_blk * advantage_blk.detach())
        value_loss  = 0.5 * advantage_blk.pow(2)
        total_loss  = policy_loss + value_loss - ENTROPY_BETA * sum_entropy_blk

        optimizer.zero_grad(set_to_none=True)
        local_model.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), GRAD_CLIP)

        # push local grads -> global, step, then sync back
        # (keeps behavior A3C-style and avoids in-place graph issues)
        async def _apply():
            async with opt_lock:
                for gp, lp in zip(global_model.parameters(), local_model.parameters()):
                    if lp.grad is not None:
                        gp.grad = lp.grad.clone()
                optimizer.step()
                local_model.load_state_dict(global_model.state_dict())
        # run the tiny async section synchronously here
        run_coro(_apply())

        # reset block buffers and set new block start state
        block_steps.clear()
        block_rewards.clear()

    # seed: start up to one initial job per server
    for i in range(N_SERVERS):
        if remaining <= 1e-9:
            break
        schedule_for_server(i)

    # event loop: whoever finishes gets the next chunk immediately
    #keep looping while there are active downloads
    while active:
        #done = the set of tasks (one or more) that just completed.
        done, _ = await asyncio.wait(active.values(), return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            #i = which server index finished
            #chunk_mb = the chunk size that server was downloading,
            #obs_det = the observation used when choosing that action (detached tensor),
            #a_idx = the action index (which chunk-size option was chosen).
            i, chunk_mb, obs_det, a_idx = task_to_info.pop(t)
            #this server is no longer busy; remove it from the active set.
            del active[i]
            res = t.result()
            sim_t = float(res["sim_time"])

            # --- compute shaped reward using elapsed BEFORE and AFTER this completion ---
            prev_elapsed = server_elapsed.copy()
            after_elapsed = server_elapsed.copy()
            after_elapsed[i] += sim_t

            # record busy time and totals
            server_elapsed[i] += sim_t
            total_mb += float(chunk_mb)

            # update last metrics for that server
            last_time[i] = sim_t
            last_thr[i]  = float(chunk_mb) / max(sim_t, 1e-6)

            # compute per-step reward and store
            r_step = step_reward(prev_elapsed, after_elapsed, chunk_mb, sim_t, remaining)
            step_rewards.append(r_step)

            # accumulate into current block
            block_steps.append((obs_det, i, a_idx))
            block_rewards.append(r_step)
            since_update += 1

            # if a full set (N_SERVERS) completed, update policy NOW
            if since_update == N_SERVERS:
                do_block_update()
                since_update = 0
                obs_block_start = build_observation(last_time, last_thr).detach()

            # immediately schedule next chunk for this server if any remains
            schedule_for_server(i)

    mean_chunks  = [float(np.mean(ch)) if len(ch) else 0.0 for ch in per_server_chunks]
    total_shaped_return = float(np.sum(step_rewards)) if step_rewards else 0.0

    return total_mb, server_elapsed, mean_chunks, total_shaped_return

# ----------------------------
# A3C worker coroutine
# ----------------------------
async def a3c_worker(worker_id, n_episodes, global_model, optimizer, opt_lock, plot_lock):
    global episodes_done
    local_model = ActorCritic(N_SERVERS, K)
    local_model.load_state_dict(global_model.state_dict())

    for _ in range(n_episodes):
        speeds = sample_speeds()

        # Run a full pipelined episode with in-episode per-block updates
        total_mb, server_elapsed, mean_chunks, shaped_R = \
            await run_pipelined_episode_by_size(
                speeds, local_model, global_model, optimizer, opt_lock, FILE_SIZE_MB
            )

        # Legacy metrics for visibility/plots
        R_report, throughput, imbalance, round_time = reward_from_totals(total_mb, server_elapsed)

        # Log NEW shaped reward for training progress
        reward_history.append(float(shaped_R))
        variance_history.append(float(np.var(server_elapsed)))
        throughput_history.append(float(throughput))
        chunk_history.append(mean_chunks)

        if (worker_id == 0) and (episodes_done <= 3 or episodes_done % PRINT_EVERY == 0):
            sizes_str = ", ".join(f"S{i+1}:~{mean_chunks[i]:.1f}MB" for i in range(N_SERVERS))
            print(f"[Ep {episodes_done:4d}] R_shaped={shaped_R:.3f}  thr={throughput:.2f}MB/s  "
                  f"imb={imbalance:.3f}s  roundT={round_time:.2f}s | avg chunks [{sizes_str}]")

        async with plot_lock:
            episodes_done += 1
            if (worker_id == 0) and (episodes_done % LIVE_PLOT_EVERY == 0 or episodes_done == EPISODES):
                clear_output(wait=True)
                fig = plt.figure(figsize=(18, 8))

                # Reward (now NEW shaped reward)
                ax1 = plt.subplot(2, 2, 1)
                ax1.plot(reward_history, label="Shaped reward / episode")
                if len(reward_history) >= SMOOTH_WINDOW:
                    kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
                    smooth = np.convolve(reward_history, kernel, mode="valid")
                    ax1.plot(range(SMOOTH_WINDOW-1, SMOOTH_WINDOW-1+len(smooth)),
                             smooth, label=f"Smoothed ({SMOOTH_WINDOW})")
                ax1.set_title("Reward Progress (Shaped)"); ax1.set_xlabel("Episode"); ax1.set_ylabel("Reward")
                ax1.grid(True); ax1.legend()

                # Variance
                ax2 = plt.subplot(2, 2, 2)
                ax2.plot(variance_history, label="Server Busy-Time Variance")
                ax2.set_title("Timing Variance"); ax2.set_xlabel("Episode"); ax2.set_ylabel("Variance (s²)")
                ax2.grid(True); ax2.legend()

                # Mean chunk size per server
                ax3 = plt.subplot(2, 1, 2)
                ch = np.array(chunk_history)
                for i in range(N_SERVERS):
                    ax3.plot(ch[:, i], label=f"S{i+1}")
                ax3.set_title("Mean Chosen Chunk Size per Server")
                ax3.set_xlabel("Episode"); ax3.set_ylabel("Chunk Size (MB)")
                ax3.grid(True); ax3.legend()

                plt.tight_layout()
                plt.show()

# ----------------------------
# Training entry (spawns workers)
# ----------------------------
def train():
    global reward_history, variance_history, chunk_history, throughput_history, episodes_done
    reward_history, variance_history, chunk_history, throughput_history = [], [], [], []
    episodes_done = 0

    global_model = ActorCritic(N_SERVERS, K)
    optimizer = optim.Adam(global_model.parameters(), lr=LR)

    async def run_workers():
        opt_lock = asyncio.Lock()
        plot_lock = asyncio.Lock()

        base = EPISODES // WORKERS
        rem = EPISODES % WORKERS
        per_worker = [base + (1 if i < rem else 0) for i in range(WORKERS)]

        tasks = [
            asyncio.create_task(a3c_worker(i, per_worker[i], global_model, optimizer, opt_lock, plot_lock))
            for i in range(WORKERS)
        ]
        await asyncio.gather(*tasks)

    run_coro(run_workers())
    return global_model, reward_history, variance_history, chunk_history, throughput_history

# ---- RUN TRAINING (notebook) ----
model, rewards, variances, chunks, thr = train()
print("Training done.")

# ----------------------------
# OPTIONAL: save checkpoint for test
# ----------------------------
torch.save({
    "state_dict": model.state_dict(),
    "n_servers": N_SERVERS,
    "k": K,
    "chunk_choices_mb": CHUNK_CHOICES_MB.tolist(),
    "file_size_mb": FILE_SIZE_MB,
    "base_speeds": BASE_SPEEDS.tolist(),
    "speed_noise_std": SPEED_NOISE_STD,
    # store the shaping knobs too
    "thr_w": THR_W,
    "sync_w": SYNC_W,
    "final_w": FINAL_W,
    "near_end_mb": NEAR_END_MB,
}, "a3c_multiserver_last_time_thr.pt")
print("Saved to a3c_multiserver_last_time_thr.pt")

# ----------------------------
# Greedy evaluation helper (uses the same per-decision state)
# (Note: 'reward' in this report is the legacy throughput-imbalance metric for clarity)
# ----------------------------
def eval_once_with_speeds(model, speeds_array, file_size_mb=FILE_SIZE_MB):
    speeds = np.asarray(speeds_array, dtype=float)
    assert speeds.shape[0] == N_SERVERS, f"Expected {N_SERVERS} speeds, got {speeds.shape[0]}"

    # Simple eval path (no training updates here)
    async def _eval():
        # Reuse the same episode machinery but without optimizer steps:
        # We'll use a dummy optimizer & lock and skip block updates by setting N_SERVERS=0 logic
        # Instead, we implement a minimal eval episode here:
        remaining = float(file_size_mb)
        server_elapsed = np.zeros(N_SERVERS, dtype=float)
        last_time = np.zeros(N_SERVERS, dtype=float)
        last_thr = np.zeros(N_SERVERS, dtype=float)
        total_mb = 0.0

        active = {}
        def pick_and_schedule(i):
            nonlocal remaining, total_mb
            if remaining <= 1e-9: return False
            obs = build_observation(last_time, last_thr)
            logits, _ = model(obs)
            a = torch.argmax(logits[i]).item()  # greedy for eval
            chunk_mb = float(min(CHUNK_CHOICES_MB[a], remaining))
            if chunk_mb <= 0: return False
            remaining -= chunk_mb
            t = asyncio.create_task(fetch_one(f"S{i+1}", float(speeds[i]), chunk_mb))
            active[i] = (t, chunk_mb)
            return True

        for i in range(N_SERVERS):
            pick_and_schedule(i)
        while active:
            done, _ = await asyncio.wait([v[0] for v in active.values()], return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                i = [k for k, v in active.items() if v[0] is t][0]
                chunk_mb = active[i][1]
                del active[i]
                sim_t = float(t.result()["sim_time"])
                server_elapsed[i] += sim_t
                total_mb += chunk_mb
                last_time[i] = sim_t
                last_thr[i] = chunk_mb / max(sim_t, 1e-6)
                pick_and_schedule(i)

        return total_mb, server_elapsed

    total_mb, server_elapsed = run_coro(_eval())
    R_legacy, thr, imb, roundT = reward_from_totals(total_mb, server_elapsed)
    return {
        "throughput": thr,
        "makespan_s": roundT,
        "imbalance_std": imb,
        "reward": R_legacy,          # legacy metric for inspection
        "speeds": speeds.tolist(),
    }

# Example quick test:
test_speeds = np.array([50.0, 29.0, 11.2])
report = eval_once_with_speeds(model, test_speeds)
print("Greedy test report:", report)

