import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

# ---------------- CONFIG ----------------
c_min, c_max = 1.0, 128.0
CHUNK_CHOICES_MB = torch.tensor([8.0, 16.0, 32.0, 64.0])  # (not used directly, kept for reference)
lr = 1e-4                           # Learning rate
gamma = 0.99                        # Discount factor
max_episodes = 2000                 # Number of episodes

# File size + safety cap
FILE_SIZE_MB = 4096                 # e.g., 4 GB = 4096 MB
MAX_STEPS_PER_EPISODE = 10_000      # safety cap

# Reward balance knob
ALPHA = 0.6                         # weight on imbalance penalty (tune as needed)

# ------------- ENVIRONMENT -------------
def simulate_server_speeds():
    """Simulate three server speeds with noise (acts like MB/s)."""
    base = np.array([50.0, 11.5, 28.0])               # S1 > S2 > S3
    noise = np.random.normal(0, 0.1, size=3)          # small variation
    # Fix: use a realistic upper cap (was 5.0 which crippled speeds)
    return np.clip(base + noise, 0.5, 200.0)

def measure_throughput(speeds, chunk_sizes):
    """
    Measure throughput as chunk_size / download time.
    Here: time = size / speed, throughput ~ speed with noise.
    """
    times = chunk_sizes / speeds
    throughputs = chunk_sizes / times                      # = speeds ideally
    throughputs += np.random.normal(0, 0.1 * throughputs, size=3)  # 10% noise
    throughputs = np.clip(throughputs, 0.1, None)
    return times, throughputs

def scale_action(x):
    """Scale agent output [-1,1] to [c_min,c_max]. Works with tensor or float."""
    return c_min + 0.5 * (x + 1.0) * (c_max - c_min)

# ----------- ACTOR-CRITIC NET ----------
class ACModel(nn.Module):
    """A3C-style Actor-Critic (shared torso, Gaussian policy)."""
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )
        self.mu = nn.Linear(64, 1)     # mean of action (after tanh -> [-1,1])
        self.value = nn.Linear(64, 1)  # state value

    def forward(self, x):
        x = self.shared(x)
        mu = torch.tanh(self.mu(x))
        value = self.value(x)
        return mu, value

# ---------- PLOTTING HELPER (no logic change) ----------
def render_plots(episode_rewards, variance_log, c1_log, c2_log, c3_log, episode, every=5):
    if episode % every != 0 and episode != max_episodes - 1:
        return
    clear_output(wait=True)
    plt.figure(figsize=(18, 8))

    # Reward plot
    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards, label="Avg Reward/Step")
    if len(episode_rewards) > 10:
        sm = np.convolve(episode_rewards, np.ones(10)/10, mode='valid')
        plt.plot(range(9, 9+len(sm)), sm, label="Smoothed")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward Progress")
    plt.grid(True); plt.legend()

    # Variance/imbalance proxy plot (we keep logging time variance)
    plt.subplot(2, 2, 2)
    plt.plot(variance_log, label="Finish Time Variance")
    plt.xlabel("Steps")
    plt.ylabel("Variance")
    plt.title("Variance Reduction")
    plt.grid(True); plt.legend()

    # Combined chunk sizes plot
    plt.subplot(2, 2, 3)
    plt.plot(c1_log, label="Chunk S1 (Fastest)")
    plt.plot(c2_log, label="Chunk S2")
    plt.plot(c3_log, label="Chunk S3")
    plt.xlabel("Steps")
    plt.ylabel("Chunk Size (MB)")
    plt.title("Chunk Sizes per Server")
    plt.grid(True); plt.legend()

    # Separate chunk size graph
    plt.subplot(2, 2, 4)
    plt.plot(c1_log, label="Server 1")
    plt.plot(c2_log, label="Server 2")
    plt.plot(c3_log, label="Server 3")
    plt.xlabel("Steps")
    plt.ylabel("Chunk Size (MB)")
    plt.title("Chunk Sizes Over Time")
    plt.grid(True); plt.legend()

    plt.tight_layout()
    plt.show()
    plt.pause(0.001)

# --------------- TRAINING ---------------
def train_agent(global_model, optimizer):
    local_model = ACModel()
    local_model.load_state_dict(global_model.state_dict())

    episode_rewards = []
    variance_log = []
    c1_log, c2_log, c3_log = [], [], []

    try:
        for episode in range(max_episodes):
            log_probs, values, rewards = [], [], []
            total_episode_reward = 0.0

            # Reset file + time
            remaining = float(FILE_SIZE_MB)  # MB left
            total_time = 0.0                 # seconds
            steps_taken = 0

            # Initial probe: observe state = (throughputs of a small probe)
            speeds = simulate_server_speeds()
            probe_chunks = np.array([4.0, 4.0, 4.0])
            _, throughputs = measure_throughput(speeds, probe_chunks)
            state = torch.FloatTensor(throughputs)

            # Run until file is fully downloaded (or safety cap)
            while remaining > 1e-9 and steps_taken < MAX_STEPS_PER_EPISODE:
                mu, value = local_model(state)
                dist = torch.distributions.Normal(mu, 0.1)  # fixed std
                action = dist.sample()
                log_prob = dist.log_prob(action).sum()      # scalar

                # Agent decides c1 (fastest server chunk, MB)
                c1 = float(scale_action(action).item())

                # Compute chunk sizes proportional to throughput ratios (fastest baseline)
                fastest_idx = int(np.argmax(throughputs))
                ratios = throughputs / throughputs[fastest_idx]
                c2 = c1 * float(ratios[1])
                c3 = c1 * float(ratios[2])
                chunk_sizes = np.array([c1, c2, c3], dtype=float)  # MB

                # If overshooting remaining, scale down proportionally
                total_chunk = float(chunk_sizes.sum())
                if total_chunk > remaining:
                    scale = remaining / total_chunk
                    chunk_sizes *= scale

                # Per-server times for this step (epoch of 3 chunks)
                finish_times = chunk_sizes / speeds                # seconds
                step_time = float(np.max(finish_times))            # we wait for all

                # ----- NEW REWARD (per step) -----
                # throughput over the step (MB/s)
                step_throughput = total_chunk / max(step_time, 1e-9)
                # imbalance = end-spread (seconds)
                imbalance = float(np.max(finish_times) - np.min(finish_times))
                # reward = throughput - ALPHA * imbalance
                reward_val = step_throughput - ALPHA * imbalance
                reward = torch.tensor([reward_val], dtype=torch.float32)

                # (Optional) still log variance as a proxy to see dispersion trending
                variance = np.var(finish_times)

                # Logs
                c1_log.append(chunk_sizes[0]); c2_log.append(chunk_sizes[1]); c3_log.append(chunk_sizes[2])
                variance_log.append(variance)

                # Store for A3C
                log_probs.append(log_prob)
                values.append(value.squeeze(-1))                   # make scalar shape
                rewards.append(reward)
                total_episode_reward += float(reward_val)

                # Advance environment
                remaining -= total_chunk
                total_time += step_time
                steps_taken += 1

                # Next observation: new speeds + small probe to estimate throughputs
                speeds = simulate_server_speeds()
                probe_chunks = np.array([4.0, 4.0, 4.0])
                _, throughputs = measure_throughput(speeds, probe_chunks)
                state = torch.FloatTensor(throughputs)

            # ----- A3C UPDATE -----
            with torch.no_grad():
                _, next_value = local_model(state)

            # Compute returns
            R = next_value.detach()
            returns = []
            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)
            returns = torch.cat(returns) if returns else torch.zeros(1)

            values = torch.stack(values).unsqueeze(-1) if values else torch.zeros_like(returns)
            log_probs = torch.stack(log_probs) if log_probs else torch.zeros(1)

            advantage = returns - values                      # [T,1]
            actor_loss = -(log_probs * advantage.detach().squeeze(-1)).mean()
            critic_loss = advantage.pow(2).mean()
            total_loss = actor_loss + 0.5 * critic_loss

            optimizer.zero_grad()
            total_loss.backward()
            # Push local grads to global
            for gp, lp in zip(global_model.parameters(), local_model.parameters()):
                if lp.grad is not None:
                    gp.grad = lp.grad.clone()
            optimizer.step()
            local_model.load_state_dict(global_model.state_dict())

            # Track & report
            avg_reward = total_episode_reward / max(1, steps_taken)
            print(f"=== Episode {episode} | steps {steps_taken} | "
                  f"avg reward/step {avg_reward:.4f} | total time {total_time:.3f}s ===")
            episode_rewards.append(avg_reward)

            # Draw/update plots
            render_plots(episode_rewards, variance_log, c1_log, c2_log, c3_log, episode, every=5)

    except KeyboardInterrupt:
        print("\n Interrupted by user — rendering final plots before exit...")
        render_plots(episode_rewards, variance_log, c1_log, c2_log, c3_log, episode=max_episodes-1, every=1)

# --------------- RUN --------------------
if __name__ == "__main__":
    global_model = ACModel()
    optimizer = optim.Adam(global_model.parameters(), lr=lr)
    train_agent(global_model, optimizer)
