# -*- coding: utf-8 -*-
"""A3C for Optimized File Downloads (3 actions)

Each step the policy outputs 3 independent Beta actions y∈(0,1),
mapped to 3 chunk sizes in MB (one per server). The environment
consumes the 3-vector, simulates per-server times, and returns a
reward that penalizes both the fastest round time and inter-server
lateness:  reward = - (t_fast + sum_i max(0, t_i - t_fast))^2
"""

import os
import time
import threading
from collections import deque

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Beta

import matplotlib.pyplot as plt
from IPython.display import clear_output, display

# -------------------------------------------------
# Constants for action space (per server)
# -------------------------------------------------
ACTION_MIN_MB = 10.0
ACTION_MAX_MB = 128.0


# -------------------------------------------------
# Environment
# -------------------------------------------------
class DownloadSimEnv:
    def __init__(self,
                 file_size=4096.0,
                 initial_chunks=(10.0, 10.0, 10.0),
                 base_speed=(50.0, 11.5, 28.0),
                 lambda_delay=1e-3):
        # Static config
        self.file_size = float(file_size)
        self.initial_chunks = np.asarray(initial_chunks, dtype=float)
        self.base_speed = np.asarray(base_speed, dtype=float)
        self.lambda_delay = float(lambda_delay)
        self.n = len(self.base_speed)

        # Runtime state
        self.remaining = None
        self.server_elapsed = None
        self.speeds = None
        self.reward = 0.0
        self.done = False

    # --------------- Noise model ---------------
    def jitter_lognormal(self, base, rel_std=0.20):
        rng = np.random.default_rng()
        base = np.asarray(base, dtype=float)
        sigma = np.sqrt(np.log1p(rel_std**2))
        factors = rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma, size=base.shape)
        return base * factors

    def fetch_data(self, chunk_mb, speed_mb_s):
        chunk_mb = np.asarray(chunk_mb, dtype=float)
        speed_mb_s = np.asarray(speed_mb_s, dtype=float)
        return chunk_mb / np.maximum(speed_mb_s, 1e-9)

    # --------------- Reset ---------------
    def reset(self, *, seed=None, options=None):
        self.speeds = self.base_speed.copy()
        self.remaining = float(self.file_size)
        self.server_elapsed = np.zeros(self.n, dtype=float)
        self.reward = 0.0
        self.done = False
        obs = self.speeds.astype(np.float32)
        info = {}
        return obs, info

    # --------------- Step ---------------
    def step(self, action):
        """
        action: np.ndarray shape [n] of per-server chunk sizes in MB
        """
        # action must be a vector of length n
        action = np.asarray(action, dtype=float).reshape(-1)
        assert action.size == self.n, f"Expected {self.n} actions, got {action.size}"

        # Copy and clamp to nonnegative
        chunk_sizes = np.maximum(action, 0.0).astype(float)  # [n]
        print("chunk_sizes????????",chunk_sizes)
        remaining = float(self.remaining)

        # Sample current speeds with jitter
        self.speeds = self.jitter_lognormal(self.base_speed, rel_std=0.12)

        # Cap by remaining file size (scale all proportionally if over-allocated)
        total_c = float(np.sum(chunk_sizes))
        if total_c > remaining and total_c > 0:
            scale = remaining / total_c
            chunk_sizes *= scale

        # Per-server times this round
        round_dt = self.fetch_data(chunk_sizes, self.speeds)  # [n]
        server_elapsed = np.zeros(self.n, dtype=float)
        server_elapsed += round_dt

        # Fastest finisher this round
        fastest_server_time_id = int(np.argmin(round_dt))
        remaining -= float(np.sum(chunk_sizes))

        # Reward: -(t_fast + delays)^2, delays = sum_i max(0, t_i - t_fast)
        t_fast = float(round_dt[fastest_server_time_id])
        delays = float(np.sum(np.maximum(round_dt - t_fast, 0.0)))
        reward = -((t_fast + delays) ** 2)

        # Final allocation if near end (optional but safe)
        # Use min positive chunk to detect "small last step"
        min_pos = float(np.min(np.where(chunk_sizes > 1e-9, chunk_sizes, np.inf)))
        if (remaining > 0.0) and (remaining < (min_pos if np.isfinite(min_pos) else np.inf)):
            final_chunks = np.zeros(self.n, dtype=float)
            final_chunks[fastest_server_time_id] = remaining
            final_dt = self.fetch_data(final_chunks, self.speeds)
            server_elapsed += final_dt

            t_fast_f = float(final_dt[fastest_server_time_id])
            delays_f = float(np.sum(np.maximum(final_dt - t_fast_f, 0.0)))
            # Keep the ^2 form for consistency
            reward = -((t_fast_f + self.lambda_delay * delays_f) ** 2)
            remaining = 0.0

        total_simulated_seconds = float(np.max(server_elapsed)) if server_elapsed.size else 0.0
        done = remaining <= 0.0

        # Write back
        self.remaining = remaining
        self.server_elapsed = server_elapsed
        self.reward = reward
        self.done = done

        obs = self.speeds.astype(np.float32)
        info = {
            "chunks": chunk_sizes.tolist(),           # fixed key
            "round_time": round_dt.tolist(),
            "fastest_server_id": fastest_server_time_id,
            "total_time": total_simulated_seconds,
            "remaining": remaining
        }
        return obs, float(reward), bool(done), info


def env_factory():
    return DownloadSimEnv(
        file_size=4096.0,
        initial_chunks=(10.0, 10.0, 10.0),
        base_speed=(50.0, 11.5, 28.0),
        lambda_delay=1e-3
    )


# -------------------------------------------------
# Shared Adam (compatible with multiprocessing)
# -------------------------------------------------
class SharedAdam(T.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0, amsgrad=False):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        # Initialize optimizer state in shared memory (step must be a tensor)
        for group in self.param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue
                state = self.state[p]
                state['step'] = T.zeros(1)
                state['exp_avg'] = T.zeros_like(p.data, memory_format=T.preserve_format)
                state['exp_avg_sq'] = T.zeros_like(p.data, memory_format=T.preserve_format)
                if amsgrad:
                    state['max_exp_avg_sq'] = T.zeros_like(p.data, memory_format=T.preserve_format)

                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
                if amsgrad:
                    state['max_exp_avg_sq'].share_memory_()


# -------------------------------------------------
# Actor-Critic (3-action Beta policy)
# -------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99):
        super().__init__()
        self.gamma = gamma
        self.n_actions = n_actions

        # Backbone
        self.fc1 = nn.Linear(input_dims, 128)
        self.fc2 = nn.Linear(128, 128)

        # Actor: independent Beta heads per action dim
        self.actor_alpha = nn.Linear(128, n_actions)
        self.actor_beta = nn.Linear(128, n_actions)

        # Critic
        self.critic = nn.Linear(128, 1)

        # Rollout buffers
        self.rewards = []
        self.actions = []  # store y vectors in (0,1), shape [n_actions]
        self.states = []   # obs vectors

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        alpha = F.softplus(self.actor_alpha(x)) + 1.0
        beta = F.softplus(self.actor_beta(x)) + 1.0
        value = self.critic(x)
        return (alpha, beta), value

    # -------- Memory helpers --------
    def remember(self, state, action_vec, reward):
        # state: np.ndarray [D], action_vec: np.ndarray [n_actions]
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(np.asarray(action_vec, dtype=np.float32))
        self.rewards.append(float(reward))

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []

    # -------- Returns / Loss --------
    def calc_R(self, done):
        # Robust when no steps were gathered
        if not self.states:
            return T.zeros(1, 1)

        states = T.tensor(self.states, dtype=T.float32)
        _, v = self.forward(states)  # [T,1]
        # Bootstrap with last value if not done
        R = v[-1] * (0 if done else 1)
        batch_return = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            batch_return.append(R)
        batch_return.reverse()
        return T.stack(batch_return)  # [T,1]

    def calc_loss(self, done):
        states = T.tensor(self.states, dtype=T.float32)           # [T, D]
        actions = T.tensor(self.actions, dtype=T.float32)         # [T, n_actions], y in (0,1)

        returns = self.calc_R(done).squeeze(-1)                   # [T]
        (alpha, beta), values = self.forward(states)              # alpha,beta: [T,n_actions]; values: [T,1]
        values = values.squeeze(-1)                               # [T]

        dist = Beta(alpha, beta)                                  # independent Betas
        log_probs = dist.log_prob(actions).sum(dim=1)             # [T], sum over action dims
        entropy = dist.entropy().sum(dim=1).mean()                # scalar (encourage exploration)

        advantages = returns - values                             # [T]
        actor_loss = -(log_probs * advantages.detach()).mean() - 0.01 * entropy
        critic_loss = (advantages ** 2).mean()
        return actor_loss, critic_loss

    # -------- Action selection --------
    def choose_action(self, obs_vec: np.ndarray) -> np.ndarray:
        obs = T.tensor(obs_vec, dtype=T.float32).unsqueeze(0)     # [1, D]
        (alpha, beta), _ = self.forward(obs)                      # [1, n_actions]
        alpha, beta = alpha.squeeze(0), beta.squeeze(0)           # [n_actions]
        dist = Beta(alpha, beta)
        y = dist.sample().clamp(1e-6, 1 - 1e-6)                   # [n_actions]
        # Map y∈(0,1) -> MB range
        chunk_sizes = ACTION_MIN_MB + y * (ACTION_MAX_MB - ACTION_MIN_MB)  # [n_actions]
        return chunk_sizes.detach().cpu().numpy().astype(np.float32)


# -------------------------------------------------
# A3C Worker
# -------------------------------------------------
class Agent(mp.Process):
    def __init__(self, global_actor_critic, optimizer, input_dims, n_actions,
                 gamma, lr, name, global_ep_idx, env_factory, log_queue):
        super().__init__()
        self.local_actor_critic = ActorCritic(input_dims, n_actions, gamma)
        self.local_actor_critic.load_state_dict(global_actor_critic.state_dict())

        self.global_actor_critic = global_actor_critic
        self.name = f"w{name:02d}"
        self.episode_idx = global_ep_idx
        self.optimizer = optimizer
        self.env_factory = env_factory
        self.log_queue = log_queue

    def run(self):
        env = self.env_factory()
        t_step = 1
        UPDATE_EVERY = 20  # T_max

        while self.episode_idx.value < 3000:
            done = False
            observation, _info = env.reset()
            observation = np.asarray(observation, dtype=np.float32)

            score = 0.0
            episode_losses = []
            episode_steps = 0
            self.local_actor_critic.clear_memory()

            while not done:
                # ---- Choose vector action: 3 chunks (MB) ----
                chunk_mb_vec = self.local_actor_critic.choose_action(observation)  # [3]
                # Optionally reduce noisy prints across workers
                # if self.name == "w00":
                #     print(f"{self.name} step {episode_steps} picks MB:", chunk_mb_vec)

                # ---- Env step ----
                observation_, reward, done, info = env.step(chunk_mb_vec)
                observation_ = np.asarray(observation_, dtype=np.float32)

                score += float(reward)
                episode_steps += 1

                # ---- Store transition in Beta-space y∈(0,1) per action ----
                denom = (ACTION_MAX_MB - ACTION_MIN_MB)
                y_vec = (chunk_mb_vec - ACTION_MIN_MB) / max(denom, 1e-6)
                y_vec = np.clip(y_vec, 1e-6, 1.0 - 1e-6).astype(np.float32)  # [3]

                self.local_actor_critic.remember(observation, y_vec, float(reward))

                # ---- Local->Global update ----
                if (t_step % UPDATE_EVERY == 0) or done:
                    actor_loss, critic_loss = self.local_actor_critic.calc_loss(done)
                    loss = actor_loss + 0.5 * critic_loss
                    episode_losses.append(loss.item())

                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.local_actor_critic.parameters(), max_norm=0.5)

                    # Push local grads to global
                    for lp, gp in zip(self.local_actor_critic.parameters(),
                                      self.global_actor_critic.parameters()):
                        if gp.grad is not None:
                            gp.grad.zero_()
                        gp._grad = lp.grad

                    self.optimizer.step()

                    # Sync local with global
                    self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict())
                    self.local_actor_critic.clear_memory()

                observation = observation_
                t_step += 1

            # Episode done: bump global ep index
            with self.episode_idx.get_lock():
                self.episode_idx.value += 1
                current_ep = self.episode_idx.value

            avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0

            # Send metrics to main process
            self.log_queue.put({
                'episode': current_ep,
                'reward': score,
                'length': episode_steps,
                'loss': avg_loss,
                'worker': self.name
            })


# -------------------------------------------------
# Visualization (main process)
# -------------------------------------------------
class LogVisualizer:
    def __init__(self, log_dir='runs'):
        self.rewards = []
        self.episode_lengths = []
        self.losses = []
        self.worker_rewards = {}
        self.running_reward = deque(maxlen=100)
        os.makedirs('plots', exist_ok=True)

    def add_data(self, data):
        episode = data['episode']
        reward = data['reward']
        length = data['length']
        loss = data.get('loss', 0.0)
        worker = data['worker']

        self.rewards.append((episode, reward))
        self.episode_lengths.append((episode, length))
        if loss > 0:
            self.losses.append((episode, loss))
        self.running_reward.append(reward)

        if worker not in self.worker_rewards:
            self.worker_rewards[worker] = []
        self.worker_rewards[worker].append((episode, reward))

        if episode % 20 == 0:
            self.plot(intermediate=True)

    def plot(self, intermediate=False):
        if not self.rewards:
            return

        fig, axes = plt.subplots(3, 1, figsize=(12, 12))

        # Reward
        ax = axes[0]
        eps, rews = zip(*sorted(self.rewards))
        ax.plot(eps, rews, alpha=0.3, label='Raw rewards')
        if len(rews) >= 50:
            window = 50
            smoothed = np.convolve(rews, np.ones(window) / window, mode='valid')
            ax.plot(eps[window - 1:], smoothed, label=f'{window}-ep avg', linewidth=2)
        for worker, data in self.worker_rewards.items():
            if data:
                w_eps, w_rews = zip(*sorted(data))
                ax.plot(w_eps, w_rews, alpha=0.4, linestyle='--', label=f'{worker}')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Training Reward')
        ax.legend()
        ax.grid(True)

        # Episode length
        ax = axes[1]
        eps_len, lens = zip(*sorted(self.episode_lengths))
        ax.plot(eps_len, lens, alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Episode Length')
        ax.grid(True)

        # Loss
        ax = axes[2]
        if self.losses:
            eps_loss, loss_vals = zip(*sorted(self.losses))
            ax.plot(eps_loss, loss_vals, alpha=0.6)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss')
            ax.grid(True)
        else:
            ax.set_title('No loss data yet')

        plt.tight_layout()
        filename = f"plots/training_progress{'_temp' if intermediate else ''}.png"
        plt.savefig(filename, dpi=150)
        if intermediate:
            clear_output(wait=True)
        display(fig)
        plt.close(fig)


def logging_thread_fn(log_queue, visualizer, max_episodes):
    while True:
        try:
            data = log_queue.get(timeout=5)
            visualizer.add_data(data)
            if data['episode'] >= max_episodes:
                break
        except Exception:
            continue
    visualizer.plot(intermediate=False)


# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method('fork', force=True)

    # Model dims
    input_dims = 3          # observation = current speeds of 3 servers
    n_actions = 3           # one chunk size per server

    # Training hyperparams
    lr = 3e-4
    MAX_EPISODES = 3000
    NUM_WORKERS = 3

    # Global model + optimizer
    global_actor_critic = ActorCritic(input_dims, n_actions)
    global_actor_critic.share_memory()
    optim = SharedAdam(global_actor_critic.parameters(), lr=lr, betas=(0.92, 0.999))

    # Shared episode counter
    global_ep = mp.Value('i', 0)

    # Logging infra
    log_queue = mp.Queue()
    visualizer = LogVisualizer()
    log_thread = threading.Thread(
        target=logging_thread_fn, args=(log_queue, visualizer, MAX_EPISODES), daemon=True
    )
    log_thread.start()

    # Spawn workers
    workers = [
        Agent(global_actor_critic, optim, input_dims, n_actions,
              gamma=0.99, lr=lr, name=i, global_ep_idx=global_ep,
              env_factory=env_factory, log_queue=log_queue)
        for i in range(NUM_WORKERS)
    ]

    print(f"Starting training with {NUM_WORKERS} workers...")
    start_time = time.time()

    [w.start() for w in workers]
    try:
        [w.join() for w in workers]
    except KeyboardInterrupt:
        print("\nTraining interrupted. Stopping gracefully...")

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time/60:.1f} minutes")

    # Save trained model
    model_path = 'models'
    os.makedirs(model_path, exist_ok=True)
    T.save(global_actor_critic.state_dict(), f'{model_path}/a3c_download_model.pth')
    print(f"Model saved to {model_path}/a3c_download_model.pth")
    print("\nCheck visualization plots in the 'plots' directory")
