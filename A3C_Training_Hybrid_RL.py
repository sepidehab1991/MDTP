# -*- coding: utf-8 -*-
"""A3C for Optimized File Downloads

Implementation of A3C algorithm for optimizing file downloads across multiple servers
"""

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
from torch.distributions import Beta
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import os
import time
import threading
from collections import deque

# Constants for action space
ACTION_MIN_MB = 10.0
ACTION_MAX_MB = 128.0

# ------------------------------
# Environment
# ------------------------------
class DownloadSimEnv:
    def __init__(self,
                 file_size=4096.0,
                 initial_chunks=(10, 10, 10),
                 base_speed=(50.0, 11.5, 28.0),
                 lambda_delay=1e-8,
                 default_large_chunk=40.0):
        # Static config
        self.file_size = float(file_size)
        self.initial_chunks = np.asarray(initial_chunks, dtype=float)
        self.base_speed = np.asarray(base_speed, dtype=float)
        self.lambda_delay = float(lambda_delay)
        self.n = len(self.base_speed)

        # Runtime state
        self.large_chunk = float(default_large_chunk)
        self.remaining = None
        self.server_elapsed = None
        self.speeds = None
        self.replicas_time = None
        self.reward = 0.0
        self.done = False

    # ------------------------------
    # Noise model
    # ------------------------------
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

    # ------------------------------
    # Reset
    # ------------------------------
    def reset(self, *, seed=None, options=None):
        self.speeds = self.base_speed.copy()
        self.remaining = float(self.file_size)
        self.server_elapsed = np.zeros(self.n, dtype=float)
        self.reward = 0.0
        self.done = False
        obs = self.speeds.astype(np.float32)
        info = {}
        return obs, info

    # ------------------------------
    # Step
    # ------------------------------
    def step(self, action):
        # === Init robust defaults ===
        reward = 0.0
        done = False
        delays = 0.0

        # --- Inputs/state ---
        base_speed = self.base_speed
        remaining = float(self.remaining)
        large_chunk = float(np.asarray(action).reshape(-1)[0]) if action is not None else self.large_chunk
        print("Large Chunk:", large_chunk)

        # --- Sample current server speeds ---
        self.speeds = self.jitter_lognormal(base_speed, rel_std=0.15)

        # --- Choose fastest by current speed ---
        fastest_by_speed = int(np.argmax(self.speeds))
        fastest_server_time = large_chunk / max(self.speeds[fastest_by_speed], 1e-9)

        # --- Allocate chunks to equalize finish times this round ---
        chunk_size = np.zeros(self.n, dtype=float)
        for i in range(self.n):
            if i == fastest_by_speed:
                chunk_size[i] = large_chunk
            else:
                chunk_size[i] = fastest_server_time * self.speeds[i]

        # --- Cap by remaining file size ---
        total_c = float(np.sum(chunk_size))
        if total_c > remaining:
            scale = max(0.0, min(1.0, remaining / total_c))
            chunk_size *= scale

        # --- Measure times for this round ---
        round_dt = self.fetch_data(chunk_size, self.speeds)
        server_elapsed = np.zeros(self.n, dtype=float)
        server_elapsed += round_dt

        # --- Fastest finish this round (by round_dt) ---
        fastest_server_time_id = int(np.argmin(round_dt))
        remaining -= float(np.sum(chunk_size))

        # --- Reward for this round: minimize fastest finish time + weighted delays ---
        # delays = sum of positive (time_i - time_fastest)
        delays = float(np.sum(np.maximum(round_dt - round_dt[fastest_server_time_id], 0.0)))
        reward = -((float(round_dt[fastest_server_time_id]) + delays)**2)

        # --- Final allocation if near the end ---
        if (remaining < large_chunk) and (remaining > 0.0):
            final_chunks = np.zeros(self.n, dtype=float)
            final_chunks[fastest_server_time_id] = remaining

            final_dt = self.fetch_data(final_chunks, self.speeds)
            server_elapsed += final_dt

            # recompute reward on final step
            delays = float(np.sum(np.maximum(final_dt - final_dt[fastest_server_time_id], 0.0)))
            reward = -((float(final_dt[fastest_server_time_id]) + delays)**2)

            remaining -= remaining  # -> 0

        total_simulated_seconds = float(np.max(server_elapsed)) if server_elapsed.size else 0.0

        # --- Episode termination ---
        if remaining <= 0.0:
            done = True
        # else: done stays False

        # --- Write back ---
        self.remaining = remaining
        self.server_elapsed = server_elapsed
        self.reward = reward
        self.done = done

        # --- Observation & info ---
        obs = self.speeds.astype(np.float32)
        info = {
            "chunks": chunk_size.tolist(),
            "round_time": round_dt.tolist(),
            "fastest_server_id": fastest_server_time_id,
            "total_time": total_simulated_seconds,
            "remaining": remaining
        }
        return obs, float(reward), bool(done), info

def env_factory():
    return DownloadSimEnv(
        file_size=4096.0,
        initial_chunks=(10, 10, 10),
        base_speed=(50.0, 11.5, 28.0),
        lambda_delay=1e-3
    )


# ------------------------------
# Shared Adam 
# ------------------------------
class SharedAdam(T.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0, amsgrad=False):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        # initialize optimizer state in shared memory (step must be a *tensor*)
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    # singleton tensors (required by newer Adam)
                    state['step'] = T.zeros(1)                        # <-- tensor, not int
                    state['exp_avg'] = T.zeros_like(p.data, memory_format=T.preserve_format)
                    state['exp_avg_sq'] = T.zeros_like(p.data, memory_format=T.preserve_format)
                    if amsgrad:
                        state['max_exp_avg_sq'] = T.zeros_like(p.data, memory_format=T.preserve_format)

                    # share across processes
                    state['step'].share_memory_()
                    state['exp_avg'].share_memory_()
                    state['exp_avg_sq'].share_memory_()
                    if amsgrad:
                        state['max_exp_avg_sq'].share_memory_()

# ------------------------------
#   Deep Neural Network
# ------------------------------
class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99):
        super().__init__()
        self.gamma = gamma

        # MLP
        self.fc1 = nn.Linear(input_dims, 128)
        self.fc2 = nn.Linear(128, 128)

        # Actor heads (Beta params) and Critic
        self.actor_alpha = nn.Linear(128, n_actions)
        self.actor_beta  = nn.Linear(128, n_actions)
        self.critic      = nn.Linear(128, 1)

        self.rewards = []
        self.actions = []
        self.states  = []

    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        alpha = F.softplus(self.actor_alpha(x)) + 1.0
        beta  = F.softplus(self.actor_beta(x))  + 1.0
        value = self.critic(x)
        return (alpha, beta), value

    def calc_R(self, done):
        states = T.tensor(self.states, dtype=T.float32)
        _, v = self.forward(states)
        R = v[-1] * (1 - int(done))
        batch_return = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            batch_return.append(R)
        batch_return.reverse()
        return T.stack(batch_return).detach()  # shape [T, 1] or [T]

    def calc_loss(self, done):
        states  = T.tensor(self.states,  dtype=T.float32)
        actions = T.tensor(self.actions, dtype=T.float32)

        returns = self.calc_R(done).squeeze(-1)       # [T]
        (alpha, beta), values = self.forward(states)  # values: [T,1]
        values = values.squeeze(-1)                   # [T]

        dist = Beta(alpha, beta)
        log_probs = dist.log_prob(actions).squeeze(-1)  # [T]
        entropy = dist.entropy().mean() # Add entropy calculation

        advantages = returns - values
        actor_loss  = -(log_probs * advantages.detach()).mean() - 0.01 * entropy
        critic_loss = (advantages ** 2).mean()
        return actor_loss, critic_loss

    def choose_action(self, obs_vec: np.ndarray) -> float:
        obs = T.tensor(obs_vec, dtype=T.float32).unsqueeze(0)  # [1,D]
        (alpha, beta), _ = self.forward(obs)                   # [1,1] each
        dist = Beta(alpha.squeeze(0), beta.squeeze(0))
        y = dist.sample().clamp(1e-6, 1-1e-6)                  # [1]
        large_chunk = ACTION_MIN_MB + y * (ACTION_MAX_MB - ACTION_MIN_MB)
        return float(large_chunk.item())

# ------------------------------
# Agent class without TensorFlow dependencies
# ------------------------------
class Agent(mp.Process):
    def __init__(self, global_actor_critic, optimizer, input_dims, n_actions,
                 gamma, lr, name, global_ep_idx, env_factory, log_queue):
        super(Agent, self).__init__()
        self.local_actor_critic = ActorCritic(input_dims, n_actions, gamma)
        # start with global weights
        self.local_actor_critic.load_state_dict(global_actor_critic.state_dict())

        self.global_actor_critic = global_actor_critic
        self.name = f"w{name:02d}"
        self.episode_idx = global_ep_idx
        self.optimizer = optimizer
        self.env_factory = env_factory
        self.log_queue = log_queue  # Queue for logging data to main process

    def run(self):
        # Build a fresh env in this process
        self.env = self.env_factory()

        t_step = 1
        UPDATE_EVERY = 20  # like A3C T_max

        while self.episode_idx.value < 3000:
            done = False
            observation, _info = self.env.reset()
            # ensure np float32
            observation = np.asarray(observation, dtype=np.float32)

            score = 0.0
            episode_losses = []
            episode_steps = 0
            self.local_actor_critic.clear_memory()

            while not done:
                # ---- Choose action (large_chunk in MB) ----
                large_chunk_mb = self.local_actor_critic.choose_action(observation)
                print(f"{self.name} chooses {large_chunk_mb:.1f} MB on step:{episode_steps} of {self.episode_idx.value}")
                # ---- Step environment ----
                observation_, reward, done, info = self.env.step(large_chunk_mb)
                observation_ = np.asarray(observation_, dtype=np.float32)

                score += float(reward)
                episode_steps += 1

                # ---- Store transition for policy update ----
                # IMPORTANT: store the action in *Beta space* y∈(0,1)
                denom = (ACTION_MAX_MB - ACTION_MIN_MB)
                y = (large_chunk_mb - ACTION_MIN_MB) / max(denom, 1e-6)
                # clamp to valid open interval
                y = float(np.clip(y, 1e-6, 1.0 - 1e-6))

                self.local_actor_critic.remember(observation, y, float(reward))

                # ---- Perform local->global update every few steps or at episode end ----
                if (t_step % UPDATE_EVERY == 0) or done:
                    actor_loss, critic_loss = self.local_actor_critic.calc_loss(done)
                    loss = actor_loss + 0.5 * critic_loss
                    episode_losses.append(loss.item())

                    self.optimizer.zero_grad()
                    loss.backward()

                    T.nn.utils.clip_grad_norm_(self.local_actor_critic.parameters(), max_norm=0.5)

                    # Push local grads to global
                    for lp, gp in zip(self.local_actor_critic.parameters(),
                                      self.global_actor_critic.parameters()):
                        if gp.grad is not None:
                            gp.grad.zero_()
                        gp._grad = lp.grad

                    self.optimizer.step()

                    # Pull updated global weights back to local
                    self.local_actor_critic.load_state_dict(
                        self.global_actor_critic.state_dict()
                    )
                    self.local_actor_critic.clear_memory()

                # next step
                observation = observation_
                t_step += 1

            # episode done
            with self.episode_idx.get_lock():
                self.episode_idx.value += 1
                current_ep = self.episode_idx.value

            # Calculate average loss if any updates were made
            avg_loss = np.mean(episode_losses) if episode_losses else 0.0
            
            # Send metrics to main process via queue
            self.log_queue.put({
                'episode': current_ep,
                'reward': score,
                'length': episode_steps,
                'loss': avg_loss,
                'worker': self.name
            })
            
            print(f"{self.name} episode {current_ep}: reward={score:.1f}, steps={episode_steps}")

# ------------------------------
# Visualization in main process
# ------------------------------
class LogVisualizer:
    def __init__(self, log_dir='runs'):
        self.rewards = []
        self.episode_lengths = []
        self.losses = []
        self.worker_rewards = {}
        self.running_reward = deque(maxlen=100)
        
        # Create plots directory
        os.makedirs('plots', exist_ok=True)
    
    def add_data(self, data):
        """Process a single data point from workers"""
        episode = data['episode']
        reward = data['reward']
        length = data['length']
        loss = data.get('loss', 0)
        worker = data['worker']
        
        # Store raw data
        self.rewards.append((episode, reward))
        self.episode_lengths.append((episode, length))
        if loss > 0:
            self.losses.append((episode, loss))
        self.running_reward.append(reward)
        
        # Track per-worker performance
        if worker not in self.worker_rewards:
            self.worker_rewards[worker] = []
        self.worker_rewards[worker].append((episode, reward))
        
        # Plot every 20 episodes
        if episode % 20 == 0:
            self.plot(intermediate=True)
    
    def plot(self, intermediate=False):
        """Create training plots"""
        if not self.rewards:
            return  # No data yet
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        
        # PLOT 1: Reward curves
        ax = axes[0]
        
        # Main reward curve
        eps, rews = zip(*sorted(self.rewards))
        ax.plot(eps, rews, alpha=0.3, color='gray', label='Raw rewards')
        
        # Smoothed reward
        if len(rews) >= 50:
            window = 50
            smoothed = np.convolve(rews, np.ones(window)/window, mode='valid')
            ax.plot(eps[window-1:], smoothed, label=f'{window}-ep avg', 
                    linewidth=2, color='blue')
        
        # Per-worker rewards
        for worker, data in self.worker_rewards.items():
            if data:
                w_eps, w_rews = zip(*sorted(data))
                ax.plot(w_eps, w_rews, alpha=0.5, label=f'Worker {worker}', 
                        linestyle='--')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Training Reward')
        ax.legend()
        ax.grid(True)
        
        # PLOT 2: Episode length
        ax = axes[1]
        eps_len, lens = zip(*sorted(self.episode_lengths))
        ax.plot(eps_len, lens, alpha=0.7, color='green')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Episode Length')
        ax.grid(True)
        
        # PLOT 3: Loss values
        ax = axes[2]
        if self.losses:
            eps_loss, loss_vals = zip(*sorted(self.losses))
            ax.plot(eps_loss, loss_vals, alpha=0.6, color='red')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss')
            ax.grid(True)
        else:
            ax.set_title('No loss data yet')
        
        plt.tight_layout()
        
        # Save the figure
        filename = f"plots/training_progress{'_temp' if intermediate else ''}.png"
        plt.savefig(filename, dpi=150)
        
        # Display in notebook
        if intermediate:
            clear_output(wait=True)
        display(fig)
        plt.close(fig)

def logging_thread_fn(log_queue, visualizer, max_episodes):
    """Background thread for processing log data"""
    while True:
        try:
            data = log_queue.get(timeout=5)
            visualizer.add_data(data)
            
            # Exit condition
            if data['episode'] >= max_episodes:
                break
        except:
            # Queue timeout
            continue
    
    # Final plot
    visualizer.plot(intermediate=False)

# ------------------------------
# Main function
# ------------------------------
if __name__ == "__main__":
    mp.set_start_method('fork', force=True)
    
    # Model dimensions
    input_dims = 3
    n_actions = 1
    
    # Training hyperparameters
    lr = 3e-4 # increase from 1e-4 to 3e-4
    MAX_EPISODES = 2000
    
    # Global (shared) model + optimizer
    global_actor_critic = ActorCritic(input_dims, n_actions)
    global_actor_critic.share_memory()
    optim = SharedAdam(global_actor_critic.parameters(), lr=lr, betas=(0.92, 0.999))
    
    # Shared episode counter
    global_ep = mp.Value('i', 0)
    
    # Create logging queue
    log_queue = mp.Queue()
    
    # Create visualizer
    visualizer = LogVisualizer()
    
    # Start logging thread
    log_thread = threading.Thread(
        target=logging_thread_fn,
        args=(log_queue, visualizer, MAX_EPISODES),
        daemon=True
    )
    log_thread.start()
    
    # Create workers
    NUM_WORKERS = 3
    workers = [
        Agent(global_actor_critic,
              optim,
              input_dims,
              n_actions,
              gamma=0.99,
              lr=lr,
              name=i,
              global_ep_idx=global_ep,
              env_factory=env_factory,
              log_queue=log_queue)
        for i in range(NUM_WORKERS)
    ]
    
    print(f"Starting training with {NUM_WORKERS} workers...")
    start_time = time.time()
    
    # Start all workers
    [w.start() for w in workers]
    
    try:
        # Wait for workers to finish
        [w.join() for w in workers]
    except KeyboardInterrupt:
        print("\nTraining interrupted. Stopping gracefully...")
    
    # Training complete
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time/60:.1f} minutes")
    
    # Save trained model
    model_path = 'models'
    os.makedirs(model_path, exist_ok=True)
    T.save(global_actor_critic.state_dict(),
           f'{model_path}/a3c_download_model.pth')
    print(f"Model saved to {model_path}/a3c_download_model.pth")
    
    print("\nCheck visualization plots in the 'plots' directory")
    
    # Optional: For TensorBoard visualization later (uncomment if needed)
    # %tensorboard --logdir=runs