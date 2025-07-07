import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical
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
import os
import numpy as np

# --------------------------- ENVIRONMENT ---------------------------

class DownloaderEnv:
    def __init__(self, file_size_str, iteration, num_servers, folder_name):
        self.iteration = iteration
        self.num_servers = num_servers
        self.folder_name = folder_name
        self.file_size_str = file_size_str
        #self.urls = [
        #    "http://10.134.133.2/", "http://10.129.130.2/",
        #    "http://10.135.4.2/", "http://10.138.4.2/",
        #    "http://10.132.1.2/", "http://10.147.3.2/"
        #]
        self.urls = ["http://10.129.138.2/",
                "http://10.141.8.2/",
                "http://10.135.141.2/"]
        
        self.file_ips = self.urls[:self.num_servers]
        self.file_sources = [ip + file_size_str for ip in self.file_ips]

        self.chunk_sizes = [20, 40, 80, 160]  # MB
        self.large_chunk = 20 * 1024 * 1024  # default
        self.range_dict = {}
        self.throughput = {}

        # Determine actual file size
        response = requests.head(self.file_sources[0])
        self.file_size = int(response.headers.get('Content-Length', 0))
        print("File size:", self.file_size)

    def reset(self):
        now = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = f"{self.folder_name}/A3C_{now}_{self.num_servers}_{self.iteration}_{self.file_size_str}.log"
        os.makedirs(self.folder_name, exist_ok=True)
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
        self.start_byte = 0
        self.end_byte = 0
        self.throughput = {}
        self.range_dict = {}
        self.number_of_chunk = 0
        self.fast_downloadTime = 0
        return np.array([0.0, 0.0], dtype=np.float32)

    async def fetch_range(self, session, url, s_b, e_b):
        headers = {'Range': f"bytes={s_b}-{e_b}"}
        start_time = time.time()
        try:
            async with async_timeout.timeout(500):
                async with session.get(url, headers=headers) as response:
                    data = await response.read()
                    end_time = time.time()
                    delay = end_time - start_time
                    self.throughput[url] = (e_b - s_b + 1) / delay
                    self.range_dict[s_b] = data
                    self.start_byte += e_b - s_b + 1
                    self.end_byte = e_b
                    self.number_of_chunk += 1
                    return data, delay
        except Exception as e:
            logging.error(f"Error fetching data from {url}: {e}")
            return None, 5.0

    async def download_file(self, session, url):
        logging.info(f"Downloading file from URL: {url}")
        initial_chunk = 5 * 1024 * 1024
        self.end_byte = self.start_byte + initial_chunk - 1
        await self.fetch_range(session, url, self.start_byte, self.end_byte)

        fast_dic = {}
        slow_dic = {}

        while self.end_byte < self.file_size and self.start_byte < self.file_size:
            values = list(self.throughput.values())
            if not values:
                break
            throughput_gm = gmean(values)
            tp = self.throughput[url]
            if math.ceil(tp) >= math.ceil(throughput_gm):
                fast_dic[url] = tp
                fastest = max(fast_dic.values())
                self.fast_downloadTime = self.large_chunk / fastest
                large_s = self.start_byte
                large_e = min(large_s + self.large_chunk - 1, self.file_size)
                await self.fetch_range(session, url, large_s, large_e)
            else:
                slow_dic[url] = tp
                small_chunk = math.floor(self.fast_downloadTime * tp)
                small_s = self.start_byte
                small_e = min(small_s + small_chunk - 1, self.file_size)
                await self.fetch_range(session, url, small_s, small_e)

    async def step_async(self, action_idx):
        self.large_chunk = self.chunk_sizes[action_idx] * 1024 * 1024

        filename = f"{self.folder_name}/output_file_{dt.datetime.now().strftime('%H%M%S')}"
        start_time = time.time()
        session_dict = {url: aiohttp.ClientSession() for url in self.file_sources}
        self.throughput = {}
        self.range_dict = {}

        await asyncio.gather(
            *(self.download_file(session_dict[url], url) for url in self.file_sources)
        )
        for session in session_dict.values():
            await session.close()

        with open(filename, 'wb') as f:
            for start_b in sorted(self.range_dict.keys()):
                f.write(self.range_dict[start_b])
        end_time = time.time()

        download_time = end_time - start_time
        avg_throughput = np.mean(list(self.throughput.values())) if self.throughput else 1e-6
        obs = np.array([download_time, avg_throughput], dtype=np.float32)
        reward = avg_throughput / download_time
        done = True  # full file downloaded
        return obs, reward, done, {}

# --------------------------- MODEL ---------------------------

class SharedAdam(T.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = T.zeros_like(p.data)
                state['exp_avg_sq'] = T.zeros_like(p.data)
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99):
        super().__init__()
        self.gamma = gamma
        self.pi1 = nn.Linear(*input_dims, 128)
        self.v1 = nn.Linear(*input_dims, 128)
        self.pi = nn.Linear(128, n_actions)
        self.v = nn.Linear(128, 1)
        self.rewards, self.actions, self.states = [], [], []

    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_memory(self):
        self.states, self.actions, self.rewards = [], [], []

    def forward(self, state):
        pi1 = F.relu(self.pi1(state))
        v1 = F.relu(self.v1(state))
        return self.pi(pi1), self.v(v1)

    def calc_R(self, done):
        states = T.tensor(self.states, dtype=T.float32)
        with T.no_grad():
            _, v = self.forward(states)
        R = v[-1] * (1 - int(done))
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return T.tensor(returns, dtype=T.float32)

    def calc_loss(self, done):
        states = T.tensor(self.states, dtype=T.float32)
        actions = T.tensor(self.actions, dtype=T.int64)
        returns = self.calc_R(done)
        pi, values = self.forward(states)
        values = values.squeeze()
        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs * (returns - values.detach())
        critic_loss = F.mse_loss(values, returns, reduction='none')
        return (actor_loss + critic_loss).mean()

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float32)
        with T.no_grad():
            pi, _ = self.forward(state)
            probs = T.softmax(pi, dim=1)
            dist = Categorical(probs)
            action = dist.sample().item()
        return action

# --------------------------- AGENT ---------------------------

class Agent(mp.Process):
    def __init__(self, global_model, optimizer, input_dims, n_actions, gamma, lr,
                 name, global_ep_idx, file_size_str, iteration, num_servers, folder_name):
        super().__init__()
        self.local_model = ActorCritic(input_dims, n_actions, gamma)
        self.global_model = global_model
        self.optimizer = optimizer
        self.name = f'w{name:02d}'
        self.global_ep_idx = global_ep_idx
        self.file_size_str = file_size_str
        self.iteration = iteration
        self.num_servers = num_servers
        self.folder_name = folder_name

    def run(self):
        while self.global_ep_idx.value < 1000:
            env = DownloaderEnv(self.file_size_str, self.iteration, self.num_servers, self.folder_name)
            obs = env.reset()
            done, score = False, 0
            self.local_model.clear_memory()

            action = self.local_model.choose_action(obs)
            obs_, reward, done, _ = asyncio.run(env.step_async(action))
            score += reward
            self.local_model.remember(obs, action, reward)

            loss = self.local_model.calc_loss(done)
            self.optimizer.zero_grad()
            loss.backward()
            for lp, gp in zip(self.local_model.parameters(), self.global_model.parameters()):
                gp._grad = lp.grad
            self.optimizer.step()
            self.local_model.load_state_dict(self.global_model.state_dict())
            self.local_model.clear_memory()

            with self.global_ep_idx.get_lock():
                self.global_ep_idx.value += 1
            print(self.name, 'Episode', self.global_ep_idx.value, 'Reward: %.4f' % score)
            env.reset()

# --------------------------- MAIN ---------------------------

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    # Get parameters from command-line
    if len(sys.argv) != 5:
        print("Usage: python script.py <file_size_str> <iteration> <num_servers> <folder_name>")
        sys.exit(1)

    file_size_str = sys.argv[1]
    iteration = sys.argv[2]
    num_servers = int(sys.argv[3])
    folder_name = sys.argv[4]

    input_dims = [2]  # [download_time, throughput]
    n_actions = 4     # chunk sizes: 20MB, 40MB, 80MB, 160MB
    lr = 1e-4

    global_model = ActorCritic(input_dims, n_actions)
    global_model.share_memory()
    optimizer = SharedAdam(global_model.parameters(), lr=lr)
    global_ep_idx = mp.Value('i', 0)

    workers = [
        Agent(global_model, optimizer, input_dims, n_actions, gamma=0.99, lr=lr,
              name=i, global_ep_idx=global_ep_idx, file_size_str=file_size_str,
              iteration=iteration, num_servers=num_servers, folder_name=folder_name)
        for i in range(mp.cpu_count())
    ]

    [w.start() for w in workers]
    [w.join() for w in workers]

