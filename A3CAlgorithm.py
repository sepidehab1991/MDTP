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

        self.urls = ["http://10.129.138.2/",
                "http://10.141.8.2/",
                "http://10.135.141.2/"]
        self.file_ips = self.get_file_sources(self.urls, self.num_servers)
        self.file_sources = [ip + file_size_str for ip in self.file_ips]
        print(self.file_sources)

        response = requests.head(self.file_sources[0])
        self.file_size = int(response.headers.get('Content-Length', 0))
        print("File size:", self.file_size)

        self.start_byte = 0
        self.end_byte = 0
        self.currentRequestedByte = 0
        self.throughput_gm = 0
        self.number_of_chunk = 0
        self.fast_downloadTime = 0
        self.range_dict = {}
        self.throughput = {}

    def get_file_sources(self, urls, num_sources):
        if num_sources > len(urls):
            raise ValueError("Number of sources exceeds the number of available URLs.")
        return urls[:num_sources]

    def reset(self):
        now = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = f"{self.folder_name}/Dynamic_Chunking_Async_{now}_{self.num_servers}_{self.iteration}_{self.file_size_str}.log"
        os.makedirs(self.folder_name, exist_ok=True)
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
        self.start_byte = 0
        self.end_byte = 0
        self.throughput = {}
        self.range_dict = {}
        self.number_of_chunk = 0
        self.fast_downloadTime = 0
        return np.array([0.0, 0.0], dtype=np.float32)

    async def fetch_range(self, session, url, s_b, e_b, throughput, range_dict):
        logging.info(f"Fetching range for URL: {url}, start_byte={s_b}, end_byte={e_b}")
        local_start_byte = s_b
        local_end_byte = e_b
        self.start_byte += e_b - s_b + 1
        self.end_byte = e_b
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
                    throughput[url] = (local_end_byte - local_start_byte) / delay
                    range_dict[local_start_byte] = data
                    logging.info(f"Successfully fetched range {local_start_byte}-{local_end_byte} from {url}")
                    logging.info(f"Throughput: {throughput}")
                    self.number_of_chunk += 1
                    logging.info(f"number_of_chunk: {self.number_of_chunk}")
                    logging.info(f"Fetched range: start_byte={local_start_byte}, end_byte={local_end_byte}, delay={delay:.4f}, throughput={throughput[url]}")
                return data, throughput
        except Exception as e:
            logging.error(f"Time out Error fetching data from {url}: {e}")

    async def download_file(self, session, url, throughput, range_dict, large_chunk):
        logging.info(f"Downloading file from URL: {url}")
        initial_chunk = 5 * 1024 * 1024
        self.end_byte = self.start_byte + initial_chunk - 1
        await self.fetch_range(session, url, self.start_byte, self.end_byte, throughput, range_dict)

        fast_dic = {}
        slow_dic = {}

        while self.end_byte < self.file_size and self.start_byte < self.file_size:
            values = list(throughput.values())
            self.throughput_gm = gmean(values)
            logging.info(f"Geometric mean of throughput: {self.throughput_gm}")
            logging.info(f"start and end byte are: {self.start_byte} and {self.end_byte}")
            logging.info(f"Current throughput for {url}: {throughput[url]}")
            if math.ceil(throughput[url]) >= math.ceil(self.throughput_gm):
                fast_dic[url] = throughput[url]
                fastest_server = max(fast_dic.values())
                logging.info(f"Fast server throughput: {fastest_server}")
                self.fast_downloadTime = large_chunk / fastest_server
                logging.info(f"Fast download time: {self.fast_downloadTime}")
                large_start_byte = self.start_byte
                logging.info(f"Large chunk start byte: {large_start_byte}")
                large_end_byte = large_start_byte + large_chunk + 1
                logging.info(f"Large chunk end byte: {large_end_byte}")
                if large_end_byte > self.file_size:
                    large_end_byte = self.file_size
                if large_start_byte > self.file_size:
                    break
                logging.info(f"Large_start_byte, Large_end_byte: {large_start_byte} and {large_end_byte}")
                await self.fetch_range(session, url, large_start_byte, large_end_byte, throughput, range_dict)

            if throughput[url] < self.throughput_gm:
                slow_dic[url] = throughput[url]
                logging.info(f"Slow server throughput: {slow_dic[url]}")
                small_chunk = math.floor(self.fast_downloadTime * (slow_dic[url]))
                logging.info(f"Small chunk size: {small_chunk}")
                small_start_byte = self.start_byte
                small_end_byte = small_start_byte + small_chunk + 1
                if small_end_byte > self.file_size:
                    small_end_byte = self.file_size
                if small_start_byte > self.file_size:
                    break
                logging.info(f"small_start_byte, small_end_byte: {small_start_byte} and {small_end_byte}")
                await self.fetch_range(session, url, small_start_byte, small_end_byte, throughput, range_dict)

    async def step_async(self, large_chunk):

        print("RL gave large chunk size (bytes):", large_chunk)
        filename = f"{self.folder_name}/output_file_{dt.datetime.now().strftime('%H%M%S')}"
        start_time = time.time()
        session_dict = {url: aiohttp.ClientSession() for url in self.file_sources}
        throughput = {}
        range_dict = {}

        await asyncio.gather(
            *(self.download_file(session_dict[url], url, throughput, range_dict, large_chunk) for url in self.file_sources)
        )
        for session in session_dict.values():
            await session.close()

        logging.info("All chunks downloaded, writing to disk.")
        start_Disk_time = time.time()
        with open(filename, 'wb') as f:
            for start_b in sorted(range_dict.keys()):
                f.write(range_dict[start_b])
        end_time = time.time()
        delay = end_time - start_time
        end_Disk_time = time.time()
        Disk_delay = end_Disk_time - start_Disk_time
        print("they printed all")
        logging.info("They printed all")
        print("Disk_delay:", Disk_delay)
        print("File size,{},Delay,{}".format(self.file_size, delay))
        logging.info(f"Delay {delay}")
        logging.info(f"Total download delay: {delay}")
        logging.info(f"Disk_delay: {Disk_delay}")

        avg_throughput = np.mean(list(throughput.values())) if throughput else 1e-6
        obs = np.array([delay, avg_throughput], dtype=np.float32)
        reward = avg_throughput / delay
        done = True
        return obs, reward, done, {}

# --------------------------- ACTOR-CRITIC ---------------------------

class ActorCritic(nn.Module):
    def __init__(self, input_dims, gamma=0.99):
        super().__init__()
        self.gamma = gamma
        self.pi1 = nn.Linear(*input_dims, 128)
        self.v1 = nn.Linear(*input_dims, 128)
        self.pi = nn.Linear(128, 1)
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
        actions = T.tensor(self.actions, dtype=T.float32)  # Actual chunk sizes in MB
        predicted_chunks, values = self.forward(states)
        values = values.squeeze()
        returns = self.calc_R(done)

        # Actor loss: Mean squared error between predicted chunk sizes and actions taken
        actor_loss = F.mse_loss(predicted_chunks.squeeze(), actions, reduction='mean')

        # Advantage = returns - values
        advantage = returns - values.detach()

        # Weight the actor loss by advantage (to encourage good predictions)
        actor_loss = actor_loss * advantage.mean()

        # Critic loss: how close predicted value is to return
        critic_loss = F.mse_loss(values, returns, reduction='mean')

        # Total loss
        return actor_loss + critic_loss

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float32)
        with T.no_grad():
            chunk_size_pred, _ = self.forward(state)
            chunk_size_mb = T.clamp(chunk_size_pred, 20, 160).item()  # clamp between 20MB and 160MB
            print(f"RL predicted large chunk size: {chunk_size_mb} MB")
            return int(chunk_size_mb * 1024 * 1024)  # convert MB → bytes


# --------------------------- AGENT ---------------------------

class Agent(mp.Process):
    def __init__(self, global_model, optimizer, input_dims, gamma, lr,
                 name, global_ep_idx, file_size_str, iteration, num_servers, folder_name):
        super().__init__()
        self.local_model = ActorCritic(input_dims, gamma)
        self.global_model = global_model
        self.optimizer = optimizer
        self.name = f'w{name:02d}'
        self.global_ep_idx = global_ep_idx
        self.file_size_str = file_size_str
        self.iteration = iteration
        self.num_servers = num_servers
        self.folder_name = folder_name
        

    def run(self):
        while self.global_ep_idx.value < 1:
            env = DownloaderEnv(self.file_size_str, self.iteration, self.num_servers, self.folder_name)
            obs = env.reset()
            done, score = False, 0
            self.local_model.clear_memory()

            large_chunk = self.local_model.choose_action(obs)
            obs_, reward, done, _ = asyncio.run(env.step_async(large_chunk))
            score += reward
            self.local_model.remember(obs, large_chunk / (1024 * 1024), reward)


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

    if len(sys.argv) != 5:
        #print(\"Usage: python script.py <file_size_str> <iteration> <num_servers> <folder_name>\")
        print("Enter five item as inputs")
        sys.exit(1)

    file_size_str = sys.argv[1]
    iteration = sys.argv[2]
    num_servers = int(sys.argv[3])
    folder_name = sys.argv[4]

    input_dims = [2]  # [download_time, throughput]
    n_actions = 4     # chunk sizes: 20MB, 40MB, 80MB, 160MB
    lr = 1e-4


    global_model = ActorCritic(input_dims)
    global_model.share_memory()
    optimizer = T.optim.Adam(global_model.parameters(), lr=lr)
    global_ep_idx = mp.Value('i', 0)

    workers = [
        Agent(global_model, optimizer, input_dims, gamma=0.99, lr=lr,
              name=i, global_ep_idx=global_ep_idx, file_size_str=file_size_str,
              iteration=iteration, num_servers=num_servers, folder_name=folder_name)
        for i in range(mp.cpu_count())
    ]

    [w.start() for w in workers]
    [w.join() for w in workers]

