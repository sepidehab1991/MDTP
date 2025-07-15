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
from multiprocessing import Barrier, Manager
import matplotlib.pyplot as plt

class DownloaderEnv:
    def __init__(self, file_size_str, iteration, num_servers, folder_name, urls, sync_barrier):
        self.iteration = iteration
        self.num_servers = num_servers
        self.folder_name = folder_name
        self.file_size_str = file_size_str
        self.urls = urls
        self.sync_barrier = sync_barrier
        
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
        self.fast_dic = {}
        self.slow_dic = {}
        self.output_filename = f"{self.folder_name}/output_file_{dt.datetime.now().strftime('%H%M%S')}"

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
        self.fast_dic = {}
        self.slow_dic = {}
        self.number_of_chunk = 0
        self.fast_downloadTime = 0
        return np.array([0.0, 0.0], dtype=np.float32)

    async def fetch_range(self, session, url, s_b, e_b):
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
                    self.throughput[url] = (local_end_byte - local_start_byte) / delay
                    self.range_dict[local_start_byte] = data
                    logging.info(f"Successfully fetched range {local_start_byte}-{local_end_byte} from {url}")
                    logging.info(f"Throughput: {self.throughput}")
                    self.number_of_chunk += 1
                    logging.info(f"number_of_chunk: {self.number_of_chunk}")
                    logging.info(f"Fetched range: start_byte={local_start_byte}, end_byte={local_end_byte}, delay={delay:.4f}, throughput={self.throughput[url]}")

        except Exception as e:
            logging.error(f"Time out Error fetching data from {url}: {e}")

    async def download_file(self, session, url, large_chunk):
        logging.info(f"Downloading file from URL: {url}")
        initial_chunk = 5 * 1024 * 1024
        self.end_byte = self.start_byte + initial_chunk - 1
        await self.fetch_range(session, url, self.start_byte, self.end_byte)

        while self.end_byte < self.file_size and self.start_byte < self.file_size:
            values = list(self.throughput.values())
            self.throughput_gm = gmean(values)
            logging.info(f"Geometric mean of throughput: {self.throughput_gm}")
            logging.info(f"start and end byte are: {self.start_byte} and {self.end_byte}")
            logging.info(f"Current throughput for {url}: {self.throughput[url]}")
            if math.ceil(self.throughput[url]) >= math.ceil(self.throughput_gm):
                self.fast_dic[url] = self.throughput[url]
                fastest_server = max(self.fast_dic.values())
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
                await self.fetch_range(session, url, large_start_byte, large_end_byte)

            if self.throughput[url] < self.throughput_gm:
                self.slow_dic[url] = self.throughput[url]
                logging.info(f"Slow server throughput: {self.slow_dic[url]}")
                small_chunk = math.floor(self.fast_downloadTime * (self.slow_dic[url]))
                logging.info(f"Small chunk size: {small_chunk}")
                small_start_byte = self.start_byte
                small_end_byte = small_start_byte + small_chunk + 1
                if small_end_byte > self.file_size:
                    small_end_byte = self.file_size
                if small_start_byte > self.file_size:
                    break
                logging.info(f"small_start_byte, small_end_byte: {small_start_byte} and {small_end_byte}")
                await self.fetch_range(session, url, small_start_byte, small_end_byte)

    async def step_async(self, worker_id, large_chunk):

        print("RL gave large chunk size (bytes):", large_chunk)
        start_time = time.time()
        print(f"worker_id: {worker_id}")
        print(f"self.file_sources: {self.file_sources}")
        url = self.file_sources[worker_id]
        session =  aiohttp.ClientSession()

        await self.download_file(session, url, large_chunk)
        await session.close()

        logging.info("All chunks downloaded, writing to disk.")
        start_Disk_time = time.time()
        wait_workers = True
        if worker_id == 0:
            logging.info("Worker 0 writing to disk.")
            with open(self.output_filename, 'wb') as f:
                for start_b in sorted(self.range_dict.keys()):
                    f.write(self.range_dict[start_b])
            logging.info("Worker 0 finished writing.")
            
        # All workers wait here until worker 0 finishes
        self.sync_barrier.wait()
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
        avg_throughput = np.mean(list(self.throughput.values())) if self.throughput else 1e-6
        obs = np.array([delay, avg_throughput], dtype=np.float32)
        reward = avg_throughput / delay
        done = True
        return obs, reward, done, {}

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

class Agent(mp.Process):
    def __init__(self, global_model, optimizer, input_dims, gamma, lr,
                 name, worker_id, global_ep_idx, file_size_str, iteration, num_servers, 
                 folder_name, urls, sync_barrier, reward_history, loss_history):
        super().__init__()
        self.local_model = ActorCritic(input_dims, gamma)
        self.global_model = global_model
        self.optimizer = optimizer
        self.worker_id = worker_id
        self.name = f'w{name:02d}'
        self.global_ep_idx = global_ep_idx
        self.file_size_str = file_size_str
        self.iteration = iteration
        self.num_servers = num_servers
        self.folder_name = folder_name
        self.urls = urls
        self.sync_barrier = sync_barrier
        self.reward_history = reward_history
        self.loss_history = loss_history


    def run(self):
        while self.global_ep_idx.value < 3:
            env = DownloaderEnv(self.file_size_str, self.iteration, self.num_servers, self.folder_name, self.urls, self.sync_barrier)
            obs = env.reset()
            done, score = False, 0
            self.local_model.clear_memory()
            large_chunk = self.local_model.choose_action(obs)
            obs_, reward, done, _ = asyncio.run(env.step_async( self.worker_id, large_chunk))
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
                ep_idx = self.global_ep_idx.value
                self.global_ep_idx.value += 1

            self.reward_history.append((ep_idx, score))
            self.loss_history.append((ep_idx, loss.item()))

            print(self.name, 'Episode', ep_idx, 'Reward: %.4f' % score, 'Loss: %.4f' % loss.item())
            self.sync_barrier.wait()  # Synchronize workers
            

