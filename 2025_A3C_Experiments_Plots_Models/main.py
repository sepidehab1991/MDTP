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
from a3c_module import DownloaderEnv, ActorCritic, Agent
from multiprocessing import Barrier, Manager
import matplotlib.pyplot as plt

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    file_size_str = "4G"
    iteration = 10
    num_servers = 3
    folder_name = "Log"
    
    urls = ["http://10.129.138.2/",
                "http://10.141.8.2/",
                "http://10.135.141.2/"]
    sync_barrier = Barrier(num_servers)
    manager = Manager()
    reward_history = manager.list()
    loss_history = manager.list()
    
    input_dims = [2]  # [download_time, throughput]
    n_actions = 4     # chunk sizes: 20MB, 40MB, 80MB, 160MB
    lr = 1e-4
  
    
    
    global_model = ActorCritic(input_dims)
    global_model.share_memory()
    optimizer = T.optim.Adam(global_model.parameters(), lr=lr)
    global_ep_idx = mp.Value('i', 0)
    
    workers = [
        Agent(global_model, optimizer, input_dims, gamma=0.99, lr=lr,
              name=idx, worker_id=idx, global_ep_idx=global_ep_idx, file_size_str=file_size_str,
              iteration=iteration, num_servers=num_servers, folder_name=folder_name, urls=urls, sync_barrier=sync_barrier,
                  reward_history=reward_history, loss_history=loss_history)
        for idx in range(num_servers)
    ]
    
    [w.start() for w in workers]
    [w.join() for w in workers]

    # Convert shared lists to normal Python lists
    rewards = sorted(list(reward_history), key=lambda x: x[0])
    losses = sorted(list(loss_history), key=lambda x: x[0])
    episodes, reward_vals = zip(*rewards)
    _, loss_vals = zip(*losses)
    
    # Plot Rewards
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, reward_vals, label='Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Convergence - Rewards')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, loss_vals, color='red', label='Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Convergence - Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

