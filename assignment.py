import asyncio
import aiohttp
import logging
from collections import deque
import time
import sys
import datetime as dt
import requests
from scipy.stats import gmean
import hashlib
import math
import statistics
import itertools
import queue
import threading
import async_timeout



file_sources= [
    "http://10.135.140.2/2G", "http://10.141.1.2/2G", "http://10.135.132.2/2G",
    "http://10.132.132.2/2G",   "http://10.136.130.2/2G","http://10.134.131.2/2G"
]

response = requests.head(file_sources[0])
file_size = int(response.headers.get('Content-Length', 0))
#print("File size:", file_size)

start_byte = 0
end_byte = 0
currentRequestedByte = 0
throughput_gm = 0
number_of_chunk = 0
count = 0
remaining_chunk_queue = []
time_download = []
combination_time = []
wait_flag = False

# --- NEW: combination state ---
# Maps url -> 'fast' or 'slow'
server_roles = {}
# The computed greedy assignment: maps url -> list of chunk sizes to download
greedy_assignment = {}
# Lock for reading/writing greedy_assignment and server_roles
combo_lock = threading.Lock()
# Tracks the last ranking snapshot that triggered a recalculation
last_ranking_snapshot = ()
# Background thread handle
combo_thread = None


# ---------------------------------------------------------------------------
# CHANGED: Replaced find_chunk_combinations + find_best_server_combination
#          with a simple greedy assignment computed in a background thread.
# ---------------------------------------------------------------------------

def classify_servers(throughput):
    """
    Simple O(n) classification: compare each server to the geometric mean.
    Returns a dict {url: 'fast'|'slow'} and the ranking tuple (sorted urls by speed).
    """
    if not throughput:
        return {}, ()
    values = list(throughput.values())
    gm = gmean(values)
    roles = {}
    for url, speed in throughput.items():
        roles[url] = 'fast' if speed >= gm else 'slow'
    # ranking tuple: sorted urls by descending throughput (used to detect changes)
    ranking = tuple(sorted(throughput, key=lambda u: throughput[u], reverse=True))
    return roles, ranking


def compute_greedy_assignment(throughput, big_chunk, small_chunk, urls):
    """
    Greedy O(n) assignment:
      - Fast servers  -> big chunks
      - Slow servers  -> small chunks
    Returns a dict {url: chunk_size}.
    This runs in a background thread so it never blocks the download coroutines.
    """
    roles, _ = classify_servers(throughput)
    assignment = {}
    for url in urls:
        if roles.get(url) == 'fast':
            assignment[url] = big_chunk
        else:
            assignment[url] = small_chunk
    logging.info(f"Greedy assignment computed: { {u: assignment[u]//(1024*1024) for u in assignment} } MB per server")
    return assignment


def _background_compute(throughput_snapshot, big_chunk, small_chunk, urls):
    """
    Runs in a daemon thread. Computes the greedy assignment and writes it to
    greedy_assignment / server_roles under combo_lock.
    """
    global greedy_assignment, server_roles, last_ranking_snapshot
    new_assignment = compute_greedy_assignment(throughput_snapshot, big_chunk, small_chunk, urls)
    new_roles, new_ranking = classify_servers(throughput_snapshot)
    with combo_lock:
        greedy_assignment = new_assignment
        server_roles = new_roles
        last_ranking_snapshot = new_ranking
    #print(f"[combo thread] assignment updated: { {u.split('/')[-2]: new_assignment[u]//(1024*1024) for u in new_assignment} } MB")


def maybe_trigger_recompute(throughput, big_chunk, small_chunk, urls):
    """
    Called after every fetch. If the fast/slow ranking has changed since the
    last computation, spawn a new background thread to recompute.
    Does NOT block the caller.
    """
    global combo_thread, last_ranking_snapshot
    if not throughput:
        return
    _, current_ranking = classify_servers(throughput)
    with combo_lock:
        ranking_unchanged = (current_ranking == last_ranking_snapshot)
    if ranking_unchanged:
        return  # reuse existing assignment
    # Ranking changed — recompute in background
    if combo_thread is not None and combo_thread.is_alive():
        return  # a recompute is already in flight; let it finish
    throughput_snapshot = dict(throughput)  # snapshot to avoid race
    combo_thread = threading.Thread(
        target=_background_compute,
        args=(throughput_snapshot, big_chunk, small_chunk, urls),
        daemon=True
    )
    combo_thread.start()
    logging.info("Background combination thread started due to ranking change.")


def get_chunk_size_for(url):
    """
    Read the current greedy assignment for this url (non-blocking, under lock).
    Falls back to big_chunk if no assignment is ready yet.
    """
    big_chunk = 40 * 1024 * 1024
    with combo_lock:
        return greedy_assignment.get(url, big_chunk)
# ---------------------------------------------------------------------------


async def fetch_range(session, url, s_b, e_b, throughput, range_dict, file_path):
    global start_byte, end_byte, number_of_chunk
    local_start_byte = s_b
    local_end_byte = e_b
    #print("updating start byte")
    start_byte += e_b - s_b + 1
    end_byte = e_b
    #print("fetching")
    #print(f"Fetching range::::: {local_start_byte}-{local_end_byte} from {url}")
    logging.info(f"Fetching range::::: {local_start_byte}-{local_end_byte} from {url}")
    headers = {'Range': f"bytes={local_start_byte}-{local_end_byte}"}
    start_time = time.time()
    async with async_timeout.timeout(200):
        async with session.get(url, headers=headers) as response:
            data = await response.read()
            end_time = time.time()
            delay = end_time - start_time
            #print("delay:", delay)
            logging.info(f"Delay {delay}")
            throughput[url] = (local_end_byte - local_start_byte) / delay
            range_dict[local_start_byte] = data
            #print("throughput", throughput)
            logging.info(f"Successfully fetched range {local_start_byte}-{local_end_byte} from {url}")
            logging.info(f"Throughput: {throughput}")
            number_of_chunk = number_of_chunk + 1
            #print("number_of_chunk:", number_of_chunk)
            logging.info(f"number_of_chunk: {number_of_chunk}")
            return data, throughput


async def download_file(session, url, throughput, range_dict, file_path, urls, idx):
    big_chunk = 5 * 1024 * 1024
    global throughput_gm
    global end_byte, start_byte

    end_byte = start_byte + big_chunk - 1
    #print(start_byte, end_byte)
    data, throughput = await fetch_range(session, url, start_byte, end_byte, throughput, range_dict, file_path)

    BIG = 40 * 1024 * 1024
    SMALL = 10 * 1024 * 1024

    while end_byte < int(0.97 * file_size):
        # After every fetch, check if the ranking changed and trigger background recompute
        maybe_trigger_recompute(throughput, BIG, SMALL, urls)

        # Read current chunk size assignment for this url (non-blocking)
        chunk_size = get_chunk_size_for(url)

        chunk_start = start_byte
        chunk_end = chunk_start + chunk_size - 1
        if chunk_end > file_size:
            chunk_end = file_size
        data, throughput = await fetch_range(session, url, chunk_start, chunk_end, throughput, range_dict, file_path)

    # Remaining tail: use assigned chunk sizes for the last stretch
    while end_byte < file_size:
        maybe_trigger_recompute(throughput, BIG, SMALL, urls)
        chunk_size = get_chunk_size_for(url)
        chunk_start = start_byte
        chunk_end = min(chunk_start + chunk_size - 1, file_size)
        if chunk_start > file_size:
            break
        data, throughput = await fetch_range(session, url, chunk_start, chunk_end, throughput, range_dict, file_path)


async def main():
    start_time = time.time()
    now = dt.datetime.now()
    formatted_now = now.strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"Dynamic_Chunking_Async_{formatted_now}.log"
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
    session_dict = {url: aiohttp.ClientSession() for url in file_sources}
    filename = "output_file"
    file_path = "/Users/sabdollah42/Desktop/output1.txt"
    throughput = {}
    range_dict = {}

    await asyncio.gather(
        *(download_file(session_dict[url], url, throughput, range_dict, file_path, file_sources, idx)
          for idx, url in enumerate(file_sources))
    )
    for session in session_dict.values():
        await session.close()
    logging.info("All chunks downloaded, writing to disk.")
    with open(filename, 'wb') as f:
        for sb in sorted(range_dict.keys()):
            data = range_dict[sb]
            f.write(data)
    end_time = time.time()
    print("they printed all")
    delay = end_time - start_time
    print("Delay:", delay)
    logging.info(f"Delay {delay}")
    total_throughput = file_size / delay
    print("total_throughput:", total_throughput)
    logging.info(f"Total download throughput: {total_throughput}")


if __name__ == "__main__":
    asyncio.run(main())
