import asyncio
import aiohttp
import logging
import threading
import time
import datetime as dt
import requests
import math
import sys
import async_timeout
from scipy.stats import gmean

# ─────────────────────────────────────────────
# Command line args
# ─────────────────────────────────────────────

file_size_arg  = sys.argv[1]
iteration      = sys.argv[2]
num_servers    = int(sys.argv[3])
folder_name    = sys.argv[4]

# ─────────────────────────────────────────────
# Server URLs
# ─────────────────────────────────────────────

def get_file_sources(urls, num_sources):
    if num_sources > len(urls):
        raise ValueError("Number of sources exceeds the number of available URLs.")
    return urls[:num_sources]

urls         = ["http://10.135.140.2/", "http://10.141.1.2/", "http://10.135.132.2/",
                "http://10.132.132.2/",  "http://10.136.130.2/", "http://10.131.131.2/"]
file_ips     = get_file_sources(urls, num_servers)
file_sources = [ip + file_size_arg for ip in file_ips]
print(file_sources)

# Get actual file size from HTTP HEAD
response  = requests.head(file_sources[0])
file_size = int(response.headers.get('Content-Length', 0))
print("File size:", file_size)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

INITIAL_CHUNK      = 5 * 1024 * 1024  # 5 MB equal probe to measure throughput
REBALANCE_INTERVAL = 3.0               # seconds between rebalance checks
THROUGHPUT_ALPHA   = 0.3               # EWMA smoothing factor
MIN_CHUNK_BYTES    = 512 * 1024        # minimum bytes per server (512 KB)
STREAM_CHUNK_SIZE  = 256 * 1024        # internal read size per iteration

# ─────────────────────────────────────────────
# Shared state (protected by state_lock)
# ─────────────────────────────────────────────

server_state = {
    url: {
        "start_byte":  0,
        "end_byte":    0,
        "current_pos": 0,
        "throughput":  0.0,
        "done":        False,
        "failed":      False,
    }
    for url in file_sources
}
state_lock = threading.Lock()

# ─────────────────────────────────────────────
# Step 1: Probe — equal 5MB chunk to measure throughput
# ─────────────────────────────────────────────

async def probe_server(session, url, output_file, file_lock):
    """
    Download INITIAL_CHUNK bytes from the server's assigned start_byte.
    Measures throughput and stores it in server_state[url]["throughput"].
    Writes bytes directly to the output file at the correct offset.
    """
    with state_lock:
        s_b = server_state[url]["start_byte"]
        e_b = s_b + INITIAL_CHUNK - 1

    logging.info(f"Probe: {url} bytes {s_b:,}–{e_b:,}")
    headers    = {'Range': f"bytes={s_b}-{e_b}"}
    t_start    = time.time()
    bytes_recv = 0

    try:
        async with async_timeout.timeout(500):
            async with session.get(url, headers=headers) as response:
                async for chunk in response.content.iter_chunked(STREAM_CHUNK_SIZE):
                    chunk_len = len(chunk)

                    with state_lock:
                        pos = server_state[url]["current_pos"]
                        server_state[url]["current_pos"] += chunk_len

                    async with file_lock:
                        output_file.seek(pos)
                        output_file.write(chunk)

                    bytes_recv += chunk_len

        elapsed = max(time.time() - t_start, 1e-6)
        tp      = bytes_recv / elapsed
        with state_lock:
            server_state[url]["throughput"] = tp
        logging.info(f"Probe done: {url}  tp={tp/1e6:.2f} MB/s  bytes={bytes_recv:,}")

    except Exception as e:
        logging.error(f"Probe error {url}: {e}")
        with state_lock:
            server_state[url]["failed"] = True

# ─────────────────────────────────────────────
# Step 2: Throughput-proportional allocation
# ─────────────────────────────────────────────

def throughput_allocation():
    """
    After probe, compute:
      T = remaining_file / sum(throughputs)
      chunk_i = throughput_i * T

    Assigns new end_byte to each server starting from current_pos
    (which is after the probe chunk). Uses contiguous cursor to
    guarantee no gaps or overlaps.
    """
    with state_lock:
        active_urls = [url for url in file_sources
                       if not server_state[url]["failed"]]
        throughputs = {url: server_state[url]["throughput"] for url in active_urls}
        positions   = {url: server_state[url]["current_pos"] for url in active_urls}

    total_tp        = sum(throughputs.values())
    total_remaining = file_size - min(positions.values())

    if total_tp == 0:
        raise RuntimeError("All servers have zero throughput after probe.")

    T = total_remaining / total_tp
    logging.info(f"Throughput allocation: T={T:.1f}s  total_tp={total_tp/1e6:.2f} MB/s  "
                 f"remaining={total_remaining/1e6:.1f} MB")

    # Sort by current_pos so ranges are assigned contiguously
    sorted_urls = sorted(active_urls, key=lambda u: positions[u])
    cursor      = positions[sorted_urls[0]]

    with state_lock:
        for i, url in enumerate(sorted_urls):
            cursor = max(cursor, server_state[url]["current_pos"])

            if i == len(sorted_urls) - 1:
                new_end = file_size   # last server reaches end — no gaps
            else:
                fair_bytes = max(throughputs[url] * T, MIN_CHUNK_BYTES)
                new_end    = min(int(cursor + fair_bytes), file_size)

            server_state[url]["end_byte"] = new_end
            cursor = new_end
            logging.info(f"  {url}: pos={server_state[url]['current_pos']:,}  "
                         f"end={new_end:,}  tp={throughputs[url]/1e6:.2f} MB/s")

# ─────────────────────────────────────────────
# Step 3: fetch_range
# ─────────────────────────────────────────────

async def fetch_range(session, url, s_b, e_b, output_file, file_lock):
    """
    Fetch bytes [s_b, e_b] from url.
    Writes directly to pre-allocated file via seek(pos)+write(chunk).
    Atomically reads pos, checks trim, advances current_pos under one lock.
    """
    logging.info(f"fetch_range: {url} bytes {s_b:,}–{e_b:,}")
    headers    = {'Range': f"bytes={s_b}-{e_b}"}
    t_start    = time.time()
    bytes_recv = 0
    do_break   = False

    try:
        async with async_timeout.timeout(500):
            async with session.get(url, headers=headers) as response:
                async for chunk in response.content.iter_chunked(STREAM_CHUNK_SIZE):
                    chunk_len = len(chunk)

                    with state_lock:
                        pos     = server_state[url]["current_pos"]
                        new_end = server_state[url]["end_byte"]

                        if pos + chunk_len > new_end:
                            trimmed    = new_end - pos
                            write_data = bytes(chunk[:trimmed]) if trimmed > 0 else None
                            server_state[url]["current_pos"] += trimmed
                            bytes_recv += trimmed
                            do_break    = True
                        else:
                            write_data  = bytes(chunk)
                            server_state[url]["current_pos"] += chunk_len
                            bytes_recv += chunk_len

                    if write_data:
                        async with file_lock:
                            output_file.seek(pos)
                            output_file.write(write_data)

                    if do_break:
                        logging.info(f"fetch_range: {url} trimmed at {new_end:,}")
                        break

                    # Update EWMA throughput
                    elapsed    = max(time.time() - t_start, 1e-6)
                    instant_tp = bytes_recv / elapsed
                    with state_lock:
                        prev = server_state[url]["throughput"]
                        server_state[url]["throughput"] = (
                            instant_tp if prev == 0
                            else THROUGHPUT_ALPHA * instant_tp + (1 - THROUGHPUT_ALPHA) * prev
                        )

        elapsed = max(time.time() - t_start, 1e-6)
        logging.info(f"fetch_range done: {url} bytes {s_b:,}–{s_b+bytes_recv:,} "
                     f"tp={bytes_recv/elapsed/1e6:.2f} MB/s")

    except Exception as e:
        logging.error(f"fetch_range error {url}: {e}")
        with state_lock:
            server_state[url]["failed"] = True

# ─────────────────────────────────────────────
# Step 4: download_file — per-server coroutine
# ─────────────────────────────────────────────

async def download_file(session, url, output_file, file_lock):
    logging.info(f"download_file started: {url}")

    while True:
        with state_lock:
            pos = server_state[url]["current_pos"]
            end = server_state[url]["end_byte"]

        if pos >= end:
            with state_lock:
                server_state[url]["done"] = True
            await asyncio.sleep(0.5)
            with state_lock:
                pos = server_state[url]["current_pos"]
                end = server_state[url]["end_byte"]
            if pos >= end:
                break
            else:
                with state_lock:
                    server_state[url]["done"] = False
                logging.info(f"download_file: {url} extended to {end:,}, resuming")
                continue

        with state_lock:
            failed = server_state[url]["failed"]
        if failed:
            break

        await fetch_range(session, url, pos, end, output_file, file_lock)

    with state_lock:
        server_state[url]["done"] = True
    logging.info(f"download_file finished: {url} at pos {server_state[url]['current_pos']:,}")

# ─────────────────────────────────────────────
# Step 5: Rebalancer thread
# ─────────────────────────────────────────────

def rebalancer_thread(stop_event):
    """
    Runs every REBALANCE_INTERVAL seconds in a separate thread.
    Does NOT block the asyncio download loop.

    Algorithm:
      total_remaining = sum of (end_byte - current_pos) across active servers
      total_tp        = sum of throughputs
      T_remaining     = total_remaining / total_tp
      new_end_i       = cursor + throughput_i * T_remaining  (contiguous cursor)

    All servers finish at the same T_remaining.
    Excess from slow servers distributed to ALL servers proportionally.
    """
    rebalance_count = 0

    while not stop_event.is_set():
        time.sleep(REBALANCE_INTERVAL)
        if stop_event.is_set():
            break

        with state_lock:
            snapshot = {
                url: {
                    "pos":    s["current_pos"],
                    "end":    s["end_byte"],
                    "tp":     s["throughput"],
                    "failed": s["failed"],
                    "done":   s["done"],
                }
                for url, s in server_state.items()
            }

        active = {url: s for url, s in snapshot.items() if not s["failed"]}
        if not active:
            logging.warning("Rebalancer: all servers failed.")
            break

        measured = {url: s for url, s in active.items() if s["tp"] > 0}
        if not measured:
            logging.info("Rebalancer: no throughput data yet, skipping.")
            continue

        total_remaining = sum(max(0, s["end"] - s["pos"]) for s in active.values())
        total_tp        = sum(s["tp"] for s in measured.values())

        if total_tp == 0 or total_remaining == 0:
            continue

        T_remaining = total_remaining / total_tp
        rebalance_count += 1

        tp_values = [s["tp"] for s in measured.values()]
        tp_gm     = gmean(tp_values) if len(tp_values) > 1 else tp_values[0]
        logging.info(
            f"Rebalancer #{rebalance_count}: "
            f"remaining={total_remaining/1e6:.2f} MB  "
            f"total_tp={total_tp/1e6:.2f} MB/s  "
            f"gmean_tp={tp_gm/1e6:.2f} MB/s  "
            f"T_remaining={T_remaining:.1f}s"
        )

        sorted_urls = sorted(active.keys(), key=lambda u: active[u]["pos"])
        new_ends    = {}
        cursor      = active[sorted_urls[0]]["pos"]

        for i, url in enumerate(sorted_urls):
            cursor = max(cursor, active[url]["pos"])

            if i == len(sorted_urls) - 1:
                new_end = file_size
            else:
                tp         = active[url]["tp"] if active[url]["tp"] > 0 else 0
                fair_bytes = max(tp * T_remaining, MIN_CHUNK_BYTES)
                new_end    = min(int(cursor + fair_bytes), file_size)

            new_ends[url] = new_end
            cursor        = new_end

        with state_lock:
            for url, new_end in new_ends.items():
                old_end = server_state[url]["end_byte"]
                if new_end != old_end:
                    direction = "trimmed" if new_end < old_end else "extended"
                    logging.info(
                        f"  {url}: end_byte {old_end:,} → {new_end:,} ({direction})  "
                        f"tp={server_state[url]['throughput']/1e6:.2f} MB/s"
                    )
                    server_state[url]["end_byte"] = new_end
                    if new_end > server_state[url]["current_pos"]:
                        server_state[url]["done"] = False

# ─────────────────────────────────────────────
# Step 6: Main
# ─────────────────────────────────────────────

async def main():
    start_time = time.time()

    now           = dt.datetime.now()
    formatted_now = now.strftime("%Y-%m-%d_%H-%M-%S")
    log_file      = (f"{folder_name}/Dynamic_Rebalance_"
                     f"{formatted_now}_{num_servers}_{iteration}_{file_size}.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )

    # Set equal start positions for probe — each server probes from its own
    # equal slice so probe bytes are not wasted and land at correct file offsets
    n      = len(file_sources)
    chunk  = file_size // n
    cursor = 0
    with state_lock:
        for i, url in enumerate(file_sources):
            server_state[url]["start_byte"]  = cursor
            server_state[url]["current_pos"] = cursor
            server_state[url]["end_byte"]    = cursor + INITIAL_CHUNK
            cursor += chunk
    logging.info("Probe allocation set.")

    # Pre-allocate output file
    filename = f"{folder_name}/output_file"
    with open(filename, 'wb') as f:
        f.seek(file_size - 1)
        f.write(b'\0')
    logging.info(f"Pre-allocated output file: {filename} ({file_size:,} bytes)")

    file_lock    = asyncio.Lock()
    session_dict = {url: aiohttp.ClientSession() for url in file_sources}

    with open(filename, 'r+b') as output_file:

        # Phase 1: probe all servers simultaneously with equal 5MB chunk
        logging.info("=== Phase 1: Probing throughput ===")
        await asyncio.gather(
            *(probe_server(session_dict[url], url, output_file, file_lock)
              for url in file_sources)
        )

        # Phase 2: throughput-proportional allocation for real download
        logging.info("=== Phase 2: Throughput-proportional allocation ===")
        throughput_allocation()

        # Phase 3: start rebalancer thread then download
        logging.info("=== Phase 3: Downloading with rebalancer ===")
        stop_event = threading.Event()
        rb_thread  = threading.Thread(
            target=rebalancer_thread,
            args=(stop_event,),
            daemon=True,
            name="rebalancer"
        )
        rb_thread.start()

        await asyncio.gather(
            *(download_file(session_dict[url], url, output_file, file_lock)
              for url in file_sources)
        )

    for session in session_dict.values():
        await session.close()

    stop_event.set()
    rb_thread.join(timeout=5)

    end_time = time.time()
    delay    = end_time - start_time

    print("They printed all")
    print(f"File size,{file_size},Delay,{delay}")
    logging.info(f"Total download delay: {delay}")


if __name__ == "__main__":
    asyncio.run(main())
