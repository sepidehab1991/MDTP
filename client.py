import asyncio
import aiohttp
import logging
from collections import deque
import time
import sys
import datetime as dt

async def fetch_range(session, url, start_byte, end_byte):
    logging.info(f"Fetching range {start_byte}-{end_byte} from {url}")
    headers = {'Range': f"bytes={start_byte}-{end_byte}"}
    async with session.get(url, headers=headers) as response:
        data = await response.read()
        logging.info(f"Successfully fetched range {start_byte}-{end_byte} from {url}")
        return data, start_byte

async def download_file(session, url, chunk_queue, chunks_dict):
    while chunk_queue:
        start_byte, end_byte = chunk_queue.popleft()
        try:
            logging.info(f"Client for {url} picking up range {start_byte}-{end_byte}")
            data, offset = await fetch_range(session, url, start_byte, end_byte)
            chunks_dict[offset] = data
        except Exception as e:
            logging.error(f"An error occurred while processing range {start_byte}-{end_byte} from {url}: {e}")
            chunk_queue.append((start_byte, end_byte))

async def main():
    start_time = time.time()
    # Initialize logging
    now = dt.datetime.now()
    formatted_now = now.strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"Dynamic_Async_{formatted_now}.log"
    logging.basicConfig(filename=log_file,level=logging.INFO,format='%(asctime)s:%(levelname)s:%(message)s')

    # Configurable variables
    file_sources = ["https://mirror.dst.ca/fedora-linux/fedora/linux/releases/38/Everything/x86_64/iso/Fedora-Everything-netinst-x86_64-38-1.6.iso",
            "http://muug.ca/mirror/fedora/linux/releases/38/Everything/x86_64/iso/Fedora-Everything-netinst-x86_64-38-1.6.iso",
            "https://mirrors.bfsu.edu.cn/fedora/releases/38/Everything/x86_64/iso/Fedora-Everything-netinst-x86_64-38-1.6.iso"]
    num_chunks = int(sys.argv[1])
    filename = "output_file"

    chunk_queue = deque()
    chunks_dict = {}
    session_dict = {url: aiohttp.ClientSession() for url in file_sources}

    # Get file size from the first source
    async with session_dict[file_sources[0]].head(file_sources[0]) as response:
        file_size = int(response.headers['Content-Length'])
    logging.info(f"File size determined as {file_size} bytes.")

    chunk_size = file_size // num_chunks
    for i in range(num_chunks):
        start_byte = i * chunk_size
        end_byte = start_byte + chunk_size - 1 if i < num_chunks - 1 else file_size - 1
        chunk_queue.append((start_byte, end_byte))

    logging.info("Initialized chunk queue and starting download.")
    await asyncio.gather(
        *(download_file(session_dict[url], url, chunk_queue, chunks_dict) for url in file_sources)
    )

    # Close all sessions
    for session in session_dict.values():
        await session.close()

    # Combine and write all chunks to disk
    logging.info("All chunks downloaded, writing to disk.")
    with open(filename, 'wb') as f:
        for offset in sorted(chunks_dict.keys()):
            f.write(chunks_dict[offset])

    logging.info("File download and assembly complete.")

    end_time = time.time()
    delay=end_time-start_time
    logging.info(f"Number_of_chunk:{num_chunks}")
    logging.info("The number of server:3 Slow")
    logging.info(f"Delay:{delay}")
    print("Delay:",delay)

if __name__ == "__main__":
    asyncio.run(main())
