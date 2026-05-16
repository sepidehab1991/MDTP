import asyncio
import aiohttp
import logging
from collections import deque
import time
import sys
import datetime as dt
import hashlib

def calculate_checksum(data):
    if not isinstance(data, bytes):
        data = str(data).encode('utf-8')
    sha256 = hashlib.sha256()
    sha256.update(data)
    return sha256.hexdigest().encode('utf-8')

# It reads original checksum from the file line by line
def read_original_checksum(file, line_number):
    with open(file, "r") as fp:
        cur_line = 0
        while True:
            line = fp.readline().strip()
            if not line:
                break
            elif cur_line == line_number:
                return line
            cur_line = cur_line + 1

async def compare_files(url, data, chunk_size, offset , file_path):

    chunks_to_retransmit = []
    line_number=int(offset/chunk_size)
    print("line_number:", line_number)
    #read each chunk line by line from the original file
    checksum1 = read_original_checksum(file_path, line_number)
    # decode for converting from binary to string
    checksum2 = calculate_checksum(data).decode()
    print(f"[{checksum1}] {type(checksum1)}")
    print(f"[{checksum2}] {type(checksum2)}")

    if checksum1 == checksum2:
        print(f"Checksums match for line {line_number}.")
    else:
        print(f"Checksums do not match for line {line_number}.")
        print(f"Checksums do not match for offset {offset}.")
        print(f"Line {line_number} in file1: {line_number}")
        chunks_to_retransmit.append((offset, offset+chunk_size-1))

    await retransmit_chunks(url, chunks_to_retransmit)

async def retransmit_chunks(url, chunks_to_retransmit):
    for start_byte, end_byte in chunks_to_retransmit:
        # Open a new session for each chunk download
        new_session = aiohttp.ClientSession()
        try:
            await fetch_range(new_session, url, start_byte, end_byte)
            print("The new downloaded chunk:", start_byte, end_byte)
        except Exception as e:
            logging.error(f"An error occurred while retransmitting range {start_byte}-{end_byte} from {url}: {e}")
        finally:
            await new_session.close()        
    
async def fetch_range(session, url, start_byte, end_byte):
    logging.info(f"Fetching range {start_byte}-{end_byte} from {url}")
    headers = {'Range': f"bytes={start_byte}-{end_byte}"}
    async with session.get(url, headers=headers) as response:
        data = await response.read()
        logging.info(f"Successfully fetched range {start_byte}-{end_byte} from {url}")
        return data, start_byte

async def download_file(session, url, chunk_queue, chunks_dict, chunk_size, start_byte, file_path):
    while chunk_queue:
        start_byte, end_byte = chunk_queue.popleft()
        try:
            logging.info(f"Client for {url} picking up range {start_byte}-{end_byte}")
            data, offset = await fetch_range(session, url, start_byte, end_byte)
            await compare_files(url, data, chunk_size, offset , file_path)
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
    file_sources = ["https://mirror.dst.ca/fedora-linux/fedora/linux/releases/38/Everything/x86_64/iso/Fedora-Everything-netinst-x86_64-38-1.6.iso",
            "http://muug.ca/mirror/fedora/linux/releases/38/Everything/x86_64/iso/Fedora-Everything-netinst-x86_64-38-1.6.iso"
            ]
    num_chunks = int(sys.argv[1])
    filename = "output_file"
    file_path= "/Users/sabdollah42/Desktop/output2.txt"
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
        *(download_file(session_dict[url], url, chunk_queue, chunks_dict, chunk_size, num_chunks, file_path) for url in file_sources)
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
