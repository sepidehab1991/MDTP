import asyncio
import aiohttp
import logging
from collections import deque
import time
import datetime as dt
import requests
from scipy.stats import gmean
import math
import hashlib


#file_sources = ["http://10.132.2.2/ubuntu-23.10.1-desktop-amd64.iso", "http://10.133.34.2/ubuntu-23.10.1-desktop-amd64.iso","http://10.137.24.2/ubuntu-23.10.1-desktop-amd64.iso","http://10.130.156.2/ubuntu-23.10.1-desktop-amd64.iso", "http://10.135.136.2/ubuntu-23.10.1-desktop-amd64.iso", "http://10.137.132.2/ubuntu-23.10.1-desktop-amd64.iso"]
file_sources = ["https://fedora.mirror.digitalpacific.com.au/linux/releases/38/Everything/x86_64/iso/Fedora-Everything-netinst-x86_64-38-1.6.iso",
                "https://ftp-stud.hs-esslingen.de/pub/Mirrors/archive.fedoraproject.org/fedora/linux/releases/38/Everything/x86_64/iso/Fedora-Everything-netinst-x86_64-38-1.6.iso"] 
response = requests.head(file_sources[0])
file_size = int(response.headers.get('Content-Length', 0))
print("File size:", file_size)

start_byte=0
end_byte=0
currentRequestedByte=0
throughput_gm= 0
number_of_chunk=0
fast_downloadTime= 0

def calculate_checksum(data):
    sha256 = hashlib.sha256()
    sha256.update(data)
    return sha256.hexdigest()


def read_original_checksum(file, line_number):
    with open(file, "r") as fp:
        for cur_line, line in enumerate(fp):
            if cur_line == line_number:
                text = line.strip()
                print(text)
                return text.split(" ")[-1]
async def compare_files(url, data, start_byte, end_byte, file_path):
    print("22222222222222")
    print("start Byte, end_byte:", start_byte, end_byte)
    print("hiiiii")
    chunks_to_retransmit = []
    chunk_size = 1048576
    print("??????????")
    if (end_byte - start_byte) == chunk_size-1:
        #print("end_byte - start_byte",end_byte - start_byte)
        print("start byte, end byte:",start_byte, end_byte)
        print("@@@@@@@@@@@")
        line_number = ((start_byte//chunk_size))
        print("line_number:", line_number)
        #read each chunk line by line from the original file
        checksum1 = read_original_checksum(file_path, line_number)
        print("checksum1:", checksum1)
        # decode for converting from binary to string
        checksum2 = calculate_checksum(data)
        logging.info(f"checksum1, checksum2: {checksum1}, {checksum2}")
        print("checksum2:", checksum2)
        # print(f"[{checksum1}] {type(checksum1)}")
        # print(f"[{checksum2}] {type(checksum2)}")
        print("///////////")
        if checksum1 == checksum2:
            print(f"Checksums match for line {line_number}.")
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        else:
            print(f"Checksums do not match for line {line_number}.")
            print(f"Checksums do not match for offset {start_byte}.")
            print(f"Line {line_number} in file1: {line_number}")
            chunks_to_retransmit.append((start_byte, start_byte+chunk_size-1))

    else:
        print("**************")
        print("start Byte, end_byte:", start_byte, end_byte)
        coefficient = (end_byte - start_byte+1)//chunk_size
        print("&&&&&&&&&coefficient:", coefficient)
        for i in range(coefficient):
            print("!!!!!!!!!!!!!!!")
            offset = start_byte + i * chunk_size
            data_part = data[i*chunk_size:i*chunk_size+1*1024*1024]
            line_number = int(offset / chunk_size)
            print("line_number:", line_number)
            print("hhhhhhhhhhh",i*chunk_size, i*chunk_size+1*1024*1024 -1)
            checksum1 = read_original_checksum(file_path, line_number)
            # decode for converting from binary to string
            checksum2 = calculate_checksum(data_part)
            print(f"[{checksum1}] {type(checksum1)}")
            print(f"[{checksum2}] {type(checksum2)}")
            logging.info(f"checksum1, checksum2: {checksum1}, {checksum2}")
            if checksum1 == checksum2:
                print(f"Checksums match for line {line_number}.")
                print("helooooooooooooooooooooooooooo")
            else:
                print(f"Checksums do not match for line {line_number}.")
                print(f"Checksums do not match for offset {start_byte}.")
                print(f"Line {line_number} in file1: {line_number}")
                chunks_to_retransmit.append((start_byte, start_byte+chunk_size-1))

async def fetch_range(session, url, s_b, e_b, throughput, range_dict, file_path):
    logging.info(f"Fetching range for URL: {url}, start_byte={s_b}, end_byte={e_b}")
    global start_byte, end_byte, number_of_chunk
    local_start_byte= s_b
    local_end_byte = e_b
    print("updating start byte")
    start_byte += e_b - s_b +1
    end_byte = e_b
    print("fetching")
    print(f"Fetching range::::: {local_start_byte}-{local_end_byte} from {url}")
    logging.info(f"Fetching range::::: {local_start_byte}-{local_end_byte} from {url}")
    headers = {'Range': f"bytes={local_start_byte}-{local_end_byte}"}
    start_time = time.time()
    async with session.get(url, headers=headers) as response:
        data = await response.read()
        end_time= time.time()
        delay = end_time - start_time
        print("delay:", delay)
        logging.info(f"Delay {delay}")
        await compare_files(url, data,local_start_byte, local_end_byte, file_path)
        throughput[url]= (local_end_byte - local_start_byte)/delay
        range_dict[local_start_byte]= data
        print("throughput",throughput)
        logging.info(f"Successfully fetched range {local_start_byte}-{local_end_byte} from {url}")
        logging.info(f"Throughput: {throughput}")
        number_of_chunk= number_of_chunk + 1
        print("number_of_chunk:",number_of_chunk)
        logging.info(f"number_of_chunk: {number_of_chunk}")
        logging.info(f"Fetched range: start_byte={local_start_byte}, end_byte={local_end_byte}, delay={delay:.4f}, throughput={throughput[url]}")
        return data, throughput
    
async def download_file(session, url, throughput, range_dict, file_path):
    logging.info(f"Downloading file from URL: {url}")
    global fast_downloadTime
    initial_chunk= 10* 1024 * 1024
    large_chunk = 10 * 1024 * 1024
    global throughput_gm
    global end_byte, start_byte
    end_byte= start_byte + initial_chunk -1
    print ("111111111111",start_byte, end_byte)
    data, throughput = await fetch_range(session, url, start_byte, end_byte, throughput, range_dict, file_path)
    
    fast_dic={}
    slow_dic={}

    while end_byte < file_size:
        values= list(throughput.values())
        throughput_gm = gmean(values)
        print("throughput_gm", throughput_gm)
        logging.info(f"Geometric mean of throughput: {throughput_gm}")
        logging.info(f"start and end byte are: {start_byte} and {end_byte}")
        logging.info(f"Current throughput for {url}: {throughput[url]}")
        if math.ceil(throughput[url])>= math.ceil(throughput_gm):
            fast_dic[url]= throughput[url]
            fastest_server = max(fast_dic.values())
            print("fast server throughput:", fastest_server)
            logging.info(f"Fast server throughput: {fastest_server}")
            fast_downloadTime = large_chunk/fastest_server
            print("fast_downloadTime", fast_downloadTime)
            logging.info(f"Fast download time: {fast_downloadTime}")
            large_start_byte = start_byte
            logging.info(f"Large chunk start byte: {large_start_byte}")
            large_end_byte = large_start_byte + large_chunk -1
            logging.info(f"Large chunk end byte: {large_end_byte}")
            if large_end_byte>file_size:
                large_end_byte=file_size
            logging.info(f"Large_start_byte, Large_end_byte: {large_start_byte} and {large_end_byte}")
            print("large_start_byte and large_end_byte:", large_start_byte, large_end_byte)
            data, throughput = await fetch_range(session, url, large_start_byte, large_end_byte, throughput, range_dict, file_path)



        if throughput[url]< throughput_gm:
            #print("UUUUUU:", throughput[url])
            slow_dic[url]= throughput[url]
            print("values:",slow_dic[url])
            logging.info(f"Slow server throughput: {slow_dic[url]}")
            #print("fast_downloadTime",fast_downloadTime)
            #small_chunk = math.floor(fast_downloadTime* (slow_dic[url]))
            small_chunk =1024*1024 * round((fast_downloadTime* (slow_dic[url]))/(1024*1024))
            print("small chunk", small_chunk)
            logging.info(f"Small chunk size: {small_chunk}")
            small_start_byte = start_byte
            small_end_byte = small_start_byte + small_chunk -1
            if small_end_byte>file_size:
                small_end_byte=file_size
            logging.info(f"small_start_byte, small_end_byte: {small_start_byte} and {small_end_byte}")
            print("small_start_byte, small_end_byte:", small_start_byte, small_end_byte)
            data, throughput = await fetch_range(session, url, small_start_byte, small_end_byte, throughput, range_dict, file_path)
            

async def main():
    start_time = time.time()
    # Initialize logging
    now = dt.datetime.now()
    formatted_now = now.strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"Dynamic_Chunking_Async_{formatted_now}.log"
    logging.basicConfig(filename=log_file,level=logging.INFO,format='%(asctime)s:%(levelname)s:%(message)s')
    session_dict = {url: aiohttp.ClientSession() for url in file_sources}
    filename = "output_file"
    file_path= "output1.txt"
    throughput= {}
    range_dict = {}
    await asyncio.gather(
        *(download_file(session_dict[url], url, throughput, range_dict, file_path) for url in file_sources)
    )
    logging.info("All chunks downloaded, writing to disk.")
    with open(filename, 'wb') as f:
        for start_byte in sorted(range_dict.keys()):
            data = range_dict[start_byte]
            if isinstance(data, str):
                data = data.encode('utf-8')  # encode the string to bytes if it's a string
            f.write(data)
    end_time= time.time()
    print("they printed all")
    delay= end_time - start_time
    total_throughput= file_size/delay
    print("Delay:", delay)
    print("total_throughput:", total_throughput)
    logging.info(f"Delay {delay}")
    logging.info(f"Total download delay: {delay}")
    logging.info(f"Total download throughput: {total_throughput}")


if __name__ == "__main__":
    asyncio.run(main())
