import hashlib
import time
import os

total_time = 0.0
class Chunk:
    """A chunk of data with its associated metadata"""

    def __init__(self, number, data, start_byte, end_byte):
        self.number = number
        self.data = data
        self.start_byte = start_byte
        self.end_byte = end_byte
        start_time= time.time()
        
        self.checksum = self._generate_checksum()
        end_time=time.time()
        global total_time
        delay = end_time-start_time
        total_time+=delay
        print("Delay, total time, start_byte, end_byte:", delay, total_time, start_byte, end_byte)
  
        
        
        
    def _generate_checksum(self):
        """Generate checksum for the chunk's data."""
        sha256 = hashlib.sha256()
        sha256.update(self.data)
        return sha256.hexdigest()

def divide_file_into_chunks(file_path, chunks_size):
    """Divide a file into the specified number of chunks and calculate checksums."""
    chunks = []
    with open(file_path, 'rb') as f:
        start_byte = 0
        i =1
        file_size = os.path.getsize(file_path)
        print("size:", file_size)
        while i<= file_size/chunks_size:
            data = f.read(chunks_size)
            end_byte = start_byte + len(data) - 1
            chunk = Chunk(chunks_size, data, start_byte, end_byte)
            chunks.append(chunk)
            start_byte = end_byte + 1
            i +=1
        chunk= Chunk(file_size - end_byte, f.read(file_size-end_byte+1) ,end_byte+1, file_size )
        chunks.append(chunk)
        #print("chunk:",chunks)
        return chunks

def write_checksums_to_file(chunks, output_file):
    """Write checksums and byte range information to the output file."""
    with open(output_file, 'w') as f:
        for chunk in chunks:
            f.write(f"Chunk {chunk.number} - Start Byte: {chunk.start_byte}, End Byte: {chunk.end_byte}, Checksum: {chunk.checksum}\n")
            #f.write(f"{chunk.checksum}\n\n")

# Example usage


input_file = "/Users/sabdollah42/Desktop/Fedora-Everything-netinst-x86_64-38-1.6.iso"
chunks_size = 1 * 1024 * 1024
output_file = "output1.txt"
chunks = divide_file_into_chunks(input_file, chunks_size)

write_checksums_to_file(chunks, output_file)
print("Total time:",total_time)
