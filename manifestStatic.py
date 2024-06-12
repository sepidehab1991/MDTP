import hashlib
import time

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

def divide_file_into_chunks(file_size,file_path, num_chunks):
    """Divide a file into the specified number of chunks and calculate checksums."""
    chunks = []
    chunk_size = os.path.getsize(file_path) // num_chunks
    with open(file_path, 'rb') as f:
        start_byte = 0
        for chunk_num in range(num_chunks):
            data = f.read(chunk_size)
            end_byte = start_byte + len(data) - 1
            chunk = Chunk(chunk_num, data, start_byte, end_byte)
            chunks.append(chunk)
            start_byte = end_byte + 1
        chunk= Chunk(file_size - end_byte, f.read(file_size-end_byte+1) ,end_byte+1, file_size )
        chunks.append(chunk)
    return chunks

def write_checksums_to_file(chunks, output_file):
    """Write checksums and byte range information to the output file."""
    with open(output_file, 'w') as f:
        for chunk in chunks:
            f.write(f"Chunk {chunk.number} - Start Byte: {chunk.start_byte}, End Byte: {chunk.end_byte}\n")
            #f.write(f"Checksum: {chunk.checksum}\n\n")
            f.write(f"{chunk.checksum}\n")

# Example usage
import os

input_file = "/Users/sabdollah42/Desktop/ICC-Data/updatedManifest/Fedora-Everything-netinst-x86_64-38-1.6.iso"
file_size= os.path.getsize(input_file)
num_chunks = 74  # Specify the number of chunks you want
output_file = "output2.txt"
chunks = divide_file_into_chunks(file_size, input_file, num_chunks)

write_checksums_to_file(chunks, output_file)
print("Total time:",total_time)
