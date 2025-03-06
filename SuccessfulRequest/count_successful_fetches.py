import re
import pandas as pd
import sys
def count_successful_fetches(log_file_path):
    """
    Reads a log file, extracts successful fetch entries, and counts occurrences for each unique IP.
    
    :param log_file_path: Path to the log file.
    :return: A dictionary with IPs as keys and counts as values.
    """
    pattern = r"Successfully fetched range \d+-\d+ from (http://\d+\.\d+\.\d+\.\d+/32G)"
    
    # Read log file
    with open(log_file_path, 'r') as file:
        log_content = file.read()
    
    # Extract all IPs that had successful fetches
    matches = re.findall(pattern, log_content)
    
    # Count occurrences of each unique IP
    ip_counts = pd.Series(matches).value_counts().to_dict()
    
    return ip_counts

if __name__ == "__main__":
    log_file_path = sys.argv[1]  # Change this to your actual log file path
    result = count_successful_fetches(log_file_path)
    
    print("Successful Fetch Counts:")
    for ip, count in result.items():
        print(f"{ip}: {count}")

