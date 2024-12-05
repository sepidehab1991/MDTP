#!/bin/bash

# Get the current date
printf -v date '%(%Y-%m-%d)T\n' -1

# Output files
log_file="$2/output.txt"
csv_file="$2/output.csv"

# Redirect all output to log_file
exec > "$log_file" 2>&1

# Create the CSV file and write the header
echo "Iteration,FileSize,Delay" > "$csv_file"

# Loop to run the program 5 times
for ((i=1;i<=5;i++)); do
    for file_size in 1M 2M 4M 8M 16M 32M 64M 128M 1G 2G 4G 8G 16G 32G 64G 128G; do
        echo "Running new_binpacking.py: Iteration $i, FileSize $file_size, Iteration $i, Num Servers $1, Folder Name $2"
        python3 new_binpacking.py $file_size $i $1 $2
        echo "Finished Iteration $i, FileSize $file_size, Num Servers $num_servers"
        echo "----------------------"
        sleep 5
    done
done

# Parse the log file to extract Iteration, FileSize, and Delay
prev_line=""
iteration=""
file_size=""

while read -r line; do
    # Check if the line contains "Iteration" and "FileSize"
    if [[ $line =~ Iteration && $line =~ FileSize ]]; then
        iteration=$(echo "$line" | grep -oP '(?<=Iteration )\d+')
        file_size=$(echo "$line" | grep -oP '(?<=FileSize )\S+')
    fi

    # Check if the previous line contains "Delay" and current line contains "Finished"
    if [[ $prev_line =~ Delay && $line =~ Finished ]]; then
        # Extract the delay from the previous line
        delay=$(echo "$prev_line" | grep -oP '(?<=Delay,)\S+')
        # Append to the CSV file
        echo "$iteration,$file_size,$delay" >> "$csv_file"
    fi

    # Update the previous line
    prev_line="$line"
done < "$log_file"

echo "CSV file generated: $csv_file"

