#!/bin/bash

# Output CSV file
output_file="delay_results.csv"

# Write header to CSV file
echo "Run,Delay" > "$output_file"

# Run the Python script 10 times
for i in {1..10}
do
    # Capture the output of the Python script
    delay=$(python3 async.py 204)
    
    # Append the result to the CSV file
    echo "$i,$delay" >> "$output_file"

    # Optional: Print output to terminal
    echo "Run $i: Delay = $delay"
done

echo "Results saved to $output_file"

