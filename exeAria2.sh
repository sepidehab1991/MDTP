#!/bin/bash
#!/bin/bash

# Output CSV file
output_file="elapsed_time_results.csv"

# Write header to CSV file
echo "Run,Elapsed_Time" > "$output_file"

# Run the script 10 times
for i in {1..2}
do
    # Run the script and capture output
    output=$(./DiskAria2.sh)
    
    # Extract ELAPSED_TIME from the output
    elapsed_time=$(echo "$output" | grep -oP '(?<=Delay: )\d+')

    # Append the result to the CSV file
    echo "$i,$elapsed_time" >> "$output_file"

    # Optional: Print output to terminal
    echo "Run $i: Elapsed Time = $elapsed_time"
done

echo "Results saved to $output_file"

