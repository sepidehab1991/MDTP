#!/bin/bash
#!/bin/bash

# List of folder names
folders=("2_log" "3_log" "4_log" "5_log" "6_log")

# Iterate over each folder name
for folder in "${folders[@]}"; do
    # Create the folder
    mkdir -p "$folder"

    # Extract the number of servers from the folder name
    num_servers=${folder%%_*}  # Extracts the number before the underscore (_)

    # Run run.sh with the extracted number of servers and folder name
    echo "Running run.sh with arguments: $num_servers $folder"
    bash run.sh "$num_servers" "$folder"
done

echo "All tasks completed."

