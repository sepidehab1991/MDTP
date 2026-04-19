#!/bin/bash
#!/bin/bash

# List of folder names
folders=("8_log")

# Iterate over each folder name
for folder in "${folders[@]}"; do
    # Create the folder
    mkdir -p "$folder"

    # Extract the number of servers from the folder name
    num_servers=6  # Extracts the number before the underscore (_)

    # Run run.sh with the extracted number of servers and folder name
    echo "Running run.sh with arguments: $num_servers $folder"
    bash run.sh "$num_servers" "$folder"
done

echo "All tasks completed."

