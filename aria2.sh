#!/bin/bash

# URLs to download the file from
URL1="http://10.132.141.2/32G"
URL2="http://10.133.138.2/32G"
URL3="http://10.138.135.2/32G"
URL4="http://10.147.165.2/32G"
URL5="http://10.136.136.2/32G"
URL6="http://10.130.7.2/32G"

# Start time
START_TIME=$(date +%s)

# Download files using aria2c but discard output and disable fragmentation
aria2c --file-allocation=none --allow-overwrite=true -o /dev/null \
       $URL1 $URL2 $URL3 $URL4 $URL5 $URL6

# End time
END_TIME=$(date +%s)

# Calculate the elapsed time
ELAPSED_TIME=$(($END_TIME - $START_TIME))

# Display the time taken
echo "Download time (excluding disk I/O, no fragmentation): $ELAPSED_TIME seconds"

