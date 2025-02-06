#!/bin/bash

FILENAME="$1"
LOGFILE="${FILENAME}_log"
CSV_FILE="download_time.csv"

# Capture start time with nanosecond precision
START=$(date +%s.%N)

# Download the file using aria2c
aria2c --seed-ratio=0 --seed-time=0 "$FILENAME" --log-level=info -l "$LOGFILE"

# Capture end time
END=$(date +%s.%N)

# Calculate elapsed time using bc (supports floating-point arithmetic)
ELAPSED_TIME=$(echo "$END - $START" | bc)

# Display the elapsed time
echo "Download time: $ELAPSED_TIME seconds"

# Check if the CSV file exists; if not, add headers
if [ ! -f "$CSV_FILE" ]; then
    echo "Filename,Start Time,End Time,Elapsed Time (seconds)" > "$CSV_FILE"
fi

# Append the result to the CSV file
echo "$FILENAME,$ELAPSED_TIME" >> "$CSV_FILE"

