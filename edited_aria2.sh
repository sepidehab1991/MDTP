#!/bin/bash
# URLs to download the file from
URL1="http://10.135.140.2/4G"
URL2="http://10.141.1.2/4G"
URL3="http://10.135.132.2/4G"
URL4="http://10.132.132.2/4G"
URL5="http://10.131.131.2/4G"
URL6="http://10.136.130.2/4G"

LOG_FILE="/tmp/aria2.log"
> "$LOG_FILE"  # Clear log at start

# Start time (nanosecond precision)
START_TIME=$(date +%s%N)

# Download files using aria2c with 6 replicas and chunk logging
aria2c \
  --file-allocation=none \
  --allow-overwrite=true \
  --split=6 \
  --min-split-size=1M \
  --max-connection-per-server=6 \
  --max-concurrent-downloads=6 \
  --log="$LOG_FILE" \
  --log-level=debug \
  -o /dev/null \
  $URL1 $URL2 $URL3 $URL4 $URL5 $URL6

# End time (nanosecond precision)
END_TIME=$(date +%s%N)

# Calculate elapsed time in seconds with millisecond precision
ELAPSED_NS=$(( END_TIME - START_TIME ))
ELAPSED_SEC=$(echo "scale=3; $ELAPSED_NS / 1000000000" | bc)

# Parse and display successfully downloaded chunks from log
echo ""
echo "=== Successfully Downloaded Segments ==="
grep -E "Download complete|NOTICE.*download" "$LOG_FILE" | while read -r line; do
  echo "[CHUNK OK] $line"
done

# Display the time taken
echo ""
echo "Download time (excluding disk I/O, no fragmentation): ${ELAPSED_SEC} seconds"
echo "Full chunk log saved to: $LOG_FILE"
