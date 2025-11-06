#!/bin/bash
set -e

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: Please run as root (use sudo)"
    exit 1
fi

# Parse arguments
INTERFACE=${1:-eth0}
TOTAL_BANDWIDTH=${2:-100}  # Total bandwidth in Mbps
OVERHEAD_FACTOR=${3:-1.25}

# Calculate speed range (25% to 75% of total bandwidth)
MIN_SPEED=$(awk -v total="$TOTAL_BANDWIDTH" 'BEGIN { printf "%.0f", total * 0.25 }')
MAX_SPEED=$(awk -v total="$TOTAL_BANDWIDTH" 'BEGIN { printf "%.0f", total * 0.75 }')

# Validate interface exists
if ! ip link show "$INTERFACE" &> /dev/null; then
    echo "ERROR: Interface $INTERFACE not found"
    echo ""
    echo "Available interfaces:"
    ip -br link show
    exit 1
fi

echo "=========================================="
echo "  Dynamic Bandwidth Fluctuation Simulator"
echo "=========================================="
echo "Interface:         $INTERFACE"
echo "Total Bandwidth:   ${TOTAL_BANDWIDTH} Mbps"
echo "Throttle Range:    ${MIN_SPEED}-${MAX_SPEED} Mbps (25%-75%)"
echo "Overhead Factor:   ${OVERHEAD_FACTOR}x"
echo "=========================================="
echo ""
echo "Press Ctrl+C to stop and cleanup"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "=========================================="
    echo "Cleaning up and removing throttling..."
    tc qdisc del dev "$INTERFACE" root 2>/dev/null || true
    echo "Cleanup complete. Network restored to normal."
    echo "=========================================="
    exit 0
}

# Set trap for cleanup on script exit
trap cleanup SIGINT SIGTERM EXIT

# Function to apply throttling
apply_throttle() {
    local speed=$1
    local percentage=$(awk -v speed="$speed" -v total="$TOTAL_BANDWIDTH" \
        'BEGIN { printf "%.0f", (speed / total) * 100 }')
    local configured_speed=$(awk -v target="$speed" -v factor="$OVERHEAD_FACTOR" \
        'BEGIN { printf "%.2f", target * factor }')
    local speed_kbps=$(awk -v speed="$configured_speed" 'BEGIN { printf "%.0f", speed * 1000 }')
    local burst_kb=$(awk -v kbps="$speed_kbps" 'BEGIN { printf "%.0f", (kbps * 20) / 8 }')
    
    # Remove existing qdisc
    tc qdisc del dev "$INTERFACE" root 2>/dev/null || true
    
    # Apply HTB throttling
    tc qdisc add dev "$INTERFACE" root handle 1: htb default 10
    tc class add dev "$INTERFACE" parent 1: classid 1:10 htb \
        rate "${speed_kbps}kbit" \
        ceil "${speed_kbps}kbit" \
        burst "${burst_kb}k"
    tc qdisc add dev "$INTERFACE" parent 1:10 handle 10: fq_codel
    
    echo "[$(date '+%H:%M:%S')] ⬇️  THROTTLED to ${speed} Mbps (${percentage}% of capacity)"
}

# Function to remove throttling
remove_throttle() {
    tc qdisc del dev "$INTERFACE" root 2>/dev/null || true
    echo "[$(date '+%H:%M:%S')] ⬆️  UNTHROTTLED - Full speed restored (100%)"
}

# Main loop
cycle=1
while true; do
    echo ""
    echo "--- Cycle $cycle ---"
    
    # Random throttle speed between 25% and 75% of bandwidth
    throttle_speed=$(shuf -i ${MIN_SPEED}-${MAX_SPEED} -n 1)
    
    # Random duration for throttling (5-20 seconds)
    throttle_duration=$(shuf -i 5-20 -n 1)
    
    # Random duration for unthrottled state (3-15 seconds)
    unthrottle_duration=$(shuf -i 3-15 -n 1)
    
    # Apply throttling
    apply_throttle "$throttle_speed"
    echo "   Maintaining for ${throttle_duration} seconds..."
    sleep "$throttle_duration"
    
    # Remove throttling
    remove_throttle
    echo "   Maintaining for ${unthrottle_duration} seconds..."
    sleep "$unthrottle_duration"
    
    ((cycle++))
done
