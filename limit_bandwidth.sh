
INTERFACE="enp7s0"  # Change this to your network interface (use `ip a` to check)
IP_ADDRESS="10.132.141.2"  # Change this to the IP you want to limit
RATE="500mbit"  # Set the bandwidth limit
BURST="10k"  # Small burst buffer for smooth control

# Clear previous rules (to prevent duplicates)
echo "Removing old traffic control rules..."
sudo tc qdisc del dev $INTERFACE ingress 2>/dev/null

# Add Ingress Policing to the interface
echo "Adding ingress policing to $INTERFACE..."
sudo tc qdisc add dev $INTERFACE handle ffff: ingress

# Apply bandwidth limit for the specific IP
echo "Limiting incoming bandwidth for $IP_ADDRESS to $RATE..."
sudo tc filter add dev $INTERFACE parent ffff: protocol ip prio 1 u32 \
	match ip src $IP_ADDRESS flowid 1:1 \
	police rate $RATE burst $BURST drop

echo "Bandwidth limit applied successfully!"
