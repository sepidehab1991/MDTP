cd /home/ubuntu/Binpacking/Aria2 && ./DiskAria2.sh  && \
sleep 1 && \
sudo tc qdisc add dev enp7s0 root handle 1: prio && \
sudo tc filter add dev enp7s0 parent 1: protocol ip u32 match ip dst 10.147.165.2 flowid 1:1 && \
sudo tc qdisc add dev enp7s0 parent 1:1 handle 10: netem delay 500ms
