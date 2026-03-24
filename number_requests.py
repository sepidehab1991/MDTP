import re
from collections import defaultdict

cuid_ip = {}
ip_success = defaultdict(int)

with open('/tmp/aria2.log', 'r') as f:
    for line in f:
        # Map CUID -> IP
        m = re.search(r'CUID#(\d+) - Connecting to ([\d.]+):', line)
        if m:
            cuid, ip = m.groups()
            cuid_ip[cuid] = ip

        # Count successful completed segments per CUID, then map to IP
        m = re.search(r'CUID#(\d+) - The download for one segment completed successfully', line)
        if m:
            cuid = m.group(1)
            ip = cuid_ip.get(cuid, 'unknown')
            ip_success[ip] += 1

print("=== Successfully Downloaded Segments per Replica ===")
print(f"{'IP':<20} {'Segments':>10} {'Size (MB)':>12}")
print("-" * 44)

total = 0
for ip, count in sorted(ip_success.items(), key=lambda x: -x[1]):
    print(f"  {ip:<18} {count:>10}   {count:>8} MB")
    total += count

print("-" * 44)
print(f"  {'TOTAL':<18} {total:>10}   {total:>8} MB")
