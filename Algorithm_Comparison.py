import numpy as np
import matplotlib.pyplot as plt

# Set figure size (balanced, not too big)
fig, ax = plt.subplots(figsize=(6, 4), dpi=150, constrained_layout=True)

# X-axis labels as numbers
Time_Duration = ["1th", "2th", "3th", "4th"]

# Algorithm names for the legend
Algorithm_Names = ["MDTP", "Aria2", "Static_Chunking", "BitTorrent"]

# Data values
MDTP = [273.3606533]
Aria2 = [271.7]
StaticChunking = [284.2020649]
BitTorrent = [356.5208429]

# Standard deviations
std_MDTP = [5.632]
std_Aria2 = [6.766]
std_StaticChunking = [9.838]
std_BitTorrent = [183.580]

# Compact bar positions
positions = np.arange(len(Time_Duration)) * 0.85 

# Colors and hatches
colors = ['#ffb6c1', '#c0c0c0', '#add8e6', '#f0e68c']
hatches = ['//', '\\', '|', '-']

# Plot bars with error bars
for i, (value, std, color, hatch, algo) in enumerate(zip([MDTP, Aria2, StaticChunking, BitTorrent], 
                                                          [std_MDTP, std_Aria2, std_StaticChunking, std_BitTorrent], 
                                                          colors, hatches, Algorithm_Names)):
    ax.bar(positions[i], value, color=color, label=algo, yerr=std, 
           width=0.4, edgecolor='black', capsize=4, hatch=hatch)
    
    # Display standard deviation values above the error bars
    ax.text(positions[i], value[0] + std[0] + 12, f"{std[0]:.2f}", 
            ha='center', fontsize=10, fontweight='bold', color='black')

# Customizations (minimalist, clear)
ax.set_ylim(0, 600)  
ax.set_xlabel('Algorithms', fontsize=12, fontweight='bold')
ax.set_ylabel('Delay (s)', fontsize=12, fontweight='bold')
ax.set_xticks(positions)
ax.set_xticklabels(Time_Duration, fontsize=11)  # X-axis is now 1, 2, 3, 4
ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.7)  # Light dashed grid

# Title (clean, no excess spacing)
ax.set_title("Comparison of Different Algorithms", fontsize=13, fontweight='bold', pad=10)

# Adjust legend higher with algorithm names
ax.legend(fontsize=10, loc='upper right', bbox_to_anchor=(0.35, 1), frameon=False)

# Save and show
plt.savefig('compact_comparison_plot.pdf', format='pdf', bbox_inches='tight')
plt.show()



