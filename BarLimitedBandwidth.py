# Re-import necessary libraries after execution state reset
import matplotlib.pyplot as plt
import numpy as np

# Data
categories = ["Full Bandwidth", "Limited Bandwidth"]
MDTP = [231.9742307, 282.54592]
Aria2 = [303.8, 353.9]

# Positioning
x = np.arange(len(categories))
bar_width = 0.25
hatches = ['//', '\\']
# Define colors
colors = ['#ffb6c1', '#c0c0c0']  # MDTP and Aria2

# Create figure and axes with adjusted size (6x4 inches)
fig, ax = plt.subplots(figsize=(6, 4), dpi=150, constrained_layout=True)

# Create bar chart
ax.bar(x - bar_width / 2, MDTP, width=bar_width, color=colors[0], edgecolor='black', label="MDTP", hatch=hatches[0])
ax.bar(x + bar_width / 2, Aria2, width=bar_width, color=colors[1], edgecolor='black', label="Aria2", hatch=hatches[1])

# Labels and title
ax.set_xlabel('Bandwidth Condition', fontsize=12, fontweight='bold')
ax.set_ylabel('Delay (s)', fontsize=12, fontweight='bold')
ax.set_title('Different Bandwidths Comparison', fontsize=13, fontweight='bold', pad=10)

# Customizing x-axis
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=10)

# Grid for better readability
ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.7)

# Adding legend
ax.legend(fontsize=10, loc="upper left", frameon=False)

# Save the plot as a PDF
plt.savefig('comparison_bandwidth_conditions.pdf', format='pdf', bbox_inches='tight')

# Show the plot
plt.show()

