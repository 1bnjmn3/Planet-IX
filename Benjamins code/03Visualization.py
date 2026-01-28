import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
INPUT_FILE = 'processed_etnos.csv'
script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, INPUT_FILE)

if not os.path.exists(input_path):
    print("Error: processed_etnos.csv not found.")
    exit()

df = pd.read_csv(input_path)

# Filter for the High-Quality Objects only (matching your Step 2 filter)
# Adjust 'a' cutoff if you changed it in step 01 (e.g. 150 or 230)
df_hq = df[ (df['a'] > 230) & (df['q'] > 30) ].copy()

print(f"Plotting {len(df_hq)} High-Quality Objects...")

# --- PLOT: The Rose Diagram ---
plt.figure(figsize=(10, 10))
ax = plt.subplot(111, polar=True)

# 1. Plot the Histogram (The Fan)
# We want to see if the angles cluster around 60 degrees
num_bins = 24
counts, bin_edges = np.histogram(df_hq['varpi_rad'], bins=num_bins, range=(0, 2*np.pi))
widths = np.diff(bin_edges)

# Color the bars based on count (Darker = More Clustering)
bars = ax.bar(bin_edges[:-1], counts, width=widths, bottom=0.0, alpha=0.8, edgecolor='black')

# 2. Add the "Planet 9 Prediction" line
# Planet 9 is suspected to be at varpi ~241 degrees.
# The cluster should be Anti-Aligned (~61 degrees).
p9_angle = np.radians(241)
cluster_angle = np.radians(61)

ax.axvline(cluster_angle, color='red', linewidth=3, linestyle='--', label='Predicted Cluster (Anti-P9)')
ax.axvline(p9_angle, color='green', linewidth=3, linestyle='--', label='Planet 9 Location')

# 3. Highlight your Top 3 Anomalies (From your previous result)
# Object 13 (110 deg), Object 3 (59 deg), Object 9 (25 deg)
anomalies = [np.radians(110), np.radians(59), np.radians(25)]
ax.scatter(anomalies, [max(counts)]*3, c='yellow', s=150, edgecolors='black', zorder=10, label='ML Top Anomalies')

# Styling
ax.set_theta_zero_location("E") # Set 0 degrees to East
ax.set_theta_direction(1)       # Counter-clockwise
ax.set_title("Longitude of Perihelion Clustering\n(The 'Smoking Gun' for Planet 9)", va='bottom', fontsize=14)
ax.legend(loc='lower left', bbox_to_anchor=(-0.1, -0.1))

plt.show()