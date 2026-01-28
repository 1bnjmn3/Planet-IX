import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import os

# --- CONFIGURATION ---
INPUT_FILE = "live_mpc_data.csv" # Or mega_dataset.csv
if not os.path.exists(INPUT_FILE):
    print("Error: Data file not found. Run Script 11 first.")
    exit()

df = pd.read_csv(INPUT_FILE)
# Filter for Extreme/Detached again
df_hq = df[ (df['a'] > 150) & (df['q'] > 30) ].copy()

print(f"Scanning {len(df_hq)} objects for Orbital Warp (Inclination Clustering)...")

# --- THE WARP PHYSICS ---
# Planet 9 / Planet Y predicts that orbits should cluster in the (i, Node) plane.
# This corresponds to a "common orbital plane" for the distant solar system.

# 1. Prepare Data
# We use Inclination (i) and Node (Omega). 
# Note: Node is cyclic (0-360), so we use Sin/Cos components for clustering.
X = pd.DataFrame({
    'i': df_hq['i'],
    'node_sin': np.sin(np.radians(df_hq['Node'])),
    'node_cos': np.cos(np.radians(df_hq['Node']))
})

# 2. ML Clustering (DBSCAN) to find "Knots" in the plane
# We normalize 'i' to have similar weight to the Node components
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)

# Search for clusters
db = DBSCAN(eps=0.5, min_samples=4).fit(X_scaled)
labels = db.labels_

# Count clusters (ignoring noise -1)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"ML identified {n_clusters} distinct Orbital Planes (Warps).")

df_hq['warp_cluster'] = labels

# 3. VISUALIZATION
plt.figure(figsize=(10, 8))

# Plot Noise (Grey)
noise = df_hq[df_hq['warp_cluster'] == -1]
plt.scatter(noise['Node'], noise['i'], c='gray', alpha=0.3, label='Random Background')

# Plot Clusters (Colored)
if n_clusters > 0:
    clustered = df_hq[df_hq['warp_cluster'] != -1]
    # Use a distinct colormap for clusters
    sc = plt.scatter(clustered['Node'], clustered['i'], c=clustered['warp_cluster'], 
                     cmap='tab10', s=100, edgecolors='black', label='Detected Warp/Plane')
    
    # Calculate the "Mean Plane" for the biggest cluster
    top_cluster = clustered['warp_cluster'].mode()[0]
    mean_i = clustered[clustered['warp_cluster'] == top_cluster]['i'].mean()
    mean_node = clustered[clustered['warp_cluster'] == top_cluster]['Node'].mean()
    
    plt.scatter(mean_node, mean_i, c='red', marker='X', s=200, label=f'Warp Center (i={mean_i:.1f})')
    print(f"Primary Warp Detected at: Inclination={mean_i:.1f} deg, Node={mean_node:.1f} deg")

plt.xlabel("Longitude of Ascending Node (\u03A9) [deg]")
plt.ylabel("Inclination (i) [deg]")
plt.title(f"Warp Signature Search: Is there a common orbital plane?\n(N={len(df_hq)} objects)")

plt.grid(True, alpha=0.3)
plt.xlim(0, 360)
plt.ylim(0, 60) # TNOs usually low inclination, but P9 allows high i
plt.legend()
plt.show()