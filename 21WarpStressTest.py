import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import os

# --- CONFIGURATION ---
INPUT_FILE = "live_mpc_data.csv"
if not os.path.exists(INPUT_FILE):
    print("Error: Run Script 11 first.")
    exit()

df = pd.read_csv(INPUT_FILE)

# 1. THE IRON FILTER (Requested by Critique)
# They want stable ETNOs: a > 250, q > 40
df_iron = df[ (df['a'] > 250) & (df['q'] > 40) ].copy()
print(f"Applying Iron Filter (a>250, q>40)...")
print(f"Original Count: {len(df)}")
print(f"Stable Survivors: {len(df_iron)}")

if len(df_iron) < 10:
    print("WARNING: Sample size too small for clustering. Proceeding with caution.")

# 2. PREPARE WARP SPACE
# We cluster in (Inclination, Node_Sin, Node_Cos)
X = pd.DataFrame({
    'i': df_iron['i'],
    'node_sin': np.sin(np.radians(df_iron['Node'])),
    'node_cos': np.cos(np.radians(df_iron['Node']))
})

# Scale features (Critical for GMM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. GAUSSIAN MIXTURE MODEL (GMM)
# We test 1 to 5 potential planes (components) and pick the best fit using BIC
# BIC (Bayesian Information Criterion) penalizes overfitting.
lowest_bic = np.inf
best_gmm = None
best_n = 0

print("\n--- GMM MODEL SELECTION ---")
for n in range(1, 6):
    gmm = GaussianMixture(n_components=n, random_state=42, n_init=10)
    gmm.fit(X_scaled)
    bic = gmm.bic(X_scaled)
    print(f"Components: {n} | BIC: {bic:.1f}")
    if bic < lowest_bic:
        lowest_bic = bic
        best_gmm = gmm
        best_n = n

print(f"\nWinner: {best_n} Orbital Populations Detected.")

# 4. ANALYZE THE WARP
# Predict which plane each object belongs to
labels = best_gmm.predict(X_scaled)
probs = best_gmm.predict_proba(X_scaled)
df_iron['cluster'] = labels
df_iron['warp_prob'] = probs.max(axis=1) # Confidence

# Get parameters of the main cluster
# We unscale the means to get physical degrees
# (This is approximate as we scaled inputs, simpler to re-calc mean of labeled data)
print("\n--- DETECTED PLANES (Iron Filter) ---")
for c in range(best_n):
    cluster_data = df_iron[df_iron['cluster'] == c]
    count = len(cluster_data)
    mean_i = cluster_data['i'].mean()
    
    # Calculate mean angle for Node properly (vector mean)
    s = np.mean(np.sin(np.radians(cluster_data['Node'])))
    c_val = np.mean(np.cos(np.radians(cluster_data['Node'])))
    mean_node = np.degrees(np.arctan2(s, c_val)) % 360
    
    print(f"Plane {c+1}: N={count} | Inclination={mean_i:.1f} deg | Node={mean_node:.1f} deg")

# 5. VISUALIZATION
plt.figure(figsize=(10, 8))

# Plot by Cluster
colors = ['red', 'blue', 'green', 'purple', 'orange']
for c in range(best_n):
    subset = df_iron[df_iron['cluster'] == c]
    plt.scatter(subset['Node'], subset['i'], c=colors[c], s=100, alpha=0.7, 
                edgecolors='black', label=f'Plane {c+1} (i={subset["i"].mean():.1f})')

plt.xlabel("Longitude of Ascending Node (deg)")
plt.ylabel("Inclination (deg)")
plt.title(f"Robustness Check: GMM Clustering on Stable ETNOs\n(a > 250 AU, q > 40 AU, N={len(df_iron)})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 360)
plt.ylim(0, 60)
plt.show()