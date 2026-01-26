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
# Apply the "Iron Filter" (Stable objects only)
df_stable = df[ (df['a'] > 250) & (df['q'] > 40) ].copy()
print(f"Starting Bayesian Bootstrap on {len(df_stable)} stable objects...")

# Prepare Data
X = pd.DataFrame({
    'i': df_stable['i'],
    'node_sin': np.sin(np.radians(df_stable['Node'])),
    'node_cos': np.cos(np.radians(df_stable['Node']))
})
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- THE BAYESIAN BOOTSTRAP ---
n_iterations = 2000 # Number of "Parallel Universes" to simulate
warp_detections = []

print(f"Running {n_iterations} Resampling Trials...")

for k in range(n_iterations):
    # 1. Resample Data (Bootstrap)
    # We pick N objects from the dataset with replacement.
    # This simulates "what if we had slightly different data?"
    X_resampled = X.sample(n=len(X), replace=True, random_state=k)
    X_res_scaled = scaler.transform(X_resampled) # Use original scaler
    
    # 2. Run GMM Clustering
    # We look for the "Best Fit" number of planes (1 to 3)
    best_bic = np.inf
    best_model = None
    
    for n_comp in range(1, 4): # Check 1, 2, or 3 planes
        gmm = GaussianMixture(n_components=n_comp, random_state=k)
        gmm.fit(X_res_scaled)
        bic = gmm.bic(X_res_scaled)
        if bic < best_bic:
            best_bic = bic
            best_model = gmm
    
    # 3. Analyze the Winner
    # Does it have a "Warp" component? (Cluster with i > 10 and i < 30)
    means = best_model.means_
    # Unscale the means to get rough inclination
    # Note: Inverse transform is tricky for just one column, so we approximate
    # We know 'i' is column 0.
    # real_i = scaled_i * std + mean
    mean_incs = means[:, 0] * scaler.scale_[0] + scaler.mean_[0]
    
    # Check if ANY cluster is a "P9 Warp" (between 12 and 22 degrees)
    has_warp = np.any((mean_incs > 12) & (mean_incs < 22))
    warp_detections.append(has_warp)

    if k % 200 == 0:
        print(f"Trial {k}: {'Warp Found' if has_warp else 'Noise'}")

# --- FINAL STATISTICS ---
confidence = np.mean(warp_detections) * 100
print(f"\n--- FINAL BAYESIAN CONFIDENCE ---")
print(f"Probability that the 17.9 deg Warp is REAL: {confidence:.1f}%")

# --- PLOT CONFIDENCE ---
plt.figure(figsize=(8, 5))
plt.bar(['Real Warp', 'Random Noise'], [confidence, 100-confidence], color=['green', 'gray'])
plt.ylabel("Probability (%)")
plt.title(f"Statistical Robustness of Warp Detection\n(Confidence: {confidence:.1f}%)")
plt.ylim(0, 100)
plt.show()