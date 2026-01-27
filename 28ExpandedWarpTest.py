import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import os

# --- CONFIGURATION ---
INPUT_FILE = "smart_dataset.csv"
if not os.path.exists(INPUT_FILE):
    print("Error: Run Script 27 first.")
    exit()

df = pd.read_csv(INPUT_FILE)
print(f"Loading Smart Dataset: {len(df)} Objects")

# 1. PREPARE DATA (Inclination & Node)
# We use the same phase space as before
X = pd.DataFrame({
    'i': df['i'],
    'node_sin': np.sin(np.radians(df['Node'])),
    'node_cos': np.cos(np.radians(df['Node']))
})

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. FIND BEST FIT MODEL (GMM)
# We test 1-5 components and pick the best BIC
best_bic = np.inf
best_gmm = None
best_n = 0

print("\n--- CLUSTERING ANALYSIS ---")
for n in range(1, 6):
    gmm = GaussianMixture(n_components=n, random_state=42, n_init=10)
    gmm.fit(X_scaled)
    bic = gmm.bic(X_scaled)
    if bic < best_bic:
        best_bic = bic
        best_gmm = gmm
        best_n = n

print(f"Optimal Model Complexity: {best_n} Planes Detected")

# 3. EXTRACT WARP PARAMETERS
# Predict labels
labels = best_gmm.predict(X_scaled)
df['cluster'] = labels

# Analyze clusters
found_warp = False
warp_params = {}

print("\n--- DETECTED STRUCTURES ---")
for c in range(best_n):
    subset = df[df['cluster'] == c]
    count = len(subset)
    mean_i = subset['i'].mean()
    
    # Vector mean for Node
    s = np.mean(np.sin(np.radians(subset['Node'])))
    c_val = np.mean(np.cos(np.radians(subset['Node'])))
    mean_node = np.degrees(np.arctan2(s, c_val)) % 360
    
    print(f"Cluster {c+1}: N={count} | Inclination={mean_i:.1f} deg | Node={mean_node:.1f} deg")
    
    # Check if this matches our "Warp Profile" (i between 12 and 25)
    if 12 < mean_i < 25:
        found_warp = True
        warp_params = {'i': mean_i, 'node': mean_node, 'n': count}
        print(f"  >>> MATCHES PLANET 9/Y PROFILE <<<")

# 4. BAYESIAN BOOTSTRAP (The Robustness Check)
# We run 1000 trials to see how stable this detection is
if found_warp:
    print(f"\n--- STRESS TESTING WARP ({warp_params['i']:.1f} deg) ---")
    trials = 1000
    successes = 0
    
    for k in range(trials):
        # Resample with replacement
        X_res = X.sample(n=len(X), replace=True, random_state=k)
        X_res_s = scaler.transform(X_res)
        
        # Fit GMM (Fixed to best_n components for stability)
        gmm_b = GaussianMixture(n_components=best_n, random_state=k)
        gmm_b.fit(X_res_s)
        
        # Check means
        means = gmm_b.means_
        # Unscale Inclination (approx)
        # i is col 0
        real_i = means[:, 0] * scaler.scale_[0] + scaler.mean_[0]
        
        # Is there a cluster in the 12-25 deg range?
        if np.any((real_i > 12) & (real_i < 25)):
            successes += 1
            
    confidence = (successes / trials) * 100
    print(f"Bootstrap Confidence (N={len(df)}): {confidence:.1f}%")
else:
    print("\nNo Warp Candidate found in this dataset.")

# 5. POLE MAP (Visual Check)
plt.figure(figsize=(10, 6))

# Plot Poles
# Pole definition:
# Lx = sin(i)sin(Node), Ly = -sin(i)cos(Node), Lz = cos(i)
# RA = atan2(Ly, Lx), Dec = asin(Lz)
i_rad = np.radians(df['i'])
n_rad = np.radians(df['Node'])
lx = np.sin(i_rad) * np.sin(n_rad)
ly = -np.sin(i_rad) * np.cos(n_rad)
lz = np.cos(i_rad)
pole_dec = np.degrees(np.arcsin(lz))
pole_ra = np.degrees(np.arctan2(ly, lx)) % 360


plt.scatter(pole_ra, pole_dec, c=labels, cmap='viridis', s=80, edgecolors='black', label='Objects')

if found_warp:
    # Plot the Center of the Warp
    w_i = np.radians(warp_params['i'])
    w_n = np.radians(warp_params['node'])
    w_lx = np.sin(w_i) * np.sin(w_n)
    w_ly = -np.sin(w_i) * np.cos(w_n)
    w_lz = np.cos(w_i)
    w_dec = np.degrees(np.arcsin(w_lz))
    w_ra = np.degrees(np.arctan2(w_ly, w_lx)) % 360
    plt.scatter(w_ra, w_dec, c='red', marker='X', s=200, label=f'Warp Center ({warp_params["i"]:.1f} deg)')

plt.xlabel("Pole RA (deg)")
plt.ylabel("Pole Dec (deg)")
plt.title(f"New Dataset Validation (N={len(df)})\nDoes the Warp Persist?")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 360)
plt.ylim(0, 90)
plt.show()