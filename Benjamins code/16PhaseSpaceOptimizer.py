import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import os

# --- CONFIGURATION ---
INPUT_FILE = "mega_dataset.csv" # Or "live_mpc_data.csv"
if not os.path.exists(INPUT_FILE):
    INPUT_FILE = "live_mpc_data.csv"

if not os.path.exists(INPUT_FILE):
    print("Error: No data found. Run 11 or 13 first.")
    exit()

df = pd.read_csv(INPUT_FILE)
# Filter for the "Sweet Spot" range where dynamics are most active
# Typically a > 150 is where the P9 coupling dominates
df_hq = df[ (df['a'] > 150) & (df['q'] > 30) ].copy()

print(f"Analyzing Phase Space Dynamics of {len(df_hq)} Objects...")

# --- THE PHYSICS: KOZAI-LIDOV COUPLING ---
# In P9 theory, e and varpi are coupled.
# We measure this using "Mutual Information" (MI).
# High MI = Variables are linked (Physics).
# Low MI = Variables are independent (Random).

def calc_coupling_strength(dataframe):
    # We look for relationship between Eccentricity and Angle
    # We must handle the cyclic nature of angles for the metric
    
    # 1. Prepare Data
    e = dataframe['e'].values.reshape(-1, 1)
    
    # We decompose angle into components to capture cyclic relationships
    varpi_sin = np.sin(np.radians(dataframe['varpi'])).values
    varpi_cos = np.cos(np.radians(dataframe['varpi'])).values
    
    # 2. Calculate Mutual Information
    # How much does knowing 'e' tell us about 'varpi'?
    mi_sin = mutual_info_regression(e, varpi_sin, random_state=42)[0]
    mi_cos = mutual_info_regression(e, varpi_cos, random_state=42)[0]
    
    return mi_sin + mi_cos

# 1. Measure Real Data Coupling
real_score = calc_coupling_strength(df_hq)
print(f"\nReal Data Coupling Score (e vs varpi): {real_score:.4f}")

# 2. Measure Random "Null Hypothesis" Coupling (Monte Carlo)
# We shuffle 'varpi' to break any physical link while keeping the same values.
print("Benchmarking against Random Noise...")
random_scores = []
for _ in range(1000):
    df_fake = df_hq.copy()
    # Shuffle angles independently of eccentricity
    df_fake['varpi'] = np.random.permutation(df_hq['varpi'].values)
    score = calc_coupling_strength(df_fake)
    random_scores.append(score)

avg_random = np.mean(random_scores)
p_value_coupling = np.sum(np.array(random_scores) >= real_score) / 1000.0

print(f"Average Random Coupling Score: {avg_random:.4f}")
print(f"P-Value (Is Real > Random?): {p_value_coupling:.4f}")

# --- DIAGNOSIS ---
if p_value_coupling < 0.05:
    print("\nSTATUS: DETECTED.")
    print("Eccentricity and Perihelion are physically coupled!")
    print("This implies a dynamical perturber (Planet 9 or Resonance).")
else:
    print("\nSTATUS: FLATLINE.")
    print("Orbits appear dynamically decoupled (Random Dust).")

# --- VISUALIZATION: The Phase Space Plot ---
# P9 predicts specific tracks in this space
plt.figure(figsize=(10, 6))

# Plot Background (Random Noise Simulation for comparison)
# We generate fake points to show what "Random" looks like
fake_e = np.random.uniform(df_hq['e'].min(), df_hq['e'].max(), 500)
fake_varpi = np.random.uniform(0, 360, 500)
plt.scatter(fake_varpi, fake_e, c='gray', alpha=0.1, label='Random Background')

# Plot Real Data

sc = plt.scatter(df_hq['varpi'], df_hq['e'], c=df_hq['a'], cmap='plasma', s=80, edgecolors='black', label='Real TNOs')

plt.colorbar(sc, label='Semi-Major Axis (a) [AU]')
plt.xlabel("Longitude of Perihelion (deg)")
plt.ylabel("Eccentricity (e)")
plt.title(f"Phase Space Diagnostics: e vs \u03D6 Coupling\n(Coupling p-value: {p_value_coupling:.3f})")

# Draw P9 "Resonance Islands" (Approximate locations)
# P9 creates islands of stability at high 'e' near aligned/anti-aligned angles
plt.axvline(60, color='red', linestyle='--', alpha=0.3, label='Anti-Aligned Zone')
plt.axvline(240, color='blue', linestyle='--', alpha=0.3, label='Aligned Zone')

plt.xlim(0, 360)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.show()