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

# 1. Filter High-Quality Objects
# Use the strict filter (a > 230) to remove Neptune noise
df_hq = df[ (df['a'] > 230) & (df['q'] > 30) ].copy()
angles_rad = np.radians(df_hq['varpi']) # Convert to radians
n_objects = len(df_hq)

print(f"Analyzing {n_objects} High-Quality Objects...")

# --- TEST 1: The Rayleigh Test (Vector Sum) ---
# Calculate the "Mean Resultant Length" (R_bar)
# If R_bar is close to 1, they are pointing in the same direction.
# If R_bar is close to 0, they are random.
C = np.sum(np.cos(angles_rad))
S = np.sum(np.sin(angles_rad))
R = np.sqrt(C**2 + S**2)
R_bar = R / n_objects

# Calculate P-value using the Rayleigh approximation
# Z = n * R_bar^2
Z = n_objects * (R_bar**2)
p_value_rayleigh = np.exp(-Z) * (1 + (2*Z - Z**2)/(4*n_objects) - (24*Z - 132*Z**2 + 76*Z**3 - 9*Z**4)/(288*n_objects**2))

print("\n--- RAYLEIGH TEST RESULTS ---")
print(f"Mean Vector Length (0-1): {R_bar:.4f}")
print(f"Z-Statistic:              {Z:.4f}")
print(f"P-Value (Analytical):     {p_value_rayleigh:.5f}")


# --- TEST 2: Monte Carlo Simulation (The "Gold Standard") ---
# We generate 100,000 random sets of N objects and see how often they cluster this well.
print("\n--- RUNNING MONTE CARLO SIMULATION (100k Runs) ---")
n_simulations = 100000
random_R_bars = []

# Generate random angles for all simulations at once for speed
random_angles = np.random.uniform(0, 2*np.pi, (n_simulations, n_objects))
# Calculate vector sums for all simulations
sim_C = np.sum(np.cos(random_angles), axis=1)
sim_S = np.sum(np.sin(random_angles), axis=1)
sim_R = np.sqrt(sim_C**2 + sim_S**2)
sim_R_bars = sim_R / n_objects

# Count how many random simulations beat our real score
better_sims = np.sum(sim_R_bars >= R_bar)
p_value_mc = better_sims / n_simulations

print(f"Simulations that beat your data: {better_sims} out of {n_simulations}")
print(f"Monte Carlo P-Value: {p_value_mc:.5f}")

# --- INTERPRETATION ---
print("\n--- FINAL VERDICT ---")
if p_value_mc < 0.01:
    print("STATUS: CONFIRMED. (99%+ Confidence)")
    print("The clustering is statistically real.")
elif p_value_mc < 0.05:
    print("STATUS: PROMISING. (95% Confidence)")
    print("Likely real, but more data would help.")
else:
    print("STATUS: RANDOM.")
    print(f"There is a {p_value_mc*100:.1f}% chance this is just noise.")

# --- PLOT: Visual Proof ---
plt.figure(figsize=(10, 6))
plt.hist(sim_R_bars, bins=50, color='gray', alpha=0.5, label='Random Chance (Monte Carlo)')
plt.axvline(R_bar, color='red', linestyle='dashed', linewidth=2, label=f'Your Data (R={R_bar:.2f})')
plt.xlabel("Vector Clustering Strength (R_bar)")
plt.ylabel("Frequency")
plt.title(f"Is your cluster real? (P = {p_value_mc:.5f})")
plt.legend()
plt.show()