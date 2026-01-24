import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest, uniform
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
# (Using the same strict cutoffs as before)
df_hq = df[ (df['a'] > 230) & (df['q'] > 30) ].copy()
print(f"Analyzing Statistical Significance of {len(df_hq)} Objects...")

# 2. Prepare Data
# Normalize angles to 0-1 range for the KS test (0 = 0 deg, 1 = 360 deg)
# We use varpi (Longitude of Perihelion)
angles_normalized = df_hq['varpi'] / 360.0

# 3. The Kolmogorov-Smirnov (KS) Test
# Null Hypothesis: The angles are uniformly distributed (Random).
# Alternative: They are NOT uniform (Clustered).
statistic, p_value = kstest(angles_normalized, 'uniform')

print("\n--- STATISTICAL VERDICT ---")
print(f"KS Statistic: {statistic:.4f} (How 'bunchy' the data is)")
print(f"P-Value:      {p_value:.5f}")

print("\n--- INTERPRETATION ---")
if p_value < 0.01:
    print("RESULT: HIGHLY SIGNIFICANT CLUSTERING (99%+ Confidence)")
    print("This is publishable evidence. The random model is rejected.")
elif p_value < 0.05:
    print("RESULT: Significant Clustering (95% Confidence)")
    print("Strong signal, likely real.")
else:
    print("RESULT: Indistinguishable from Randomness.")
    print("Current sample size may be too small or noise is too high.")

# 4. Visual Cumulative Distribution (CDF) Plot
# If random, the blue line should follow the diagonal dashed line.
# If clustered, the blue line will look like 'stairs' or curve away.
plt.figure(figsize=(8, 6))
plt.plot(np.sort(angles_normalized), np.linspace(0, 1, len(angles_normalized), endpoint=False), label='Your Data (CDF)')
plt.plot([0, 1], [0, 1], 'r--', label='Random Uniform (Expected)')
plt.xlabel("Normalized Angle (0 to 360 deg)")
plt.ylabel("Cumulative Probability")
plt.title(f"KS Test for Clustering (p={p_value:.4f})")
plt.legend()
plt.grid(True)
plt.show()