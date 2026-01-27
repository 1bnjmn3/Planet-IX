import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os

# --- CONFIGURATION ---
INPUT_FILE = "smart_dataset.csv"
if not os.path.exists(INPUT_FILE):
    print("Error: Run Script 27 first.")
    exit()

df = pd.read_csv(INPUT_FILE)
print(f"Analyzing Distribution of {len(df)} Objects...")

# 1. SETUP HISTOGRAM DATA
# We are looking strictly at Inclination (i)
incs = df['i']

# 2. CALCULATE KERNEL DENSITY ESTIMATION (KDE)
# This creates a smooth curve representing the probability distribution
# Bandwidth method 'scott' or 'silverman' determines smoothness
density = gaussian_kde(incs)
x_vals = np.linspace(0, 60, 200)
density_vals = density(x_vals)

# Find the Peak of the Curve (The "Most Likely" Warp Angle)
peak_idx = np.argmax(density_vals)
peak_inc = x_vals[peak_idx]

print(f"\n--- DISTRIBUTION ANALYSIS ---")
print(f"Primary Peak Detected at: {peak_inc:.1f} degrees")

# Check for secondary peaks (simple method)
# Find local maxima
from scipy.signal import find_peaks
peaks, _ = find_peaks(density_vals, height=0.01)
peak_angles = x_vals[peaks]

print(f"All Density Spikes: {peak_angles}")

# 3. PLOT THE "NAKED EYE" TEST
plt.figure(figsize=(10, 6))

# Histogram (The raw counts)
plt.hist(incs, bins=range(0, 60, 3), density=True, alpha=0.3, color='gray', label='Raw Counts (Bins)')

# KDE Curve (The smooth reality)
plt.plot(x_vals, density_vals, 'b-', linewidth=3, label='Probability Density (KDE)')

# Mark the Peaks
for p in peak_angles:
    plt.axvline(p, color='red', linestyle='--', alpha=0.6)
    plt.text(p, max(density_vals)*1.02, f"{p:.1f}°", color='red', ha='center', fontweight='bold')

# Reference Lines (The Competing Theories)
plt.axvline(18.0, color='green', linestyle=':', alpha=0.5, label='Theory: Planet 9 (~18°)')
plt.axvline(10.0, color='orange', linestyle=':', alpha=0.5, label='Theory: Planet Y (~10°)')

plt.xlabel("Inclination (degrees)")
plt.ylabel("Probability Density")
plt.title(f"The 'Naked Eye' Test: Is the Warp Visible?\n(N={len(df)} Stable Objects)")
plt.legend()
plt.xlim(0, 50)
plt.grid(True, alpha=0.3)
plt.show()