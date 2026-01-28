import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import anderson_ksamp
import os

# --- CONFIGURATION ---
INPUT_FILE = "smart_dataset.csv"
if not os.path.exists(INPUT_FILE):
    print("Error: Run Script 27 first.")
    exit()

df_real = pd.read_csv(INPUT_FILE)
real_incs = df_real['i'].values

print(f"--- BIAS EMULATOR: TESTING NULL HYPOTHESIS ---")
print(f"Real Data: N={len(real_incs)} objects")

# 1. GENERATE THE "NULL UNIVERSE" (Flat Solar System)
# We assume a standard "scattered disk" distribution:
# Inclinations follow a sin(i)*Gaussian distribution (Brown 2001)
# Null Hypothesis: The underlying population is "Standard" (peak ~10 deg, broad)
# NOT Warped at 16 deg.
n_sim = 10000
print(f"Simulating {n_sim} synthetic objects (Null Hypothesis)...")

# Generate standard scattered disk inclinations (Brown 2001, sigma=10)
sigma_i = 10.0
raw_incs = np.random.rayleigh(sigma_i, n_sim) 
# Note: Rayleigh matches the sin(i)*gaussian physics of a hot disk

# Generate random sky positions (RA/Dec) for these objects
# An object at Inclination 'i' spends most time at max declination +/- i
# We approximate Dec ~ i * sin(random_phase)
phase = np.random.uniform(0, 2*np.pi, n_sim)
raw_decs = raw_incs * np.sin(phase)
raw_ras = np.random.uniform(0, 360, n_sim)

# 2. APPLY SURVEY BIAS (The "Window")
# We filter these fake objects through the "Survey Mask"
# Approximate footprints of major surveys (OSSOS, DES, etc.)
# Real surveys heavily favor the Ecliptic (Dec +/- 10)
# But they have "blocks" off-ecliptic.

def is_in_survey(ra, dec):
    # Ecliptic Survey (The "Band")
    if abs(dec) < 5: return True
    
    # OSSOS-like Blocks (Approximate)
    if (10 < ra < 50) and (abs(dec) < 15): return True
    if (300 < ra < 350) and (abs(dec) < 15): return True
    
    # High-Latitude checks (Very rare)
    # Most surveys MISS high-i objects because they look at the ecliptic.
    return False

# Filter the Null Universe
visible_indices = [k for k in range(n_sim) if is_in_survey(raw_ras[k], raw_decs[k])]
biased_incs = raw_incs[visible_indices]

print(f"Simulated Survey Detection Rate: {len(biased_incs)/n_sim*100:.1f}%")

# 3. ANDERSON-DARLING TEST
# Critique Requirement: "Reject warp if p > 0.05"
# We compare the "Real" distribution to the "Biased Null" distribution
# If they are DIFFERENT, then Bias cannot explain your data.

statistic, critical_values, significance_level = anderson_ksamp([real_incs, biased_incs])

print("\n--- STATISTICAL VERDICT (Anderson-Darling) ---")
print(f"Statistic: {statistic:.4f}")
print(f"Significance Levels: {significance_level}")
print(f"Critical Values:     {critical_values}")

# Interpretation logic for k-sample AD test
# If statistic > critical_value at 5%, we reject the null (They are different).
is_different = statistic > critical_values[2] # Index 2 is usually 5% level

if is_different:
    print("\n>>> RESULT: REJECT NULL HYPOTHESIS (p < 0.05)")
    print("Survey bias ALONE cannot explain the observed warp.")
    print("The 15.7 deg structure is statistically distinct from a biased flat disk.")
else:
    print("\n>>> RESULT: CANNOT REJECT NULL (p > 0.05)")
    print("WARNING: The 'Warp' looks just like a biased selection of normal objects.")
    print("Critique was right: This might be an observational illusion.")

# 4. PLOT COMPARISON
plt.figure(figsize=(10, 6))

# Real Data (KDE)
from scipy.stats import gaussian_kde
x_eval = np.linspace(0, 50, 200)
kde_real = gaussian_kde(real_incs)
plt.plot(x_eval, kde_real(x_eval), 'r-', linewidth=3, label='Real Data (Warped?)')

# Biased Null (KDE)
kde_null = gaussian_kde(biased_incs)
plt.plot(x_eval, kde_null(x_eval), 'k--', linewidth=2, label='Biased Null Model (Standard Disk)')

plt.axvline(15.7, color='red', alpha=0.3, label='Detected Warp (15.7 deg)')

plt.xlabel("Inclination (deg)")
plt.ylabel("Probability Density")
plt.title(f"Bias Quantification: Real Data vs. Survey-Biased Null\n(AD Statistic={statistic:.2f})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()