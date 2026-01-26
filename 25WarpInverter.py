import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# --- CONFIGURATION ---
INPUT_FILE = "live_mpc_data.csv"
if not os.path.exists(INPUT_FILE):
    print("Error: Run Script 11 first.")
    exit()

df = pd.read_csv(INPUT_FILE)
# 1. ISOLATE THE 9 "WARP" OBJECTS
# We apply the Iron Filter (a>250, q>40)
# And we filter for the specific Warp Plane (i approx 15-20, Node approx 100-130)
# based on your previous GMM results (Plane 2).
df_warp = df[ 
    (df['a'] > 250) & 
    (df['q'] > 40) & 
    (df['i'] > 14) & (df['i'] < 22) # The broad "Warp Zone"
].copy()

print(f"Isolating Warp Candidates...")
print(f"Found {len(df_warp)} objects fitting the profile.")

if len(df_warp) < 4:
    print("Not enough points to fit a curve. Aborting.")
    exit()

# 2. DEFINE THE PHYSICS MODEL (The "Inverse" Function)
# We want to solve for 'planet_inc' and 'planet_mass'
# x = particle_distance (a)
# y = particle_inclination (i)

def secular_model(particle_a, planet_inc, planet_mass_earth):
    # Constants
    m_sun = 1.0
    planet_a = 500.0 # We fix distance at 500 AU (hard to constrain both mass/dist)
    planet_m = planet_mass_earth * 3e-6 # Convert Earth Mass to Solar Mass
    
    # Torque Balance Formula (Same as Script 23)
    # Torque Giants ~ 1/a^3.5
    torque_giants = 1.0 / (particle_a**3.5) * 1e8 
    # Torque P9 ~ a^2 / planet_a^3
    torque_p9 = (planet_m / m_sun) * (particle_a**2) / (planet_a**3) * 1e10
    
    # Equilibrium Inclination
    i_forced = (torque_p9 * planet_inc) / (torque_giants + torque_p9)
    return i_forced

# 3. RUN THE OPTIMIZER (Curve Fit)
# We provide initial guesses: i=20, Mass=5
# The code will adjust these to best fit the Real Data.
p0 = [20.0, 5.0] 
# Bounds: Inc [0, 90], Mass [0.1, 20]
bounds = ([0, 0.1], [90, 20])

popt, pcov = curve_fit(secular_model, df_warp['a'], df_warp['i'], p0=p0, bounds=bounds)

best_inc = popt[0]
best_mass = popt[1]

print(f"\n--- INVERSION RESULTS ---")
print(f"The data implies Planet 9 has:")
print(f"  Inclination: {best_inc:.2f} degrees")
print(f"  Mass:        {best_mass:.2f} Earth Masses")

# 4. PLOT THE FIT
plt.figure(figsize=(10, 6))

# Plot Real Data
plt.scatter(df_warp['a'], df_warp['i'], color='red', s=100, label='Real Objects (Stable)')

# Plot the Optimized Curve
x_range = np.linspace(200, 600, 100)
y_pred = secular_model(x_range, best_inc, best_mass)
plt.plot(x_range, y_pred, 'k--', linewidth=2, label=f'Best Fit Model (i={best_inc:.1f})')

plt.axhline(best_inc, color='blue', linestyle=':', label='Predicted P9 Plane')

plt.xlabel("Semi-Major Axis (a) [AU]")
plt.ylabel("Inclination (deg)")
plt.title(f"Warp Inverter: Determining P9 Parameters from Data\n(N={len(df_warp)})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()