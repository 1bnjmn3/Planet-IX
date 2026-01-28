import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# --- CONFIGURATION ---
TARGET_WARP = 15.7       # The observed signal
ETNO_DIST = 400.0        # Where the rocks are
GIANT_J2 = 38023.0       # Strength of Jupiter/Saturn/etc (from Script 34)

print(f"--- FINAL SHOWDOWN: PLANET Y vs PLANET 9 ---")

# --- PHYSICS ENGINE 1: PLANET Y (Inner, i=65) ---
def get_y_mass(d, inc=65.0):
    if TARGET_WARP >= inc: return np.nan
    # J2 Balance Model
    req_strength = (TARGET_WARP * GIANT_J2) / (inc - TARGET_WARP)
    mass = req_strength / (d**2)
    return mass

# --- PHYSICS ENGINE 2: PLANET 9 (Outer, i=20) ---
def get_p9_mass(d, inc=20.0):
    if TARGET_WARP >= inc: return np.nan
    # External Torque Model
    tau_giants = 1.0 / (ETNO_DIST**3.5) * 1e8
    req_tau_b = (TARGET_WARP * tau_giants) / (inc - TARGET_WARP)
    mass_solar = req_tau_b * (d**3) / (ETNO_DIST**2) / 1e10
    mass_earth = mass_solar / 3e-6
    return mass_earth

# --- GENERATE CURVES ---
# We scan 50 to 1500 AU to catch everything
dists = np.linspace(50, 1500, 500)

y_curve = [get_y_mass(d, 65.0) for d in dists]
p9_curve = [get_p9_mass(d, 20.0) for d in dists]

# --- FIND SOLUTIONS ---
# 1. Planet Y (1.0 Me)
try:
    y_solver = interp1d(y_curve, dists, fill_value="extrapolate")
    opt_y = float(y_solver(1.0))
except:
    opt_y = 0

# 2. Planet 9 (5.0 Me)
try:
    p9_solver = interp1d(p9_curve, dists, fill_value="extrapolate")
    opt_p9 = float(p9_solver(5.0))
except:
    opt_p9 = 0

print("\n--- THE VERDICT ---")
print(f"PLANET Y SCENARIO (Target: 1 Me, Inc: 65°)")
print(f"  > Required Distance: {opt_y:.1f} AU")
print(f"  > Plausibility: {'HIGH' if 80 < opt_y < 150 else 'LOW'}")

print(f"\nPLANET 9 SCENARIO (Target: 5 Me, Inc: 20°)")
print(f"  > Required Distance: {opt_p9:.1f} AU")
print(f"  > Plausibility: {'HIGH' if 400 < opt_p9 < 800 else 'LOW'}")

# --- PLOT ---
plt.figure(figsize=(12, 6))

# Plot Y
plt.plot(dists, y_curve, 'g-', linewidth=3, label='Planet Y Theory (i=65°)')
plt.scatter([opt_y], [1.0], color='green', s=150, zorder=10, edgecolors='black')
plt.text(opt_y, 1.2, f"  1 $M_E$ @ {opt_y:.0f} AU", color='green', fontweight='bold')

# Plot 9
plt.plot(dists, p9_curve, 'r-', linewidth=3, label='Planet 9 Theory (i=20°)')
plt.scatter([opt_p9], [5.0], color='red', s=150, zorder=10, edgecolors='black')
plt.text(opt_p9, 5.5, f"  5 $M_E$ @ {opt_p9:.0f} AU", color='red', fontweight='bold')

# Zones
plt.axhspan(0.8, 1.2, color='green', alpha=0.1, label='Earth-Mass Range')
plt.axhspan(4.5, 5.5, color='red', alpha=0.1, label='Super-Earth Range')

plt.title(f"Which Planet Fits the {TARGET_WARP}° Warp Best?")
plt.xlabel("Distance (AU)")
plt.ylabel("Required Mass (Earth Masses)")
plt.yscale('log')
plt.ylim(0.1, 50)
plt.xlim(0, 1200)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()