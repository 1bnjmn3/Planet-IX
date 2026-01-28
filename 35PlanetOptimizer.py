import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# --- CONFIGURATION ---
TARGET_WARP = 15.7       # From Script 29
ETNO_DIST = 400.0        # Average distance of our stable rocks
GIANT_J2 = 38023.0       # Strength of Giants (Earth Mass * AU^2) from Script 34

print(f"--- DUAL-REGIME PLANET OPTIMIZER ---")
print(f"Target: Explain the {TARGET_WARP}° Warp.")

# --- PHYSICS ENGINE 1: PLANET Y (INNER REGIME) ---
# Formula: Weighted Average of Planes (J2 Balance)
def solve_inner_mass(planet_dist, planet_inc):
    if TARGET_WARP >= planet_inc: return np.inf
    # warp = (Strength_Y * Inc_Y) / (Strength_G + Strength_Y)
    # Strength_Y * (Inc_Y - warp) = warp * Strength_G
    # Strength_Y = (warp * Strength_G) / (Inc_Y - warp)
    # Strength_Y = Mass * dist^2
    
    req_strength = (TARGET_WARP * GIANT_J2) / (planet_inc - TARGET_WARP)
    mass = req_strength / (planet_dist**2)
    return mass

# --- PHYSICS ENGINE 2: PLANET 9 (OUTER REGIME) ---
# Formula: External Torque Balance (Laplace-Lagrange)
def solve_outer_mass(planet_dist, planet_inc):
    if TARGET_WARP >= planet_inc: return np.inf
    
    # Torque A (Giants trying to flatten): Proportional to 1/r^3.5
    tau_giants = 1.0 / (ETNO_DIST**3.5) * 1e8
    
    # Torque B (Planet 9 trying to lift): Proportional to Mass * r^2 / P_dist^3
    # Balance: Warp = (Tau_B * Inc_P) / (Tau_A + Tau_B)
    # Tau_B = (Warp * Tau_A) / (Inc_P - Warp)
    
    req_tau_b = (TARGET_WARP * tau_giants) / (planet_inc - TARGET_WARP)
    
    # Invert Torque B to get Mass
    # Tau_B = (Mass_Solar) * (ETNO_a^2) / (Planet_a^3) * 1e10
    mass_solar = req_tau_b * (planet_dist**3) / (ETNO_DIST**2) / 1e10
    mass_earth = mass_solar / 3e-6
    return mass_earth

# --- RUN THE SOLVERS ---

# 1. OPTIMIZE PLANET Y (Fixed at 65 deg inclination)
y_dists = np.linspace(60, 150, 100)
y_masses = [solve_inner_mass(d, 65.0) for d in y_dists]

# 2. OPTIMIZE PLANET 9 (Fixed at 65 deg inclination)
p9_dists = np.linspace(300, 1200, 100)
p9_masses = [solve_outer_mass(d, 65.0) for d in p9_dists]

# --- FIND "LITERATURE MATCHES" ---
# Planet Y is theorized to be Earth-Mass (~1.0). Where does that happen?
y_solver = interp1d(y_masses, y_dists, fill_value="extrapolate")
optimal_y_dist = float(y_solver(1.0))

# Planet 9 is theorized to be 5 Earth-Masses. Where does that happen?
p9_solver = interp1d(p9_masses, p9_dists, fill_value="extrapolate")
optimal_p9_dist = float(p9_solver(5.0))

print("\n--- OPTIMIZATION RESULTS ---")
print(f"THEORY Y (Inner Perturber, i=65°):")
print(f"  To match the data with 1.0 Earth Mass, it MUST be at: {optimal_y_dist:.1f} AU")

print(f"\nTHEORY 9 (Outer Shepherd, i=65°):")
print(f"  To match the data with 5.0 Earth Masses, it MUST be at: {optimal_p9_dist:.1f} AU")

# --- VISUALIZATION ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot Y
ax1.plot(y_dists, y_masses, 'g-', linewidth=3)
ax1.scatter([optimal_y_dist], [1.0], color='black', s=100, zorder=5)
ax1.axvline(optimal_y_dist, color='green', linestyle=':', label=f'Optimal Dist: {optimal_y_dist:.0f} AU')
ax1.set_title("Planet Y Optimization (i=65°)")
ax1.set_xlabel("Distance (AU)")
ax1.set_ylabel("Required Mass (Earth Masses)")
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 9
ax2.plot(p9_dists, p9_masses, 'r-', linewidth=3)
ax2.scatter([optimal_p9_dist], [5.0], color='black', s=100, zorder=5)
ax2.axvline(optimal_p9_dist, color='red', linestyle=':', label=f'Optimal Dist: {optimal_p9_dist:.0f} AU')
ax2.set_title("Planet 9 Optimization (i=65°)")
ax2.set_xlabel("Distance (AU)")
ax2.set_ylabel("Required Mass (Earth Masses)")
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.show()