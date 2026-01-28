import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# --- CONFIGURATION ---
TARGET_WARP = 15.7      # The observed angle (from Script 29)
PLANET_INC = 65.0       # The hypothesized rigid inclination
PARTICLE_DIST = 400.0   # Average distance of our ETNOs

print(f"--- HIGH-INCLINATION SOLVER (DEBUG MODE) ---")
print(f"Goal: Create a {TARGET_WARP}° Warp using a {PLANET_INC}° Planet.")

# 1. DEFINE THE PHYSICS
# We use the same 'Tune' from Script 25 that gave us 1.65 Me @ 500 AU = 17 deg
# Calibration Factor derived from Script 25 result:
# Torque_ratio = (1.65 * 3e-6 * 400**2 / 500**3 * 1e10) / (1/400**3.5 * 1e8) approx 1.0
# We stick to the relative formula to ensure consistency.

def get_required_mass(planet_dist, target_warp, planet_inc):
    # Torque Giant Planets (Force trying to flatten orbit)
    torque_giants = 1.0 / (PARTICLE_DIST**3.5) * 1e8
    
    # Torque Planet X (Force trying to lift orbit)
    # We solve for Torque_P in the balance equation:
    # Warp = (Torque_P * Inc_P) / (Torque_G + Torque_P)
    # Warp * Torque_G = Torque_P * (Inc_P - Warp)
    # Torque_P = (Warp * Torque_G) / (Inc_P - Warp)
    
    if target_warp >= planet_inc: return np.inf
    
    required_torque_p = (target_warp * torque_giants) / (planet_inc - target_warp)
    
    # Convert Torque -> Mass
    # Torque_P = (Mass_Solar) * (Part_a^2) / (Planet_a^3) * 1e10
    # Mass_Solar = Torque_P * Planet_a^3 / (Part_a^2 * 1e10)
    
    mass_solar = required_torque_p * (planet_dist**3) / (PARTICLE_DIST**2) / 1e10
    mass_earth = mass_solar / 3e-6 # Convert to Earth Masses
    return mass_earth

# 2. CALCULATE CURVE
dist_range = np.linspace(100, 2000, 500) # Scan 100 to 2000 AU
mass_curve = [get_required_mass(d, TARGET_WARP, PLANET_INC) for d in dist_range]

# Check bounds to avoid the "Straight Line" error
print(f"Min Mass Found: {min(mass_curve):.3f} Earth Masses")
print(f"Max Mass Found: {max(mass_curve):.3f} Earth Masses")

# 3. FIND EXACT SOLUTIONS (Interpolation)
# We create a function d(m) to find distance for a given mass
dist_from_mass = interp1d(mass_curve, dist_range, bounds_error=False, fill_value="extrapolate")

sol_1M = float(dist_from_mass(1.0))
sol_5M = float(dist_from_mass(5.0))

print("\n--- SOLUTIONS ---")
if 100 < sol_1M < 2000:
    print(f"1. A 1.0 Earth-Mass Planet (Planet Y) must be at: {sol_1M:.0f} AU")
else:
    print(f"1. A 1.0 Earth-Mass Planet is impossible in this range (Need >2000 AU).")

if 100 < sol_5M < 2000:
    print(f"2. A 5.0 Earth-Mass Planet (Planet 9) must be at: {sol_5M:.0f} AU")
else:
    print(f"2. A 5.0 Earth-Mass Planet is impossible in this range (Need >2000 AU).")

# 4. PLOT
plt.figure(figsize=(10, 6))
plt.plot(dist_range, mass_curve, 'b-', linewidth=2, label=f'Required Mass for {TARGET_WARP}° Warp')

# Mark solutions
if 100 < sol_1M < 2000:
    plt.scatter([sol_1M], [1.0], color='green', s=100, zorder=5, label='Planet Y Solution')
    plt.axvline(sol_1M, color='green', linestyle=':', alpha=0.5)

if 100 < sol_5M < 2000:
    plt.scatter([sol_5M], [5.0], color='red', s=100, zorder=5, label='Planet 9 Solution')
    plt.axvline(sol_5M, color='red', linestyle=':', alpha=0.5)

plt.axhline(1.0, color='green', alpha=0.1)
plt.axhline(5.0, color='red', alpha=0.1)

plt.xlabel("Planet Distance (AU)")
plt.ylabel("Required Mass (Earth Masses)")
plt.title(f"Trade-Off: If Planet is at {PLANET_INC}°, where must it be?")
plt.grid(True, alpha=0.3)
plt.legend()
plt.yscale('log') # Log scale is crucial here
plt.show()