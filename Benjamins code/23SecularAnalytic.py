import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# REAL DATA TARGETS (From your GMM Analysis)
REAL_P9_WARP = {'i': 17.9, 'node': 114.2}
REAL_PY_WARP = {'i': 11.2, 'node': 212.3}

print("--- ANALYTIC SECULAR SOLVER ---")

def calc_forced_plane(planet_m, planet_a, planet_i, planet_node, particle_a):
    """
    Calculates the 'Forced Inclination' and 'Forced Node' imposed by a distant planet.
    Based on linear secular theory (Lagrange-Laplace).
    """
    # Constants
    m_sun = 1.0
    
    # The 'Strength' of the planet's perturbation (coefficient B)
    # Proportional to mass / a^3
    # For a particle inside the planet's orbit (a < a_p)
    # Approx: B ~ (m_p / m_sun) * (a / a_p^2) * b_laplace
    
    # Simplified Secular Equilibrium (The "Fixed Point"):
    # If a particle is dominated by the planet, its orbit aligns with the planet's orbit.
    # The 'Forced Plane' is essentially the Planet's Orbital Plane.
    
    # However, the Giant Planets (J/S/U/N) try to keep it at i=0.
    # The result is a weighted average between i=0 (Giants) and i=Planet (P9).
    
    # Torque from Giants (J2 moment approx) vs Torque from P9
    # Torque_Giants ~ 1 / particle_a^(7/2)
    # Torque_P9 ~ particle_a^2 / planet_a^3
    
    # As particle_a increases, P9 wins.
    # We calculate the 'Forced Inclination' (i_forced) at the particle's distance.
    
    # This is a toy model approximation of the Linear Secular solution
    # i_forced = (Torque_P9 * i_P9) / (Torque_Giants + Torque_P9)
    
    torque_giants = 1.0 / (particle_a**3.5) * 1e8 # Scaling factor for inner system rigidity
    torque_p9 = (planet_m / m_sun) * (particle_a**2) / (planet_a**3) * 1e10
    
    i_forced = (torque_p9 * planet_i) / (torque_giants + torque_p9)
    
    # The Node aligns with the dominant torquer (P9)
    # If P9 dominates, Node -> Planet_Node
    # If Giants dominate, Node precession is uniform (no lock)
    # We model Node alignment as a transition
    node_forced = planet_node if torque_p9 > torque_giants else np.random.uniform(0, 360)
    
    return i_forced, node_forced

# --- TEST 1: PLANET 9 MODEL ---
# 5 Earth Mass, 500 AU, i=20, Node=114
print("\nTesting Planet 9 Model...")
p9_i_results = []
test_distances = np.linspace(150, 500, 50)

for a in test_distances:
    i_f, n_f = calc_forced_plane(5e-5, 500, 20, 114, a)
    p9_i_results.append(i_f)

# --- TEST 2: PLANET Y MODEL ---
# 1 Earth Mass, 150 AU, i=10, Node=212
print("Testing Planet Y Model...")
py_i_results = []
for a in test_distances:
    i_f, n_f = calc_forced_plane(3e-6, 150, 10, 212, a)
    py_i_results.append(i_f)

# --- PLOT COMPARISON ---
plt.figure(figsize=(10, 6))

# Plot Real Data Targets (Horizontal Lines)
plt.axhline(REAL_P9_WARP['i'], color='red', linestyle='--', label=f'Real Warp A (i={REAL_P9_WARP["i"]})')
plt.axhline(REAL_PY_WARP['i'], color='green', linestyle='--', label=f'Real Warp B (i={REAL_PY_WARP["i"]})')

# Plot Theoretical Curves
plt.plot(test_distances, p9_i_results, 'r-', linewidth=3, label='Theoretical P9 Influence')
plt.plot(test_distances, py_i_results, 'g-', linewidth=3, label='Theoretical Planet Y Influence')

plt.xlabel("Distance from Sun (AU)")
plt.ylabel("Forced Inclination (deg)")
plt.title("Analytic Check: Which Planet Creates the Observed Warp?")
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 30)
plt.show()