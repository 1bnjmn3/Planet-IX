import rebound
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# We run two parallel simulations
# 1. P9 Model (The Heavyweight)
P9_PARAMS = {'m': 5e-5, 'a': 500, 'e': 0.25, 'inc': 20, 'omega': 114} # Node aligned w/ Plane 2

# 2. Planet Y Model (The Lightweight - based on Critique/Plane 1)
PY_PARAMS = {'m': 3e-6, 'a': 150, 'e': 0.1, 'inc': 11, 'omega': 212} # Node aligned w/ Plane 1 ~212

print("--- MODEL FACE-OFF: PLANET 9 VS PLANET Y ---")

def run_simulation(planet_params, label):
    print(f"Running {label} Simulation...")
    sim = rebound.Simulation()
    sim.add(m=1.0) # Sun
    # Add Giant Planets (Approx secular influence)
    sim.add(m=0.00095, a=5.2)
    sim.add(m=0.00028, a=9.5)
    sim.add(m=0.00004, a=19.2)
    sim.add(m=0.00005, a=30.0)
    
    # Add The Candidate
    sim.add(m=planet_params['m'], a=planet_params['a'], e=planet_params['e'], 
            inc=np.radians(planet_params['inc']), Omega=np.radians(planet_params['omega']))
    
    # Add Test Particles (Stable Zone: a > 250)
    final_nodes = []
    final_incs = []
    
    # Inject 50 particles
    for _ in range(50):
        sim.add(a=np.random.uniform(250, 400), e=np.random.uniform(0.1, 0.4), 
                inc=np.radians(np.random.uniform(0, 30)), Omega=np.random.uniform(0, 2*np.pi))
        
    sim.move_to_com()
    # Run for 250,000 years (Short secular check)
    sim.integrate(250000)
    
    for p in sim.particles[6:]:
        final_nodes.append(np.degrees(p.Omega) % 360)
        final_incs.append(np.degrees(p.inc))
        
    return final_nodes, final_incs

# Run Both
p9_nodes, p9_incs = run_simulation(P9_PARAMS, "Planet 9")
py_nodes, py_incs = run_simulation(PY_PARAMS, "Planet Y")

# --- PLOT THE RESULTS ---
plt.figure(figsize=(12, 6))

# Plot Real Data Targets
plt.scatter(114.2, 17.9, s=200, c='red', marker='X', label='Real Plane 2 (P9 Candidate)')
plt.scatter(212.3, 11.2, s=200, c='green', marker='X', label='Real Plane 1 (Planet Y Candidate)')

# Plot P9 Results
plt.scatter(p9_nodes, p9_incs, c='red', alpha=0.3, label='P9 Simulation Particles')

# Plot PY Results
plt.scatter(py_nodes, py_incs, c='green', alpha=0.3, label='Planet Y Sim Particles')

plt.xlabel("Longitude of Node (deg)")
plt.ylabel("Inclination (deg)")
plt.title("Dynamical Forensics: Which Planet Matches the Data?")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 360)
plt.ylim(0, 40)
plt.show()