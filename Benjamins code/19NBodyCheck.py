import rebound
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- CONFIGURATION ---
# Your Detected Warp Parameters (from Script 18)
TARGET_INC = 16.8  # degrees
TARGET_NODE = 154.3 # degrees

print(f"--- INITIATING N-BODY SIMULATION ---")
print(f"Testing if Planet 9 can create a warp at i={TARGET_INC}, Node={TARGET_NODE}")

# 1. Setup Simulation
sim = rebound.Simulation()
sim.units = ('yr', 'AU', 'Msun')

# Add Sun
sim.add(m=1.0)

# Add Giant Planets (approximate for secular speed)
# Jupiter, Saturn, Uranus, Neptune
sim.add(m=0.0009543, a=5.2)
sim.add(m=0.0002857, a=9.5)
sim.add(m=0.0000436, a=19.2)
sim.add(m=0.0000515, a=30.0)

# Add CANDIDATE PLANET 9
# We place it at parameters that *should* theoretically cause a warp
# Mass ~ 5-10 Earth masses
# a ~ 400-500 AU
# i ~ 20 deg (similar to your warp)
sim.add(m=5e-5, a=500, e=0.25, inc=np.radians(20), Omega=np.radians(TARGET_NODE + 180)) 
# Note: We put P9's node 180 deg away to see if it shepherded objects to the opposite side

# Add Test Particles (The TNOs)
# We add them Randomly to see if they get "pushed" into the warp
n_particles = 100
print(f"Injecting {n_particles} random test particles...")
for _ in range(n_particles):
    rand_a = np.random.uniform(150, 400)
    rand_e = np.random.uniform(0.1, 0.5)
    rand_i = np.random.uniform(0, 40) # Random inclination
    rand_node = np.random.uniform(0, 2*np.pi) # Random node
    rand_w = np.random.uniform(0, 2*np.pi)
    
    sim.add(a=rand_a, e=rand_e, inc=np.radians(rand_i), Omega=rand_node, omega=rand_w)

# 2. Run Integration
# We use IAS15 (high precision) or WHFast (fast). For secular trends, IAS15 is safer.
# We run for a short "epoch" to calculate forces/trends, not 4Gyr (too slow for script).
# We look at the "Secular Torque".
sim.move_to_com()

times = np.linspace(0, 500000, 50) # Run for 500k years (Quick check)
print(f"Integrating for {times[-1]} years (Fast-Forward)...")

inclinations = []
nodes = []

for i, time in enumerate(times):
    sim.integrate(time)
    
    # Snapshot of particles (indices 5 to end are test particles)
    # 0=Sun, 1-4=Giants, 5=P9
    current_incs = []
    current_nodes = []
    
    for p in sim.particles[6:]:
        current_incs.append(np.degrees(p.inc))
        current_nodes.append(np.degrees(p.Omega))
        
    inclinations.append(current_incs)
    nodes.append(current_nodes)
    
    if i % 10 == 0:
        print(f"Progress: {i/len(times)*100:.0f}%...")

# 3. Analyze Results
# Did the particles drift towards the target?
final_incs = np.array(inclinations[-1])
final_nodes = np.array(nodes[-1])

# Calculate "Closeness" to your Warp
# We measure the distance in (i, Node) space
dist_to_warp = np.sqrt((final_incs - TARGET_INC)**2 + (final_nodes - TARGET_NODE)**2)
captured = np.sum(dist_to_warp < 20) # Within 20 degrees is "Captured"

print(f"\n--- SIMULATION RESULTS ---")
print(f"Particles 'Captured' by the P9 Warp Field: {captured}/{n_particles}")

# 4. Plot Evolution
plt.figure(figsize=(10, 8))

# Plot Starting Positions (Grey)
start_incs = inclinations[0]
start_nodes = nodes[0]
plt.scatter(start_nodes, start_incs, c='gray', alpha=0.3, label='Start (Random)')

# Plot Final Positions (Blue)

plt.scatter(final_nodes, final_incs, c='blue', alpha=0.6, label=f'End ({int(times[-1])} yrs)')

# Plot Your Warp Target
plt.scatter(TARGET_NODE, TARGET_INC, c='red', marker='X', s=200, label='REAL DATA Warp Center')

# Draw Arrows showing movement
for k in range(n_particles):
    # connecting line
    plt.plot([start_nodes[k], final_nodes[k]], [start_incs[k], final_incs[k]], 'k-', alpha=0.1)

plt.xlabel("Longitude of Ascending Node (deg)")
plt.ylabel("Inclination (deg)")
plt.title(f"N-Body Test: Does P9 create the observed Warp?\n(Target: i={TARGET_INC}, Node={TARGET_NODE})")
plt.legend()
plt.xlim(0, 360)
plt.ylim(0, 60)
plt.show()