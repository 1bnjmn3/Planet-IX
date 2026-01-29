import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
INPUT_FILE = "live_mpc_data.csv"
if not os.path.exists(INPUT_FILE):
    print("Error: Run Script 11 first.")
    exit()

df = pd.read_csv(INPUT_FILE)
# Apply the Iron Filter (Stable only)
df_stable = df[ (df['a'] > 250) & (df['q'] > 40) ].copy()

print(f"Mapping Orbital Poles of {len(df_stable)} stable objects...")

# 1. CALCULATE ORBITAL POLES
# The pole is a vector perpendicular to the orbit.
# In Cartesian coords (x,y,z):
# Lx = sin(i) * sin(Node)
# Ly = -sin(i) * cos(Node)
# Lz = cos(i)
# (Note: This points to the orbital "North")

def get_pole_coords(row):
    i_rad = np.radians(row['i'])
    node_rad = np.radians(row['Node'])
    
    # Vector components
    lx = np.sin(i_rad) * np.sin(node_rad)
    ly = -np.sin(i_rad) * np.cos(node_rad)
    lz = np.cos(i_rad)
    
    # Convert Vector -> RA/Dec on the Sky
    # This tells us where the "Axle" of the orbit points
    dec = np.degrees(np.arcsin(lz))
    ra = np.degrees(np.arctan2(ly, lx)) % 360
    
    return pd.Series([ra, dec])

df_stable[['Pole_RA', 'Pole_Dec']] = df_stable.apply(get_pole_coords, axis=1)

# 2. THE VISUAL TEST
plt.figure(figsize=(10, 8))

# Plot the Poles
# If random: Scattered everywhere.
# If Warp: Clustered in one tight spot.
plt.scatter(df_stable['Pole_RA'], df_stable['Pole_Dec'], c='blue', s=100, edgecolors='black', label='Orbital Poles')

# Plot the "Average Pole" of your Warp Candidate
# Warp was approx i=17.9, Node=114
warp_i = np.radians(17.9)
warp_node = np.radians(114)
w_lx = np.sin(warp_i) * np.sin(warp_node)
w_ly = -np.sin(warp_i) * np.cos(warp_node)
w_lz = np.cos(warp_i)
w_dec = np.degrees(np.arcsin(w_lz))
w_ra = np.degrees(np.arctan2(w_ly, w_lx)) % 360

plt.scatter(w_ra, w_dec, c='red', marker='X', s=300, label='Predicted Warp Center')

plt.xlabel("Pole Right Ascension (deg)")
plt.ylabel("Pole Declination (deg)")
plt.title(f"The Naked Eye Test: Do the Orbital Poles Cluster?\n(N={len(df_stable)})")

plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(0, 360)
plt.ylim(-90, 90)

# Add "Target Zones"
# If points are here, they support the Warp
circle = plt.Circle((w_ra, w_dec), 10, color='red', fill=False, linestyle='--', label='10-deg Confidence Zone')
plt.gca().add_patch(circle)

plt.show()