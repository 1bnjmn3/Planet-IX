import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- CONFIGURATION ---
# THE FINAL CONFIRMED PARAMETERS (From Script 29)
FINAL_INC = 15.7   # The "Primary Peak"
FINAL_NODE = 114.0 # We stick with the Node from the "High Warp" group (Plane 2) which was the strongest signal

print(f"--- GENERATING FINAL PUBLICATION TRACK ---")
print(f"Targeting Orbital Plane: i={FINAL_INC}, Node={FINAL_NODE}")

def get_orbit_track(inc_deg, node_deg):
    inc = np.radians(inc_deg)
    node = np.radians(node_deg)
    v = np.linspace(0, 2*np.pi, 500)
    
    # Orbital Frame -> Ecliptic Frame
    x_ecl = np.cos(node)*np.cos(v) - np.sin(node)*np.sin(v)*np.cos(inc)
    y_ecl = np.sin(node)*np.cos(v) + np.cos(node)*np.sin(v)*np.cos(inc)
    z_ecl = np.sin(v)*np.sin(inc)
    
    # Ecliptic -> Equatorial (RA/Dec)
    epsilon = np.radians(23.439)
    x_eq = x_ecl
    y_eq = y_ecl * np.cos(epsilon) - z_ecl * np.sin(epsilon)
    z_eq = y_ecl * np.sin(epsilon) + z_ecl * np.cos(epsilon)
    
    ra = np.degrees(np.arctan2(y_eq, x_eq)) % 360
    dec = np.degrees(np.arcsin(z_eq))
    
    return pd.DataFrame({'RA': ra, 'Dec': dec}).sort_values('RA')

p9_track = get_orbit_track(FINAL_INC, FINAL_NODE)
gal_track = get_orbit_track(62.87, 282.85) # Galactic Plane

# --- PLOT FOR PUBLICATION ---
plt.figure(figsize=(12, 7))

# 1. The Galaxy (Avoidance Zone)
plt.scatter(gal_track['RA'], gal_track['Dec'], c='gray', s=10, alpha=0.3, label='Galactic Plane (High Stellar Density)')

# 2. The Final Search Track
plt.plot(p9_track['RA'], p9_track['Dec'], 'r-', linewidth=3, label=f'Planet Y Search Track (i={FINAL_INC}°)')

# 3. Mark the "Best Bet" Zones
# These are regions where the track is highest in the sky (best visibility) and far from the galaxy
plt.axvspan(30, 90, color='green', alpha=0.1, label='Prime Observation Window (Fall/Winter)')
plt.axvspan(210, 270, color='green', alpha=0.1)

plt.xlabel("Right Ascension (deg)")
plt.ylabel("Declination (deg)")
plt.title(f"Targeting Map for Earth-Mass Perturber\n(Based on Secular Warp at i={FINAL_INC}°)")
plt.grid(True, alpha=0.3)
plt.xlim(0, 360)
plt.ylim(-60, 60)
plt.legend(loc='lower right')
plt.show()

# --- OUTPUT COORDINATES TABLE ---
print("\n--- TELESCOPE TARGET LIST (Top 5 Vectors) ---")
# Sample points every 45 degrees of RA for quick reference
targets = p9_track.iloc[::50] # Downsample
print(targets[['RA', 'Dec']].to_string(index=False))