import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import io

# --- CONFIGURATION ---
MPC_URL = "https://www.minorplanetcenter.net/iau/MPCORB/Distant.txt"

print("--- RE-ACQUIRING DATA WITH EPHEMERIS ---")

# 1. Fetch Data & Extract Mean Anomaly (M)
response = requests.get(MPC_URL)
raw_lines = response.text.split('\n')

data = []
for line in raw_lines:
    if len(line) < 100: continue
    try:
        # MPCORB Format
        # M (Mean Anomaly) is usually cols 26-35
        m_val = float(line[26:35])
        w_val = float(line[37:46])
        node_val = float(line[48:57])
        i_val = float(line[59:68])
        e_val = float(line[70:79])
        n_val = float(line[80:91])
        
        # Calculate 'a'
        if n_val > 0:
            a_val = (0.9856 / n_val) ** (2/3)
        else:
            continue
            
        # Filter (High Quality ETNOs)
        q_val = a_val * (1 - e_val)
        if a_val > 150 and q_val > 30:
             data.append([a_val, e_val, i_val, node_val, w_val, m_val])
             
    except ValueError:
        continue

df = pd.DataFrame(data, columns=['a', 'e', 'i', 'Node', 'w', 'M'])
print(f"Calculated positions for {len(df)} objects.")

# 2. PHYSICS ENGINE: Calculate Sky Coordinates
# We need to convert Orbit Elements -> Ecliptic Longitude/Latitude

def solve_kepler(M, e):
    # Newton-Raphson solver for E = M + e*sin(E)
    E = M # Initial guess
    for _ in range(10):
        f = E - e * np.sin(E) - M
        fp = 1 - e * np.cos(E)
        E = E - f / fp
    return E

def get_sky_pos(row):
    # Convert degrees to radians
    i = np.radians(row['i'])
    om = np.radians(row['Node']) # Omega (Long. Ascending Node)
    w = np.radians(row['w'])     # Argument of Perihelion
    M = np.radians(row['M'])     # Mean Anomaly
    e = row['e']
    
    # 1. Solve Kepler (Mean Anomaly -> Eccentric Anomaly)
    E = solve_kepler(M, e)
    
    # 2. True Anomaly (v)
    # v = 2 * atan( sqrt((1+e)/(1-e)) * tan(E/2) )
    v = 2 * np.arctan(np.sqrt((1+e)/(1-e)) * np.tan(E/2))
    
    # 3. Heliocentric Coordinates in Orbital Plane
    # r = a * (1 - e*cos(E))
    r = row['a'] * (1 - e*np.cos(E))
    
    # Position in orbital plane (z_orb = 0)
    # x_orb = r * cos(v)
    # y_orb = r * sin(v)
    
    # 4. Rotate to Ecliptic Frame
    # We combine the rotations: 
    # u = w + v (Argument of Latitude)
    u = w + v
    
    x_ecl = r * (np.cos(om)*np.cos(u) - np.sin(om)*np.sin(u)*np.cos(i))
    y_ecl = r * (np.sin(om)*np.cos(u) + np.cos(om)*np.sin(u)*np.cos(i))
    z_ecl = r * (np.sin(u)*np.sin(i))
    
    # 5. Convert XYZ -> Lon/Lat
    lon = np.degrees(np.arctan2(y_ecl, x_ecl)) % 360
    lat = np.degrees(np.arcsin(z_ecl / r))
    
    return pd.Series([lon, lat])

# Apply Physics
df[['Sky_Lon', 'Sky_Lat']] = df.apply(get_sky_pos, axis=1)

# 3. THE "NUCLEAR OPTION" PLOT
plt.figure(figsize=(15, 8))

# A. The Galaxy (Avoidance Zone)
x_gal = np.linspace(0, 360, 500)
# Approx Sine wave for Milky Way in Ecliptic coords
y_gal = 60 * np.sin(np.radians(x_gal - 280)) 
plt.fill_between(x_gal, y_gal-15, y_gal+15, color='gray', alpha=0.3, label='Milky Way (Cannot Observe)')

# B. Survey Fields (Where we looked)
# OSSOS/DES roughly target these longitudes
plt.axvspan(300, 360, ymin=0.1, ymax=0.9, color='blue', alpha=0.1, label='Major Survey Windows')
plt.axvspan(0, 60, ymin=0.1, ymax=0.9, color='blue', alpha=0.1)

# C. Real Objects

plt.scatter(df['Sky_Lon'], df['Sky_Lat'], c='red', s=80, edgecolors='black', label='Real ETNO Positions')

plt.xlabel("Ecliptic Longitude (deg)")
plt.ylabel("Ecliptic Latitude (deg)")
plt.title(f"The 'Nuclear Option': Sky Position of {len(df)} ETNOs vs Survey Bias")
plt.legend(loc='lower center', ncol=3)
plt.grid(True, alpha=0.3)
plt.xlim(0, 360)
plt.ylim(-60, 60)

plt.show()