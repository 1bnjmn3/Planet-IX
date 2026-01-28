import pandas as pd
import numpy as np
import requests
import io
import os

# --- CONFIGURATION ---
MPC_URL = "https://www.minorplanetcenter.net/iau/MPCORB/Distant.txt"
SAVE_FILE = "live_mpc_data.csv"

print("--- INITIATING DATA HARVEST ---")
print(f"Target: {MPC_URL}")

try:
    # 1. Download the latest data
    response = requests.get(MPC_URL)
    response.raise_for_status()
    print("Download successful. Parsing data...")
    
    # 2. Parse Fixed-Width Format (MPC Standard)
    # The file has a header, then data. We need to handle it carefully.
    raw_lines = response.text.split('\n')
    
    data = []
    for line in raw_lines:
        if len(line) < 100: continue # Skip headers/empty lines
        
        try:
            # MPCORB Format slicing
            # a (Semi-major axis) is usually around column 92-103
            # e (Eccentricity) around 70-79
            # i (Inclination) around 59-68
            # Node (Omega) around 48-57
            # ArgPeri (w) around 37-46
            
            # Note: These positions are standard for MPCORB.DAT format
            # We strip whitespace to handle alignment
            id_str = line[0:7].strip()
            w = float(line[37:46])
            node = float(line[48:57])
            i = float(line[59:68])
            e = float(line[70:79])
            
            # 'n' (mean motion) is often used to calc 'a', or 'a' is explicit
            # In Distant.txt, 'a' is often at the end or calculated.
            # Let's try to extract 'a' from column 92-103 if it exists, else calc from n
            try:
                a = float(line[92:103])
            except:
                # Fallback: Calculate a from mean motion n (degrees/day)
                # Kepler's 3rd Law: n^2 * a^3 = k (approx)
                # a = (0.9856076686 / n)^(2/3) roughly
                n_val = float(line[80:91])
                if n_val > 0:
                    a = (0.9856 / n_val) ** (2/3)
                else:
                    continue

            # Calculate Derived Physics
            q = a * (1.0 - e) # Perihelion
            varpi = (w + node) % 360 # Longitude of Perihelion
            
            data.append([id_str, a, q, e, i, varpi, w, node])
            
        except ValueError:
            continue

    df = pd.DataFrame(data, columns=['ID', 'a', 'q', 'e', 'i', 'varpi', 'w', 'Node'])
    
    # 3. Filter for our project (High Quality ETNOs)
    # Critique suggested checking a > 150 again to be inclusive
    df_etno = df[ (df['a'] > 150) & (df['q'] > 30) ].copy()
    
    print(f"\nTotal Objects Parsed: {len(df)}")
    print(f"ETNOs Found (a > 150, q > 30): {len(df_etno)}")
    
    # Save
    df_etno.to_csv(SAVE_FILE, index=False)
    print(f"Saved live dataset to: {SAVE_FILE}")
    print("Top 5 Fresh Objects:")
    print(df_etno[['ID', 'a', 'q', 'varpi']].head())

except Exception as e:
    print(f"CRITICAL FAILURE: {e}")
    print("Check your internet connection or MPC URL.")