import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
# Use absolute path to be safe, or relative if you are in the folder
FILE_NAME = 'distant_extended.dat' 
OUTPUT_FILE = 'processed_etnos.csv'

def load_mpc_data(filepath):
    data = []
    
    if not os.path.exists(filepath):
        print(f"CRITICAL ERROR: Could not find file at: {filepath}")
        return pd.DataFrame()

    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    print(f"Reading {len(lines)} lines from file...")
    
    for line in lines:
        # Skip short lines or headers
        if len(line) < 50: continue
        
        # METHOD: Split by whitespace
        # Your file format appears to be:
        # ID, H, G, Epoch, M, w, Node, i, e, n, a, ...
        parts = line.split()
        
        try:
            # We need at least 11 columns to get 'a' (index 10)
            if len(parts) < 11: continue
            
            # Extract Orbital Elements based on your file structure
            # parts[5] = Argument of Perihelion (w)
            # parts[6] = Longitude of Ascending Node (Node)
            # parts[7] = Inclination (i)
            # parts[8] = Eccentricity (e)
            # parts[10] = Semi-major Axis (a)
            
            w_val = float(parts[5])
            node_val = float(parts[6])
            i_val = float(parts[7])
            e_val = float(parts[8])
            a_val = float(parts[10])
            
            # Calculate Perihelion Distance (q) -> q = a * (1 - e)
            q_val = a_val * (1.0 - e_val)
            
            data.append([a_val, q_val, e_val, i_val, w_val, node_val])
            
        except (ValueError, IndexError):
            # If a line is messy or a header, skip it
            continue

    df = pd.DataFrame(data, columns=['a', 'q', 'e', 'i', 'w', 'Node'])
    return df

def filter_etnos(df):
    # STRICTER FILTER:
    # a > 230 AU (Removes Neptune scattering noise)
    # q > 30 AU (Detached)
    # This aligns closer to the Batygin/Brown "metastable" population
    etnos = df[ (df['a'] > 230) & (df['q'] > 30) ].copy()
    return etnos

def calculate_features(df):
    # Calculate Longitude of Perihelion (varpi)
    df['varpi'] = (df['w'] + df['Node']) % 360
    
    # Convert to Radians and Components
    df['varpi_rad'] = np.radians(df['varpi'])
    df['i_rad'] = np.radians(df['i'])
    df['varpi_sin'] = np.sin(df['varpi_rad'])
    df['varpi_cos'] = np.cos(df['varpi_rad'])
    
    return df

# --- EXECUTION ---
# Get the absolute path of the current script to find the .dat file next to it
script_dir = os.path.dirname(os.path.abspath(__file__))
full_path = os.path.join(script_dir, FILE_NAME)

print(f"Looking for data at: {full_path}")

df_raw = load_mpc_data(full_path)

if df_raw.empty:
    print("ERROR: No data loaded. Check the file content.")
else:
    df_etnos = filter_etnos(df_raw)
    
    if df_etnos.empty:
        print("WARNING: Data loaded, but NO objects met the ETNO criteria (a>150, q>30).")
        print("Max 'a' found in file:", df_raw['a'].max())
    else:
        df_ready = calculate_features(df_etnos)
        
        print(f"Total Objects Parsed: {len(df_raw)}")
        print(f"Extreme TNOs (ETNOs) Found: {len(df_ready)}")
        print(df_ready[['a', 'e', 'varpi']].head())
        
        # SAVE THE FILE
        output_path = os.path.join(script_dir, OUTPUT_FILE)
        df_ready.to_csv(output_path, index=False)
        print(f"Saved ML-ready data to: {output_path}")