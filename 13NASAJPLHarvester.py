import pandas as pd
import numpy as np
import requests
import os

# --- CONFIGURATION ---
MPC_FILE = "live_mpc_data.csv"
SAVE_FILE = "mega_dataset.csv"
JPL_API_URL = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"

print("--- INITIATING NASA JPL DATA RAID ---")

# 1. Query NASA JPL for ALL Trans-Neptunian Objects (TNOs)
# We fetch specific fields: object name, a, e, i, w, Node, q
params = {
    "sb-class": "TNO", # Limit to Trans-Neptunian Objects
    "fields": "full_name,a,e,i,w,om,q",
    "full-prec": "true"
}

try:
    print("Contacting JPL Servers...")
    response = requests.get(JPL_API_URL, params=params)
    response.raise_for_status()
    data = response.json()
    
    # 2. Parse JSON Response
    # JPL data comes in a 'data' list and 'fields' header
    cols = data['fields'] # ['full_name', 'a', 'e', 'i', 'w', 'om', 'q']
    raw_rows = data['data']
    
    print(f"JPL returned {len(raw_rows)} total TNOs.")
    
    # Convert to DataFrame
    df_jpl = pd.DataFrame(raw_rows, columns=cols)
    
    # Convert numeric columns (they come as strings)
    for c in ['a', 'e', 'i', 'w', 'om', 'q']:
        df_jpl[c] = pd.to_numeric(df_jpl[c], errors='coerce')
        
    # Rename columns to match our MPC format
    df_jpl = df_jpl.rename(columns={'om': 'Node', 'full_name': 'ID'})
    
    # Calculate varpi (Longitude of Perihelion)
    df_jpl['varpi'] = (df_jpl['w'] + df_jpl['Node']) % 360
    
    # 3. Apply the "Wider Net" Filter
    # a > 100 (Grab the transition objects)
    # q > 35 (Stay detached from Neptune)
    df_jpl_filtered = df_jpl[ (df_jpl['a'] > 100) & (df_jpl['q'] > 35) ].copy()
    print(f"JPL Objects meeting criteria (a>100, q>35): {len(df_jpl_filtered)}")

    # 4. Merge with MPC Data (If exists)
    if os.path.exists(MPC_FILE):
        print("Merging with MPC data...")
        df_mpc = pd.read_csv(MPC_FILE)
        
        # Standardize IDs for duplicate checking
        # MPC IDs look like "(523735) 2014 QX441"
        # JPL IDs look like "523735 (2014 QX441)"
        # We'll rely on numerical comparison of 'a', 'e', 'i' to detect duplicates
        # because names are messy.
        
        # Combine lists
        df_combined = pd.concat([df_mpc, df_jpl_filtered], ignore_index=True)
        
        # Remove duplicates based on orbital elements (rounded to 3 decimals)
        # If two objects have the same a, e, i, they are the same object.
        df_final = df_combined.drop_duplicates(subset=['a', 'e', 'i', 'varpi'], keep='first')
        
        # Re-filter again just to be safe (ensure MPC data also meets new q>35 check)
        df_final = df_final[ (df_final['a'] > 100) & (df_final['q'] > 35) ]
        
    else:
        df_final = df_jpl_filtered

    print(f"\n--- HARVEST COMPLETE ---")
    print(f"Total Unique Objects in Mega Dataset: {len(df_final)}")
    
    df_final.to_csv(SAVE_FILE, index=False)
    print(f"Saved to {SAVE_FILE}")
    print(df_final[['ID', 'a', 'q', 'varpi']].sort_values('a', ascending=False).head(5))

except Exception as e:
    print(f"NASA RAID FAILED: {e}")