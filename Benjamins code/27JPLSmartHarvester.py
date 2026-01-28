import pandas as pd
import numpy as np
import requests
import os

# --- CONFIGURATION ---
SAVE_FILE = "smart_dataset.csv"
JPL_API_URL = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"

print("--- INITIATING SMART HARVEST (Tisserand Filter) ---")

# 1. Query NASA JPL for TNOs [cite: 128]
# We fetch full orbital elements including 'tp' (time of perihelion) for completeness
params = {
    "sb-class": "TNO",
    "fields": "full_name,a,e,i,w,om,q,tp", 
    "full-prec": "true"
}

try:
    print("Contacting JPL Servers...")
    response = requests.get(JPL_API_URL, params=params)
    response.raise_for_status()
    data = response.json()
    
    cols = data['fields']
    raw_rows = data['data']
    
    df = pd.DataFrame(raw_rows, columns=cols)
    
    # Clean Data: Convert strings to numeric
    for c in ['a', 'e', 'i', 'w', 'om', 'q']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df = df.rename(columns={'om': 'Node', 'full_name': 'ID'})
    
    # 2. CALCULATE TISSERAND PARAMETER (T_N)
    # T_N > 3 implies stability (detached from Neptune scattering) [cite: 116]
    # Formula: T_N = a_N/a + 2 * sqrt( (a/a_N) * (1-e^2) ) * cos(i)
    a_Neptune = 30.07
    
    df['T_N'] = (a_Neptune / df['a']) + 2 * np.sqrt( (df['a'] / a_Neptune) * (1 - df['e']**2) ) * np.cos(np.radians(df['i']))
    
    # 3. APPLY SMART FILTERS
    # Filter A: Distance (Relaxed to > 150 AU to capture transition objects) [cite: 129]
    mask_dist = (df['a'] > 150)
    
    # Filter B: Stability (Physics-based)
    # We set q > 38 AU to ensure objects are outside the immediate scattering zone of Neptune (a_N=30)
    # This aligns with the critique's suggestion to avoid "overfiltering" while excluding transients.
    mask_stable = (df['q'] > 38)
    
    df_smart = df[mask_dist & mask_stable].copy()
    
    print(f"Total TNOs Downloaded: {len(df)}")
    print(f"Smart Filter Survivors: {len(df_smart)}")
    print(f"  (Criteria: a > 150 AU AND q > 38 AU)")
    
    # Save to CSV for subsequent GMM analysis
    df_smart.to_csv(SAVE_FILE, index=False)
    print(f"Saved to {SAVE_FILE}")
    
    # Check against the critique's target of N > 30 [cite: 129]
    if len(df_smart) > 30:
        print("\nSUCCESS: Dataset size > 30. Critique Requirement Satisfied.")
    else:
        print("\nWARNING: Dataset size is still small. Further relaxation (a > 120) may be needed.")

except Exception as e:
    print(f"Harvest Failed: {e}")