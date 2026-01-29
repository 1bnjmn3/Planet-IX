import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import os

# --- CONFIGURATION ---
# We use the Mega Dataset (Script 13 output) for maximum reach
INPUT_FILE = "mega_dataset.csv" 
# Fallback to live data if mega doesn't exist
if not os.path.exists(INPUT_FILE):
    INPUT_FILE = "live_mpc_data.csv"

if not os.path.exists(INPUT_FILE):
    print("Error: No data found. Run Script 13 or 11 first.")
    exit()

df_full = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df_full)} objects. Initiating Sensitivity Scan...")

# --- PHYSICS: THE TISSERAND PARAMETER (Smart Noise Filter) ---
def calc_tisserand(df):
    # T_N = a_N / a + 2 * sqrt( (a/a_N) * (1-e^2) ) * cos(i)
    # Neptune a_N = 30.07 AU
    a_N = 30.07
    return a_N / df['a'] + 2 * np.sqrt( (df['a']/a_N) * (1 - df['e']**2) ) * np.cos(np.radians(df['i']))

df_full['T_N'] = calc_tisserand(df_full)

# --- THE SIMULATION ENGINE ---
# We use a simplified/fast version of the Bimodal test for speed
def get_p9_probability(df_slice):
    if len(df_slice) < 10: return 0.5 # Too few data points to judge
    
    # 1. Generate Synthetic Worlds based on THIS slice's orbital stats
    n_sim = 5000
    # World A: Bimodal P9 (60 deg + 240 deg)
    a = np.random.choice(df_slice['a'], n_sim)
    e = np.random.choice(df_slice['e'], n_sim)
    i = np.random.choice(df_slice['i'], n_sim) # Use raw degrees if that's what's in CSV
    
    # Angles
    n_anti = int(n_sim * 0.6)
    n_align = int(n_sim * 0.3)
    n_scat = n_sim - n_anti - n_align
    v = np.concatenate([
        np.random.vonmises(np.radians(60), 2.5, n_anti),
        np.random.vonmises(np.radians(240), 2.0, n_align),
        np.random.uniform(0, 2*np.pi, n_scat)
    ])
    np.random.shuffle(v)
    
    X_p9 = pd.DataFrame({'a':a, 'varpi_sin':np.sin(v), 'varpi_cos':np.cos(v)})
    X_p9['label'] = 1
    
    # World B: Random
    v_rnd = np.random.uniform(0, 2*np.pi, n_sim)
    X_rnd = pd.DataFrame({'a':a, 'varpi_sin':np.sin(v_rnd), 'varpi_cos':np.cos(v_rnd)})
    X_rnd['label'] = 0
    
    # Train
    train = pd.concat([X_p9, X_rnd])
    clf = RandomForestClassifier(n_estimators=50, max_depth=5) # Light model
    clf.fit(train[['a', 'varpi_sin', 'varpi_cos']], train['label'])
    
    # Test on Real Data
    X_real = pd.DataFrame({
        'a': df_slice['a'],
        'varpi_sin': np.sin(np.radians(df_slice['varpi'])),
        'varpi_cos': np.cos(np.radians(df_slice['varpi']))
    })
    
    probs = clf.predict_proba(X_real)[:, 1]
    return np.mean(probs)

# --- THE GRID SEARCH ---
# We sweep 'a' from 100 to 300
a_cuts = range(100, 310, 10)
results_q30 = [] # For q > 30 (Standard)
results_q35 = [] # For q > 35 (Detached)
results_tisserand = [] # For T_N filtering (Smart)

print("Scanning Limits...")

for cut in a_cuts:
    # Scenario 1: q > 30
    subset_30 = df_full[ (df_full['a'] > cut) & (df_full['q'] > 30) ]
    prob_30 = get_p9_probability(subset_30)
    results_q30.append(prob_30)
    
    # Scenario 2: q > 35 (Cleaner)
    subset_35 = df_full[ (df_full['a'] > cut) & (df_full['q'] > 35) ]
    prob_35 = get_p9_probability(subset_35)
    results_q35.append(prob_35)
    
    # Scenario 3: Tisserand Filter (Ignore 'q' cut, rely on Physics)
    # Keep objects where T_N is NOT between 2.9 and 3.1 (Neptune Resonance Zone)
    # And a > cut
    subset_t = df_full[ (df_full['a'] > cut) & ((df_full['T_N'] < 2.9) | (df_full['T_N'] > 3.1)) ]
    prob_t = get_p9_probability(subset_t)
    results_tisserand.append(prob_t)
    
    print(f"a > {cut}: N={len(subset_30)} | P9_Prob={prob_35:.2f}")

# --- PLOT THE TRADEOFF ---
plt.figure(figsize=(10, 6))

plt.plot(a_cuts, results_q30, 'b--o', label='Standard (q > 30)', alpha=0.5)
plt.plot(a_cuts, results_q35, 'g-o', label='Detached (q > 35)', linewidth=2)
plt.plot(a_cuts, results_tisserand, 'r-o', label='Tisserand Smart Filter', linewidth=2)

# Mark the 50% "Coin Flip" line
plt.axhline(0.5, color='gray', linestyle=':')
plt.axhspan(0.6, 1.0, color='green', alpha=0.1, label='Strong Evidence Zone')

plt.xlabel("Minimum Semi-Major Axis (a) [AU]")
plt.ylabel("AI Confidence in Planet 9 Model")
plt.title("Limit Optimizer: Where is the Signal?")
plt.legend()
plt.grid(True)
plt.show()