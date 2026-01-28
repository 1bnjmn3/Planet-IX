import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

# --- CONFIGURATION ---
INPUT_FILE = "live_mpc_data.csv" # Use the fresh data!

if not os.path.exists(INPUT_FILE):
    print("Error: Run Script 11 first to get data.")
    exit()

df = pd.read_csv(INPUT_FILE)
# Filter: The critique suggested we might have been too strict.
# Let's check the a > 150 range again, but maybe weight higher 'a' more in the model later.
df_real = df.copy() 
print(f"Analyzing {len(df_real)} ETNOs from Live Data.")

# --- THE NEW SIMULATION (BIMODAL) ---
def generate_advanced_p9_world(n_samples=10000):
    """
    Simulates a 'Sophisticated' Planet 9 influence:
    - 60% of objects in the Anti-Aligned Cluster (60 deg)
    - 30% of objects in the Aligned/Resonant Cluster (240 deg)
    - 10% Scatter
    """
    # Physics basis
    a = np.random.choice(df_real['a'], n_samples)
    e = np.random.choice(df_real['e'], n_samples)
    i = np.random.choice(df_real['i'], n_samples)
    
    # Bimodal Angles
    n_anti = int(n_samples * 0.6)
    n_aligned = int(n_samples * 0.3)
    n_scatter = n_samples - n_anti - n_aligned
    
    # Camp A (Anti-Aligned ~ 60 deg)
    v1 = np.random.vonmises(np.radians(60), 2.5, n_anti)
    # Camp B (Aligned ~ 240 deg)
    v2 = np.random.vonmises(np.radians(240), 2.0, n_aligned)
    # Scatter
    v3 = np.random.uniform(0, 2*np.pi, n_scatter)
    
    varpi = np.concatenate([v1, v2, v3])
    np.random.shuffle(varpi)
    
    return pd.DataFrame({
        'a': a, 'e': e, 'i_rad': np.radians(i),
        'varpi_sin': np.sin(varpi), 'varpi_cos': np.cos(varpi),
        'label': 1 # LABEL 1 = PLANET 9 EXISTS
    })

def generate_random_world(n_samples=10000):
    """The Null Hypothesis (Pure Noise)"""
    a = np.random.choice(df_real['a'], n_samples)
    e = np.random.choice(df_real['e'], n_samples)
    i = np.random.choice(df_real['i'], n_samples)
    varpi = np.random.uniform(0, 2*np.pi, n_samples)
    
    return pd.DataFrame({
        'a': a, 'e': e, 'i_rad': np.radians(i),
        'varpi_sin': np.sin(varpi), 'varpi_cos': np.cos(varpi),
        'label': 0 # LABEL 0 = NO PLANET 9
    })

# 1. Build Training Data
print("Generating 'Advanced P9' Simulation vs Random Noise...")
df_p9 = generate_advanced_p9_world(20000)
df_rnd = generate_random_world(20000)
df_train = pd.concat([df_p9, df_rnd]).sample(frac=1).reset_index(drop=True)

# 2. Train Classifier
print("Training Bimodal Classifier...")
X = df_train[['a', 'e', 'i_rad', 'varpi_sin', 'varpi_cos']]
y = df_train['label']
clf = RandomForestClassifier(n_estimators=150, max_depth=12)
clf.fit(X, y)

# 3. Test on Real Live Data
X_real = pd.DataFrame({
    'a': df_real['a'],
    'e': df_real['e'],
    'i_rad': np.radians(df_real['i']),
    'varpi_sin': np.sin(np.radians(df_real['varpi'])),
    'varpi_cos': np.cos(np.radians(df_real['varpi']))
})

probs = clf.predict_proba(X_real)[:, 1]
avg_prob = np.mean(probs)

print(f"\n--- NEW RESULTS (BIMODAL MODEL) ---")
print(f"Average Support for Planet 9: {avg_prob*100:.1f}%")

if avg_prob > 0.65:
    print("STATUS: CONFIRMED.")
    print("Incorporating the 'Aligned' cluster theory explains the data!")
    print("The critique was right: bimodality IS the signal.")
elif avg_prob < 0.45:
    print("STATUS: DEAD.")
    print("Even with the Bimodal theory, the data looks random.")
else:
    print("STATUS: AMBIGUOUS.")

# 4. Visualization: The "Two Camps" Check
plt.figure(figsize=(10, 10))
ax = plt.subplot(111, polar=True)
sc = ax.scatter(np.radians(df_real['varpi']), df_real['a'], c=probs, cmap='viridis', s=100, edgecolors='black')
plt.title(f"Bimodal P9 Probability Map\n(Avg Score: {avg_prob*100:.1f}%)")
plt.colorbar(sc, label="Probability (Yellow=Fits Bimodal P9)")

# Mark the expected zones
ax.fill_between(np.linspace(np.radians(30), np.radians(90), 50), 0, df_real['a'].max(), color='red', alpha=0.1, label='Anti-Aligned')
ax.fill_between(np.linspace(np.radians(210), np.radians(270), 50), 0, df_real['a'].max(), color='blue', alpha=0.1, label='Aligned')
plt.legend(loc='lower left', bbox_to_anchor=(-0.1, -0.1))
plt.show()