import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import os

INPUT_FILE = "mega_dataset.csv"

if not os.path.exists(INPUT_FILE):
    print("Run Script 13 first!")
    exit()

df_real = pd.read_csv(INPUT_FILE)
print(f"Processing Mega Dataset: {len(df_real)} Objects")

# --- SIMULATION (Bimodal P9 vs Random) ---
# We keep the same sophisticated Bimodal model
def generate_bimodal_p9(n=10000):
    # Sample from real 'a' and 'e' to match the new distribution
    a = np.random.choice(df_real['a'], n)
    e = np.random.choice(df_real['e'], n)
    i = np.random.choice(df_real['i'], n)
    
    # 60% Anti-Aligned (60 deg), 30% Aligned (240 deg), 10% Scatter
    n1 = int(n*0.6); n2 = int(n*0.3); n3 = n - n1 - n2
    v1 = np.random.vonmises(np.radians(60), 2.5, n1)
    v2 = np.random.vonmises(np.radians(240), 2.0, n2)
    v3 = np.random.uniform(0, 2*np.pi, n3)
    varpi = np.concatenate([v1, v2, v3])
    np.random.shuffle(varpi)
    
    return pd.DataFrame({'a':a, 'e':e, 'i_rad':np.radians(i), 
                         'varpi_sin':np.sin(varpi), 'varpi_cos':np.cos(varpi), 'label':1})

def generate_random(n=10000):
    a = np.random.choice(df_real['a'], n)
    e = np.random.choice(df_real['e'], n)
    i = np.random.choice(df_real['i'], n)
    varpi = np.random.uniform(0, 2*np.pi, n)
    return pd.DataFrame({'a':a, 'e':e, 'i_rad':np.radians(i), 
                         'varpi_sin':np.sin(varpi), 'varpi_cos':np.cos(varpi), 'label':0})

# Train
print("Training on 40,000 synthetic solar systems...")
df_train = pd.concat([generate_bimodal_p9(20000), generate_random(20000)]).sample(frac=1)
clf = RandomForestClassifier(n_estimators=200, max_depth=15) # Deeper trees for more complex data
clf.fit(df_train[['a', 'e', 'i_rad', 'varpi_sin', 'varpi_cos']], df_train['label'])

# Test
X_real = pd.DataFrame({
    'a': df_real['a'], 'e': df_real['e'], 'i_rad': np.radians(df_real['i']),
    'varpi_sin': np.sin(np.radians(df_real['varpi'])),
    'varpi_cos': np.cos(np.radians(df_real['varpi']))
})

probs = clf.predict_proba(X_real)[:, 1]
avg_prob = np.mean(probs)

print(f"\n--- MEGA DATASET VERDICT ---")
print(f"Data Count: {len(df_real)}")
print(f"P9 Probability: {avg_prob*100:.2f}%")

# Visualization
plt.figure(figsize=(10, 10))
ax = plt.subplot(111, polar=True)
# Plot the new data

sc = ax.scatter(np.radians(df_real['varpi']), df_real['a'], c=probs, cmap='inferno', s=80, alpha=0.8)
plt.colorbar(sc, label="P9 Confidence")
plt.title(f"Mega Dataset Analysis (N={len(df_real)})\nP9 Probability: {avg_prob*100:.1f}%")
plt.show()