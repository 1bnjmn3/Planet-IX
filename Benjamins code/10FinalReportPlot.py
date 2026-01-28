import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

# --- CONFIGURATION ---
INPUT_FILE = 'processed_etnos.csv'
script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, INPUT_FILE)

# 1. Load & Filter
df = pd.read_csv(input_path)
df_real = df[ (df['a'] > 230) & (df['q'] > 30) ].copy()

# 2. Retrain the Judge (Quickly replication Step 09 logic)
# We need the probabilities for the plot
def generate_p9_world(n=5000):
    varpi = np.random.vonmises(np.radians(60), 2.0, n)
    return pd.DataFrame({
        'a': np.random.choice(df_real['a'], n), 'e': np.random.choice(df_real['e'], n),
        'i_rad': np.random.choice(df_real['i_rad'], n),
        'varpi_sin': np.sin(varpi), 'varpi_cos': np.cos(varpi), 'label': 1
    })

def generate_rnd_world(n=5000):
    varpi = np.random.uniform(0, 2*np.pi, n)
    return pd.DataFrame({
        'a': np.random.choice(df_real['a'], n), 'e': np.random.choice(df_real['e'], n),
        'i_rad': np.random.choice(df_real['i_rad'], n),
        'varpi_sin': np.sin(varpi), 'varpi_cos': np.cos(varpi), 'label': 0
    })

df_train = pd.concat([generate_p9_world(), generate_rnd_world()]).sample(frac=1)
clf = RandomForestClassifier(n_estimators=100, max_depth=10)
clf.fit(df_train[['a', 'e', 'i_rad', 'varpi_sin', 'varpi_cos']], df_train['label'])

# 3. Get Probabilities for Real Data
probs = clf.predict_proba(df_real[['a', 'e', 'i_rad', 'varpi_sin', 'varpi_cos']])[:, 1]
df_real['P9_Prob'] = probs

# --- THE MONEY SHOT ---
plt.figure(figsize=(10, 10))
ax = plt.subplot(111, polar=True)

# Plot objects, color-coded by AI Confidence
# RED = High P9 Probability (Camp A)
# BLUE = Low P9 Probability (Camp B / Random)
sc = ax.scatter(np.radians(df_real['varpi']), df_real['a'], 
                c=df_real['P9_Prob'], cmap='coolwarm', 
                s=100, edgecolors='black', alpha=0.9, vmin=0, vmax=1)

# Annotations
ax.set_ylim(0, df_real['a'].max() + 100)
ax.set_yticklabels([]) # Hide radial labels for cleanliness
ax.set_theta_zero_location("E")

# The "Camp A" Zone (Planet 9 Prediction)
ax.fill_between(np.linspace(np.radians(20), np.radians(100), 100), 
                0, df_real['a'].max()+100, color='red', alpha=0.1, label='Predicted P9 Cluster')

# Add Colorbar
cbar = plt.colorbar(sc, pad=0.1)
cbar.set_label('AI Probability: Is this object influenced by Planet 9?', rotation=270, labelpad=20)

plt.title(f"The Final Verdict: Why the AI Rejected the Theory\n(Avg P9 Probability: {probs.mean()*100:.1f}%)", fontsize=14)
plt.legend(loc='lower left', bbox_to_anchor=(-0.1, -0.1))

plt.show()