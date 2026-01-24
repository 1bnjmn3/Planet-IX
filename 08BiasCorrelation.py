import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

INPUT_FILE = 'processed_etnos.csv'
script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, INPUT_FILE)

df = pd.read_csv(input_path)
df_hq = df[ (df['a'] > 230) & (df['q'] > 30) ].copy()

# Recover your Top 12 Anomalies (Camp A & B)
# We need their specific angles (varpi) and their "Node" (Longitude of Ascending Node)
# to see where they are in the sky relative to the Galaxy.
# (Note: We are approximating survey bias using Galactic Latitude exclusion zones here)

# Re-run the quick scorer to get the top 12 indices
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
feat_cols = ['a', 'e', 'i_rad', 'varpi_sin', 'varpi_cos']

# Quick synthetic train (just to get scores back)
# Ideally you'd save scores in previous step, but this is fast:
syn_varpi = np.random.uniform(0, 2*np.pi, 5000)
X_syn = scaler.fit_transform(pd.DataFrame({
    'a': np.random.choice(df_hq['a'], 5000),
    'e': np.random.choice(df_hq['e'], 5000),
    'i_rad': np.random.choice(df_hq['i_rad'], 5000),
    'varpi_sin': np.sin(syn_varpi), 'varpi_cos': np.cos(syn_varpi)
}))
X_real = scaler.transform(df_hq[feat_cols])

# Simple Autoencoder (Replicating your architecture)
import tensorflow as tf
from tensorflow.keras import layers, models
inp = layers.Input(shape=(5,))
e = layers.Dense(12, activation='relu')(inp)
b = layers.Dense(2, activation='linear')(e)
d = layers.Dense(12, activation='relu')(b)
out = layers.Dense(5, activation='sigmoid')(d)
m = models.Model(inp, out)
m.compile(loss='mse', optimizer='adam')
m.fit(X_syn, X_syn, epochs=30, verbose=0)

# Get Scores
recon = m.predict(X_real, verbose=0)
df_hq['score'] = np.mean(np.power(X_real - recon, 2), axis=1)
top_12 = df_hq.sort_values('score', ascending=False).head(12)

# --- THE "KILL SHOT" PLOT ---
# We plot the Longitude of Perihelion (varpi) vs Longitude of Ascending Node (Node)
# Survey biases often show up as correlations here.

plt.figure(figsize=(10, 6))

# 1. Plot all background objects
plt.scatter(df_hq['Node'], df_hq['varpi'], c='gray', alpha=0.3, label='Background Objects')

# 2. Plot the "Anomalies"
plt.scatter(top_12['Node'], top_12['varpi'], c='red', s=100, edgecolors='black', label='Top 12 Anomalies')

# 3. Mark the "Galactic Plane" Avoidance Zones (Rough Approximation)
# Surveys usually avoid the Milky Way plane because it's too crowded with stars.
# This creates "gaps" in data.
plt.axvspan(0, 40, color='blue', alpha=0.1, label='Common Survey Zone A')
plt.axvspan(140, 180, color='blue', alpha=0.1, label='Common Survey Zone B')

plt.xlabel("Longitude of Ascending Node (deg)")
plt.ylabel("Longitude of Perihelion (deg)")
plt.title("Do Anomalies correlate with Observation Windows?")
plt.legend()
plt.grid(True)
plt.show()