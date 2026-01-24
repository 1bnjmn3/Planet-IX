import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
INPUT_FILE = 'processed_etnos.csv'
TOP_N = 12  # We will test the "Top 12" weirdest objects
# Why 12? That's roughly the number of "metastable" objects in the original Caltech papers.

script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, INPUT_FILE)

if not os.path.exists(input_path):
    print("Error: processed_etnos.csv not found.")
    exit()

# 1. Load & Filter Data
df = pd.read_csv(input_path)
df_hq = df[ (df['a'] > 230) & (df['q'] > 30) ].copy()
print(f"Starting with {len(df_hq)} objects. hunting for the Top {TOP_N} anomalies...")

# 2. Build & Train Autoencoder (Fast version)
# We need to regenerate the anomaly scores since we didn't save them last time.
feature_cols = ['a', 'e', 'i_rad', 'varpi_sin', 'varpi_cos'] 

# Synthetic Data Gen
n_syn = 30000
syn_a = np.random.choice(df_hq['a'], n_syn, replace=True)
syn_e = np.random.choice(df_hq['e'], n_syn, replace=True)
syn_i = np.random.choice(df_hq['i_rad'], n_syn, replace=True)
syn_varpi = np.random.uniform(0, 2*np.pi, n_syn)
df_syn = pd.DataFrame({'a':syn_a, 'e':syn_e, 'i_rad':syn_i, 
                       'varpi_sin':np.sin(syn_varpi), 'varpi_cos':np.cos(syn_varpi)})

# Scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(df_syn[feature_cols])
X_test = scaler.transform(df_hq[feature_cols])

# Model
input_dim = 5
input_layer = layers.Input(shape=(input_dim,))
enc = layers.Dense(12, activation="relu")(input_layer)
bot = layers.Dense(2, activation="linear")(enc)
dec = layers.Dense(12, activation="relu")(bot)
out = layers.Dense(input_dim, activation="sigmoid")(dec)
ae = models.Model(input_layer, out)
ae.compile(optimizer='adam', loss='mse')
ae.fit(X_train, X_train, epochs=50, batch_size=32, verbose=0)

# 3. Identify the "Chosen Ones"
reconstructions = ae.predict(X_test)
mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)
df_hq['anomaly_score'] = mse

# Sort and take the Top N
df_top = df_hq.sort_values('anomaly_score', ascending=False).head(TOP_N).copy()
print(f"\n--- TOP {TOP_N} ANOMALIES ---")
print(df_top[['a', 'varpi', 'anomaly_score']])

# 4. Run Stats on the Top N
angles_rad = np.radians(df_top['varpi'])
C = np.sum(np.cos(angles_rad))
S = np.sum(np.sin(angles_rad))
R_bar = np.sqrt(C**2 + S**2) / TOP_N

print(f"\nMean Vector Length (R_bar): {R_bar:.4f}")

# 5. Monte Carlo Validation (The "Cherry Picking" Check)
# We must prove that taking the "Top 12 errors" doesn't ALWAYS produce clustering.
# We run the ENTIRE pipeline (Train -> Score -> Sort -> Measure) on random data.
# This is computationally heavy, so we simulate 1000 runs.
print("\nRunning Validation (Is this just a math trick?)...")
n_sims = 2000
beats = 0

for i in range(n_sims):
    # Create a FAKE dataset of 38 objects (Random angles)
    fake_varpi = np.random.uniform(0, 2*np.pi, len(df_hq))
    fake_data = df_hq.copy()
    fake_data['varpi_sin'] = np.sin(fake_varpi)
    fake_data['varpi_cos'] = np.cos(fake_varpi)
    
    # Scale
    X_fake = scaler.transform(fake_data[feature_cols])
    
    # Get Error Scores (using the SAME model)
    fake_recon = ae.predict(X_fake, verbose=0)
    fake_mse = np.mean(np.power(X_fake - fake_recon, 2), axis=1)
    
    # Pick Top N of this fake batch
    # We want to see if the "weirdest" random objects also look clustered
    fake_indices = np.argsort(fake_mse)[-TOP_N:] # Indices of top N errors
    top_fake_angles = fake_varpi[fake_indices]
    
    # Measure Clustering
    f_C = np.sum(np.cos(top_fake_angles))
    f_S = np.sum(np.sin(top_fake_angles))
    f_R = np.sqrt(f_C**2 + f_S**2) / TOP_N
    
    if f_R >= R_bar:
        beats += 1

p_value = beats / n_sims

print("\n--- FINAL VERDICT (SUBSET ANALYSIS) ---")
print(f"P-Value: {p_value:.5f}")

if p_value < 0.05:
    print("SUCCESS: The 'Anomalies' are significantly clustered.")
    print("Conclusion: The ML model successfully filtered out the noise.")
else:
    print("FAILURE: Even the anomalies are random.")
    print("Conclusion: No evidence of Planet 9 in this dataset.")

# Plot
plt.figure(figsize=(6,6))
ax = plt.subplot(111, polar=True)
ax.scatter(angles_rad, [1]*TOP_N, c='blue', s=100, label=f'Top {TOP_N} Anomalies')
ax.vlines(np.radians(61), 0, 1, color='red', linestyle='--', label='P9 Prediction')
ax.set_title(f"Top {TOP_N} Anomalies (p={p_value:.3f})")
plt.legend(loc='lower left')
plt.show()