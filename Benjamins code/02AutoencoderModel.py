import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
INPUT_FILE = 'processed_etnos.csv'
FEATURE_COLS = ['a', 'e', 'i_rad', 'varpi_sin', 'varpi_cos'] 

# --- PATH SETUP ---
script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, INPUT_FILE)

if not os.path.exists(input_path):
    print(f"CRITICAL ERROR: {INPUT_FILE} missing. Run 01 first.")
    exit()

df_real = pd.read_csv(input_path)
print(f"Loaded {len(df_real)} High-Quality ETNOs.")

# 1. Generate Synthetic Data (The "Null Hypothesis")
def generate_synthetic_data(n_samples=30000, real_df=None):
    syn_a = np.random.choice(real_df['a'], n_samples, replace=True)
    syn_e = np.random.choice(real_df['e'], n_samples, replace=True)
    syn_i = np.random.choice(real_df['i_rad'], n_samples, replace=True)
    
    # Randomize angles (No Planet 9)
    syn_varpi = np.random.uniform(0, 2*np.pi, n_samples)
    
    syn_data = pd.DataFrame({
        'a': syn_a,
        'e': syn_e,
        'i_rad': syn_i,
        'varpi_sin': np.sin(syn_varpi),
        'varpi_cos': np.cos(syn_varpi)
    })
    return syn_data

print("Generating Synthetic Data...")
df_synthetic = generate_synthetic_data(n_samples=30000, real_df=df_real)

# 2. Preprocessing
scaler = MinMaxScaler()
X_train = scaler.fit_transform(df_synthetic[FEATURE_COLS])
X_test = scaler.transform(df_real[FEATURE_COLS])

# 3. Build the Autoencoder
input_dim = len(FEATURE_COLS)
encoding_dim = 2 # SQUEEZE HARDER: Force data into 2D space

input_layer = layers.Input(shape=(input_dim,))
encoder = layers.Dense(12, activation="relu")(input_layer)
bottleneck = layers.Dense(encoding_dim, activation="linear", name="bottleneck")(encoder) 

decoder = layers.Dense(12, activation="relu")(bottleneck)
output_layer = layers.Dense(input_dim, activation="sigmoid")(decoder)

autoencoder = models.Model(inputs=input_layer, outputs=output_layer)
encoder_model = models.Model(inputs=input_layer, outputs=bottleneck) # Separate model to see the "brain"

autoencoder.compile(optimizer='adam', loss='mse')

# 4. Train
print("Training...")
history = autoencoder.fit(
    X_train, X_train,
    epochs=60,
    batch_size=16,
    shuffle=True,
    verbose=0
)

# 5. Analysis
reconstructions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)
df_real['anomaly_score'] = mse

# Extract the "Latent Space" (The mental map of the AI)
latent_real = encoder_model.predict(X_test)
latent_syn = encoder_model.predict(X_train[:2000])

print("\n--- RESULTS ---")
print("Top 3 Candidates for Planet 9 Interaction:")
print(df_real.sort_values('anomaly_score', ascending=False)[['a', 'varpi', 'anomaly_score']].head(3))

# --- PLOTTING ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Histogram
ax1.hist(df_real['anomaly_score'], bins=10, alpha=0.7, color='blue', label='Real ETNOs', density=True)
syn_mse = np.mean(np.power(X_train[:2000] - autoencoder.predict(X_train[:2000]), 2), axis=1)
ax1.hist(syn_mse, bins=30, alpha=0.3, color='red', label='Synthetic Random', density=True)
ax1.set_title("Reconstruction Error")
ax1.legend()

# Plot 2: Latent Space (The "Cluster Map")
# If Planet 9 exists, Real points (Blue) should be CLUMPED, not scattered like Red.
ax2.scatter(latent_syn[:,0], latent_syn[:,1], c='red', alpha=0.05, label='Random Noise')
ax2.scatter(latent_real[:,0], latent_real[:,1], c='blue', alpha=0.8, edgecolors='black', label='Real ETNOs')
ax2.set_title("Latent Space Map (Clustering = Planet 9)")
ax2.legend()

plt.show()