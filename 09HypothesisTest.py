import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os

# --- CONFIGURATION ---
INPUT_FILE = 'processed_etnos.csv'
script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, INPUT_FILE)

if not os.path.exists(input_path):
    print("Error: Data file not found.")
    exit()

df = pd.read_csv(input_path)
# Strict Filter again (High Quality Objects only)
df_real = df[ (df['a'] > 230) & (df['q'] > 30) ].copy()
print(f"Judgement Day for {len(df_real)} Real Objects.")

# --- STEP 1: CREATE THE WORLDS ---

def generate_p9_world(n_samples=5000):
    """Generates data assuming Planet 9 IS REAL (Caltech Model)"""
    # Orbits are roughly the same shape (a, e, i) as real data
    a = np.random.choice(df_real['a'], n_samples)
    e = np.random.choice(df_real['e'], n_samples)
    i = np.random.choice(df_real['i_rad'], n_samples)
    
    # THE KEY: Clustering!
    # P9 confines angles to ~60 degrees (Anti-aligned)
    # We use a Von Mises distribution (Gaussian on a circle) centered at 60 deg
    cluster_center = np.radians(60) 
    kappa = 2.0 # Concentration (Higher = tighter cluster)
    varpi = np.random.vonmises(cluster_center, kappa, n_samples)
    
    return pd.DataFrame({
        'a': a, 'e': e, 'i_rad': i,
        'varpi_sin': np.sin(varpi), 'varpi_cos': np.cos(varpi),
        'label': 1 # Label 1 = PLANET 9 WORLD
    })

def generate_random_world(n_samples=5000):
    """Generates data assuming Planet 9 is FAKE (Null Hypothesis)"""
    a = np.random.choice(df_real['a'], n_samples)
    e = np.random.choice(df_real['e'], n_samples)
    i = np.random.choice(df_real['i_rad'], n_samples)
    
    # Random angles (Uniform)
    varpi = np.random.uniform(0, 2*np.pi, n_samples)
    
    return pd.DataFrame({
        'a': a, 'e': e, 'i_rad': i,
        'varpi_sin': np.sin(varpi), 'varpi_cos': np.cos(varpi),
        'label': 0 # Label 0 = RANDOM NOISE
    })

# Generate Training Data
print("Simulating Planet 9 Physics...")
df_p9 = generate_p9_world(n_samples=10000)
df_rnd = generate_random_world(n_samples=10000)

# Combine and Shuffle
df_train = pd.concat([df_p9, df_rnd]).sample(frac=1).reset_index(drop=True)

X = df_train[['a', 'e', 'i_rad', 'varpi_sin', 'varpi_cos']]
y = df_train['label']

# --- STEP 2: TRAIN THE JUDGE ---
print("Training Classifier...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = RandomForestClassifier(n_estimators=100, max_depth=10)
clf.fit(X_train, y_train)

# Sanity Check: Can the AI actually tell the difference?
acc = clf.score(X_test, y_test)
print(f"Classifier Accuracy on Simulation: {acc*100:.1f}%")
if acc < 0.6:
    print("WARNING: The AI can't tell the difference. P9 signal might be too weak.")

# --- STEP 3: THE VERDICT ---
# Feed the Real Data into the model
X_real = df_real[['a', 'e', 'i_rad', 'varpi_sin', 'varpi_cos']]
probs = clf.predict_proba(X_real) # Returns [Prob_Random, Prob_P9]

# The "Planet 9 Probability" for each object
p9_likelihoods = probs[:, 1]
avg_p9_score = np.mean(p9_likelihoods)

df_real['P9_Probability'] = p9_likelihoods

print("\n--- FINAL VERDICT ---")
print(f"Average P9 Likelihood of your Data: {avg_p9_score*100:.1f}%")

# Count how many objects are "Suspiciously P9-like" (> 80% confidence)
strong_candidates = df_real[df_real['P9_Probability'] > 0.8]
print(f"Number of 'Strong P9 Candidates' found: {len(strong_candidates)}")

if avg_p9_score > 0.6:
    print("CONCLUSION: SUPPORT.")
    print("The data looks more like Planet 9 than Random Noise.")
elif avg_p9_score < 0.4:
    print("CONCLUSION: REJECT.")
    print("The data looks more like Random Noise.")
else:
    print("CONCLUSION: INCONCLUSIVE.")
    print("The data is messy (Camp A vs Camp B is confusing the AI).")

# --- VISUALIZATION ---
plt.figure(figsize=(10, 6))
plt.hist(p9_likelihoods, bins=10, range=(0,1), color='purple', alpha=0.7, edgecolor='black')
plt.axvline(0.5, color='red', linestyle='--', label='Indecisive')
plt.xlabel("Probability of being 'Planet 9 Aligned' (0=Random, 1=P9)")
plt.ylabel("Count of Objects")
plt.title(f"AI Classification of Real Data (Avg: {avg_p9_score*100:.1f}%)")
plt.legend()
plt.show()

# Print the "Most Likely" candidates for the report
print("\nTop 5 Objects that fit the Planet 9 Model best:")
print(df_real.sort_values('P9_Probability', ascending=False)[['a', 'varpi', 'P9_Probability']].head(5))