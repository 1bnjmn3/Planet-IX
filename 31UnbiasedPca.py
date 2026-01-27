import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import seaborn as sns
import os

# --- CONFIGURATION ---
INPUT_FILE = "smart_dataset.csv"
if not os.path.exists(INPUT_FILE):
    print("Error: Run Script 27 first.")
    exit()

df = pd.read_csv(INPUT_FILE)
print(f"--- UNBIASED DECOMPOSITION (N={len(df)}) ---")

# 1. PREPARE FEATURE SPACE (Full Orbital Elements)
# Critique requested: a, e, i, Omega (Node), omega (Peri) [cite: 121]
# We convert angles to sin/cos components to avoid the 0/360 discontinuity
features = pd.DataFrame({
    'a_norm': df['a'], # We will scale this
    'e': df['e'],
    'i': df['i'],
    'node_sin': np.sin(np.radians(df['Node'])),
    'node_cos': np.cos(np.radians(df['Node'])),
    'peri_sin': np.sin(np.radians(df['w'])),
    'peri_cos': np.cos(np.radians(df['w']))
})

# 2. STANDARDIZATION (Critical for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# 3. PRINCIPAL COMPONENT ANALYSIS (PCA)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Analyze Variance
explained_variance = pca.explained_variance_ratio_
print(f"\nVariance Explained by PC1: {explained_variance[0]*100:.1f}%")
print(f"Variance Explained by PC2: {explained_variance[1]*100:.1f}%")

# Analyze Loadings (What is PC1 made of?)
loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=features.columns)
print("\n--- PC1 DRIVERS (Correlations) ---")
print(loadings['PC1'].sort_values(ascending=False))
# If 'i' or 'node' are high, the Warp is the dominant feature of the solar system.

# 4. HIERARCHICAL CLUSTERING (Agglomerative) 
# We test 2 to 5 clusters to find the best Silhouette Score
best_score = -1
best_k = 0
best_labels = None

print("\n--- HIERARCHICAL CLUSTERING ---")
for k in range(2, 6):
    clusterer = AgglomerativeClustering(n_clusters=k)
    labels = clusterer.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"Clusters: {k} | Silhouette Score: {score:.3f}")
    
    if score > best_score:
        best_score = score
        best_k = k
        best_labels = labels

print(f"\nWinner: {best_k} Clusters with Score {best_score:.3f}")

# 5. VALIDATE THE CRITIQUE'S THRESHOLD
if best_score > 0.5:
    print(">>> VALIDITY CHECK PASSED (Score > 0.5) <<<")
    print("The structure is mathematically real and distinct.")
else:
    print(">>> VALIDITY CHECK FAILED (Score < 0.5) <<<")
    print("The critique is right: The data is likely random noise.")

# 6. VISUALIZATION
df['cluster'] = best_labels
plt.figure(figsize=(10, 7))
sns.scatterplot(x=df_pca['PC1'], y=df_pca['PC2'], hue=df['cluster'], palette='viridis', s=100, edgecolor='black')

# Annotate with feature vectors (Biplot style)
# This shows us physically what PC1 and PC2 mean
for i, feature in enumerate(features.columns):
    # Scale vectors for visibility
    plt.arrow(0, 0, pca.components_[0, i]*3, pca.components_[1, i]*3, color='r', alpha=0.5)
    plt.text(pca.components_[0, i]*3.2, pca.components_[1, i]*3.2, feature, color='r')

plt.xlabel(f"PC1 ({explained_variance[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({explained_variance[1]*100:.1f}%)")
plt.title(f"PCA Decomposition: The Underlying Geometry of the Outer Solar System\n(N={len(df)}, Silhouette={best_score:.3f})")
plt.grid(True, alpha=0.3)
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.show()

# 7. CHARACTERIZE THE GROUPS
print("\n--- GROUP PROFILES ---")
for c in range(best_k):
    subset = df[df['cluster'] == c]
    print(f"Group {c}: N={len(subset)} | Mean Inc={subset['i'].mean():.1f} | Mean Node={subset['Node'].mean():.1f}")