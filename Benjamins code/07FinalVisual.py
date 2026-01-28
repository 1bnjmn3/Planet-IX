import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

INPUT_FILE = 'processed_etnos.csv'
script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, INPUT_FILE)
df = pd.read_csv(input_path)

# Filter High Quality
df_hq = df[ (df['a'] > 230) & (df['q'] > 30) ].copy()

# Recover the anomalies from your logs (Hardcoded for visualization based on your data)
# We highlight the two camps
camp_a = [25, 38, 59, 61, 110, 122] # The "Planet 9" side
camp_b = [196, 218, 270, 305, 306, 317] # The "Opposite" side

plt.figure(figsize=(10, 10))
ax = plt.subplot(111, polar=True)

# 1. Plot All High Quality Objects (Grey background)
angles_all = np.radians(df_hq['varpi'])
ax.scatter(angles_all, [1]*len(angles_all), c='gray', alpha=0.3, s=50, label='Background Noise')

# 2. Plot Camp A (Red - Matches P9)
ax.scatter(np.radians(camp_a), [1.1]*len(camp_a), c='red', s=150, edgecolors='black', label='Camp A: Aligned w/ Prediction')

# 3. Plot Camp B (Blue - Contradicts P9)
ax.scatter(np.radians(camp_b), [1.1]*len(camp_b), c='blue', s=150, edgecolors='black', label='Camp B: Anti-Aligned')

# 4. Annotations
ax.axvline(np.radians(61), color='red', linestyle='--', alpha=0.5, label='Planet 9 Prediction')
ax.set_title("The 'Bimodal' Problem:\nML Finds Stable Objects on BOTH Sides", va='bottom', fontsize=14)
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15))

plt.show()