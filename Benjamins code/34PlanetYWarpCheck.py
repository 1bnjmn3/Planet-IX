import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# Team Giants (Mass in Earth Masses, Distance in AU)
giants = [
    {'name': 'Jupiter', 'm': 317.8, 'a': 5.2},
    {'name': 'Saturn',  'm': 95.2,  'a': 9.5},
    {'name': 'Uranus',  'm': 14.5,  'a': 19.2},
    {'name': 'Neptune', 'm': 17.1,  'a': 30.1}
]

# Team Planet Y Candidates (Literature values)
candidates = [
    {'name': 'Planet Y (Lower)', 'm': 1.0, 'a': 60, 'inc': 65},
    {'name': 'Planet Y (Upper)', 'm': 1.0, 'a': 80, 'inc': 65},
    {'name': 'Super-Earth Y',    'm': 2.5, 'a': 100, 'inc': 65} # Some papers suggest up to 3Me
]

print("--- PLANET Y PHYSICS CHECK (INNER PERTURBER MODEL) ---")

# 1. CALCULATE GIANT PLANET STRENGTH (J2 Moment)
# Strength proportional to Mass * a^2
strength_giants = 0
for p in giants:
    s = p['m'] * (p['a']**2)
    strength_giants += s
    # print(f"{p['name']}: Strength = {s:.0f}")

print(f"Total Strength of Known Solar System (J2): {strength_giants:.0f}")

# 2. TEST CANDIDATES
print("\n--- PREDICTED WARP ANGLES ---")
predicted_warps = []
names = []

for p in candidates:
    strength_y = p['m'] * (p['a']**2)
    
    # The Weighted Average Formula
    # Angle = (Strength_Y * Inc_Y + Strength_Giants * 0) / (Strength_Y + Strength_Giants)
    warp_angle = (strength_y * p['inc']) / (strength_y + strength_giants)
    
    print(f"Candidate: {p['name']} ({p['m']} Me @ {p['a']} AU)")
    print(f"  > Strength: {strength_y:.0f}")
    print(f"  > Predicted Warp: {warp_angle:.2f} degrees")
    
    predicted_warps.append(warp_angle)
    names.append(p['name'])

# 3. VISUALIZE THE MATCH
plt.figure(figsize=(10, 6))

# Plot Candidates
plt.bar(names, predicted_warps, color=['green', 'blue', 'purple'], alpha=0.7)

# Plot Your Data
plt.axhline(15.7, color='red', linewidth=3, linestyle='--', label='Your Data (15.7°)')
plt.axhspan(14.0, 17.5, color='red', alpha=0.1, label='Error Margin')

plt.ylabel("Predicted Warp Inclination (deg)")
plt.title("Does 'Planet Y' Explain Your Data?")
plt.legend()
plt.ylim(0, 30)
plt.grid(axis='y', alpha=0.3)

plt.show()