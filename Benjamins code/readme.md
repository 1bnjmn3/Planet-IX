Here is the comprehensive `README.md` for your project. You can save this file in your project root folder. It covers everything from setting up the environment to running the full discovery pipeline.

---

# Project: Dynamical Validation of Trans-Neptunian Perturbers (Planet 9 / Planet Y)

This repository contains the full Machine Learning and N-Body simulation pipeline used to detect and characterize the ** Secular Warp** in the Extreme Trans-Neptunian Object (ETNO) population.

The codebase evolves from initial geometric clustering (Phase I) to advanced secular dynamics and mass inversion (Phase IV), culminating in the detection of a candidate **1.65 Earth-Mass perturber** ("Planet Y").

## 📋 Prerequisites

* **Python 3.8+** (Required)
* **Visual Studio Code** (Recommended IDE)
* **Internet Connection** (Required for fetching live data from NASA JPL/MPC)

---

## 🛠️ Setup Guide (VS Code)

Follow these steps to set up a clean virtual environment and install all necessary physics and ML libraries.

### 1. Open Project in VS Code

Open your terminal in VS Code (`Ctrl + ~` or `Cmd + ~`) and navigate to your project folder.

### 2. Create the Virtual Environment

Run the following command to create a contained Python environment (prevents conflicts with your system Python):

* **Mac/Linux:**
```bash
python3 -m venv .venv

```


* **Windows:**
```bash
python -m venv .venv

```



### 3. Activate the Environment

You must activate the environment every time you open the project.

* **Mac/Linux:**
```bash
source .venv/bin/activate

```


* **Windows:**
```bash
.venv\Scripts\activate

```



*(You will know it worked if you see `(.venv)` appear at the start of your command line).*

### 4. Install Dependencies

Run this command to install all scientific and physics packages required for the scripts:

```bash
pip install numpy pandas matplotlib scipy scikit-learn seaborn requests rebound

```

* **Note on REBOUND:** This is the N-Body integrator. If the installation fails on Windows, you may need a C compiler. Usually, `pip install rebound` works out of the box for most users.

---

## 🚀 How to Run the Pipeline

The scripts are numbered logically. You do **not** need to run all 31 scripts. Below is the "Golden Path" to reproduce the final Planet Y discovery.

### Step 1: Data Harvesting

Download the latest orbital data from NASA JPL.

```bash
python 27_Smart_Harvester.py

```

* *Output:* `smart_dataset.csv` (Contains stable TNOs with  AU,  AU).

### Step 2: Unbiased Statistical Test (The Critique Check)

Verify that the signal is real using Principal Component Analysis (PCA) rather than biased clustering.

```bash
python 31_Unbiased_PCA.py

```

* *Look for:* A Silhouette Score > 0.5 and PC1 being driven by Inclination (`i`).

### Step 3: The "Naked Eye" Verification

Visualize the warp directly without complex algorithms.

```bash
python 29_Naked_Eye_Histogram.py

```

* *Look for:* A distinct probability spike at ****.

### Step 4: Physical Characterization (Mass & Orbit)

Invert the physics to calculate the mass of the planet causing the warp.

```bash
python 25_Warp_Inverter.py

```

* *Output:* Derived Mass () and Inclination.

### Step 5: Generate the Treasure Map

Create the final observation track for telescopes.

```bash
python 30_Final_Sky_Track.py

```

* *Output:* A plot showing exactly where to look in the night sky (RA/Dec).

---

## 📂 File Structure Overview

### Phase I: The Null Result (Clustering)

* `11_Live_MPC.py`: Original data scraper (replaced by Script 27).
* `15_Autoencoder.py`: ML model that found the "false" clustering signal.
* `17_Nuclear_Bias.py`: The map that proved the clustering was just survey bias.

### Phase II: The Secular Pivot (Discovery)

* `18_Warp_Detector.py`: First detection of the  plane.
* `19_NBody_Check.py`: Simulation proving a planet *can* cause this warp.
* `20_Locator_Map.py`: Prototype sky map.

### Phase III: Robustness & Planet Y

* `21_Warp_Stress_Test.py`: GMM analysis that split the signal into Planet 9 vs. Planet Y.
* `23_Secular_Analytic.py`: Mathematical proof of the "Forced Inclination" curve.
* `26_Pole_Mapper.py`: 3D visualization of orbital poles ("The Cone").

### Phase IV: Final Confirmation

* `27_Smart_Harvester.py`: **[CRITICAL]** The current data loader.
* `28_Expanded_Warp_Test.py`: Bootstrap analysis on the expanded dataset.
* `30_Final_Sky_Track.py`: **[FINAL OUTPUT]** The publication-ready map.
* `31_Unbiased_PCA.py`: **[FINAL VALIDATION]** The hypothesis-neutral check.

---

## ❓ Troubleshooting

* **"Module not found: rebound"**: Ensure you activated your venv (`source .venv/bin/activate`) before running python.
* **"Script not finding csv file"**: You must run the Harvester (`27_Smart_Harvester.py`) *before* running any analysis scripts like 28, 29, or 31.
* **VS Code Greyed Out Imports**: Press `Ctrl+Shift+P` -> `Python: Select Interpreter` -> Select the one that says `('.venv': venv)`.