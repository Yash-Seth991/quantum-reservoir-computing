# Quantum Reservoir Computing: Does Noise Help or Hurt?

An empirical study of quantum reservoir computing (QRC) on chaotic time-series prediction, examining how depolarizing noise affects performance — and whether real quantum hardware noise can *enhance* reservoir expressibility beyond what a perfect simulator achieves.

Includes a standalone **noise characterization tool** that estimates any IBM quantum backend's effective depolarizing noise level using a QRC-based noise curve.

**TKS Create Project | April 2026**

---

## Key Findings

1. **Simulated noise monotonically degrades performance** — no beneficial noise sweet spot was found under uniform depolarizing noise
2. **Real quantum hardware outperformed the classical baseline by 3x** on the Lorenz system (MSE: 0.000288 vs 0.000881), suggesting real hardware noise has a structured, correlated character that depolarizing noise does not capture
3. **Circuit choice matters at low noise, not at high noise** — variance across random seeds collapses from std 0.005849 at 0% noise to 0.001912 at 20%, meaning noise overwhelms any circuit-specific advantage
4. **Quantum advantage is task-dependent** — quantum hardware wins on deterministic chaotic data (Lorenz), classical ESN wins on quasi-periodic real-world data (sunspots)

---

## Results Summary

### Lorenz System (Chaotic Time-Series)

| Model | Test MSE |
|---|---|
| Classical ESN | 0.000881 |
| QR — 0% noise (sim, mean ± std) | 0.010019 ± 0.005849 |
| QR — 1% noise (sim, mean ± std) | 0.013063 ± 0.008389 |
| QR — 5% noise (sim, mean ± std) | 0.024008 ± 0.015876 |
| QR — 10% noise (sim, mean ± std) | 0.033453 ± 0.013191 |
| QR — 20% noise (sim, mean ± std) | 0.046369 ± 0.001912 |
| **Real Hardware — ibm_fez** | **0.000288 ✅** |

### Sunspot Prediction (Real Astronomical Data, 2001–2026)

| Model | Test MSE |
|---|---|
| Classical ESN | 0.013408 ✅ |
| Real Hardware — ibm_fez | 0.016486 |

---

## Repository Structure

```
quantum-reservoir-computing/
│
├── 📊 Experiments
│   ├── week1_qrc.py           # Lorenz data generation + Classical ESN baseline
│   ├── week2_qrc.py           # Quantum reservoir at 0% noise (simulator)
│   ├── week3_qrc.py           # Noise sensitivity curve (0–20%)
│   ├── week4a_seeds.py        # Multi-seed experiment with error bars
│   ├── week4b_hardware.py     # Real IBM hardware validation — Lorenz
│   └── week4c_sunspots.py     # Sunspot prediction: ESN vs Real Hardware
│
├── 🛠️ Tool
│   └── qrc_noise_characterizer.py   # Standalone noise characterization tool
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Noise Characterization Tool

`qrc_noise_characterizer.py` is a standalone tool that estimates the effective depolarizing noise level of any IBM quantum backend using a QRC-based noise curve.

### How it works

1. Generates a Lorenz system benchmark dataset
2. Builds a fixed random quantum reservoir circuit
3. Runs the reservoir at multiple simulated noise levels → builds a noise curve
4. Runs the same circuit on your real IBM backend
5. Uses linear interpolation to estimate where the hardware result falls on the curve
6. Outputs: **"Your backend is operating at ~X% effective depolarizing noise"**

### Why this is useful

Full quantum process tomography is expensive and complex. This tool gives researchers a fast, practical estimate of a device's effective noise level using only a simple reservoir computing pipeline — no tomography required.

### Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Edit credentials and settings at the top of the file
# Then run:
python qrc_noise_characterizer.py
```

### Settings

At the top of `qrc_noise_characterizer.py`:

```python
IBM_API_KEY    = "your_api_key"
IBM_INSTANCE   = "your_crn"

N_QUBITS       = 6        # reservoir size
N_LAYERS       = 3        # circuit depth
CIRCUIT_SEED   = 123      # random seed
SHOTS          = 1024     # measurement shots
NOISE_LEVELS   = [0.0, 0.01, 0.05, 0.10, 0.20]
N_LORENZ_STEPS = 300      # keep short to conserve IBM compute time
```

### Example output

```
============================================================
  RESULTS
============================================================

  Backend: ibm_fez

  Simulated noise curve:
     0%  noise  →  MSE: 0.003252
     1%  noise  →  MSE: 0.004554
     5%  noise  →  MSE: 0.009863
    10%  noise  →  MSE: 0.015708
    20%  noise  →  MSE: 0.047670

  Real hardware  →  MSE: 0.000288

  ┌─────────────────────────────────────────────────┐
  │  Estimated effective noise: < 0%                │
  │  (hardware noise is lower than simulated min)   │
  └─────────────────────────────────────────────────┘
```

---

## Running the Experiments

### Setup

```bash
# Create a clean virtual environment
python -m venv qrc_env

# Activate (Windows)
qrc_env\Scripts\activate

# Activate (Mac/Linux)
source qrc_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run in order

```bash
python week1_qrc.py        # ~1 min  — generates baseline
python week2_qrc.py        # ~15 min — quantum reservoir, 0% noise
python week3_qrc.py        # ~45 min — full noise sweep
python week4a_seeds.py     # ~90 min — multi-seed with error bars
python week4b_hardware.py  # IBM time — real hardware, Lorenz
python week4c_sunspots.py  # IBM time — real hardware, sunspots
```

### IBM Quantum Setup

1. Create a free account at [quantum.ibm.com](https://quantum.ibm.com)
2. Generate an API key
3. Copy your instance CRN
4. Paste both into the credentials section at the top of `week4b_hardware.py`, `week4c_sunspots.py`, and `qrc_noise_characterizer.py`

---

## Quantum Reservoir Architecture

| Parameter | Value |
|---|---|
| Qubits | 6 |
| Layers | 3 |
| Gates per layer | RX + RY + RZ (random angles) + CNOT chain |
| Input encoding | RY rotation: angle = 2π × u |
| Readout | Qubit \|1⟩ probabilities → Ridge regression |
| Shots | 1024 |
| Washout | 100 steps (simulator) / 50 steps (hardware) |

---

## Datasets

- **Lorenz system** — generated via `scipy.integrate.odeint`, σ=10, ρ=28, β=8/3, 5000 steps at dt=0.02, X-dimension only
- **Sunspot data** — monthly international sunspot numbers from [SILSO, Royal Observatory of Belgium](https://www.sidc.be/silso/datafiles), last 300 months (2001–2026)

---

## Hardware

Real quantum experiments ran on **ibm_fez** (156-qubit Heron r2, us-east) via IBM Quantum open plan.

---

## Dependencies

```
numpy
scipy
matplotlib
scikit-learn
pandas
qiskit
qiskit-aer
qiskit-ibm-runtime
```

Install with: `pip install -r requirements.txt`
