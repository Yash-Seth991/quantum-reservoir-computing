"""
Week 4c - Sunspot Prediction
==============================
Classical ESN vs Real Quantum Hardware on real astronomical data.
Uses seed 123 (our best performing circuit) on IBM hardware.
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import urllib.request
import ssl
from scipy.integrate import odeint
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.compiler import transpile

# ── 1. YOUR IBM CREDENTIALS ───────────────────────────────────────────────────

IBM_API_KEY  = "PASTE_YOUR_API_KEY_HERE"
IBM_INSTANCE = "PASTE_YOUR_CRN_HERE"

# ── 2. LOAD SUNSPOT DATA ──────────────────────────────────────────────────────

print("Loading sunspot data...")
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
data = urllib.request.urlopen(
    'https://www.sidc.be/silso/DATA/SN_m_tot_V2.0.txt', context=ctx
).read().decode()
lines = [l.split() for l in data.strip().split('\n')]
df = pd.DataFrame([[float(l[2]), float(l[3])] for l in lines],
                  columns=['year', 'sunspots'])

# Use last 300 months to keep IBM compute time low
# (same length as our Lorenz hardware run)
sunspots = df['sunspots'].values[-300:]
years    = df['year'].values[-300:]

print(f"Using {len(sunspots)} months of sunspot data")
print(f"Date range: {years[0]:.2f} to {years[-1]:.2f}")

# Normalize
scaler = MinMaxScaler()
s_scaled = scaler.fit_transform(sunspots.reshape(-1, 1)).flatten()

# Plot raw data
plt.figure(figsize=(14, 3))
plt.plot(years, sunspots, color='steelblue', lw=0.8)
plt.title("Monthly Sunspot Numbers (last 300 months)", fontsize=12)
plt.xlabel("Year")
plt.ylabel("Sunspot Count")
plt.tight_layout()
plt.savefig("sunspot_raw.png", dpi=150)
plt.show()
print("Saved: sunspot_raw.png\n")

# Train/test split
input_seq  = s_scaled[:-1]
target_seq = s_scaled[1:]
split      = 200
u_train, u_test = input_seq[:split], input_seq[split:]
y_train, y_test = target_seq[:split], target_seq[split:]

# ── 3. CLASSICAL ESN ON SUNSPOTS ──────────────────────────────────────────────

print("Training Classical ESN on sunspots...")

class EchoStateNetwork:
    def __init__(self, n_reservoir=200, spectral_radius=0.95,
                 input_scaling=0.5, sparsity=0.1, seed=42):
        self.n_reservoir = n_reservoir
        rng = np.random.RandomState(seed)
        self.W_in = rng.uniform(-1, 1, (n_reservoir, 1)) * input_scaling
        W = rng.uniform(-1, 1, (n_reservoir, n_reservoir))
        mask = rng.rand(n_reservoir, n_reservoir) > sparsity
        W[mask] = 0
        eigenvalues = np.linalg.eigvals(W)
        sr = np.max(np.abs(eigenvalues))
        self.W_res = W * (spectral_radius / sr)
        self.W_out = None

    def _run_reservoir(self, u_sequence):
        n_steps = len(u_sequence)
        states = np.zeros((n_steps, self.n_reservoir))
        state = np.zeros(self.n_reservoir)
        for t_idx, u in enumerate(u_sequence):
            pre = self.W_res @ state + self.W_in @ np.array([u])
            state = np.tanh(pre)
            states[t_idx] = state
        return states

    def fit(self, u_train, y_train, washout=50, alpha=1e-4):
        states = self._run_reservoir(u_train)
        states = states[washout:]
        y_train = y_train[washout:]
        ridge = Ridge(alpha=alpha, fit_intercept=True)
        ridge.fit(states, y_train)
        self.W_out = ridge
        return self

    def predict(self, u_test):
        states = self._run_reservoir(u_test)
        return self.W_out.predict(states)

WASHOUT = 50
esn = EchoStateNetwork(n_reservoir=200, spectral_radius=0.95, seed=42)
esn.fit(u_train, y_train, washout=WASHOUT)
esn_pred = esn.predict(u_test)
esn_mse  = mean_squared_error(y_test, esn_pred)
print(f"Classical ESN Sunspot MSE: {esn_mse:.6f}")

# ── 4. BUILD QUANTUM RESERVOIR (seed 123) ─────────────────────────────────────

N_QUBITS = 6
N_LAYERS = 3
SHOTS    = 1024

def build_reservoir_circuit(n_qubits, n_layers, seed):
    rng = np.random.RandomState(seed)
    qc = QuantumCircuit(n_qubits)
    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.rx(rng.uniform(0, 2 * np.pi), q)
            qc.ry(rng.uniform(0, 2 * np.pi), q)
            qc.rz(rng.uniform(0, 2 * np.pi), q)
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
    return qc

reservoir_circuit = build_reservoir_circuit(N_QUBITS, N_LAYERS, seed=123)
print(f"\nQuantum reservoir built (seed 123): {N_QUBITS} qubits, {N_LAYERS} layers")

# ── 5. BUILD ALL CIRCUITS ─────────────────────────────────────────────────────

def build_all_circuits(input_sequence, n_qubits, reservoir_qc):
    circuits = []
    for u in input_sequence:
        qc = QuantumCircuit(n_qubits)
        for q in range(n_qubits):
            qc.ry(2 * np.pi * u, q)
        qc.compose(reservoir_qc, inplace=True)
        qc.measure_all()
        circuits.append(qc)
    return circuits

print("Building circuits...")
train_circuits = build_all_circuits(u_train, N_QUBITS, reservoir_circuit)
test_circuits  = build_all_circuits(u_test,  N_QUBITS, reservoir_circuit)
all_circuits   = train_circuits + test_circuits
print(f"Total circuits: {len(all_circuits)}")

# ── 6. RUN ON REAL IBM HARDWARE ───────────────────────────────────────────────

print("\nConnecting to IBM Quantum...")
service = QiskitRuntimeService(
    channel="ibm_cloud",
    token=IBM_API_KEY,
    instance=IBM_INSTANCE
)
backend = service.least_busy(operational=True, simulator=False)
print(f"Selected backend: {backend.name}")
print(f"Queue: {backend.status().pending_jobs} jobs ahead")

print("Transpiling circuits...")
transpiled = transpile(all_circuits, backend=backend, optimization_level=1)

print("Submitting job to IBM Quantum...")
sampler = Sampler(backend)
job = sampler.run(transpiled, shots=SHOTS)
print(f"Job ID: {job.job_id()}")
print("Waiting for results... (don't close the terminal)")
result = job.result()
print("Job complete!")

# ── 7. EXTRACT RESULTS ────────────────────────────────────────────────────────

def extract_probs(result, n_circuits, n_qubits, shots):
    all_probs = []
    for i in range(n_circuits):
        pub_result = result[i]
        counts = pub_result.data.meas.get_counts()
        probs = np.zeros(n_qubits)
        for bitstring, count in counts.items():
            bits = bitstring.replace(" ", "")[::-1]
            for q, bit in enumerate(bits[:n_qubits]):
                if bit == "1":
                    probs[q] += count
        probs /= shots
        all_probs.append(probs)
    return np.array(all_probs)

print("Extracting results...")
all_states   = extract_probs(result, len(all_circuits), N_QUBITS, SHOTS)
train_states = all_states[:len(train_circuits)]
test_states  = all_states[len(train_circuits):]

ridge = Ridge(alpha=1e-4, fit_intercept=True)
ridge.fit(train_states[WASHOUT:], y_train[WASHOUT:])
qr_pred = ridge.predict(test_states)
qr_mse  = mean_squared_error(y_test, qr_pred)

# ── 8. RESULTS ────────────────────────────────────────────────────────────────

print(f"\n{'='*50}")
print(f"SUNSPOT PREDICTION RESULTS")
print(f"{'='*50}")
print(f"Classical ESN MSE:         {esn_mse:.6f}")
print(f"Quantum Hardware MSE:      {qr_mse:.6f}")
winner = "Quantum Hardware" if qr_mse < esn_mse else "Classical ESN"
print(f"Winner:                    {winner}")
print(f"{'='*50}")

# ── 9. PLOTS ──────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
show = len(y_test)
test_years = years[split+1:split+1+show]

# ESN predictions
axes[0].plot(test_years, y_test[:show],    label="True",          color="steelblue", lw=1.5)
axes[0].plot(test_years, esn_pred[:show],  label="ESN Predicted", color="tomato",    lw=1.2, linestyle="--")
axes[0].set_title(f"Classical ESN — Sunspot Prediction | MSE: {esn_mse:.6f}", fontsize=12)
axes[0].legend(); axes[0].set_xlabel("Year"); axes[0].set_ylabel("Normalized Sunspots")

# Quantum predictions
axes[1].plot(test_years, y_test[:show],   label="True",                    color="steelblue",  lw=1.5)
axes[1].plot(test_years, qr_pred[:show],  label="Quantum Hardware Predicted", color="darkorange", lw=1.2, linestyle="--")
axes[1].set_title(f"Quantum Hardware ({backend.name}) — Sunspot Prediction | MSE: {qr_mse:.6f}", fontsize=12)
axes[1].legend(); axes[1].set_xlabel("Year"); axes[1].set_ylabel("Normalized Sunspots")

plt.tight_layout()
plt.savefig("sunspot_predictions.png", dpi=150)
plt.show()
print("Saved: sunspot_predictions.png")

# Bar chart comparison
fig2, ax = plt.subplots(figsize=(7, 5))
models = ["Classical ESN", f"Quantum Hardware\n({backend.name})"]
mses   = [esn_mse, qr_mse]
colors = ["steelblue", "darkorange"]
bars   = ax.bar(models, mses, color=colors, width=0.4)
ax.set_ylabel("Test MSE (lower is better)")
ax.set_title("Sunspot Prediction: ESN vs Quantum Hardware")
for bar, val in zip(bars, mses):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.0001,
            f"{val:.6f}", ha="center", fontsize=11)
plt.tight_layout()
plt.savefig("sunspot_comparison.png", dpi=150)
plt.show()
print("Saved: sunspot_comparison.png")

print("\n✅ Sunspot experiment complete.")
print("   You now have results on both synthetic (Lorenz) and real (sunspot) data.")