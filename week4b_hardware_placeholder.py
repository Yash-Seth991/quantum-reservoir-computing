print("starting...")

"""
Week 4b - Real Hardware Validation
====================================
Run the quantum reservoir on a real IBM quantum backend
and compare the result to our simulated noise curve.
Uses seed 123 (our best performing circuit).
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.compiler import transpile

# ── 1. YOUR IBM CREDENTIALS ───────────────────────────────────────────────────
# Paste your API key and CRN here (keep this file private)

IBM_API_KEY = "PASTE API HERE"
IBM_INSTANCE = "PASTE CRN HERE"

# ── 2. LORENZ DATA (short sequence to conserve compute time) ──────────────────

def lorenz(state, t, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

def generate_lorenz(n_steps=5000, dt=0.02, seed=42):
    np.random.seed(seed)
    t = np.linspace(0, n_steps * dt, n_steps)
    states = odeint(lorenz, [1.0, 1.0, 1.0], t)
    return states

print("Generating Lorenz data...")
states = generate_lorenz()
x_series = states[:, 0]
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x_series.reshape(-1, 1)).flatten()

# Use a SHORT sequence to conserve IBM compute time
# 300 steps total: 200 train, 100 test
N_STEPS   = 300
x_short   = x_scaled[:N_STEPS]
input_seq  = x_short[:-1]
target_seq = x_short[1:]
split      = 200
u_train, u_test = input_seq[:split], input_seq[split:]
y_train, y_test = target_seq[:split], target_seq[split:]

# ── 3. BUILD RESERVOIR CIRCUIT (seed 123 — our best performer) ────────────────

N_QUBITS = 6
N_LAYERS = 3
SHOTS    = 1024
WASHOUT  = 50   # shorter washout for shorter sequence

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
print(f"Reservoir circuit built (seed 123): {N_QUBITS} qubits, {N_LAYERS} layers")

# ── 4. BUILD ALL CIRCUITS UPFRONT ─────────────────────────────────────────────
# We batch all circuits together and submit as one job to IBM
# This is much more efficient than submitting one at a time

def build_all_circuits(input_sequence, n_qubits, reservoir_qc):
    """Build one circuit per timestep, return as a list."""
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
print(f"Total circuits to run: {len(all_circuits)}")

# ── 5. CONNECT TO IBM AND RUN ON REAL HARDWARE ────────────────────────────────

print("\nConnecting to IBM Quantum...")
service = QiskitRuntimeService(
    channel="ibm_cloud",
    token=IBM_API_KEY,
    instance=IBM_INSTANCE
)

# Pick the least busy backend automatically
backend = service.least_busy(operational=True, simulator=False)
print(f"Selected backend: {backend.name}")
print(f"Backend qubits: {backend.num_qubits}")
print(f"Queue: {backend.status().pending_jobs} jobs ahead")

# Transpile circuits for the real backend
print("\nTranspiling circuits for real hardware...")
transpiled = transpile(all_circuits, backend=backend, optimization_level=1)
print("Transpilation complete.")

# Run as a single batched job
print("\nSubmitting job to IBM Quantum (this may queue for a few minutes)...")
sampler = Sampler(backend)
job = sampler.run(transpiled, shots=SHOTS)
print(f"Job ID: {job.job_id()}")
print("Waiting for results... (don't close the terminal)")

result = job.result()
print("Job complete!")

# ── 6. EXTRACT PROBABILITIES FROM RESULTS ─────────────────────────────────────

def extract_probs(result, n_circuits, n_qubits, shots):
    """Extract qubit-1 probabilities from SamplerV2 results."""
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

print("\nExtracting results...")
all_states   = extract_probs(result, len(all_circuits), N_QUBITS, SHOTS)
train_states = all_states[:len(train_circuits)]
test_states  = all_states[len(train_circuits):]

# ── 7. TRAIN READOUT AND EVALUATE ─────────────────────────────────────────────

ridge = Ridge(alpha=1e-4, fit_intercept=True)
ridge.fit(train_states[WASHOUT:], y_train[WASHOUT:])
y_pred_test = ridge.predict(test_states)
hw_mse = mean_squared_error(y_test, y_pred_test)

print(f"\n{'='*50}")
print(f"REAL HARDWARE RESULT (seed 123, {backend.name})")
print(f"{'='*50}")
print(f"Hardware MSE:          {hw_mse:.6f}")
print(f"Simulated 0%  noise:   0.003252")
print(f"Simulated 1%  noise:   0.004554")
print(f"Simulated 5%  noise:   0.009863")
print(f"Simulated 10% noise:   0.015708")
print(f"Classical ESN:         0.000881")
print(f"{'='*50}")

# Estimate effective noise level from curve
sim_noise_mses = [0.003252, 0.004554, 0.009863, 0.015708, 0.047670]
sim_labels     = ["0%", "1%", "5%", "10%", "20%"]
closest_idx    = np.argmin(np.abs(np.array(sim_noise_mses) - hw_mse))
print(f"\nHardware result is closest to simulated noise level: {sim_labels[closest_idx]}")
print(f"→ Estimated effective hardware noise: ~{sim_labels[closest_idx]}")

# ── 8. PLOT ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Left: place hardware result on noise curve
noise_x     = [0, 1, 5, 10, 20]
sim_mse_123 = [0.003252, 0.004554, 0.009863, 0.015708, 0.047670]

axes[0].plot(noise_x, sim_mse_123, 'o-', color='darkorange',
             linewidth=2, markersize=8, label="Simulated (Seed 123)")
axes[0].axhline(y=hw_mse, color='red', linestyle='--',
                linewidth=2, label=f"Real Hardware MSE ({hw_mse:.6f})")
axes[0].axhline(y=0.000881, color='steelblue', linestyle='--',
                linewidth=2, label="Classical ESN (0.000881)")
axes[0].set_title(f"Real Hardware vs Simulated Curve\n({backend.name})", fontsize=12)
axes[0].set_xlabel("Depolarizing Noise Level (%)")
axes[0].set_ylabel("Test MSE (lower is better)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right: prediction plot
show = min(80, len(y_test))
axes[1].plot(y_test[:show], label="True", color="steelblue", lw=1.5)
axes[1].plot(y_pred_test[:show], label="Hardware Predicted",
             color="red", lw=1.2, linestyle="--")
axes[1].set_title(f"Real Hardware Predictions\nMSE: {hw_mse:.6f}", fontsize=12)
axes[1].set_xlabel("Timestep")
axes[1].set_ylabel("Value")
axes[1].legend()

plt.tight_layout()
plt.savefig("hardware_validation.png", dpi=150)
plt.show()
print("\nSaved: hardware_validation.png")
print("\n✅ Real hardware validation complete.")
print("   Next: sunspot dataset.")