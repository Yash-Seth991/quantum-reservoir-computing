"""
Week 3 - Quantum Reservoir Computing Project
============================================
Noise sensitivity analysis.
Run the quantum reservoir at 5 different noise levels and plot the performance curve.
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
from qiskit_aer.noise import NoiseModel, depolarizing_error

# ── 1. REGENERATE LORENZ DATA ─────────────────────────────────────────────────

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

input_seq  = x_scaled[:-1]
target_seq = x_scaled[1:]
split = int(0.8 * len(input_seq))
u_train, u_test = input_seq[:split], input_seq[split:]
y_train, y_test = target_seq[:split], target_seq[split:]

# ── 2. BUILD THE FIXED RESERVOIR CIRCUIT (same as Week 2) ─────────────────────

N_QUBITS = 6
N_LAYERS = 3
SHOTS    = 1024
WASHOUT  = 100

def build_reservoir_circuit(n_qubits, n_layers, seed=42):
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

reservoir_circuit = build_reservoir_circuit(N_QUBITS, N_LAYERS)

# ── 3. NOISE MODEL BUILDER ────────────────────────────────────────────────────

def build_noise_model(noise_level):
    """
    Build a depolarizing noise model at a given error rate.
    Depolarizing noise randomly replaces a qubit's state with a completely
    random state with probability = noise_level.
    noise_level = 0.0 means perfect (no noise)
    noise_level = 0.2 means 20% chance of error per gate
    """
    if noise_level == 0.0:
        return None  # no noise model = perfect simulator

    noise_model = NoiseModel()
    # Apply noise to single-qubit gates
    single_qubit_error = depolarizing_error(noise_level, 1)
    noise_model.add_all_qubit_quantum_error(single_qubit_error, ['rx', 'ry', 'rz'])
    # Apply noise to two-qubit gates (usually higher error rate on real hardware)
    two_qubit_error = depolarizing_error(noise_level * 2, 2)
    noise_model.add_all_qubit_quantum_error(two_qubit_error, ['cx'])

    return noise_model

# ── 4. RUN RESERVOIR FUNCTION ─────────────────────────────────────────────────

def run_quantum_reservoir(input_sequence, n_qubits, reservoir_qc, shots, noise_model=None):
    simulator = AerSimulator(noise_model=noise_model)
    reservoir_states = []

    for i, u in enumerate(input_sequence):
        if i % 500 == 0:
            print(f"  Step {i}/{len(input_sequence)}")

        qc = QuantumCircuit(n_qubits)
        for q in range(n_qubits):
            qc.ry(2 * np.pi * u, q)
        qc.compose(reservoir_qc, inplace=True)
        qc.measure_all()

        job = simulator.run(qc, shots=shots)
        counts = job.result().get_counts()

        probs = np.zeros(n_qubits)
        for bitstring, count in counts.items():
            bits = bitstring.replace(" ", "")[::-1]
            for q, bit in enumerate(bits):
                if bit == "1":
                    probs[q] += count
        probs /= shots
        reservoir_states.append(probs)

    return np.array(reservoir_states)

def evaluate_at_noise_level(noise_level, label):
    """Run the full pipeline at a given noise level and return test MSE."""
    print(f"\n{'='*50}")
    print(f"Running at noise level: {label}")
    print(f"{'='*50}")

    noise_model = build_noise_model(noise_level)

    print("  Training data...")
    train_states = run_quantum_reservoir(u_train, N_QUBITS, reservoir_circuit, SHOTS, noise_model)
    print("  Test data...")
    test_states  = run_quantum_reservoir(u_test,  N_QUBITS, reservoir_circuit, SHOTS, noise_model)

    # Apply washout and train
    ridge = Ridge(alpha=1e-4, fit_intercept=True)
    ridge.fit(train_states[WASHOUT:], y_train[WASHOUT:])

    y_pred_test = ridge.predict(test_states)
    mse = mean_squared_error(y_test, y_pred_test)
    print(f"  ✅ Test MSE at {label}: {mse:.6f}")
    return mse, ridge.predict(train_states[WASHOUT:]), y_pred_test

# ── 5. RUN ALL NOISE LEVELS ───────────────────────────────────────────────────

NOISE_LEVELS = [0.0, 0.01, 0.05, 0.10, 0.20]
NOISE_LABELS = ["0%", "1%", "5%", "10%", "20%"]
ESN_MSE      = 0.000881

results = {}
for level, label in zip(NOISE_LEVELS, NOISE_LABELS):
    mse, _, y_pred_test = evaluate_at_noise_level(level, label)
    results[label] = {"mse": mse, "y_pred_test": y_pred_test}

# ── 6. PRINT SUMMARY TABLE ────────────────────────────────────────────────────

print("\n")
print("=" * 45)
print(f"{'Model':<25} {'Test MSE':>10}")
print("=" * 45)
print(f"{'Classical ESN':<25} {ESN_MSE:>10.6f}")
print("-" * 45)
for label in NOISE_LABELS:
    mse = results[label]["mse"]
    marker = " ← best quantum" if mse == min(r["mse"] for r in results.values()) else ""
    print(f"{'QR ' + label + ' noise':<25} {mse:>10.6f}{marker}")
print("=" * 45)

# ── 7. PLOT THE NOISE CURVE ───────────────────────────────────────────────────

mse_values = [results[label]["mse"] for label in NOISE_LABELS]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Left: noise sensitivity curve
axes[0].plot(NOISE_LABELS, mse_values, marker='o', color='darkorange',
             linewidth=2, markersize=8, label="Quantum Reservoir")
axes[0].axhline(y=ESN_MSE, color='steelblue', linestyle='--',
                linewidth=2, label=f"Classical ESN ({ESN_MSE:.6f})")
axes[0].set_title("Noise Sensitivity Curve", fontsize=13)
axes[0].set_xlabel("Depolarizing Noise Level")
axes[0].set_ylabel("Test MSE (lower is better)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right: bar chart of all models
all_labels = ["Classical\nESN"] + [f"QR\n{l}" for l in NOISE_LABELS]
all_mses   = [ESN_MSE] + mse_values
colors     = ["steelblue"] + ["darkorange"] * len(NOISE_LEVELS)
bars = axes[1].bar(all_labels, all_mses, color=colors, width=0.5)
axes[1].set_title("All Models Comparison", fontsize=13)
axes[1].set_ylabel("Test MSE (lower is better)")
for bar, val in zip(bars, all_mses):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.0001,
                 f"{val:.4f}", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig("noise_sensitivity_curve.png", dpi=150)
plt.show()
print("Saved: noise_sensitivity_curve.png")

# ── 8. PREDICTION PLOTS AT EACH NOISE LEVEL ──────────────────────────────────

fig, axes = plt.subplots(len(NOISE_LEVELS), 1, figsize=(14, 4 * len(NOISE_LEVELS)))
show = 300

for i, label in enumerate(NOISE_LABELS):
    y_pred = results[label]["y_pred_test"]
    mse    = results[label]["mse"]
    axes[i].plot(y_test[:show], label="True", color="steelblue", lw=1.5)
    axes[i].plot(y_pred[:show], label=f"QR Predicted",
                 color="darkorange", lw=1.2, linestyle="--")
    axes[i].set_title(f"Noise: {label} | MSE: {mse:.6f}", fontsize=11)
    axes[i].legend(loc="upper right")
    axes[i].set_ylabel("Value")

axes[-1].set_xlabel("Timestep")
plt.tight_layout()
plt.savefig("predictions_all_noise_levels.png", dpi=150)
plt.show()
print("Saved: predictions_all_noise_levels.png")

print("   Your noise sensitivity curve is your main finding.")
print("   Next: write up, figures, and presentation.")
