"""
Week 4a - Multiple Seeds Experiment
=====================================
Run the noise sensitivity analysis across 3 different random circuit seeds
to get statistically robust results with error bars.
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

# ── 1. LORENZ DATA ────────────────────────────────────────────────────────────

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

# ── 2. SETTINGS ───────────────────────────────────────────────────────────────

N_QUBITS      = 6
N_LAYERS      = 3
SHOTS         = 1024
WASHOUT       = 100
SEEDS         = [42, 123, 7]       # 3 different random circuits
NOISE_LEVELS  = [0.0, 0.01, 0.05, 0.10, 0.20]
NOISE_LABELS  = ["0%", "1%", "5%", "10%", "20%"]
ESN_MSE       = 0.000881

# ── 3. HELPER FUNCTIONS ───────────────────────────────────────────────────────

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

def build_noise_model(noise_level):
    if noise_level == 0.0:
        return None
    noise_model = NoiseModel()
    single_qubit_error = depolarizing_error(noise_level, 1)
    noise_model.add_all_qubit_quantum_error(single_qubit_error, ['rx', 'ry', 'rz'])
    two_qubit_error = depolarizing_error(noise_level * 2, 2)
    noise_model.add_all_qubit_quantum_error(two_qubit_error, ['cx'])
    return noise_model

def run_quantum_reservoir(input_sequence, n_qubits, reservoir_qc, shots, noise_model=None):
    simulator = AerSimulator(noise_model=noise_model)
    reservoir_states = []
    for i, u in enumerate(input_sequence):
        if i % 500 == 0:
            print(f"    Step {i}/{len(input_sequence)}")
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

def evaluate_one_run(reservoir_qc, noise_level):
    """Run full pipeline for one circuit + one noise level. Returns test MSE."""
    noise_model = build_noise_model(noise_level)
    train_states = run_quantum_reservoir(u_train, N_QUBITS, reservoir_qc, SHOTS, noise_model)
    test_states  = run_quantum_reservoir(u_test,  N_QUBITS, reservoir_qc, SHOTS, noise_model)
    ridge = Ridge(alpha=1e-4, fit_intercept=True)
    ridge.fit(train_states[WASHOUT:], y_train[WASHOUT:])
    y_pred = ridge.predict(test_states)
    return mean_squared_error(y_test, y_pred)

# ── 4. MAIN EXPERIMENT LOOP ───────────────────────────────────────────────────
# Shape: all_results[seed_idx][noise_idx] = MSE
all_results = np.zeros((len(SEEDS), len(NOISE_LEVELS)))

total_runs = len(SEEDS) * len(NOISE_LEVELS)
run_count  = 0

for s_idx, seed in enumerate(SEEDS):
    print(f"\n{'='*55}")
    print(f"SEED {seed}  ({s_idx+1}/{len(SEEDS)})")
    print(f"{'='*55}")
    reservoir_qc = build_reservoir_circuit(N_QUBITS, N_LAYERS, seed)

    for n_idx, (level, label) in enumerate(zip(NOISE_LEVELS, NOISE_LABELS)):
        run_count += 1
        print(f"\n  Noise: {label}  (run {run_count}/{total_runs})")
        print(f"  Training data...")
        mse = evaluate_one_run(reservoir_qc, level)
        all_results[s_idx][n_idx] = mse
        print(f"  ✅ MSE: {mse:.6f}")

# ── 5. COMPUTE STATISTICS ─────────────────────────────────────────────────────

mean_mse = np.mean(all_results, axis=0)   # average across seeds
std_mse  = np.std(all_results,  axis=0)   # standard deviation across seeds

# ── 6. PRINT SUMMARY TABLE ────────────────────────────────────────────────────

print("\n")
print("=" * 60)
print(f"{'Noise':<8} {'Seed 42':>10} {'Seed 123':>10} {'Seed 7':>10} {'Mean':>10} {'Std':>10}")
print("=" * 60)
for n_idx, label in enumerate(NOISE_LABELS):
    row = f"{label:<8}"
    for s_idx in range(len(SEEDS)):
        row += f" {all_results[s_idx][n_idx]:>10.6f}"
    row += f" {mean_mse[n_idx]:>10.6f} {std_mse[n_idx]:>10.6f}"
    print(row)
print("=" * 60)
print(f"\nClassical ESN baseline: {ESN_MSE:.6f}")

# ── 7. PLOT WITH ERROR BARS ───────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left: noise curve with error bars
axes[0].errorbar(NOISE_LABELS, mean_mse, yerr=std_mse,
                 fmt='o-', color='darkorange', linewidth=2,
                 markersize=8, capsize=6, capthick=2,
                 label="Quantum Reservoir (mean ± std)")

# Plot individual seed lines faintly behind the mean
colors = ['#ffb347', '#ff8c00', '#cc6600']
for s_idx, seed in enumerate(SEEDS):
    axes[0].plot(NOISE_LABELS, all_results[s_idx],
                 color=colors[s_idx], linewidth=0.8,
                 linestyle='--', alpha=0.4, label=f"Seed {seed}")

axes[0].axhline(y=ESN_MSE, color='steelblue', linestyle='--',
                linewidth=2, label=f"Classical ESN ({ESN_MSE:.6f})")
axes[0].set_title("Noise Sensitivity Curve (3 Seeds)", fontsize=13)
axes[0].set_xlabel("Depolarizing Noise Level")
axes[0].set_ylabel("Test MSE (lower is better)")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Right: bar chart with error bars
x = np.arange(len(NOISE_LABELS))
bars = axes[1].bar(x, mean_mse, yerr=std_mse,
                   color='darkorange', width=0.5,
                   capsize=5, label="Quantum Reservoir")
axes[1].axhline(y=ESN_MSE, color='steelblue', linestyle='--',
                linewidth=2, label=f"Classical ESN ({ESN_MSE:.6f})")
axes[1].set_xticks(x)
axes[1].set_xticklabels(NOISE_LABELS)
axes[1].set_title("Mean MSE per Noise Level (± std)", fontsize=13)
axes[1].set_ylabel("Test MSE (lower is better)")
axes[1].set_xlabel("Depolarizing Noise Level")
axes[1].legend()
for bar, val, std in zip(bars, mean_mse, std_mse):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + std + 0.0005,
                 f"{val:.4f}", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig("noise_curve_with_error_bars.png", dpi=150)
plt.show()
print("\nSaved: noise_curve_with_error_bars.png")
print("\n✅ Multiple seeds experiment complete.")
print("   Your noise sensitivity curve now has statistical error bars.")
print("   Next: real hardware validation.")