"""
╔══════════════════════════════════════════════════════════════════╗
║         QUANTUM HARDWARE NOISE CHARACTERIZATION TOOL            ║
║         Using Quantum Reservoir Computing                        ║
╚══════════════════════════════════════════════════════════════════╝

What this tool does:
  1. Generates a chaotic time-series benchmark (Lorenz system)
  2. Builds a fixed random quantum reservoir circuit
  3. Runs the reservoir at multiple simulated noise levels → noise curve
  4. Runs the same reservoir on your real IBM quantum backend
  5. Estimates your hardware's effective depolarizing noise level
     by finding where the hardware result falls on the simulated curve

Usage:
  1. Fill in your IBM credentials below
  2. Run: python qrc_noise_characterizer.py
  3. Results are printed to terminal and saved as PNG plots

Requirements:
  pip install numpy scipy matplotlib scikit-learn qiskit qiskit-aer qiskit-ibm-runtime
"""

# ══════════════════════════════════════════════════════════════════
#  ⚙️  SETTINGS — edit these before running
# ══════════════════════════════════════════════════════════════════

IBM_API_KEY  = "PASTE_YOUR_API_KEY_HERE"
IBM_INSTANCE = "PASTE_YOUR_CRN_HERE"

# Quantum reservoir settings
N_QUBITS    = 6       # number of qubits (reservoir size)
N_LAYERS    = 3       # circuit depth
CIRCUIT_SEED = 123    # random seed for reservoir circuit
SHOTS       = 1024    # measurement shots per circuit

# Noise levels to sweep (as fractions, e.g. 0.05 = 5%)
NOISE_LEVELS = [0.0, 0.01, 0.05, 0.10, 0.20]
NOISE_LABELS = ["0%", "1%", "5%", "10%", "20%"]

# Data settings
N_LORENZ_STEPS = 300   # keep short to conserve IBM compute time
WASHOUT        = 50    # reservoir warmup steps to discard
TRAIN_SPLIT    = 200   # number of steps for training

# Output
SAVE_PLOTS = True      # save plots to disk
PLOT_PREFIX = "qnc"    # prefix for output plot filenames

# ══════════════════════════════════════════════════════════════════
#  IMPORTS
# ══════════════════════════════════════════════════════════════════

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.compiler import transpile

# ══════════════════════════════════════════════════════════════════
#  STEP 1 — GENERATE BENCHMARK DATA (Lorenz system)
# ══════════════════════════════════════════════════════════════════

def generate_lorenz(n_steps, dt=0.02, seed=42):
    def lorenz(state, t, sigma=10, rho=28, beta=8/3):
        x, y, z = state
        return [sigma*(y-x), x*(rho-z)-y, x*y - beta*z]
    np.random.seed(seed)
    t = np.linspace(0, n_steps * dt, n_steps)
    states = odeint(lorenz, [1.0, 1.0, 1.0], t)
    return states[:, 0]  # return X dimension only

def prepare_data(series, train_split):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()
    inp, tgt = scaled[:-1], scaled[1:]
    return (inp[:train_split], inp[train_split:],
            tgt[:train_split], tgt[train_split:])

# ══════════════════════════════════════════════════════════════════
#  STEP 2 — BUILD QUANTUM RESERVOIR CIRCUIT
# ══════════════════════════════════════════════════════════════════

def build_reservoir(n_qubits, n_layers, seed):
    """
    Fixed random quantum circuit used as the reservoir.
    Never trained — only the classical readout layer is trained.
    Architecture: alternating random rotation gates + CNOT entanglers.
    """
    rng = np.random.RandomState(seed)
    qc  = QuantumCircuit(n_qubits)
    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.rx(rng.uniform(0, 2*np.pi), q)
            qc.ry(rng.uniform(0, 2*np.pi), q)
            qc.rz(rng.uniform(0, 2*np.pi), q)
        for q in range(n_qubits - 1):
            qc.cx(q, q+1)
    return qc

# ══════════════════════════════════════════════════════════════════
#  STEP 3 — BUILD AND RUN CIRCUITS
# ══════════════════════════════════════════════════════════════════

def build_circuits(input_seq, n_qubits, reservoir_qc):
    """Encode each input timestep as a rotation, then apply reservoir."""
    circuits = []
    for u in input_seq:
        qc = QuantumCircuit(n_qubits)
        for q in range(n_qubits):
            qc.ry(2 * np.pi * u, q)
        qc.compose(reservoir_qc, inplace=True)
        qc.measure_all()
        circuits.append(qc)
    return circuits

def extract_probs(counts, n_qubits, shots):
    """Convert measurement counts to per-qubit |1> probabilities."""
    probs = np.zeros(n_qubits)
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")[::-1]
        for q, bit in enumerate(bits[:n_qubits]):
            if bit == "1":
                probs[q] += count
    return probs / shots

def run_on_simulator(circuits, n_qubits, shots, noise_level):
    """Run circuits on local Aer simulator with optional noise."""
    noise_model = None
    if noise_level > 0:
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(
            depolarizing_error(noise_level, 1), ['rx', 'ry', 'rz'])
        nm.add_all_qubit_quantum_error(
            depolarizing_error(noise_level * 2, 2), ['cx'])
        noise_model = nm

    sim = AerSimulator(noise_model=noise_model)
    all_probs = []
    for qc in circuits:
        result = sim.run(qc, shots=shots).result()
        counts = result.get_counts()
        all_probs.append(extract_probs(counts, n_qubits, shots))
    return np.array(all_probs)

def run_on_hardware(circuits, backend, shots, n_qubits):
    """Run circuits on real IBM quantum hardware as a single batched job."""
    print(f"  Transpiling {len(circuits)} circuits...")
    transpiled = transpile(circuits, backend=backend, optimization_level=1)
    sampler = Sampler(backend)
    print(f"  Submitting job...")
    job = sampler.run(transpiled, shots=shots)
    print(f"  Job ID: {job.job_id()}")
    print(f"  Waiting for results...")
    result = job.result()
    print(f"  Job complete!")
    all_probs = []
    for i in range(len(circuits)):
        counts = result[i].data.meas.get_counts()
        all_probs.append(extract_probs(counts, n_qubits, shots))
    return np.array(all_probs)

def evaluate(train_states, test_states, y_train, y_test, washout):
    """Train Ridge readout and return test MSE + predictions."""
    ridge = Ridge(alpha=1e-4, fit_intercept=True)
    ridge.fit(train_states[washout:], y_train[washout:])
    y_pred = ridge.predict(test_states)
    return mean_squared_error(y_test, y_pred), y_pred

# ══════════════════════════════════════════════════════════════════
#  STEP 4 — ESTIMATE EFFECTIVE NOISE LEVEL
# ══════════════════════════════════════════════════════════════════

def estimate_noise_level(hw_mse, sim_mses, noise_labels, noise_levels):
    """
    Find where hardware MSE falls on the simulated noise curve.
    Uses linear interpolation between the two closest points.
    """
    sim_mses = np.array(sim_mses)

    # Direct match
    closest_idx = np.argmin(np.abs(sim_mses - hw_mse))
    closest_label = noise_labels[closest_idx]

    # Interpolate for a more precise estimate
    if hw_mse <= sim_mses[0]:
        estimate = f"< {noise_labels[0]} (hardware noise is lower than simulated minimum)"
    elif hw_mse >= sim_mses[-1]:
        estimate = f"> {noise_labels[-1]} (hardware noise exceeds simulated maximum)"
    else:
        # Find surrounding points
        for i in range(len(sim_mses) - 1):
            if sim_mses[i] <= hw_mse <= sim_mses[i+1]:
                # Linear interpolation
                frac = (hw_mse - sim_mses[i]) / (sim_mses[i+1] - sim_mses[i])
                low  = noise_levels[i] * 100
                high = noise_levels[i+1] * 100
                interp = low + frac * (high - low)
                estimate = f"~{interp:.1f}% (between {noise_labels[i]} and {noise_labels[i+1]})"
                break

    return closest_label, estimate

# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*60)
    print("  QUANTUM HARDWARE NOISE CHARACTERIZATION TOOL")
    print("="*60 + "\n")

    # ── Data ──────────────────────────────────────────────────────
    print("▶ Generating Lorenz benchmark data...")
    series = generate_lorenz(N_LORENZ_STEPS)
    u_train, u_test, y_train, y_test = prepare_data(series, TRAIN_SPLIT)
    print(f"  Train: {len(u_train)} steps | Test: {len(u_test)} steps\n")

    # ── Circuit ───────────────────────────────────────────────────
    print("▶ Building quantum reservoir circuit...")
    reservoir_qc = build_reservoir(N_QUBITS, N_LAYERS, CIRCUIT_SEED)
    train_circuits = build_circuits(u_train, N_QUBITS, reservoir_qc)
    test_circuits  = build_circuits(u_test,  N_QUBITS, reservoir_qc)
    print(f"  {N_QUBITS} qubits | {N_LAYERS} layers | seed {CIRCUIT_SEED}")
    print(f"  {len(train_circuits)+len(test_circuits)} total circuits built\n")

    # ── Simulated noise sweep ──────────────────────────────────────
    print("▶ Running simulated noise sweep...")
    sim_mses = []
    for level, label in zip(NOISE_LEVELS, NOISE_LABELS):
        print(f"  Noise: {label}...", end=" ", flush=True)
        tr = run_on_simulator(train_circuits, N_QUBITS, SHOTS, level)
        te = run_on_simulator(test_circuits,  N_QUBITS, SHOTS, level)
        mse, _ = evaluate(tr, te, y_train, y_test, WASHOUT)
        sim_mses.append(mse)
        print(f"MSE = {mse:.6f}")
    print()

    # ── Real hardware ─────────────────────────────────────────────
    print("▶ Connecting to IBM Quantum...")
    service = QiskitRuntimeService(
        channel="ibm_cloud",
        token=IBM_API_KEY,
        instance=IBM_INSTANCE
    )
    backend = service.least_busy(operational=True, simulator=False)
    print(f"  Selected backend : {backend.name}")
    print(f"  Qubits           : {backend.num_qubits}")
    print(f"  Queue            : {backend.status().pending_jobs} jobs ahead\n")

    print("▶ Running on real quantum hardware...")
    all_hw_circuits = train_circuits + test_circuits
    all_states      = run_on_hardware(all_hw_circuits, backend, SHOTS, N_QUBITS)
    hw_train = all_states[:len(train_circuits)]
    hw_test  = all_states[len(train_circuits):]
    hw_mse, hw_pred = evaluate(hw_train, hw_test, y_train, y_test, WASHOUT)
    print(f"  Hardware MSE: {hw_mse:.6f}\n")

    # ── Estimate effective noise ───────────────────────────────────
    closest, estimate = estimate_noise_level(
        hw_mse, sim_mses, NOISE_LABELS, NOISE_LEVELS)

    # ── Print results ─────────────────────────────────────────────
    print("="*60)
    print("  RESULTS")
    print("="*60)
    print(f"\n  Backend: {backend.name}")
    print(f"\n  Simulated noise curve:")
    for label, mse in zip(NOISE_LABELS, sim_mses):
        marker = " ◀" if label == closest else ""
        print(f"    {label:>5}  noise  →  MSE: {mse:.6f}{marker}")
    print(f"\n  Real hardware  →  MSE: {hw_mse:.6f}")
    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │  Estimated effective noise: {estimate}")
    print(f"  └─────────────────────────────────────────────────┘\n")

    # ── Plots ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"Quantum Hardware Noise Characterization — {backend.name}",
                 fontsize=14, fontweight='bold')

    # Left: noise curve with hardware result
    axes[0].plot(NOISE_LABELS, sim_mses, 'o-', color='darkorange',
                 linewidth=2, markersize=8, label="Simulated noise curve")
    axes[0].axhline(y=hw_mse, color='red', linestyle='--', linewidth=2,
                    label=f"Real hardware MSE ({hw_mse:.6f})")
    axes[0].fill_between(range(len(NOISE_LABELS)),
                         [m * 0.85 for m in sim_mses],
                         [m * 1.15 for m in sim_mses],
                         alpha=0.1, color='darkorange', label="±15% band")
    axes[0].set_title("Noise Sensitivity Curve", fontsize=12)
    axes[0].set_xlabel("Depolarizing Noise Level")
    axes[0].set_ylabel("Test MSE (lower is better)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].text(0.05, 0.95,
                 f"Estimated hardware noise:\n{estimate}",
                 transform=axes[0].transAxes,
                 fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Right: hardware prediction plot
    show = len(y_test)
    axes[1].plot(y_test[:show],    label="True",              color="steelblue",  lw=1.5)
    axes[1].plot(hw_pred[:show],   label="Hardware Predicted", color="red",        lw=1.2, linestyle="--")
    axes[1].set_title(f"Hardware Predictions ({backend.name})\nMSE: {hw_mse:.6f}", fontsize=12)
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Normalized Value")
    axes[1].legend()

    plt.tight_layout()
    if SAVE_PLOTS:
        fname = f"{PLOT_PREFIX}_characterization_{backend.name}.png"
        plt.savefig(fname, dpi=150)
        print(f"  Plot saved: {fname}")
    plt.show()

    print("\n✅ Characterization complete.")
    print(f"   Backend {backend.name} operates at an estimated")
    print(f"   effective depolarizing noise of {estimate}.\n")

if __name__ == "__main__":
    main()