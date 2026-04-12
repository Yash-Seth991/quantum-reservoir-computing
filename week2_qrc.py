"""
Week 2 - Quantum Reservoir Computing Project
============================================
Build a quantum reservoir and compare it to the classical ESN baseline.
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

# ── 1. REGENERATE LORENZ DATA (same as Week 1) ────────────────────────────────

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
u_train, u_test = input_seq[:split],  input_seq[split:]
y_train, y_test = target_seq[:split], target_seq[split:]

# ── 2. BUILD THE QUANTUM RESERVOIR ────────────────────────────────────────────

N_QUBITS  = 6      # size of the reservoir
N_LAYERS  = 3      # depth of the circuit (more layers = more complex transformation)
SHOTS     = 1024   # number of times we run each circuit to get measurement statistics
WASHOUT   = 100    # discard first N steps while reservoir warms up

def build_reservoir_circuit(n_qubits, n_layers, seed=42):
    """
    Build a fixed random quantum circuit to use as a reservoir.
    It has two parts that repeat for each layer:
      - Rotation gates (RX, RY, RZ): rotate each qubit by a fixed random angle
      - Entangling gates (CX): create connections between qubits
    This circuit is FIXED — we never train it, just like W_res in the ESN.
    """
    rng = np.random.RandomState(seed)
    qc = QuantumCircuit(n_qubits)

    for _ in range(n_layers):
        # Random rotations on every qubit
        for q in range(n_qubits):
            qc.rx(rng.uniform(0, 2 * np.pi), q)
            qc.ry(rng.uniform(0, 2 * np.pi), q)
            qc.rz(rng.uniform(0, 2 * np.pi), q)
        # Entangle neighbouring qubits
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)

    return qc

reservoir_circuit = build_reservoir_circuit(N_QUBITS, N_LAYERS)
print(f"Quantum reservoir: {N_QUBITS} qubits, {N_LAYERS} layers")
print(reservoir_circuit.draw())

# ── 3. ENCODE INPUT AND RUN THE RESERVOIR ────────────────────────────────────

def run_quantum_reservoir(input_sequence, n_qubits, reservoir_qc, shots, noise_model=None):
    """
    For each timestep in the input sequence:
      1. Encode the input value as RY rotation angles on all qubits
      2. Apply the fixed reservoir circuit
      3. Measure all qubits
      4. Record the probability of each qubit being |1> as the reservoir state
    Returns a matrix of shape (n_steps, n_qubits) — one row per timestep.
    """
    simulator = AerSimulator(noise_model=noise_model)
    reservoir_states = []

    print(f"Running quantum reservoir ({len(input_sequence)} steps)... this may take a minute.")

    for i, u in enumerate(input_sequence):
        if i % 200 == 0:
            print(f"  Step {i}/{len(input_sequence)}")

        # Build the full circuit for this timestep
        qc = QuantumCircuit(n_qubits)

        # Encode input: rotate each qubit by angle proportional to input value
        # u is in [0,1], so 2*pi*u maps it to a full rotation range
        for q in range(n_qubits):
            qc.ry(2 * np.pi * u, q)

        # Apply the fixed reservoir
        qc.compose(reservoir_qc, inplace=True)

        # Measure all qubits
        qc.measure_all()

        # Run the circuit
        job = simulator.run(qc, shots=shots)
        counts = job.result().get_counts()

        # Convert counts to probabilities of each qubit being |1>
        probs = np.zeros(n_qubits)
        for bitstring, count in counts.items():
            # bitstring is e.g. "001011" — right to left is qubit 0 to N
            bits = bitstring.replace(" ", "")[::-1]
            for q, bit in enumerate(bits):
                if bit == "1":
                    probs[q] += count
        probs /= shots

        reservoir_states.append(probs)

    return np.array(reservoir_states)

# ── 4. RUN AND TRAIN ──────────────────────────────────────────────────────────

print("\nRunning quantum reservoir on training data...")
train_states = run_quantum_reservoir(u_train, N_QUBITS, reservoir_circuit, SHOTS)

print("\nRunning quantum reservoir on test data...")
test_states  = run_quantum_reservoir(u_test,  N_QUBITS, reservoir_circuit, SHOTS)

# Apply washout
train_states_w = train_states[WASHOUT:]
y_train_w      = y_train[WASHOUT:]

# Train the readout layer (same Ridge regression as ESN)
print("\nTraining readout layer...")
ridge = Ridge(alpha=1e-4, fit_intercept=True)
ridge.fit(train_states_w, y_train_w)

# Predict
y_pred_train = ridge.predict(train_states_w)
y_pred_test  = ridge.predict(test_states)

mse_train = mean_squared_error(y_train_w, y_pred_train)
mse_test  = mean_squared_error(y_test,  y_pred_test)

print(f"\nQuantum Reservoir Train MSE : {mse_train:.6f}")
print(f"Quantum Reservoir Test  MSE : {mse_test:.6f}")
print(f"Classical ESN Test      MSE : 0.000881")
print(f"Difference              : {mse_test - 0.000881:+.6f}")

# ── 5. PLOT RESULTS ───────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(14, 7))
show = 300

axes[0].plot(y_train_w[-show:], label="True",               color="steelblue", lw=1.5)
axes[0].plot(y_pred_train[-show:], label="QR Predicted",    color="darkorange", lw=1.2, linestyle="--")
axes[0].set_title(f"Quantum Reservoir — Training Set | MSE: {mse_train:.6f}", fontsize=12)
axes[0].legend(); axes[0].set_xlabel("Timestep"); axes[0].set_ylabel("Value")

axes[1].plot(y_test[:show],      label="True",              color="steelblue", lw=1.5)
axes[1].plot(y_pred_test[:show], label="QR Predicted",      color="darkorange", lw=1.2, linestyle="--")
axes[1].set_title(f"Quantum Reservoir — Test Set | MSE: {mse_test:.6f}", fontsize=12)
axes[1].legend(); axes[1].set_xlabel("Timestep"); axes[1].set_ylabel("Value")

plt.tight_layout()
plt.savefig("qr_predictions.png", dpi=150)
plt.show()
print("Saved: qr_predictions.png")

# ── 6. COMPARISON BAR CHART ───────────────────────────────────────────────────

fig2, ax = plt.subplots(figsize=(7, 5))
models = ["Classical ESN", "Quantum Reservoir\n(0% noise)"]
mses   = [0.000881, mse_test]
colors = ["steelblue", "darkorange"]
bars   = ax.bar(models, mses, color=colors, width=0.4)
ax.set_ylabel("Test MSE (lower is better)")
ax.set_title("Classical ESN vs Quantum Reservoir")
for bar, val in zip(bars, mses):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00002,
            f"{val:.6f}", ha="center", fontsize=11)
plt.tight_layout()
plt.savefig("comparison_chart.png", dpi=150)
plt.show()
print("Saved: comparison_chart.png")

print("   Next: inject noise at multiple levels and plot the performance curve.")
