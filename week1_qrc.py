"""
Week 1 - Quantum Reservoir Computing Project
============================================
Step 1: Generate Lorenz system data
Step 2: Build and evaluate a classical Echo State Network (ESN) baseline
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# ── 1. GENERATE LORENZ SYSTEM ─────────────────────────────────────────────────

def lorenz(state, t, sigma=10, rho=28, beta=8/3):
    """The Lorenz equations."""
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

def generate_lorenz(n_steps=5000, dt=0.02, seed=42):
    np.random.seed(seed)
    t = np.linspace(0, n_steps * dt, n_steps)
    init = [1.0, 1.0, 1.0]
    states = odeint(lorenz, init, t)
    return t, states

print("Generating Lorenz system...")
t, states = generate_lorenz()

# Use only the x-dimension for prediction
x_series = states[:, 0]

# Normalize to [0, 1]
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x_series.reshape(-1, 1)).flatten()

# Plot the Lorenz attractor
fig = plt.figure(figsize=(14, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(states[:1000, 0], states[:1000, 1], states[:1000, 2],
         lw=0.5, color='steelblue', alpha=0.8)
ax1.set_title("Lorenz Attractor (3D)", fontsize=13)
ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

ax2 = fig.add_subplot(122)
ax2.plot(t[:500], x_scaled[:500], color='steelblue', lw=0.8)
ax2.set_title("Lorenz X-dimension (normalized)", fontsize=13)
ax2.set_xlabel("Time"); ax2.set_ylabel("Value")

plt.tight_layout()
plt.savefig("lorenz_visualization.png", dpi=150)
plt.show()
print("Saved: lorenz_visualization.png\n")

# ── 2. CLASSICAL ECHO STATE NETWORK (ESN) ────────────────────────────────────

class EchoStateNetwork:
    """
    A minimal Echo State Network.
    - W_in:  input weights (fixed, random)
    - W_res: reservoir weights (fixed, random, sparse)
    - W_out: output weights (trained via Ridge regression)
    """
    def __init__(self, n_reservoir=200, spectral_radius=0.9,
                 input_scaling=0.5, sparsity=0.1, seed=42):
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.sparsity = sparsity
        self.rng = np.random.RandomState(seed)

        # Input weights: shape (n_reservoir, 1)
        self.W_in = self.rng.uniform(-1, 1, (n_reservoir, 1)) * input_scaling

        # Reservoir weights: sparse random matrix
        W = self.rng.uniform(-1, 1, (n_reservoir, n_reservoir))
        mask = self.rng.rand(n_reservoir, n_reservoir) > sparsity
        W[mask] = 0

        # Scale to desired spectral radius
        eigenvalues = np.linalg.eigvals(W)
        sr = np.max(np.abs(eigenvalues))
        self.W_res = W * (spectral_radius / sr)

        self.W_out = None

    def _run_reservoir(self, u_sequence):
        """Drive the reservoir with input sequence u_sequence."""
        n_steps = len(u_sequence)
        states = np.zeros((n_steps, self.n_reservoir))
        state = np.zeros(self.n_reservoir)

        for t_idx, u in enumerate(u_sequence):
            pre = self.W_res @ state + self.W_in @ np.array([u])
            state = np.tanh(pre)
            states[t_idx] = state

        return states

    def fit(self, u_train, y_train, washout=100, alpha=1e-4):
        """Train output weights using Ridge regression."""
        states = self._run_reservoir(u_train)
        # Discard washout steps
        states = states[washout:]
        y_train = y_train[washout:]
        # Ridge regression
        ridge = Ridge(alpha=alpha, fit_intercept=True)
        ridge.fit(states, y_train)
        self.W_out = ridge
        return self

    def predict(self, u_test):
        """Predict on test sequence."""
        states = self._run_reservoir(u_test)
        return self.W_out.predict(states)


# ── 3. TRAIN AND EVALUATE ESN ─────────────────────────────────────────────────

# One-step-ahead prediction: input = x[t], target = x[t+1]
input_seq  = x_scaled[:-1]
target_seq = x_scaled[1:]

# Train/test split (80/20)
split = int(0.8 * len(input_seq))
u_train, u_test   = input_seq[:split],  input_seq[split:]
y_train, y_test   = target_seq[:split], target_seq[split:]

print("Training Echo State Network...")
esn = EchoStateNetwork(n_reservoir=200, spectral_radius=0.95, seed=42)
esn.fit(u_train, y_train, washout=100, alpha=1e-4)

# Predict
y_pred_train = esn.predict(u_train)
y_pred_test  = esn.predict(u_test)

mse_train = mean_squared_error(y_train[100:], y_pred_train[100:])
mse_test  = mean_squared_error(y_test, y_pred_test)

print(f"ESN Train MSE: {mse_train:.6f}")
print(f"ESN Test  MSE: {mse_test:.6f}")
print()

# ── 4. PLOT PREDICTIONS ───────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(14, 7))

# Training prediction (show last 300 steps of training)
show = 300
axes[0].plot(y_train[-show:], label="True",      color="steelblue", lw=1.5)
axes[0].plot(y_pred_train[-show:], label="ESN Predicted",
             color="tomato", lw=1.2, linestyle="--")
axes[0].set_title(f"ESN — Training Set (last {show} steps) | MSE: {mse_train:.6f}",
                  fontsize=12)
axes[0].legend(); axes[0].set_xlabel("Timestep"); axes[0].set_ylabel("Value")

# Test prediction (first 300 steps)
axes[1].plot(y_test[:show], label="True",         color="steelblue", lw=1.5)
axes[1].plot(y_pred_test[:show], label="ESN Predicted",
             color="tomato", lw=1.2, linestyle="--")
axes[1].set_title(f"ESN — Test Set (first {show} steps) | MSE: {mse_test:.6f}",
                  fontsize=12)
axes[1].legend(); axes[1].set_xlabel("Timestep"); axes[1].set_ylabel("Value")

plt.tight_layout()
plt.savefig("esn_predictions.png", dpi=150)
plt.show()
print("Saved: esn_predictions.png")
print()
print("✅ Week 1 complete. Your classical baseline MSE is:", round(mse_test, 6))
print("   This is the number your quantum reservoir needs to beat (or match) in Week 2.")