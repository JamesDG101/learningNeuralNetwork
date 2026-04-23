import numpy as np


class BasicPolicyNetwork:
    """Very small MLP policy for pendulum force control.

    This class is intentionally minimal so you can later replace it with
    PyTorch/TensorFlow or trained weights from disk.
    """

    def __init__(self, input_dim, hidden_dim=16, force_limit=6.0, seed=0):
        rng = np.random.default_rng(seed)

        # Small init keeps initial actions near zero before training.
        self.w1 = rng.normal(0.0, 0.1, size=(hidden_dim, input_dim))
        self.b1 = np.zeros(hidden_dim, dtype=float)
        self.w2 = rng.normal(0.0, 0.1, size=(1, hidden_dim))
        self.b2 = np.zeros(1, dtype=float)

        self.force_limit = float(force_limit)

    def forward(self, observation):
        x = np.asarray(observation, dtype=float).reshape(-1)
        h = np.tanh(self.w1 @ x + self.b1)
        y = self.w2 @ h + self.b2

        # Bound output to valid force range.
        return float(self.force_limit * np.tanh(y[0] / max(self.force_limit, 1e-6)))

    def random_action(self):
        return float(np.random.uniform(-self.force_limit, self.force_limit))

    def set_weights(self, w1, b1, w2, b2):
        self.w1 = np.asarray(w1, dtype=float)
        self.b1 = np.asarray(b1, dtype=float)
        self.w2 = np.asarray(w2, dtype=float)
        self.b2 = np.asarray(b2, dtype=float)

    def load_npz(self, path):
        data = np.load(path)
        self.set_weights(data['w1'], data['b1'], data['w2'], data['b2'])