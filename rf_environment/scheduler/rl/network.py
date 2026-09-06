from __future__ import annotations

import copy
import numpy as np


class MLPQNetwork:
    """Multi-Layer Perceptron Q-Network with Pure-NumPy Forward/Backward and Adam Optimizer.

    Recommended baseline architecture:
        Input(D) -> Linear(D, H1) -> ReLU -> Linear(H1, H2) -> ReLU -> Linear(H2, N)

    Zero external dependencies, completely deterministic, seed-reproducible.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        learning_rate: float = 0.001,
        seed: int | None = None,
        grad_clip: float = 5.0,
    ) -> None:
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.lr = float(learning_rate)
        self.grad_clip = float(grad_clip)

        self.rng = np.random.default_rng(seed)

        # He / Kaiming normal initialization
        self.w1 = self.rng.normal(0.0, np.sqrt(2.0 / self.input_dim), (self.input_dim, self.hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)

        self.w2 = self.rng.normal(0.0, np.sqrt(2.0 / self.hidden_dim), (self.hidden_dim, self.hidden_dim)).astype(np.float32)
        self.b2 = np.zeros(self.hidden_dim, dtype=np.float32)

        self.w3 = self.rng.normal(0.0, np.sqrt(2.0 / self.hidden_dim), (self.hidden_dim, self.output_dim)).astype(np.float32)
        self.b3 = np.zeros(self.output_dim, dtype=np.float32)

        # Adam optimizer moments
        self.m_w1 = np.zeros_like(self.w1)
        self.v_w1 = np.zeros_like(self.w1)
        self.m_b1 = np.zeros_like(self.b1)
        self.v_b1 = np.zeros_like(self.b1)

        self.m_w2 = np.zeros_like(self.w2)
        self.v_w2 = np.zeros_like(self.w2)
        self.m_b2 = np.zeros_like(self.b2)
        self.v_b2 = np.zeros_like(self.b2)

        self.m_w3 = np.zeros_like(self.w3)
        self.v_w3 = np.zeros_like(self.w3)
        self.m_b3 = np.zeros_like(self.b3)
        self.v_b3 = np.zeros_like(self.b3)

        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.t = 0  # Timestep for bias correction

        # Cache for backpropagation
        self._cache: dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray, cache: bool = False) -> np.ndarray:
        """Forward pass. Supports 1D [D] or 2D [B, D] inputs."""
        x_arr = np.asarray(x, dtype=np.float32)
        single = x_arr.ndim == 1
        if single:
            x_arr = x_arr[np.newaxis, :]  # Shape [1, D]

        z1 = np.dot(x_arr, self.w1) + self.b1
        a1 = np.maximum(0.0, z1)

        z2 = np.dot(a1, self.w2) + self.b2
        a2 = np.maximum(0.0, z2)

        q = np.dot(a2, self.w3) + self.b3

        if cache:
            self._cache = {"x": x_arr, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "q": q}

        return q[0] if single else q

    def train_step(self, states: np.ndarray, actions: np.ndarray, targets: np.ndarray) -> float:
        """Computes TD error, backpropagates gradients, and updates weights via Adam.

        Args:
            states: [B, D] float32 array of states
            actions: [B] int array of taken actions
            targets: [B] float32 array of Double DQN target values

        Returns:
            Mean squared error loss.
        """
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int64)
        targets = np.asarray(targets, dtype=np.float32)

        batch_size = len(states)
        if batch_size == 0:
            return 0.0

        # 1. Forward with caching
        q_preds = self.forward(states, cache=True)  # [B, N]

        # 2. Extract Q(s, a) for chosen actions
        chosen_q = q_preds[np.arange(batch_size), actions]  # [B]

        # 3. Loss = mean((Q(s, a) - target)^2)
        td_errors = chosen_q - targets  # [B]
        loss = float(np.mean(td_errors**2))

        # 4. Output gradient dQ [B, N]
        dq = np.zeros_like(q_preds)
        dq[np.arange(batch_size), actions] = (2.0 * td_errors) / batch_size

        # 5. Backprop through Layer 3
        a2 = self._cache["a2"]
        dw3 = np.dot(a2.T, dq)
        db3 = np.sum(dq, axis=0)

        da2 = np.dot(dq, self.w3.T)
        z2 = self._cache["z2"]
        dz2 = da2 * (z2 > 0.0)

        # 6. Backprop through Layer 2
        a1 = self._cache["a1"]
        dw2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0)

        da1 = np.dot(dz2, self.w2.T)
        z1 = self._cache["z1"]
        dz1 = da1 * (z1 > 0.0)

        # 7. Backprop through Layer 1
        x = self._cache["x"]
        dw1 = np.dot(x.T, dz1)
        db1 = np.sum(dz1, axis=0)

        # 8. Gradient clipping
        if self.grad_clip > 0.0:
            dw1 = np.clip(dw1, -self.grad_clip, self.grad_clip)
            db1 = np.clip(db1, -self.grad_clip, self.grad_clip)
            dw2 = np.clip(dw2, -self.grad_clip, self.grad_clip)
            db2 = np.clip(db2, -self.grad_clip, self.grad_clip)
            dw3 = np.clip(dw3, -self.grad_clip, self.grad_clip)
            db3 = np.clip(db3, -self.grad_clip, self.grad_clip)

        # 9. Adam optimizer step
        self.t += 1
        t = self.t

        for p, g, m, v in [
            (self.w1, dw1, self.m_w1, self.v_w1),
            (self.b1, db1, self.m_b1, self.v_b1),
            (self.w2, dw2, self.m_w2, self.v_w2),
            (self.b2, db2, self.m_b2, self.v_b2),
            (self.w3, dw3, self.m_w3, self.v_w3),
            (self.b3, db3, self.m_b3, self.v_b3),
        ]:
            m[:] = self.beta1 * m + (1.0 - self.beta1) * g
            v[:] = self.beta2 * v + (1.0 - self.beta2) * (g**2)
            m_hat = m / (1.0 - self.beta1**t)
            v_hat = v / (1.0 - self.beta2**t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        return loss

    def copy_from(self, source: MLPQNetwork) -> None:
        """Copies weights and biases from source network."""
        self.w1 = np.copy(source.w1)
        self.b1 = np.copy(source.b1)
        self.w2 = np.copy(source.w2)
        self.b2 = np.copy(source.b2)
        self.w3 = np.copy(source.w3)
        self.b3 = np.copy(source.b3)

    def clone(self) -> MLPQNetwork:
        """Creates an identical clone of this network."""
        cloned = MLPQNetwork(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            learning_rate=self.lr,
            grad_clip=self.grad_clip,
        )
        cloned.copy_from(self)
        return cloned

    def load_weights_dict(self, weights: dict[str, np.ndarray]) -> None:
        """Loads weights and biases directly from a dictionary."""
        self.w1 = np.array(weights["w1"], dtype=np.float32, copy=True)
        self.b1 = np.array(weights["b1"], dtype=np.float32, copy=True)
        self.w2 = np.array(weights["w2"], dtype=np.float32, copy=True)
        self.b2 = np.array(weights["b2"], dtype=np.float32, copy=True)
        self.w3 = np.array(weights["w3"], dtype=np.float32, copy=True)
        self.b3 = np.array(weights["b3"], dtype=np.float32, copy=True)
