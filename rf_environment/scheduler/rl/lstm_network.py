from __future__ import annotations

import copy
import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x_clipped = np.clip(x, -15.0, 15.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))


class LSTMQNetwork:
    """Pure-NumPy Recurrent Q-Network (LSTM + Dense Head + Adam Optimizer).

    Architecture:
        Input(D) -> LSTM(hidden_dim) -> Dense(hidden_dim, dense_dim) -> ReLU -> Dense(dense_dim, output_dim)

    Features:
    1. Zero external dependencies (pure NumPy, 100% deterministic, seed-controlled).
    2. Single-step mode for O(1) live inference during environment stepping.
    3. Batched sequence forward pass and analytical Backpropagation Through Time (BPTT).
    4. Target network cloning and Adam moment updates with gradient clipping.
    5. Forget gate bias initialization to +1.0 for stable gradient flow across long horizons.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        dense_dim: int = 64,
        learning_rate: float = 0.001,
        seed: int | None = None,
        grad_clip: float = 5.0,
    ) -> None:
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.dense_dim = int(dense_dim)
        self.lr = float(learning_rate)
        self.grad_clip = float(grad_clip)

        self.rng = np.random.default_rng(seed)

        # 1. LSTM Cell Weights: gates order [input (i), forget (f), output (o), cell (g)]
        # W_x: [D, 4*H], W_h: [H, 4*H], b: [4*H]
        std_x = np.sqrt(2.0 / (self.input_dim + self.hidden_dim))
        std_h = np.sqrt(2.0 / (self.hidden_dim + self.hidden_dim))

        self.w_x = self.rng.normal(0.0, std_x, (self.input_dim, 4 * self.hidden_dim)).astype(np.float32)
        self.w_h = self.rng.normal(0.0, std_h, (self.hidden_dim, 4 * self.hidden_dim)).astype(np.float32)
        self.b_lstm = np.zeros(4 * self.hidden_dim, dtype=np.float32)

        # Initialize forget gate bias to 1.0 (standard for LSTM training stability)
        self.b_lstm[self.hidden_dim : 2 * self.hidden_dim] = 1.0

        # 2. Dense Head Weights
        # Layer 1: [H, dense_dim]
        std_d1 = np.sqrt(2.0 / self.hidden_dim)
        self.w_dense = self.rng.normal(0.0, std_d1, (self.hidden_dim, self.dense_dim)).astype(np.float32)
        self.b_dense = np.zeros(self.dense_dim, dtype=np.float32)

        # Output Layer: [dense_dim, output_dim]
        std_out = np.sqrt(2.0 / self.dense_dim)
        self.w_out = self.rng.normal(0.0, std_out, (self.dense_dim, self.output_dim)).astype(np.float32)
        self.b_out = np.zeros(self.output_dim, dtype=np.float32)

        # 3. Adam Optimizer Moments
        self.params = [
            self.w_x, self.w_h, self.b_lstm,
            self.w_dense, self.b_dense,
            self.w_out, self.b_out,
        ]
        self.m_moments = [np.zeros_like(p) for p in self.params]
        self.v_moments = [np.zeros_like(p) for p in self.params]

        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.t = 0  # Timestep for bias correction

    def init_hidden(self, batch_size: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Returns initial zeros state (h_0, c_0) of shape [B, H]."""
        h = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        c = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        return h, c

    def step(
        self,
        x: np.ndarray,
        h_prev: np.ndarray,
        c_prev: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Single-step inference.

        Args:
            x: Input vector [D] or batch [B, D].
            h_prev: Previous hidden state [H] or [B, H].
            c_prev: Previous cell state [H] or [B, H].

        Returns:
            q: Q-values [N] or [B, N].
            h: Next hidden state [H] or [B, H].
            c: Next cell state [H] or [B, H].
        """
        x_arr = np.asarray(x, dtype=np.float32)
        h_arr = np.asarray(h_prev, dtype=np.float32)
        c_arr = np.asarray(c_prev, dtype=np.float32)

        single = x_arr.ndim == 1
        if single:
            x_arr = x_arr[np.newaxis, :]
            h_arr = h_arr[np.newaxis, :]
            c_arr = c_arr[np.newaxis, :]

        # Gates computation
        gates = np.dot(x_arr, self.w_x) + np.dot(h_arr, self.w_h) + self.b_lstm
        h_dim = self.hidden_dim

        i = _sigmoid(gates[:, 0:h_dim])
        f = _sigmoid(gates[:, h_dim : 2 * h_dim])
        o = _sigmoid(gates[:, 2 * h_dim : 3 * h_dim])
        g = np.tanh(gates[:, 3 * h_dim : 4 * h_dim])

        c = f * c_arr + i * g
        tanh_c = np.tanh(c)
        h = o * tanh_c

        # Head computation
        z_dense = np.dot(h, self.w_dense) + self.b_dense
        a_dense = np.maximum(0.0, z_dense)
        q = np.dot(a_dense, self.w_out) + self.b_out

        if single:
            return q[0], h[0], c[0]
        return q, h, c

    def forward_sequence(
        self,
        x_seq: np.ndarray,
        h_0: np.ndarray | None = None,
        c_0: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass over a sequence of states.

        Args:
            x_seq: Sequence array [B, L, D] or [L, D].
            h_0: Optional initial hidden state [B, H].
            c_0: Optional initial cell state [B, H].

        Returns:
            q_seq: Q-values sequence [B, L, N] (or [L, N] if single).
            h_final: Final hidden state [B, H] (or [H]).
            c_final: Final cell state [B, H] (or [H]).
        """
        x_arr = np.asarray(x_seq, dtype=np.float32)
        single = x_arr.ndim == 2
        if single:
            x_arr = x_arr[np.newaxis, :, :]  # [1, L, D]

        b_size, seq_len, _ = x_arr.shape
        if h_0 is None or c_0 is None:
            h_init, c_init = self.init_hidden(b_size)
            h = h_0 if h_0 is not None else h_init
            c = c_0 if c_0 is not None else c_init
        else:
            h = np.asarray(h_0, dtype=np.float32)
            c = np.asarray(c_0, dtype=np.float32)
            if h.ndim == 1:
                h = h[np.newaxis, :]
            if c.ndim == 1:
                c = c[np.newaxis, :]

        q_list = []
        for t_step in range(seq_len):
            x_t = x_arr[:, t_step, :]
            q_t, h, c = self.step(x_t, h, c)
            q_list.append(q_t)

        q_seq = np.stack(q_list, axis=1)  # [B, L, N]

        if single:
            return q_seq[0], h[0], c[0]
        return q_seq, h, c

    def train_step(
        self,
        x_seq: np.ndarray,
        actions_seq: np.ndarray,
        targets_seq: np.ndarray,
        h_0: np.ndarray | None = None,
        c_0: np.ndarray | None = None,
        burn_in: int = 0,
    ) -> float:
        """Executes Backpropagation Through Time (BPTT) and updates parameters via Adam.

        Args:
            x_seq: [B, L, D] sequence of feature states.
            actions_seq: [B, L] sequence of taken actions.
            targets_seq: [B, L] sequence of Double DQN target values.
            h_0: Optional initial hidden state [B, H].
            c_0: Optional initial cell state [B, H].
            burn_in: Number of prefix steps ignored in loss calculation.

        Returns:
            Mean squared error loss over evaluated steps.
        """
        x_arr = np.asarray(x_seq, dtype=np.float32)
        act_arr = np.asarray(actions_seq, dtype=np.int64)
        tgt_arr = np.asarray(targets_seq, dtype=np.float32)

        b_size, seq_len, _ = x_arr.shape
        h_dim = self.hidden_dim

        if h_0 is None or c_0 is None:
            h, c = self.init_hidden(b_size)
        else:
            h = np.copy(np.asarray(h_0, dtype=np.float32))
            c = np.copy(np.asarray(c_0, dtype=np.float32))

        # Forward pass with full tape history caching for BPTT
        tape_x = []
        tape_h_prev = []
        tape_c_prev = []
        tape_gates = []
        tape_i = []
        tape_f = []
        tape_o = []
        tape_g = []
        tape_c = []
        tape_tanh_c = []
        tape_h = []
        tape_z_dense = []
        tape_a_dense = []
        tape_q = []

        for t_step in range(seq_len):
            x_t = x_arr[:, t_step, :]
            tape_x.append(x_t)
            tape_h_prev.append(h)
            tape_c_prev.append(c)

            gates = np.dot(x_t, self.w_x) + np.dot(h, self.w_h) + self.b_lstm
            tape_gates.append(gates)

            i_gate = _sigmoid(gates[:, 0:h_dim])
            f_gate = _sigmoid(gates[:, h_dim : 2 * h_dim])
            o_gate = _sigmoid(gates[:, 2 * h_dim : 3 * h_dim])
            g_gate = np.tanh(gates[:, 3 * h_dim : 4 * h_dim])

            c = f_gate * c + i_gate * g_gate
            tanh_c = np.tanh(c)
            h = o_gate * tanh_c

            z_dense = np.dot(h, self.w_dense) + self.b_dense
            a_dense = np.maximum(0.0, z_dense)
            q_pred = np.dot(a_dense, self.w_out) + self.b_out

            tape_i.append(i_gate)
            tape_f.append(f_gate)
            tape_o.append(o_gate)
            tape_g.append(g_gate)
            tape_c.append(c)
            tape_tanh_c.append(tanh_c)
            tape_h.append(h)
            tape_z_dense.append(z_dense)
            tape_a_dense.append(a_dense)
            tape_q.append(q_pred)

        # Loss calculation on post-burn-in steps
        eval_steps = max(seq_len - burn_in, 1)
        total_loss = 0.0
        tape_dq = [np.zeros((b_size, self.output_dim), dtype=np.float32) for _ in range(seq_len)]

        for t_step in range(burn_in, seq_len):
            q_t = tape_q[t_step]
            act_t = act_arr[:, t_step]
            tgt_t = tgt_arr[:, t_step]

            chosen_q = q_t[np.arange(b_size), act_t]
            td_err = chosen_q - tgt_t
            step_loss = float(np.mean(td_err**2))
            total_loss += step_loss

            scale = (2.0 * td_err) / (b_size * eval_steps)
            dq_t = np.zeros_like(q_t)
            dq_t[np.arange(b_size), act_t] = scale
            tape_dq[t_step] = dq_t

        mean_loss = total_loss / eval_steps

        # Backward Pass (BPTT)
        dw_x = np.zeros_like(self.w_x)
        dw_h = np.zeros_like(self.w_h)
        db_lstm = np.zeros_like(self.b_lstm)
        dw_dense = np.zeros_like(self.w_dense)
        db_dense = np.zeros_like(self.b_dense)
        dw_out = np.zeros_like(self.w_out)
        db_out = np.zeros_like(self.b_out)

        dh_next = np.zeros((b_size, h_dim), dtype=np.float32)
        dc_next = np.zeros((b_size, h_dim), dtype=np.float32)

        for t_step in range(seq_len - 1, -1, -1):
            dq_t = tape_dq[t_step]
            a_dense_t = tape_a_dense[t_step]
            z_dense_t = tape_z_dense[t_step]
            h_t = tape_h[t_step]
            tanh_c_t = tape_tanh_c[t_step]
            c_t = tape_c[t_step]
            c_prev_t = tape_c_prev[t_step]
            h_prev_t = tape_h_prev[t_step]
            x_t = tape_x[t_step]
            i_t = tape_i[t_step]
            f_t = tape_f[t_step]
            o_t = tape_o[t_step]
            g_t = tape_g[t_step]

            # Head gradients
            dw_out += np.dot(a_dense_t.T, dq_t)
            db_out += np.sum(dq_t, axis=0)

            da_dense = np.dot(dq_t, self.w_out.T)
            dz_dense = da_dense * (z_dense_t > 0.0)

            dw_dense += np.dot(h_t.T, dz_dense)
            db_dense += np.sum(dz_dense, axis=0)

            dh_from_head = np.dot(dz_dense, self.w_dense.T)
            dh_total = dh_from_head + dh_next

            # LSTM cell gradients
            d_tanh_c = dh_total * o_t
            dc = d_tanh_c * (1.0 - tanh_c_t**2) + dc_next

            do = dh_total * tanh_c_t
            di = dc * g_t
            dg = dc * i_t
            df = dc * c_prev_t

            dz_o = do * o_t * (1.0 - o_t)
            dz_i = di * i_t * (1.0 - i_t)
            dz_f = df * f_t * (1.0 - f_t)
            dz_g = dg * (1.0 - g_t**2)

            dz = np.concatenate([dz_i, dz_f, dz_o, dz_g], axis=1)  # [B, 4*H]

            dw_x += np.dot(x_t.T, dz)
            dw_h += np.dot(h_prev_t.T, dz)
            db_lstm += np.sum(dz, axis=0)

            dh_next = np.dot(dz, self.w_h.T)
            dc_next = dc * f_t

        # Gradient clipping
        grads = [dw_x, dw_h, db_lstm, dw_dense, db_dense, dw_out, db_out]
        if self.grad_clip > 0.0:
            grads = [np.clip(g, -self.grad_clip, self.grad_clip) for g in grads]

        # Adam optimizer update
        self.t += 1
        t = self.t

        for p, g, m, v in zip(self.params, grads, self.m_moments, self.v_moments):
            m[:] = self.beta1 * m + (1.0 - self.beta1) * g
            v[:] = self.beta2 * v + (1.0 - self.beta2) * (g**2)
            m_hat = m / (1.0 - self.beta1**t)
            v_hat = v / (1.0 - self.beta2**t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        return mean_loss

    def copy_from(self, source: LSTMQNetwork) -> None:
        """Copies parameters from source network."""
        self.w_x = np.copy(source.w_x)
        self.w_h = np.copy(source.w_h)
        self.b_lstm = np.copy(source.b_lstm)
        self.w_dense = np.copy(source.w_dense)
        self.b_dense = np.copy(source.b_dense)
        self.w_out = np.copy(source.w_out)
        self.b_out = np.copy(source.b_out)
        self.params = [
            self.w_x, self.w_h, self.b_lstm,
            self.w_dense, self.b_dense,
            self.w_out, self.b_out,
        ]

    def clone(self) -> LSTMQNetwork:
        """Creates an identical clone of this network."""
        cloned = LSTMQNetwork(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            dense_dim=self.dense_dim,
            learning_rate=self.lr,
            grad_clip=self.grad_clip,
        )
        cloned.copy_from(self)
        return cloned

    def get_weights_dict(self) -> dict[str, np.ndarray]:
        """Returns a dictionary containing copies of all trainable parameter arrays."""
        return {
            "w_x": np.copy(self.w_x),
            "w_h": np.copy(self.w_h),
            "b_lstm": np.copy(self.b_lstm),
            "w_dense": np.copy(self.w_dense),
            "b_dense": np.copy(self.b_dense),
            "w_out": np.copy(self.w_out),
            "b_out": np.copy(self.b_out),
        }

    def load_weights_dict(self, weights: dict[str, np.ndarray]) -> None:
        """Loads weights from a dictionary, strictly validating keys and shapes."""
        for k in ["w_x", "w_h", "b_lstm", "w_dense", "b_dense", "w_out", "b_out"]:
            if k not in weights:
                raise ValueError(f"Missing required parameter key '{k}' in weights dictionary.")
            current_arr = getattr(self, k)
            loaded_arr = np.asarray(weights[k], dtype=np.float32)
            if current_arr.shape != loaded_arr.shape:
                raise ValueError(
                    f"Parameter '{k}' shape mismatch: current {current_arr.shape} != loaded {loaded_arr.shape}"
                )
            current_arr[:] = loaded_arr

        self.params = [
            self.w_x, self.w_h, self.b_lstm,
            self.w_dense, self.b_dense,
            self.w_out, self.b_out,
        ]
