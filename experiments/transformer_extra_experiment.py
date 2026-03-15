#!/usr/bin/env python3
"""Extra experiment: add Transformer-based regressor and compare with existing models.

This script does NOT modify the main report. It only prints/exports comparison results.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import autograd.numpy as anp
import numpy as np
import pandas as pd
from autograd import grad
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

SEED = 42


class TinyTransformerRegressor:
    """Minimal Transformer regressor using autograd.

    Input: 3 RSSI features as 3 tokens with scalar value each.
    Architecture: token embedding -> self-attention -> FFN -> mean pooling -> 2D regression.
    """

    def __init__(
        self,
        d_model: int = 24,
        d_ff: int = 48,
        epochs: int = 300,
        batch_size: int = 512,
        lr: float = 3e-3,
        l2: float = 1e-4,
        random_state: int = 42,
    ) -> None:
        self.d_model = d_model
        self.d_ff = d_ff
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.l2 = l2
        self.rng = np.random.default_rng(random_state)

    def _init_params(self):
        rs = self.rng
        d = self.d_model
        f = self.d_ff

        def w(shape, scale):
            return rs.normal(0.0, scale, size=shape)

        params = {
            # scalar token embedding: x_token * W_in + b_in
            "W_in": w((d,), np.sqrt(2.0 / max(d, 1))),
            "b_in": np.zeros((d,)),
            "P": w((3, d), 0.05),  # positional embedding
            # attention projections
            "Wq": w((d, d), np.sqrt(2.0 / d)),
            "Wk": w((d, d), np.sqrt(2.0 / d)),
            "Wv": w((d, d), np.sqrt(2.0 / d)),
            "Wo": w((d, d), np.sqrt(2.0 / d)),
            "bo": np.zeros((d,)),
            # feedforward
            "W1": w((d, f), np.sqrt(2.0 / d)),
            "b1": np.zeros((f,)),
            "W2": w((f, d), np.sqrt(2.0 / f)),
            "b2": np.zeros((d,)),
            # output
            "W_out": w((d, 2), np.sqrt(2.0 / d)),
            "b_out": np.zeros((2,)),
        }
        return params

    @staticmethod
    def _softmax(x, axis=-1):
        x_shift = x - anp.max(x, axis=axis, keepdims=True)
        ex = anp.exp(x_shift)
        return ex / anp.sum(ex, axis=axis, keepdims=True)

    @staticmethod
    def _relu(x):
        return anp.maximum(0.0, x)

    def _forward(self, params, x):
        # x: [N, 3]
        # embed scalar token -> vector token
        tok = x[:, :, None] * params["W_in"][None, None, :] + params["b_in"][None, None, :]
        tok = tok + params["P"][None, :, :]

        q = anp.einsum("nld,df->nlf", tok, params["Wq"])
        k = anp.einsum("nld,df->nlf", tok, params["Wk"])
        v = anp.einsum("nld,df->nlf", tok, params["Wv"])

        scores = anp.matmul(q, anp.swapaxes(k, 1, 2)) / anp.sqrt(self.d_model)
        attn = self._softmax(scores, axis=-1)
        ctx = anp.matmul(attn, v)

        ctx_proj = anp.einsum("nld,df->nlf", ctx, params["Wo"]) + params["bo"][None, None, :]
        h1 = tok + ctx_proj

        ff = self._relu(anp.einsum("nld,df->nlf", h1, params["W1"]) + params["b1"][None, None, :])
        ff2 = anp.einsum("nlf,fd->nld", ff, params["W2"]) + params["b2"][None, None, :]
        h2 = h1 + ff2

        pooled = anp.mean(h2, axis=1)
        out = anp.dot(pooled, params["W_out"]) + params["b_out"]
        return out

    def _loss(self, params, xb, yb):
        pred = self._forward(params, xb)
        mse = anp.mean((pred - yb) ** 2)
        l2_term = 0.0
        for k, v in params.items():
            if k.startswith("b"):
                continue
            l2_term = l2_term + anp.sum(v * v)
        return mse + self.l2 * l2_term

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        self.x_mean = x.mean(axis=0)
        self.x_std = x.std(axis=0) + 1e-6
        self.y_mean = y.mean(axis=0)
        self.y_std = y.std(axis=0) + 1e-6

        xn = (x - self.x_mean) / self.x_std
        yn = (y - self.y_mean) / self.y_std

        params = self._init_params()
        m = {k: np.zeros_like(v) for k, v in params.items()}
        v = {k: np.zeros_like(v_) for k, v_ in params.items()}

        grad_fn = grad(self._loss)

        n = len(xn)
        bsz = min(self.batch_size, n)

        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0

        for ep in range(self.epochs):
            idx = self.rng.permutation(n)
            for start in range(0, n, bsz):
                batch_idx = idx[start : start + bsz]
                xb = xn[batch_idx]
                yb = yn[batch_idx]

                grads = grad_fn(params, xb, yb)
                t += 1

                for k in params:
                    g = np.asarray(grads[k])
                    m[k] = beta1 * m[k] + (1.0 - beta1) * g
                    v[k] = beta2 * v[k] + (1.0 - beta2) * (g * g)

                    m_hat = m[k] / (1.0 - beta1**t)
                    v_hat = v[k] / (1.0 - beta2**t)
                    params[k] = params[k] - self.lr * m_hat / (np.sqrt(v_hat) + eps)

            # simple LR decay for stability
            if (ep + 1) % 100 == 0:
                self.lr *= 0.8

        self.params = params
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        xn = (x - self.x_mean) / self.x_std
        pred_n = np.asarray(self._forward(self.params, xn))
        return pred_n * self.y_std + self.y_mean


def evaluate_transformer_on_dataset(df: pd.DataFrame, scenario: str) -> dict:
    x = df[["rssi_ap1", "rssi_ap2", "rssi_ap3"]].to_numpy(dtype=float)
    y = df[["x", "y"]].to_numpy(dtype=float)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=SEED)

    model = TinyTransformerRegressor(
        d_model=24,
        d_ff=48,
        epochs=300,
        batch_size=512,
        lr=3e-3,
        l2=1e-4,
        random_state=SEED,
    )

    t0 = time.perf_counter()
    model.fit(x_train, y_train)
    train_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = model.predict(x_test)
    infer_s = time.perf_counter() - t1

    err = np.linalg.norm(y_pred - y_test, axis=1)

    return {
        "scenario": scenario,
        "model": "Transformer",
        "mae_m": float(np.mean(err)),
        "rmse_m": float(np.sqrt(np.mean(err**2))),
        "median_m": float(np.median(err)),
        "p90_m": float(np.percentile(err, 90)),
        "r2_x": float(r2_score(y_test[:, 0], y_pred[:, 0])),
        "r2_y": float(r2_score(y_test[:, 1], y_pred[:, 1])),
        "train_time_s": float(train_s),
        "infer_ms_per_sample": float(1000.0 * infer_s / len(x_test)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
    }


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "experiments" / "results" / "data"

    base_metrics = pd.read_csv(data_dir / "metrics_summary.csv")

    scenarios = ["ideal_empty", "complex_dynamic"]
    tf_rows = []
    for sc in scenarios:
        df = pd.read_csv(data_dir / f"dataset_{sc}.csv")
        tf_rows.append(evaluate_transformer_on_dataset(df, sc))

    tf_df = pd.DataFrame(tf_rows)
    out_df = pd.concat([base_metrics, tf_df], ignore_index=True)
    out_df.to_csv(data_dir / "metrics_with_transformer.csv", index=False)

    # summary against CNN/LightGBM for quick reading
    summary = []
    for sc in scenarios:
        sub = out_df[out_df["scenario"] == sc].set_index("model")
        tf = sub.loc["Transformer"]
        for ref in ["CNN", "LightGBM", "RandomForest", "WKNN", "KNN"]:
            r = sub.loc[ref]
            summary.append(
                {
                    "scenario": sc,
                    "vs_model": ref,
                    "Transformer_MAE_diff_m": float(tf["mae_m"] - r["mae_m"]),
                    "Transformer_RMSE_diff_m": float(tf["rmse_m"] - r["rmse_m"]),
                    "Transformer_P90_diff_m": float(tf["p90_m"] - r["p90_m"]),
                }
            )

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(data_dir / "transformer_vs_baselines.csv", index=False)

    print("Transformer extra experiment finished.")
    print("Saved:")
    print(data_dir / "metrics_with_transformer.csv")
    print(data_dir / "transformer_vs_baselines.csv")
    print("\nTransformer metrics:")
    print(tf_df.to_string(index=False))


if __name__ == "__main__":
    main()
