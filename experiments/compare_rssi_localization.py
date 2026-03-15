#!/usr/bin/env python3
"""RSSI localization benchmark for KNN / WKNN / RandomForest / LightGBM / CNN.

Scenarios:
1) ideal_empty: enclosed room without inner obstacles/humans
2) complex_dynamic: obstacles + moving human + stronger fading

Outputs:
- experiments/results/data/*.csv
- experiments/results/figures/*.png
- experiments/results/localization_experiment_report_CN.md
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
rng = np.random.default_rng(SEED)

C = 299_792_458.0
ROOM_DIMS = np.array([10.0, 10.0, 3.0])
RX_Z = 1.2

AP_LIST = [
    {"name": "AP1", "pos": np.array([1.2, 1.2, 2.7]), "tx_dbm": 18.0, "f": 2.412e9, "bw": 20e6},
    {"name": "AP2", "pos": np.array([8.8, 1.5, 2.7]), "tx_dbm": 17.0, "f": 2.422e9, "bw": 20e6},
    {"name": "AP3", "pos": np.array([5.2, 8.8, 2.7]), "tx_dbm": 19.0, "f": 2.437e9, "bw": 20e6},
]

WALL_REFLECTION_COEFF = 0.65
WALL_ABSORPTION_COEFF = 0.20
MODEL_ORDER = ["KNN", "WKNN", "RandomForest", "LightGBM", "CNN"]


@dataclass
class Obstacle:
    min_xy: Tuple[float, float]
    max_xy: Tuple[float, float]
    attenuation_db_per_m: float
    absorption_coeff: float


@dataclass
class ScenarioConfig:
    name: str
    path_loss_exp: float
    shadow_sigma_db: float
    fading: str
    rician_k_db: float
    nlos_penalty_db: float
    diffraction_penalty_db: float
    obstacles: List[Obstacle]
    enable_human: bool
    human_att_db_per_m: float
    noise_dbm: float
    bg_interf_dbm: float
    t_vec: np.ndarray


def dbm_to_mw(x_dbm: np.ndarray | float) -> np.ndarray | float:
    return 10.0 ** (np.asarray(x_dbm) / 10.0)


def mw_to_dbm(x_mw: np.ndarray | float) -> np.ndarray | float:
    x = np.maximum(np.asarray(x_mw), np.finfo(float).tiny)
    return 10.0 * np.log10(x)


def fspl_1m_db(f_hz: float) -> float:
    lam = C / f_hz
    return 20.0 * math.log10(4.0 * math.pi / lam)


def channel_overlap(a: Dict, b: Dict) -> float:
    delta_f = abs(a["f"] - b["f"])
    shared_bw = 0.5 * (a["bw"] + b["bw"])
    return float(max(0.0, 1.0 - delta_f / shared_bw))


def segment_rect_intersection_length_2d(
    p1: np.ndarray, p2: np.ndarray, rect_min: Tuple[float, float], rect_max: Tuple[float, float]
) -> float:
    d = p2 - p1
    tmin, tmax = 0.0, 1.0

    bounds = [(rect_min[0], rect_max[0]), (rect_min[1], rect_max[1])]
    for i in range(2):
        if abs(d[i]) < 1e-12:
            if p1[i] < bounds[i][0] or p1[i] > bounds[i][1]:
                return 0.0
        else:
            t1 = (bounds[i][0] - p1[i]) / d[i]
            t2 = (bounds[i][1] - p1[i]) / d[i]
            t_near, t_far = min(t1, t2), max(t1, t2)
            tmin = max(tmin, t_near)
            tmax = min(tmax, t_far)
            if tmin > tmax:
                return 0.0

    seg_len = np.linalg.norm(d)
    return max(0.0, (tmax - tmin) * seg_len)


def segment_circle_intersection_length_2d(
    p1: np.ndarray, p2: np.ndarray, center: np.ndarray, radius: float
) -> float:
    d = p2 - p1
    a = np.dot(d, d)
    if a < 1e-12:
        return 0.0

    f = p1 - center
    b = 2.0 * np.dot(f, d)
    c = np.dot(f, f) - radius**2
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return 0.0

    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)

    t_start = max(0.0, min(t1, t2))
    t_end = min(1.0, max(t1, t2))
    if t_end <= t_start:
        return 0.0
    return (t_end - t_start) * np.linalg.norm(d)


def human_position(t: float) -> np.ndarray:
    # Looping path inside the room
    waypoints = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [6.5, 4.5],
            [8.5, 7.5],
            [3.0, 8.2],
            [1.0, 2.0],
        ]
    )
    ts = np.linspace(0.0, 30.0, len(waypoints))

    t_eff = float(np.mod(t, ts[-1]))
    x = np.interp(t_eff, ts, waypoints[:, 0])
    y = np.interp(t_eff, ts, waypoints[:, 1])
    return np.array([x, y])


def reflection_paths(ap_pos: np.ndarray, rx_pos: np.ndarray, f_hz: float, n: float) -> Tuple[np.ndarray, np.ndarray]:
    # First-order image sources on x=0,x=L,y=0,y=W
    images = [
        np.array([-ap_pos[0], ap_pos[1], ap_pos[2]]),
        np.array([2 * ROOM_DIMS[0] - ap_pos[0], ap_pos[1], ap_pos[2]]),
        np.array([ap_pos[0], -ap_pos[1], ap_pos[2]]),
        np.array([ap_pos[0], 2 * ROOM_DIMS[1] - ap_pos[1], ap_pos[2]]),
    ]
    refl_gain = max(1e-3, WALL_REFLECTION_COEFF * (1.0 - WALL_ABSORPTION_COEFF))
    refl_loss_db = -20.0 * math.log10(refl_gain)
    fspl1m = fspl_1m_db(f_hz)

    dists = []
    powers_lin = []
    for img in images:
        d = np.linalg.norm(rx_pos - img)
        d = max(d, 0.5)
        pl = fspl1m + 10.0 * n * math.log10(d)
        # Reflections generally weaker and more lossy than LOS
        p_ref_dbm = -pl - refl_loss_db - 2.0
        dists.append(d)
        powers_lin.append(10.0 ** (p_ref_dbm / 10.0))

    return np.array(dists), np.array(powers_lin)


def small_scale_gain(model: str, rician_k_db: float) -> float:
    if model == "rayleigh":
        h = (rng.standard_normal() + 1j * rng.standard_normal()) / math.sqrt(2.0)
        return float(abs(h) ** 2)

    if model == "rician":
        k = 10.0 ** (rician_k_db / 10.0)
        h = math.sqrt(k / (k + 1.0)) + math.sqrt(1.0 / (2.0 * (k + 1.0))) * (
            rng.standard_normal() + 1j * rng.standard_normal()
        )
        return float(abs(h) ** 2)

    return 1.0


def compute_ap_signal_dbm(
    ap: Dict,
    rx_pos: np.ndarray,
    sc: ScenarioConfig,
    t: float,
) -> float:
    tx_pos = ap["pos"]
    d = np.linalg.norm(rx_pos - tx_pos)
    d = max(d, 0.5)

    fspl1m = fspl_1m_db(ap["f"])
    pl_los = fspl1m + 10.0 * sc.path_loss_exp * math.log10(d)

    # Obstacle attenuation in 2D projection
    obs_loss = 0.0
    blocked = False
    for obs in sc.obstacles:
        li = segment_rect_intersection_length_2d(tx_pos[:2], rx_pos[:2], obs.min_xy, obs.max_xy)
        if li > 0.0:
            blocked = blocked or li > 0.03
            obs_loss += li * obs.attenuation_db_per_m + 1.5 * li * obs.absorption_coeff

    # Human attenuation
    human_loss = 0.0
    if sc.enable_human:
        hxy = human_position(t)
        lh = segment_circle_intersection_length_2d(tx_pos[:2], rx_pos[:2], hxy, radius=0.28)
        if lh > 0.0:
            blocked = True
            human_loss += lh * sc.human_att_db_per_m

    nlos_loss = (sc.nlos_penalty_db + sc.diffraction_penalty_db) if blocked else 0.0
    shadow = sc.shadow_sigma_db * rng.standard_normal()

    p_los_dbm = ap["tx_dbm"] - pl_los - obs_loss - human_loss - nlos_loss + shadow
    p_los_lin = float(dbm_to_mw(p_los_dbm))

    # Multipath reflection with frequency-selective averaging
    d_ref, p_ref_norm_lin = reflection_paths(tx_pos, rx_pos, ap["f"], sc.path_loss_exp)
    p_ref_lin = p_ref_norm_lin * dbm_to_mw(ap["tx_dbm"])  # scale by Tx power

    sub_offsets = np.linspace(-10e6, 10e6, 8)
    p_sub = []
    for df in sub_offsets:
        f = ap["f"] + df
        e = math.sqrt(p_los_lin) * np.exp(-1j * 2.0 * math.pi * f * d / C)
        for dk, pk in zip(d_ref, p_ref_lin):
            e += math.sqrt(max(pk, 0.0)) * np.exp(-1j * 2.0 * math.pi * f * dk / C)
        p_sub.append(abs(e) ** 2)

    p_combined = float(np.mean(p_sub))
    p_final = p_combined * small_scale_gain(sc.fading, sc.rician_k_db)

    return float(mw_to_dbm(max(p_final, np.finfo(float).tiny)))


def build_scenario(name: str) -> ScenarioConfig:
    t_vec = np.arange(0.0, 30.0 + 1e-9, 1.0)

    if name == "ideal_empty":
        return ScenarioConfig(
            name="ideal_empty",
            path_loss_exp=2.0,
            shadow_sigma_db=1.8,
            fading="rician",
            rician_k_db=10.0,
            nlos_penalty_db=0.0,
            diffraction_penalty_db=0.0,
            obstacles=[],
            enable_human=False,
            human_att_db_per_m=0.0,
            noise_dbm=-96.0,
            bg_interf_dbm=-105.0,
            t_vec=t_vec,
        )

    if name == "complex_dynamic":
        return ScenarioConfig(
            name="complex_dynamic",
            path_loss_exp=2.3,
            shadow_sigma_db=3.5,
            fading="rician",
            rician_k_db=4.0,
            nlos_penalty_db=6.0,
            diffraction_penalty_db=4.0,
            obstacles=[
                Obstacle((4.5, 1.0), (4.7, 8.5), attenuation_db_per_m=13.5, absorption_coeff=0.30),
                Obstacle((7.0, 6.5), (8.4, 8.6), attenuation_db_per_m=5.5, absorption_coeff=0.42),
            ],
            enable_human=True,
            human_att_db_per_m=7.5,
            noise_dbm=-95.0,
            bg_interf_dbm=-100.0,
            t_vec=t_vec,
        )

    raise ValueError(f"Unknown scenario: {name}")


def simulate_dataset(sc: ScenarioConfig) -> pd.DataFrame:
    x_vec = np.arange(0.0, ROOM_DIMS[0] + 1e-9, 0.5)
    y_vec = np.arange(0.0, ROOM_DIMS[1] + 1e-9, 0.5)

    rows = []
    noise_lin = dbm_to_mw(sc.noise_dbm) + dbm_to_mw(sc.bg_interf_dbm)

    for t in sc.t_vec:
        for x in x_vec:
            for y in y_vec:
                rx = np.array([x, y, RX_Z])

                sig_dbm = np.array([compute_ap_signal_dbm(ap, rx, sc, t) for ap in AP_LIST], dtype=float)
                sig_lin = dbm_to_mw(sig_dbm)

                eff_dbm = np.zeros_like(sig_dbm)
                sinr_db = np.zeros_like(sig_dbm)

                for i, ap_i in enumerate(AP_LIST):
                    interf = 0.0
                    for j, ap_j in enumerate(AP_LIST):
                        if i == j:
                            continue
                        interf += channel_overlap(ap_i, ap_j) * sig_lin[j]

                    total_in = interf + noise_lin
                    sinr = sig_lin[i] / max(total_in, np.finfo(float).tiny)
                    sinr_db[i] = 10.0 * np.log10(max(sinr, np.finfo(float).tiny))

                    penalty = 10.0 * np.log10(1.0 + interf / max(sig_lin[i], np.finfo(float).tiny))
                    eff_dbm[i] = sig_dbm[i] - penalty

                rows.append(
                    {
                        "scenario": sc.name,
                        "time_s": t,
                        "x": x,
                        "y": y,
                        "z": RX_Z,
                        "rssi_ap1": eff_dbm[0],
                        "rssi_ap2": eff_dbm[1],
                        "rssi_ap3": eff_dbm[2],
                        "sinr_ap1": sinr_db[0],
                        "sinr_ap2": sinr_db[1],
                        "sinr_ap3": sinr_db[2],
                    }
                )

    df = pd.DataFrame(rows)
    return df


class SimpleCNNRegressor:
    """Lightweight 1D CNN regressor implemented in NumPy.

    Architecture:
    - Input shape: (N, 3) RSSI features
    - Conv1D: n_filters kernels, kernel_size=2, valid padding
    - ReLU
    - Flatten
    - Dense(hidden_units) + ReLU
    - Dense(2) for (x, y) regression
    """

    def __init__(
        self,
        n_filters: int = 16,
        hidden_units: int = 32,
        epochs: int = 120,
        lr: float = 1e-2,
        batch_size: int = 256,
        l2: float = 1e-4,
        random_state: int = 42,
    ) -> None:
        self.n_filters = n_filters
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.l2 = l2
        self.rng = np.random.default_rng(random_state)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    def _init_params(self) -> None:
        conv_in = 2
        flat_dim = self.n_filters * 2  # valid conv on len=3 with k=2 -> 2 positions

        self.Wc = self.rng.normal(0.0, np.sqrt(2.0 / conv_in), size=(self.n_filters, conv_in))
        self.bc = np.zeros(self.n_filters)
        self.W1 = self.rng.normal(0.0, np.sqrt(2.0 / flat_dim), size=(flat_dim, self.hidden_units))
        self.b1 = np.zeros(self.hidden_units)
        self.W2 = self.rng.normal(0.0, np.sqrt(2.0 / self.hidden_units), size=(self.hidden_units, 2))
        self.b2 = np.zeros(2)

        # Adam optimizer states
        self.m = {
            "Wc": np.zeros_like(self.Wc),
            "bc": np.zeros_like(self.bc),
            "W1": np.zeros_like(self.W1),
            "b1": np.zeros_like(self.b1),
            "W2": np.zeros_like(self.W2),
            "b2": np.zeros_like(self.b2),
        }
        self.v = {
            "Wc": np.zeros_like(self.Wc),
            "bc": np.zeros_like(self.bc),
            "W1": np.zeros_like(self.W1),
            "b1": np.zeros_like(self.b1),
            "W2": np.zeros_like(self.W2),
            "b2": np.zeros_like(self.b2),
        }
        self.t_step = 0

    def _forward(self, x: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
        # Two valid conv patches from 3-length input
        p0 = x[:, 0:2]
        p1 = x[:, 1:3]

        z0 = p0 @ self.Wc.T + self.bc
        z1 = p1 @ self.Wc.T + self.bc
        zc = np.stack([z0, z1], axis=2)  # [N, F, 2]
        ac = self._relu(zc)

        flat = ac.reshape(x.shape[0], -1)
        zh = flat @ self.W1 + self.b1
        ah = self._relu(zh)
        out = ah @ self.W2 + self.b2

        cache = (p0, p1, zc, flat, zh, ah)
        return out, cache

    def _backward(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        cache: Tuple[np.ndarray, ...],
    ) -> Dict[str, np.ndarray]:
        p0, p1, zc, flat, zh, ah = cache
        n = y_true.shape[0]

        d_out = (2.0 / n) * (y_pred - y_true)

        gW2 = ah.T @ d_out + self.l2 * self.W2
        gb2 = d_out.sum(axis=0)

        d_ah = d_out @ self.W2.T
        d_zh = d_ah * (zh > 0)

        gW1 = flat.T @ d_zh + self.l2 * self.W1
        gb1 = d_zh.sum(axis=0)

        d_flat = d_zh @ self.W1.T
        d_ac = d_flat.reshape(n, self.n_filters, 2)
        d_zc = d_ac * (zc > 0)

        d_z0 = d_zc[:, :, 0]
        d_z1 = d_zc[:, :, 1]

        gWc = d_z0.T @ p0 + d_z1.T @ p1 + self.l2 * self.Wc
        gbc = d_z0.sum(axis=0) + d_z1.sum(axis=0)

        return {"Wc": gWc, "bc": gbc, "W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}

    def _adam_step(self, grads: Dict[str, np.ndarray]) -> None:
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        self.t_step += 1

        for k in grads:
            g = grads[k]
            self.m[k] = beta1 * self.m[k] + (1.0 - beta1) * g
            self.v[k] = beta2 * self.v[k] + (1.0 - beta2) * (g * g)

            m_hat = self.m[k] / (1.0 - beta1**self.t_step)
            v_hat = self.v[k] / (1.0 - beta2**self.t_step)
            update = self.lr * m_hat / (np.sqrt(v_hat) + eps)

            setattr(self, k, getattr(self, k) - update)

    def fit(self, x: np.ndarray | pd.DataFrame, y: np.ndarray) -> "SimpleCNNRegressor":
        x_np = np.asarray(x, dtype=float)
        y_np = np.asarray(y, dtype=float)

        self.x_mean = x_np.mean(axis=0)
        self.x_std = x_np.std(axis=0) + 1e-6
        self.y_mean = y_np.mean(axis=0)
        self.y_std = y_np.std(axis=0) + 1e-6

        xn = (x_np - self.x_mean) / self.x_std
        yn = (y_np - self.y_mean) / self.y_std

        self._init_params()

        n = xn.shape[0]
        batch = min(self.batch_size, n)

        for _ in range(self.epochs):
            idx_all = self.rng.permutation(n)
            for start in range(0, n, batch):
                idx = idx_all[start : start + batch]
                xb = xn[idx]
                yb = yn[idx]

                pred, cache = self._forward(xb)
                grads = self._backward(yb, pred, cache)
                self._adam_step(grads)

        return self

    def predict(self, x: np.ndarray | pd.DataFrame) -> np.ndarray:
        x_np = np.asarray(x, dtype=float)
        xn = (x_np - self.x_mean) / self.x_std
        pred_n, _ = self._forward(xn)
        return pred_n * self.y_std + self.y_mean


def build_models() -> Dict[str, object]:
    models = {
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", KNeighborsRegressor(n_neighbors=5, weights="uniform")),
            ]
        ),
        "WKNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", KNeighborsRegressor(n_neighbors=5, weights="distance")),
            ]
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=SEED,
        ),
        "LightGBM": MultiOutputRegressor(
            LGBMRegressor(
                objective="regression",
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=SEED,
                n_jobs=-1,
                verbosity=-1,
            )
        ),
        "CNN": SimpleCNNRegressor(
            n_filters=16,
            hidden_units=32,
            epochs=120,
            lr=0.01,
            batch_size=256,
            l2=1e-4,
            random_state=SEED,
        ),
    }
    return models


def evaluate_models_for_scenario(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    x_cols = ["rssi_ap1", "rssi_ap2", "rssi_ap3"]
    y_cols = ["x", "y"]

    # Keep DataFrame for stable feature-name handling in LightGBM.
    x_data = df[x_cols]
    y_data = df[y_cols].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.3, random_state=SEED
    )

    metrics_rows = []
    raw = {}

    for model_name, model in build_models().items():
        t0 = time.perf_counter()
        model.fit(x_train, y_train)
        train_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = model.predict(x_test)
        infer_s = time.perf_counter() - t1

        err = np.linalg.norm(y_pred - y_test, axis=1)

        row = {
            "model": model_name,
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
        metrics_rows.append(row)

        raw[model_name] = {
            "y_test": y_test,
            "y_pred": y_pred,
            "err": err,
        }

    return pd.DataFrame(metrics_rows), raw


def save_heatmap(df: pd.DataFrame, out_file: Path, title: str) -> None:
    snap = df[df["time_s"] == df["time_s"].min()].copy()
    pivot = snap.pivot(index="y", columns="x", values="rssi_ap1").sort_index(ascending=True)

    plt.figure(figsize=(7, 5.5))
    plt.imshow(
        pivot.values,
        origin="lower",
        extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()],
        aspect="auto",
        cmap="turbo",
    )
    plt.colorbar(label="RSSI AP1 (dBm)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()


def save_metrics_plot(metrics: pd.DataFrame, out_file: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    metrics = metrics.copy()
    available_models = list(metrics["model"].unique())
    model_order = [m for m in MODEL_ORDER if m in available_models] + [
        m for m in available_models if m not in MODEL_ORDER
    ]

    for ax, metric, ylabel in zip(
        axes,
        ["mae_m", "rmse_m"],
        ["Mean Localization Error (m)", "RMSE (m)"],
    ):
        pivot = metrics.pivot(index="model", columns="scenario", values=metric)
        pivot = pivot.reindex(model_order)

        x = np.arange(len(pivot.index))
        scenarios = list(pivot.columns)
        width = min(0.36, 0.8 / max(len(scenarios), 1))
        offsets = (np.arange(len(scenarios)) - (len(scenarios) - 1) / 2.0) * width

        for i, sc in enumerate(scenarios):
            ax.bar(x + offsets[i], pivot[sc].values, width=width, label=sc)

        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=0)
        ax.set_ylabel(ylabel)
        ax.set_title(metric.upper())
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend()

    fig.suptitle("Localization Accuracy Comparison")
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def save_cdf_plots_by_scenario(
    error_bank: Dict[str, Dict[str, np.ndarray]], fig_dir: Path
) -> Dict[str, str]:
    out_files: Dict[str, str] = {}

    for scenario, models in error_bank.items():
        plt.figure(figsize=(8.2, 5.8))

        for model_name in [m for m in MODEL_ORDER if m in models] + [m for m in models if m not in MODEL_ORDER]:
            payload = models[model_name]
            err = np.sort(payload["err"])
            y = np.arange(1, len(err) + 1) / len(err)
            plt.plot(err, y, label=model_name)

        plt.xlabel("Localization Error (m)")
        plt.ylabel("CDF")
        plt.title(f"Error CDF - {scenario}")
        plt.grid(alpha=0.3, linestyle="--")
        plt.legend(fontsize=8)
        plt.tight_layout()

        file_name = f"error_cdf_{scenario}.png"
        plt.savefig(fig_dir / file_name, dpi=180)
        plt.close()
        out_files[scenario] = file_name

    return out_files


def save_complex_scatter(raw_complex: Dict[str, Dict[str, np.ndarray]], out_file: Path) -> None:
    names = [m for m in MODEL_ORDER if m in raw_complex] + [m for m in raw_complex if m not in MODEL_ORDER]
    n = len(names)
    ncols = 3 if n >= 3 else n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.4 * nrows), sharex=True, sharey=True)
    axes_arr = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes_arr, names):
        y_test = raw_complex[name]["y_test"]
        y_pred = raw_complex[name]["y_pred"]

        idx = np.arange(len(y_test))
        if len(idx) > 1600:
            idx = rng.choice(idx, size=1600, replace=False)

        ax.scatter(y_test[idx, 0], y_test[idx, 1], s=8, alpha=0.25, label="True")
        ax.scatter(y_pred[idx, 0], y_pred[idx, 1], s=8, alpha=0.25, label="Pred")
        ax.set_title(name)
        ax.set_xlabel("X (m)")
        ax.grid(alpha=0.25, linestyle="--")

    for ax in axes_arr[len(names) :]:
        ax.axis("off")

    axes_arr[0].set_ylabel("Y (m)")
    axes_arr[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Complex Scenario: True vs Predicted Positions")
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def build_target_gain_df(metrics: pd.DataFrame, target_model: str) -> pd.DataFrame:
    rows = []
    for sc_name in metrics["scenario"].unique():
        sub = metrics[metrics["scenario"] == sc_name].set_index("model")
        if target_model not in sub.index:
            continue

        for model in sub.index:
            if model == target_model:
                continue

            row = {
                "scenario": sc_name,
                "baseline_model": model,
                "target_model": target_model,
                "mae_improvement_pct": 100.0
                * (sub.loc[model, "mae_m"] - sub.loc[target_model, "mae_m"])
                / sub.loc[model, "mae_m"],
                "rmse_improvement_pct": 100.0
                * (sub.loc[model, "rmse_m"] - sub.loc[target_model, "rmse_m"])
                / sub.loc[model, "rmse_m"],
                "p90_improvement_pct": 100.0
                * (sub.loc[model, "p90_m"] - sub.loc[target_model, "p90_m"])
                / sub.loc[model, "p90_m"],
            }
            rows.append(row)
    return pd.DataFrame(rows)


def save_target_gain_plot(gain_df: pd.DataFrame, out_file: Path) -> None:
    if gain_df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6))
    metrics_cols = [
        ("mae_improvement_pct", "MAE Improvement"),
        ("rmse_improvement_pct", "RMSE Improvement"),
        ("p90_improvement_pct", "P90 Improvement"),
    ]

    scenario_alias = {"ideal_empty": "ideal", "complex_dynamic": "complex"}
    labels = [
        f"{scenario_alias.get(str(r['scenario']), str(r['scenario']))}-{str(r['baseline_model'])}"
        for _, r in gain_df.iterrows()
    ]
    x = np.arange(len(gain_df))
    colors = plt.cm.Set2(np.linspace(0, 1, len(gain_df)))

    for ax, (col, title) in zip(axes, metrics_cols):
        vals = gain_df[col].values
        bars = ax.bar(x, vals, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=25, ha="right")
        ax.set_ylabel("Improvement (%)")
        ax.set_title(title)
        ax.grid(alpha=0.25, linestyle="--")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.08, f"{v:.2f}%", ha="center", va="bottom", fontsize=8)

    target_model = str(gain_df["target_model"].iloc[0])
    fig.suptitle(f"Relative Gains of {target_model} Over Baselines")
    fig.tight_layout(rect=[0, 0.10, 1, 0.95])
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, cols: List[str]) -> str:
    rows = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def write_report(
    report_file: Path,
    metrics: pd.DataFrame,
    best_summary: Dict[str, Dict[str, str]],
    gain_df: pd.DataFrame,
    cdf_files: Dict[str, str],
    target_model: str,
) -> None:
    metrics_sorted = metrics.sort_values(["scenario", "mae_m", "rmse_m"]).reset_index(drop=True)
    table = markdown_table(
        metrics_sorted,
        [
            "scenario",
            "model",
            "mae_m",
            "rmse_m",
            "median_m",
            "p90_m",
            "r2_x",
            "r2_y",
            "train_time_s",
            "infer_ms_per_sample",
        ],
    )
    if gain_df.empty:
        gain_table = f"{target_model}增益表为空（未检测到可对比基线）。"
    else:
        gain_table = markdown_table(
            gain_df.sort_values(["scenario", "mae_improvement_pct"], ascending=[True, False]).reset_index(drop=True),
            [
                "scenario",
                "target_model",
                "baseline_model",
                "mae_improvement_pct",
                "rmse_improvement_pct",
                "p90_improvement_pct",
            ],
        )

    ranking_lines = []
    for sc_name in metrics_sorted["scenario"].unique():
        sub = metrics_sorted[metrics_sorted["scenario"] == sc_name].sort_values("mae_m").reset_index(drop=True)
        ranking_lines.append(f"### {sc_name}")
        for i, (_, row) in enumerate(sub.iterrows(), start=1):
            ranking_lines.append(
                f"{i}. {row['model']}  | MAE={row['mae_m']:.4f}m, RMSE={row['rmse_m']:.4f}m, P90={row['p90_m']:.4f}m"
            )
        ranking_lines.append("")
    ranking_block = "\n".join(ranking_lines).strip()
    rec_ideal = best_summary["ideal_empty"]["best_mae_model"]
    rec_complex = best_summary["complex_dynamic"]["best_mae_model"]
    cdf_ideal = cdf_files.get("ideal_empty", "error_cdf_ideal_empty.png")
    cdf_complex = cdf_files.get("complex_dynamic", "error_cdf_complex_dynamic.png")

    txt = f"""# RSSI定位算法对比实验报告（KNN / WKNN / RandomForest / LightGBM / CNN）

## 1. 实验目标

对比五种基于RSSI指纹的定位算法在两类环境下的表现：
- 场景A（ideal_empty）：空旷密闭空间（无障碍、无人）
- 场景B（complex_dynamic）：含障碍物与移动行人的复杂空间

目标是分析不同模型在精度、鲁棒性、训练成本、推理效率方面的差异，并验证CNN在复杂RSSI分布下的优势是否稳定。

## 2. 仿真与数据构建方法

### 2.1 场景与AP设置
- 房间尺寸：10m x 10m x 3m
- AP数量：3个（2.4GHz，部分频谱重叠）
- 采样网格：0.5m间隔（二维定位，接收高度固定1.2m）
- 时间步：0~30s，步长1s

### 2.2 RSSI信道模型（核心）
每条AP-点链路采用：
1. FSPL + 对数距离路径损耗
2. 墙面一阶反射（镜像源）并做复数场叠加
3. 阴影衰落（高斯dB）
4. 小尺度衰落（Rician）
5. 复杂场景中叠加障碍物穿透损耗、人体遮挡损耗与NLOS惩罚
6. AP间按频谱重叠系数计算干扰，得到有效RSSI

### 2.3 特征与标签
- 特征：`[rssi_ap1, rssi_ap2, rssi_ap3]`
- 标签：`[x, y]`
- 划分：随机 70% 训练 / 30% 测试（固定随机种子）

## 3. 算法配置
- KNN：`k=5`, 均匀权重，输入标准化
- WKNN：`k=5`, 距离加权，输入标准化
- RandomForest：300棵树
- LightGBM：500棵树，学习率0.05，`num_leaves=31`，多输出回归封装
- CNN：NumPy实现的轻量1D-CNN回归器（Conv1D-ReLU-MLP），120 epochs

## 4. 结果总表

{table}

## 5. CNN相对增益（对各基线）

{gain_table}

## 6. 插图与对比依据

### 6.1 精度柱状图（MAE / RMSE）
![accuracy](figures/accuracy_bar.png)

### 6.2 定位误差CDF（简单场景）
![cdf_ideal](figures/{cdf_ideal})

### 6.3 定位误差CDF（复杂场景）
![cdf_complex](figures/{cdf_complex})

### 6.4 复杂场景预测散点对比
![scatter](figures/complex_scatter.png)

### 6.5 RSSI场分布示意（AP1）
![ideal](figures/ideal_rssi_heatmap.png)
![complex](figures/complex_rssi_heatmap.png)

### 6.6 CNN相对提升图
![cnn_gain](figures/cnn_gain.png)

## 7. 结果解读

### 7.1 空旷场景（ideal_empty）
- 最优精度（MAE）：**{best_summary['ideal_empty']['best_mae_model']}**
- 最优鲁棒性（P90误差最低）：**{best_summary['ideal_empty']['best_p90_model']}**
- 最快推理：**{best_summary['ideal_empty']['fastest_infer_model']}**

空旷场景中RSSI空间结构较平滑，KNN通常能给出很强的局部匹配；
树模型与CNN均可学习非线性边界，但在几何更规则区域提升幅度可能有限。

### 7.2 复杂场景（complex_dynamic）
- 最优精度（MAE）：**{best_summary['complex_dynamic']['best_mae_model']}**
- 最优鲁棒性（P90误差最低）：**{best_summary['complex_dynamic']['best_p90_model']}**
- 最快推理：**{best_summary['complex_dynamic']['fastest_infer_model']}**

复杂场景中障碍物和人体引入非平稳、非线性扰动；
在本实验设置中，树模型与CNN在抗扰动能力与尾部误差控制上整体优于KNN系列方法。

## 8. 各场景按MAE排名

{ranking_block}

## 9. 各算法多维表现结论

### 9.1 KNN
- 优点：实现简单、在空旷场景可达到较高精度
- 缺点：对噪声和分布漂移敏感；样本量增加时推理成本上升

### 9.2 WKNN
- 优点：较KNN更关注近邻样本，通常能改善复杂场景精度
- 缺点：仍受局部噪声影响，样本规模增大时推理仍偏慢

### 9.3 RandomForest
- 优点：鲁棒性强，对复杂非线性关系拟合稳定
- 缺点：模型体积和训练时间较高

### 9.4 LightGBM
- 优点：在复杂场景下通常兼顾高精度与较低推理时延
- 缺点：需要参数调优；小数据下可能与RF接近

### 9.5 CNN
- 优点：可端到端学习非线性映射，便于未来接入更高维特征（如CSI/时序）
- 缺点：训练参数较多、需要调参；训练耗时通常高于KNN/WKNN

## 10. 工程建议

- 空旷/稳定环境：优先 **{rec_ideal}**（本次实验MAE最优）
- 复杂/动态环境：优先 **{rec_complex}**（本次实验MAE最优）
- 若需稳定传统树模型基线：优先 LightGBM，其次 RandomForest
- 实际部署建议使用在线校准（滑动窗口重训练/增量更新）以应对人体动态变化

## 11. 可复现实验命令

```bash
python3 experiments/compare_rssi_localization.py
```

脚本会自动生成：
- `experiments/results/data/metrics_summary.csv`
- `experiments/results/data/cnn_gain_summary.csv`
- `experiments/results/figures/*.png`
- `experiments/results/localization_experiment_report_CN.md`
"""

    report_file.write_text(txt, encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "experiments" / "results"
    fig_dir = out_dir / "figures"
    data_dir = out_dir / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    target_model = "CNN"
    scenarios = [build_scenario("ideal_empty"), build_scenario("complex_dynamic")]

    metrics_all = []
    error_bank: Dict[str, Dict[str, np.ndarray]] = {}

    for sc in scenarios:
        df = simulate_dataset(sc)
        df.to_csv(data_dir / f"dataset_{sc.name}.csv", index=False)

        metrics, raw = evaluate_models_for_scenario(df)
        metrics["scenario"] = sc.name
        metrics_all.append(metrics)
        error_bank[sc.name] = raw

        # RSSI heatmap for visual evidence
        save_heatmap(df, fig_dir / f"{sc.name.split('_')[0]}_rssi_heatmap.png", f"{sc.name}: RSSI heatmap (AP1)")

    metrics_df = pd.concat(metrics_all, ignore_index=True)
    metrics_df = metrics_df[
        [
            "scenario",
            "model",
            "mae_m",
            "rmse_m",
            "median_m",
            "p90_m",
            "r2_x",
            "r2_y",
            "train_time_s",
            "infer_ms_per_sample",
            "n_train",
            "n_test",
        ]
    ]
    metrics_df.to_csv(data_dir / "metrics_summary.csv", index=False)

    save_metrics_plot(metrics_df, fig_dir / "accuracy_bar.png")
    cdf_files = save_cdf_plots_by_scenario(error_bank, fig_dir)
    save_complex_scatter(error_bank["complex_dynamic"], fig_dir / "complex_scatter.png")
    gain_df = build_target_gain_df(metrics_df, target_model=target_model)
    gain_df.to_csv(data_dir / "cnn_gain_summary.csv", index=False)
    save_target_gain_plot(gain_df, fig_dir / "cnn_gain.png")

    best_summary = {}
    for sc_name in metrics_df["scenario"].unique():
        sub = metrics_df[metrics_df["scenario"] == sc_name]
        best_summary[sc_name] = {
            "best_mae_model": sub.loc[sub["mae_m"].idxmin(), "model"],
            "best_p90_model": sub.loc[sub["p90_m"].idxmin(), "model"],
            "fastest_infer_model": sub.loc[sub["infer_ms_per_sample"].idxmin(), "model"],
        }

    (data_dir / "best_summary.json").write_text(
        json.dumps(best_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_file = out_dir / "localization_experiment_report_CN.md"
    write_report(report_file, metrics_df, best_summary, gain_df, cdf_files, target_model)

    print("Experiment finished.")
    print(f"Metrics: {data_dir / 'metrics_summary.csv'}")
    print(f"Report : {report_file}")


if __name__ == "__main__":
    main()
