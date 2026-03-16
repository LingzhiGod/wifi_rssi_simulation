#!/usr/bin/env python3
"""Noise/outlier robustness experiment for RSSI localization models.

Experiment design:
- Use clean test set.
- Inject random outliers only into training set.
- Compare model degradation under different contamination ratios.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Reuse model builders and constants from existing experiment script
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.compare_rssi_localization import MODEL_ORDER, SEED, build_models

ROOM_BOUNDS = np.array([[0.0, 10.0], [0.0, 10.0]])
CONTAM_LEVELS = [0.00, 0.02, 0.05, 0.10, 0.20]
SCENARIOS = ["ideal_empty", "complex_dynamic"]


def inject_training_outliers(
    x_train: np.ndarray,
    y_train: np.ndarray,
    ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject feature and label outliers into training set.

    Outlier mechanism:
    - 70%: RSSI feature corruption (large additive noise + occasional random reset)
    - 30%: label corruption (position shifted or randomized in room)
    """
    x_noisy = x_train.copy()
    y_noisy = y_train.copy()

    n = len(x_noisy)
    n_out = int(round(ratio * n))
    if n_out <= 0:
        return x_noisy, y_noisy

    idx = rng.choice(n, size=n_out, replace=False)
    n_feat = int(round(0.7 * n_out))
    feat_idx = idx[:n_feat]
    lab_idx = idx[n_feat:]

    if len(feat_idx) > 0:
        # Heavy-tail like additive noise on RSSI (dB)
        noise = rng.standard_t(df=2.5, size=(len(feat_idx), x_noisy.shape[1])) * 8.0
        x_noisy[feat_idx] = x_noisy[feat_idx] + noise

        # 35% of corrupted samples: replace one AP value with random implausible value
        n_reset = int(round(0.35 * len(feat_idx)))
        if n_reset > 0:
            ridx = rng.choice(feat_idx, size=n_reset, replace=False)
            ap_col = rng.integers(0, x_noisy.shape[1], size=n_reset)
            x_noisy[ridx, ap_col] = rng.uniform(-110.0, -15.0, size=n_reset)

        # Clamp to physical-ish RSSI range
        x_noisy[feat_idx] = np.clip(x_noisy[feat_idx], -120.0, -5.0)

    if len(lab_idx) > 0:
        # Half random relabel, half shifted relabel
        n_rand = len(lab_idx) // 2
        rand_idx = lab_idx[:n_rand]
        shift_idx = lab_idx[n_rand:]

        if len(rand_idx) > 0:
            y_noisy[rand_idx, 0] = rng.uniform(ROOM_BOUNDS[0, 0], ROOM_BOUNDS[0, 1], size=len(rand_idx))
            y_noisy[rand_idx, 1] = rng.uniform(ROOM_BOUNDS[1, 0], ROOM_BOUNDS[1, 1], size=len(rand_idx))

        if len(shift_idx) > 0:
            shift = rng.normal(0.0, 2.3, size=(len(shift_idx), 2))
            y_noisy[shift_idx] = np.clip(y_noisy[shift_idx] + shift, ROOM_BOUNDS[:, 0], ROOM_BOUNDS[:, 1])

    return x_noisy, y_noisy


def evaluate_one_model(model, x_train, y_train, x_test, y_test):
    t0 = time.perf_counter()
    model.fit(x_train, y_train)
    train_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = model.predict(x_test)
    infer_s = time.perf_counter() - t1

    err = np.linalg.norm(y_pred - y_test, axis=1)
    return {
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


def run_noise_robustness(data_dir: Path) -> pd.DataFrame:
    all_rows = []

    for si, scenario in enumerate(SCENARIOS):
        df = pd.read_csv(data_dir / f"dataset_{scenario}.csv")
        feature_cols = ["rssi_ap1", "rssi_ap2", "rssi_ap3"]
        x = df[feature_cols].to_numpy(dtype=float)
        y = df[["x", "y"]].to_numpy(dtype=float)

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=SEED
        )

        for ratio in CONTAM_LEVELS:
            rng = np.random.default_rng(SEED + si * 100 + int(round(ratio * 1000)))
            x_noisy, y_noisy = inject_training_outliers(x_train, y_train, ratio, rng)
            x_noisy_df = pd.DataFrame(x_noisy, columns=feature_cols)
            x_test_df = pd.DataFrame(x_test, columns=feature_cols)

            models = build_models()
            ordered_models = [m for m in MODEL_ORDER if m in models] + [m for m in models if m not in MODEL_ORDER]

            for model_name in ordered_models:
                metrics = evaluate_one_model(
                    models[model_name],
                    x_noisy_df,
                    y_noisy,
                    x_test_df,
                    y_test,
                )
                row = {
                    "scenario": scenario,
                    "noise_ratio": ratio,
                    "model": model_name,
                    **metrics,
                }
                all_rows.append(row)

    return pd.DataFrame(all_rows)


def build_degradation_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario, model), grp in metrics_df.groupby(["scenario", "model"]):
        grp = grp.sort_values("noise_ratio").reset_index(drop=True)
        base = grp.loc[grp["noise_ratio"] == 0.0]
        if base.empty:
            continue
        base_mae = float(base.iloc[0]["mae_m"])
        base_rmse = float(base.iloc[0]["rmse_m"])
        base_p90 = float(base.iloc[0]["p90_m"])

        for _, r in grp.iterrows():
            rows.append(
                {
                    "scenario": scenario,
                    "model": model,
                    "noise_ratio": float(r["noise_ratio"]),
                    "mae_m": float(r["mae_m"]),
                    "rmse_m": float(r["rmse_m"]),
                    "p90_m": float(r["p90_m"]),
                    "mae_degrade_pct": 100.0 * (float(r["mae_m"]) - base_mae) / max(base_mae, 1e-12),
                    "rmse_degrade_pct": 100.0 * (float(r["rmse_m"]) - base_rmse) / max(base_rmse, 1e-12),
                    "p90_degrade_pct": 100.0 * (float(r["p90_m"]) - base_p90) / max(base_p90, 1e-12),
                }
            )
    return pd.DataFrame(rows)


def plot_robustness_curves(deg_df: pd.DataFrame, fig_dir: Path) -> None:
    for scenario in SCENARIOS:
        sub = deg_df[deg_df["scenario"] == scenario]
        if sub.empty:
            continue

        model_order = [m for m in MODEL_ORDER if m in sub["model"].unique()]

        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharex=True)
        metric_info = [
            ("mae_degrade_pct", "MAE Degradation (%)"),
            ("rmse_degrade_pct", "RMSE Degradation (%)"),
            ("p90_degrade_pct", "P90 Degradation (%)"),
        ]

        for ax, (col, title) in zip(axes, metric_info):
            for m in model_order:
                s = sub[sub["model"] == m].sort_values("noise_ratio")
                ax.plot(
                    s["noise_ratio"].values * 100.0,
                    s[col].values,
                    marker="o",
                    linewidth=1.8,
                    markersize=4.5,
                    label=m,
                )
            ax.set_title(title)
            ax.set_xlabel("Training Outlier Ratio (%)")
            ax.grid(alpha=0.28, linestyle="--")

        axes[0].set_ylabel("Relative Change vs Clean Training (%)")
        axes[2].legend(loc="upper left", fontsize=8)
        fig.suptitle(f"Outlier Robustness Curves - {scenario}")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(fig_dir / f"robustness_curve_{scenario}.png", dpi=180)
        plt.close(fig)


def summarize_rank_at_high_noise(deg_df: pd.DataFrame, ratio: float = 0.20) -> pd.DataFrame:
    sub = deg_df[np.isclose(deg_df["noise_ratio"], ratio)]
    rows = []
    for scenario in SCENARIOS:
        s = sub[sub["scenario"] == scenario].sort_values("mae_degrade_pct")
        for rank, (_, r) in enumerate(s.iterrows(), start=1):
            rows.append(
                {
                    "scenario": scenario,
                    "rank_by_mae_degrade": rank,
                    "model": r["model"],
                    "mae_degrade_pct": r["mae_degrade_pct"],
                    "rmse_degrade_pct": r["rmse_degrade_pct"],
                    "p90_degrade_pct": r["p90_degrade_pct"],
                }
            )
    return pd.DataFrame(rows)


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "experiments" / "results" / "data"
    out_dir = root / "experiments" / "results" / "robustness"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = run_noise_robustness(data_dir)
    metrics_path = out_dir / "noise_robustness_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    deg_df = build_degradation_table(metrics_df)
    deg_path = out_dir / "noise_robustness_degradation.csv"
    deg_df.to_csv(deg_path, index=False)

    plot_robustness_curves(deg_df, fig_dir)

    rank20 = summarize_rank_at_high_noise(deg_df, ratio=0.20)
    rank20_path = out_dir / "robustness_rank_at_20pct_noise.csv"
    rank20.to_csv(rank20_path, index=False)

    summary = {
        "noise_levels": CONTAM_LEVELS,
        "scenarios": SCENARIOS,
        "files": {
            "metrics": str(metrics_path),
            "degradation": str(deg_path),
            "rank20": str(rank20_path),
            "fig_ideal": str(fig_dir / "robustness_curve_ideal_empty.png"),
            "fig_complex": str(fig_dir / "robustness_curve_complex_dynamic.png"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Noise robustness experiment complete.")
    print(metrics_path)
    print(deg_path)
    print(rank20_path)


if __name__ == "__main__":
    main()
