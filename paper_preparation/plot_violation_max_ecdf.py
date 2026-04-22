"""
Plot ECDFs of worst-case Dirichlet violation per sample: max_{Γ} |u_pred - g|.

This complements MSE-based violation metrics by showing the *worst-case* error,
which is often what computational mechanics reviewers care about.

Inputs:
  dataset: a_test (geom, bc, ...) and u_test
  predictions: (T,N,N,1) raw space

Example:
  python paper_preparation/plot_violation_max_ecdf.py \
    --data_path data/pflow_obstacle2d_N64.pt \
    --pred_plain preds/new_fno_pred.pt \
    --pred_A1 preds/new_fnoA_pred.pt \
    --pred_A2 preds/new_cftaoA_pred.pt \
    --out_path 第二图2/viol_max_ecdf.png
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import torch


def _dir_masks_np(geom: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    N = geom.shape[0]
    bdry = np.zeros((N, N), dtype=bool)
    bdry[0, :] = True
    bdry[-1, :] = True
    bdry[:, 0] = True
    bdry[:, -1] = True
    obs = geom > 0.5
    return bdry, obs


def _ecdf(x: np.ndarray):
    xs = np.sort(x)
    ys = np.linspace(0.0, 1.0, len(xs), endpoint=True)
    return xs, ys


def _load_pred(path: str) -> np.ndarray:
    x = torch.load(path, map_location="cpu")
    if isinstance(x, dict) and "pred" in x:
        x = x["pred"]
    if not torch.is_tensor(x):
        raise ValueError(f"pred file must be Tensor or dict with key 'pred': {path}")
    return x.numpy()[..., 0]  # (T,N,N)


def _max_on_mask(v: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return float("nan")
    return float(np.max(v[mask]))


def main():
    p = argparse.ArgumentParser(description="Plot worst-case violation ECDFs (max|pred-g| on Γ).")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--pred_plain", type=str, required=True)
    p.add_argument("--pred_A1", type=str, required=True)
    p.add_argument("--pred_A2", type=str, required=True)
    p.add_argument("--label_plain", type=str, default="FNO")
    p.add_argument("--label_A1", type=str, default="FNO+A")
    p.add_argument("--label_A2", type=str, default="CFT-AO+A")
    p.add_argument("--out_path", type=str, default="第二图2/viol_max_ecdf.png")
    p.add_argument("--logx", action="store_true")
    args = p.parse_args()

    data = torch.load(args.data_path, map_location="cpu")
    a_test = data["a_test"].numpy()  # (T,N,N,C>=2)
    T, N, _, _ = a_test.shape
    geom = a_test[..., 0]
    bc = a_test[..., 1]

    pp = _load_pred(args.pred_plain)
    p1 = _load_pred(args.pred_A1)
    p2 = _load_pred(args.pred_A2)
    if pp.shape != (T, N, N) or p1.shape != (T, N, N) or p2.shape != (T, N, N):
        raise ValueError("pred shapes must match (T,N,N)")

    max_plain = np.zeros(T, dtype=np.float64)
    max_A1 = np.zeros(T, dtype=np.float64)
    max_A2 = np.zeros(T, dtype=np.float64)

    for i in range(T):
        bdry, obs = _dir_masks_np(geom[i])
        gamma = bdry | obs
        v0 = np.abs(pp[i] - bc[i])
        v1 = np.abs(p1[i] - bc[i])
        v2 = np.abs(p2[i] - bc[i])
        max_plain[i] = _max_on_mask(v0, gamma)
        max_A1[i] = _max_on_mask(v1, gamma)
        max_A2[i] = _max_on_mask(v2, gamma)

    # drop nan (shouldn't happen for obstacle datasets, but keep safe)
    max_plain = max_plain[np.isfinite(max_plain)]
    max_A1 = max_A1[np.isfinite(max_A1)]
    max_A2 = max_A2[np.isfinite(max_A2)]

    import matplotlib.pyplot as plt

    x0, y0 = _ecdf(max_plain)
    x1, y1 = _ecdf(max_A1)
    x2, y2 = _ecdf(max_A2)

    plt.figure(figsize=(6.4, 4.2))
    plt.plot(x0, y0, label=args.label_plain, linewidth=2)
    plt.plot(x1, y1, label=args.label_A1, linewidth=2)
    plt.plot(x2, y2, label=args.label_A2, linewidth=2)
    if args.logx:
        plt.xscale("log")
    plt.xlabel(r"per-sample worst-case violation  $\max_{\Gamma}|u_{\theta}-g|$")
    plt.ylabel("ECDF")
    plt.title("Worst-case Dirichlet violation (ECDF)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out_path, dpi=180)
    plt.close()
    print(f"Saved: {args.out_path}")


if __name__ == "__main__":
    main()

