"""
Plot Pareto curves for soft-penalty baselines.

Inputs:
  One or more *_metrics.pt files produced by eval_plain_fno2d.py.
  Each metrics file must contain:
    { "raw_mse": float, "viol_bdry": float, "viol_obs": float }

Example:
  python paper_preparation/plot_pareto_penalty.py \
    --metrics_paths preds/penalty/*.pt \
    --out_path 第二图2/pareto_penalty_pflow.png \
    --title "Soft-penalty baseline (pflow)"
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List, Tuple

import numpy as np
import torch


def _load_metrics(path: str) -> Tuple[float, float, float]:
    d = torch.load(path, map_location="cpu")
    if not isinstance(d, dict):
        raise ValueError(f"metrics file must be a dict: {path}")
    return float(d["raw_mse"]), float(d["viol_bdry"]), float(d["viol_obs"])


def _label_from_path(p: str) -> str:
    b = os.path.basename(p)
    # strip common suffix
    for suf in ["_metrics.pt", ".pt"]:
        if b.endswith(suf):
            b = b[: -len(suf)]
    return b


def main():
    ap = argparse.ArgumentParser(description="Plot Pareto curves for penalty baselines.")
    ap.add_argument("--metrics_paths", nargs="+", required=True, help="List of metrics files or globs.")
    ap.add_argument("--out_path", type=str, required=True)
    ap.add_argument("--title", type=str, default="Penalty Pareto trade-off")
    ap.add_argument("--logx", action="store_true", help="Log-scale for violation (x-axis).")
    ap.add_argument("--logy", action="store_true", help="Log-scale for raw MSE (y-axis).")
    args = ap.parse_args()

    # expand globs
    paths: List[str] = []
    for x in args.metrics_paths:
        if any(ch in x for ch in ["*", "?", "["]):
            paths.extend(sorted(glob.glob(x)))
        else:
            paths.append(x)
    paths = [p for p in paths if p.endswith(".pt")]
    if not paths:
        raise ValueError("No .pt metrics files found.")

    rows = []
    for p in paths:
        raw_mse, vb, vo = _load_metrics(p)
        rows.append((p, raw_mse, vb, vo))

    # sort by obstacle violation then boundary violation
    rows.sort(key=lambda t: (t[3], t[2], t[1]))

    raw = np.array([r[1] for r in rows], dtype=np.float64)
    vb = np.array([r[2] for r in rows], dtype=np.float64)
    vo = np.array([r[3] for r in rows], dtype=np.float64)
    labels = [_label_from_path(r[0]) for r in rows]

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(vo, raw, marker="o", linewidth=1.5)
    ax[0].set_xlabel("Viol(obs) MSE")
    ax[0].set_ylabel("Raw MSE(px)")
    ax[0].set_title("Obstacle trade-off")
    ax[0].grid(True, alpha=0.25)
    if args.logx:
        ax[0].set_xscale("log")
    if args.logy:
        ax[0].set_yscale("log")

    ax[1].plot(vb, raw, marker="o", linewidth=1.5, color="tab:green")
    ax[1].set_xlabel("Viol(bdry) MSE")
    ax[1].set_ylabel("Raw MSE(px)")
    ax[1].set_title("Boundary trade-off")
    ax[1].grid(True, alpha=0.25)
    if args.logx:
        ax[1].set_xscale("log")
    if args.logy:
        ax[1].set_yscale("log")

    fig.suptitle(args.title)

    # annotate a few points (avoid clutter)
    for k in [0, len(labels) // 2, len(labels) - 1]:
        if 0 <= k < len(labels):
            ax[0].annotate(labels[k], (vo[k], raw[k]), fontsize=8, xytext=(4, 4), textcoords="offset points")
            ax[1].annotate(labels[k], (vb[k], raw[k]), fontsize=8, xytext=(4, 4), textcoords="offset points")

    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {args.out_path}")


if __name__ == "__main__":
    main()

