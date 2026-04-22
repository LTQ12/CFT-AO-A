"""
Generate multiple violation heatmaps for plain FNO (or any unconstrained model),
plus aggregated heatmaps (mean / median / 90th percentile) over a subset.

We visualize |u_pred - g| on the Dirichlet set Γ = outer boundary ∪ obstacles,
and set non-Γ pixels to 0 for clarity.

Example (Colab):
  python3 paper_preparation/plot_violation_heatmap_batch.py \
    --data_path /content/data/pflow_obstacle2d_N64.pt \
    --pred_path /content/preds/new_fno_pred.pt \
    --indices 3 8 14 52 60 99 123 164 \
    --out_dir /content/viz/viol_heatmaps_pflow \
    --prefix viol_heatmap_fno
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
import torch


def _dir_masks_np(geom: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """geom: (N,N) in {0,1} or [0,1]."""
    N = geom.shape[0]
    bdry = np.zeros((N, N), dtype=bool)
    bdry[0, :] = True
    bdry[-1, :] = True
    bdry[:, 0] = True
    bdry[:, -1] = True
    obs = geom > 0.5
    return bdry, obs


def _load_pred(path: str) -> np.ndarray:
    x = torch.load(path, map_location="cpu")
    if isinstance(x, dict) and "pred" in x:
        x = x["pred"]
    if not torch.is_tensor(x):
        raise ValueError(f"pred file must be Tensor or dict with key 'pred': {path}")
    # accept (T,N,N,1) or (T,N,N)
    if x.ndim == 4:
        x = x[..., 0]
    return x.numpy()


def _save_heatmap(img: np.ndarray, out_path: str, title: str):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(4.6, 4.1))
    plt.imshow(img, cmap="inferno")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title, fontsize=10)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main():
    p = argparse.ArgumentParser(description="Batch violation heatmaps + aggregate heatmaps.")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--pred_path", type=str, required=True)
    p.add_argument("--indices", type=int, nargs="+", required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--prefix", type=str, default="viol_heatmap")
    p.add_argument("--percentile", type=float, default=90.0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    data = torch.load(args.data_path, map_location="cpu")
    a_test = data["a_test"].numpy()  # (T,N,N,C>=2)
    geom = a_test[..., 0]
    bc = a_test[..., 1]

    pred = _load_pred(args.pred_path)  # (T,N,N)
    if pred.shape[:3] != bc.shape[:3]:
        raise ValueError(f"pred shape {pred.shape} must match a_test (T,N,N).")

    idxs: List[int] = list(args.indices)
    viol_stack = []

    for i in idxs:
        bdry, obs = _dir_masks_np(geom[i])
        gamma = bdry | obs
        v = np.abs(pred[i] - bc[i]).astype(np.float64)
        img = np.zeros_like(v)
        img[gamma] = v[gamma]
        viol_stack.append(img)
        out_path = os.path.join(args.out_dir, f"{args.prefix}_idx{i:03d}.png")
        _save_heatmap(img, out_path, title=f"{args.prefix} idx={i}")

    V = np.stack(viol_stack, axis=0)  # (K,N,N)
    mean_img = V.mean(axis=0)
    med_img = np.median(V, axis=0)
    p_img = np.percentile(V, args.percentile, axis=0)

    _save_heatmap(mean_img, os.path.join(args.out_dir, f"{args.prefix}_mean.png"), title=f"{args.prefix} mean (K={len(idxs)})")
    _save_heatmap(med_img, os.path.join(args.out_dir, f"{args.prefix}_median.png"), title=f"{args.prefix} median (K={len(idxs)})")
    _save_heatmap(p_img, os.path.join(args.out_dir, f"{args.prefix}_p{int(args.percentile)}.png"), title=f"{args.prefix} p{int(args.percentile)} (K={len(idxs)})")

    print(f"Saved per-sample + aggregate heatmaps to: {args.out_dir}")


if __name__ == "__main__":
    main()

