"""
Generate *violation heatmaps* for plain (unconstrained) predictions on obstacle datasets.

We visualize the discrepancy to prescribed Dirichlet values, restricted to the Dirichlet set:
  - outer boundary pixels
  - obstacle-region pixels (geom > 0.5)

Definition:
  V(i,j) = |u_pred(i,j) - g(i,j)| for (i,j) in Γ, and masked elsewhere.

Example:
  python paper_preparation/plot_violation_heatmap.py \
    --data_path data/pflow_obstacle2d_N64.pt \
    --pred_path preds/new_fno_pred.pt \
    --out_dir 第二图2 \
    --indices 3 \
    --tag fno

This will create:
  第二图2/viol_heatmap_fno_idx003.png
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import torch


def _dirichlet_mask(geom_2d: np.ndarray) -> np.ndarray:
    """geom_2d: (N,N) float/0-1; returns (N,N) bool mask for Γ."""
    N = geom_2d.shape[0]
    bdry = np.zeros((N, N), dtype=bool)
    bdry[0, :] = True
    bdry[-1, :] = True
    bdry[:, 0] = True
    bdry[:, -1] = True
    obs = geom_2d > 0.5
    return np.logical_or(bdry, obs)


def _parse_indices(xs: List[str]) -> List[int]:
    out: List[int] = []
    for x in xs:
        out.append(int(x))
    return out


def main():
    p = argparse.ArgumentParser(description="Plot violation heatmaps |pred-g| on the Dirichlet set.")
    p.add_argument("--data_path", type=str, required=True, help="Dataset .pt with a_test (geom, bc).")
    p.add_argument("--pred_path", type=str, required=True, help="Predictions .pt (T,N,N,1) in raw space.")
    p.add_argument("--out_dir", type=str, default="第二图2", help="Output directory for PNGs.")
    p.add_argument("--indices", nargs="+", default=["3"], help="Sample indices (0-based).")
    p.add_argument("--tag", type=str, default="fno", help="Filename tag, e.g., fno.")
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--vmax", type=float, default=None, help="Colorbar vmax (after log if enabled).")
    p.add_argument("--log1p", action="store_true", help="Plot log1p(|pred-g|) to improve dynamic range.")
    args = p.parse_args()

    idx_list = _parse_indices(args.indices)
    os.makedirs(args.out_dir, exist_ok=True)

    data = torch.load(args.data_path, map_location="cpu")
    a_test = data["a_test"]  # (T,N,N,2) -> (geom, bc)
    pred = torch.load(args.pred_path, map_location="cpu")  # (T,N,N,1)

    if pred.shape[:3] != a_test.shape[:3]:
        raise ValueError(f"shape mismatch: pred {tuple(pred.shape)} vs a_test {tuple(a_test.shape)}")

    T = int(a_test.shape[0])
    geom = a_test[..., 0].numpy()
    bc = a_test[..., 1].numpy()
    pp = pred[..., 0].numpy()

    import matplotlib.pyplot as plt

    for i in idx_list:
        if i < 0 or i >= T:
            raise ValueError(f"index {i} out of range [0, {T-1}]")

        mask = _dirichlet_mask(geom[i])
        viol = np.abs(pp[i] - bc[i]).astype(np.float64)
        viol_masked = np.full_like(viol, np.nan, dtype=np.float64)
        viol_masked[mask] = viol[mask]

        show = np.log1p(viol_masked) if args.log1p else viol_masked

        fig = plt.figure(figsize=(6.0, 4.6))
        ax = plt.gca()
        cmap = plt.get_cmap("magma").copy()
        cmap.set_bad(color=(1, 1, 1, 1))  # white for masked

        im = ax.imshow(show, cmap=cmap, vmax=args.vmax, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Violation on Dirichlet set" + (" (log1p)" if args.log1p else "") + f" (idx={i})")

        # obstacle outline (optional)
        obs = geom[i] > 0.5
        if np.any(obs):
            ax.contour(obs.astype(float), levels=[0.5], colors="cyan", linewidths=0.8, alpha=0.8)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$|u_{\theta}-g|$ on $\Gamma$" + (" (log1p)" if args.log1p else ""))

        plt.tight_layout()
        out_name = f"viol_heatmap_{args.tag}_idx{i:03d}.png"
        out_path = os.path.join(args.out_dir, out_name)
        plt.savefig(out_path, dpi=int(args.dpi))
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

