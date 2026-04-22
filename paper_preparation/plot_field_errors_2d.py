"""
Plot field visualizations and error maps for 2D datasets (NHWC) used in this repo.

Supports:
  - diffusion-like scalar fields (u_test is scalar)
  - pflow streamfunction psi (also scalar), with optional velocity derived from psi:
        u = dpsi/dy,  v = -dpsi/dx

Inputs:
  --data_path: .pt with a_test (geom, bc) and u_test
  --pred_plain / --pred_A1 / --pred_A2: predicted tensors (n_test,N,N,1)
  --out_dir: output folder for pngs

It saves per-sample figures:
  sample_{idx:03d}_psi.png
  sample_{idx:03d}_velmag.png  (if --plot_vel)
and a small summary text.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

import numpy as np
import torch


def _load_pred(path: str) -> torch.Tensor:
    x = torch.load(path, map_location="cpu")
    if isinstance(x, dict) and "pred" in x:
        x = x["pred"]
    if not torch.is_tensor(x):
        raise ValueError(f"pred file {path} must be a Tensor or dict with key 'pred'")
    return x


def _vel_from_psi(psi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    psi: (N,N) on [0,1]^2 grid.
    Returns u,v on same grid using numpy gradient (central diffs inside).
    u = dpsi/dy, v = -dpsi/dx
    """
    N = psi.shape[0]
    dx = 1.0 / max(N - 1, 1)
    dpsi_dx, dpsi_dy = np.gradient(psi, dx, dx, edge_order=1)
    u = dpsi_dy
    v = -dpsi_dx
    return u, v


def _imshow(ax, img, title: str, cmap="viridis", vmin=None, vmax=None):
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    return im


def _plot_triplet(fig, axes, gt, pred, name: str, vmin=None, vmax=None, err_vmax=None):
    err = np.abs(pred - gt)
    _imshow(axes[0], gt, f"gt {name}", vmin=vmin, vmax=vmax)
    _imshow(axes[1], pred, f"pred {name}", vmin=vmin, vmax=vmax)
    _imshow(axes[2], err, f"|err| {name}", cmap="magma", vmin=0.0, vmax=err_vmax)


def main():
    p = argparse.ArgumentParser(description="Plot field and error maps for 2D experiments.")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--pred_plain", type=str, required=True)
    p.add_argument("--pred_A1", type=str, required=True)
    p.add_argument("--pred_A2", type=str, required=True)
    p.add_argument("--label_plain", type=str, default="FNO")
    p.add_argument("--label_A1", type=str, default="FNO+A")
    p.add_argument("--label_A2", type=str, default="CFT-AO+A")
    p.add_argument("--out_dir", type=str, default="viz/fields")
    p.add_argument("--n_samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--indices", type=str, default="", help="Comma-separated indices, overrides random sampling.")
    p.add_argument("--plot_vel", action="store_true", help="Also plot velocity magnitude derived from psi.")
    p.add_argument("--prefix", type=str, default="sample", help="Output filename prefix (avoid collisions across datasets).")
    p.add_argument("--tag", type=str, default="psi", help="Field tag used in titles and filenames (e.g., 'psi' or 'u').")
    args = p.parse_args()

    data = torch.load(args.data_path, map_location="cpu")
    a_test = data["a_test"].numpy()  # (T,N,N,C>=2); first two channels: geom, bc
    u_test = data["u_test"].numpy()  # (T,N,N,1)

    pred_plain = _load_pred(args.pred_plain).numpy()
    pred_A1 = _load_pred(args.pred_A1).numpy()
    pred_A2 = _load_pred(args.pred_A2).numpy()

    if pred_plain.shape != u_test.shape or pred_A1.shape != u_test.shape or pred_A2.shape != u_test.shape:
        raise ValueError("pred shapes must match u_test")

    T, N, _, _ = u_test.shape
    geom = a_test[..., 0]
    bc = a_test[..., 1]
    gt = u_test[..., 0]
    pp = pred_plain[..., 0]
    p1 = pred_A1[..., 0]
    p2 = pred_A2[..., 0]

    # choose indices
    if args.indices.strip():
        idxs = [int(x) for x in args.indices.split(",") if x.strip()]
    else:
        rng = np.random.default_rng(int(args.seed))
        idxs = rng.choice(T, size=min(int(args.n_samples), T), replace=False).tolist()

    os.makedirs(args.out_dir, exist_ok=True)

    # global color ranges for field for consistent comparison (robust to outliers)
    all_gt = gt.reshape(T, -1)
    q1, q99 = np.quantile(all_gt, [0.01, 0.99])
    field_vmin, field_vmax = float(q1), float(q99)

    import matplotlib.pyplot as plt

    for k, i in enumerate(idxs):
        # field plots
        fig, ax = plt.subplots(4, 3, figsize=(10, 10))
        # row 0: geom/bc/gt
        _imshow(ax[0, 0], geom[i], "geom", cmap="gray", vmin=0.0, vmax=1.0)
        _imshow(ax[0, 1], bc[i], "bc", cmap="coolwarm")
        _imshow(ax[0, 2], gt[i], f"gt {args.tag}", vmin=field_vmin, vmax=field_vmax)
        # row 1-3: each model
        _plot_triplet(fig, ax[1, :], gt[i], pp[i], args.label_plain, vmin=field_vmin, vmax=field_vmax)
        _plot_triplet(fig, ax[2, :], gt[i], p1[i], args.label_A1, vmin=field_vmin, vmax=field_vmax)
        _plot_triplet(fig, ax[3, :], gt[i], p2[i], args.label_A2, vmin=field_vmin, vmax=field_vmax)
        fig.suptitle(f"Sample {i} ({args.tag})")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"{args.prefix}_{i:03d}_{args.tag}.png"), dpi=160)
        plt.close(fig)

        if args.plot_vel:
            ug, vg = _vel_from_psi(gt[i])
            up, vp = _vel_from_psi(pp[i])
            u1, v1 = _vel_from_psi(p1[i])
            u2, v2 = _vel_from_psi(p2[i])
            mg = np.sqrt(ug**2 + vg**2)
            mp = np.sqrt(up**2 + vp**2)
            m1 = np.sqrt(u1**2 + v1**2)
            m2 = np.sqrt(u2**2 + v2**2)
            # robust range
            q1m, q99m = np.quantile(mg.reshape(-1), [0.01, 0.99])
            vminm, vmaxm = float(q1m), float(q99m)

            fig, ax = plt.subplots(4, 3, figsize=(10, 10))
            _imshow(ax[0, 0], geom[i], "geom", cmap="gray", vmin=0.0, vmax=1.0)
            _imshow(ax[0, 1], bc[i], "bc", cmap="coolwarm")
            _imshow(ax[0, 2], mg, "gt |v|", vmin=vminm, vmax=vmaxm)
            _plot_triplet(fig, ax[1, :], mg, mp, f"{args.label_plain} |v|", vmin=vminm, vmax=vmaxm)
            _plot_triplet(fig, ax[2, :], mg, m1, f"{args.label_A1} |v|", vmin=vminm, vmax=vmaxm)
            _plot_triplet(fig, ax[3, :], mg, m2, f"{args.label_A2} |v|", vmin=vminm, vmax=vmaxm)
            fig.suptitle(f"Sample {i} (velocity magnitude from psi)")
            fig.tight_layout()
            fig.savefig(os.path.join(args.out_dir, f"{args.prefix}_{i:03d}_velmag.png"), dpi=160)
            plt.close(fig)

    # quick summary
    with open(os.path.join(args.out_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(f"data_path: {args.data_path}\n")
        f.write(f"pred_plain: {args.pred_plain} ({args.label_plain})\n")
        f.write(f"pred_A1: {args.pred_A1} ({args.label_A1})\n")
        f.write(f"pred_A2: {args.pred_A2} ({args.label_A2})\n")
        f.write(f"indices: {idxs}\n")
        f.write(f"prefix: {args.prefix}\n")
        f.write(f"tag: {args.tag}\n")
        f.write("Outputs: {prefix}_XXX_{tag}.png and optional {prefix}_XXX_velmag.png\n")

    print("Saved field plots to:", args.out_dir)


if __name__ == "__main__":
    main()


