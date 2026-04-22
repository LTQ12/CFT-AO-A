"""
Compare two predictions on the same diffusion-like dataset (NHWC) and plot together:
  - mean |err| vs distance-to-Dirichlet (overlay)
  - per-sample MSE ECDF (overlay)
  - per-sample MSE boxplot (side-by-side)
  - worst-K samples: side-by-side (model A vs model B)

This script is dataset-agnostic as long as data has:
  a_test: (T,N,N,2) [geom, bc]
  u_test: (T,N,N,1)
and pred tensors are (T,N,N,1).
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import torch


def _try_import_edt():
    try:
        from scipy.ndimage import distance_transform_edt  # type: ignore

        return distance_transform_edt
    except Exception:
        return None


def _distance_to_dirichlet(geom: np.ndarray) -> np.ndarray:
    N = geom.shape[0]
    dir_mask = np.zeros((N, N), dtype=bool)
    dir_mask[0, :] = True
    dir_mask[-1, :] = True
    dir_mask[:, 0] = True
    dir_mask[:, -1] = True
    dir_mask = np.logical_or(dir_mask, geom > 0.5)

    edt = _try_import_edt()
    if edt is None:
        # fallback Manhattan relaxation (slow but dependency-free)
        dist = np.full((N, N), fill_value=1e9, dtype=np.float64)
        dist[dir_mask] = 0.0
        for _ in range(2 * N):
            changed = 0
            for i in range(N):
                for j in range(N):
                    best = dist[i, j]
                    if i > 0:
                        best = min(best, dist[i - 1, j] + 1)
                    if i < N - 1:
                        best = min(best, dist[i + 1, j] + 1)
                    if j > 0:
                        best = min(best, dist[i, j - 1] + 1)
                    if j < N - 1:
                        best = min(best, dist[i, j + 1] + 1)
                    if best < dist[i, j]:
                        dist[i, j] = best
                        changed += 1
            if changed == 0:
                break
        return dist

    return edt(~dir_mask)


def _bin_curve(dist: np.ndarray, err: np.ndarray, bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        m = (dist >= lo) & (dist < hi)
        xs.append(0.5 * (lo + hi))
        ys.append(np.nan if m.sum() == 0 else float(err[m].mean()))
    return np.asarray(xs), np.asarray(ys)


def _save_worst_pair(out_dir: str, k: int, geom: np.ndarray, bc: np.ndarray, gt: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, label_a: str, label_b: str):
    import matplotlib.pyplot as plt

    err_a = np.abs(pred_a - gt)
    err_b = np.abs(pred_b - gt)
    fig, ax = plt.subplots(2, 5, figsize=(16, 6))
    # row 0: A
    ax[0, 0].imshow(geom, cmap="gray"); ax[0, 0].set_title("geom")
    ax[0, 1].imshow(bc, cmap="coolwarm"); ax[0, 1].set_title("bc")
    ax[0, 2].imshow(gt, cmap="viridis"); ax[0, 2].set_title("gt")
    ax[0, 3].imshow(pred_a, cmap="viridis"); ax[0, 3].set_title(f"pred {label_a}")
    ax[0, 4].imshow(err_a, cmap="magma"); ax[0, 4].set_title(f"|err| {label_a}")
    # row 1: B
    ax[1, 0].imshow(geom, cmap="gray"); ax[1, 0].set_title("geom")
    ax[1, 1].imshow(bc, cmap="coolwarm"); ax[1, 1].set_title("bc")
    ax[1, 2].imshow(gt, cmap="viridis"); ax[1, 2].set_title("gt")
    ax[1, 3].imshow(pred_b, cmap="viridis"); ax[1, 3].set_title(f"pred {label_b}")
    ax[1, 4].imshow(err_b, cmap="magma"); ax[1, 4].set_title(f"|err| {label_b}")
    for a in ax.reshape(-1):
        a.axis("off")
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"worst_pair_{k:03d}.png"), dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Compare two A-model predictions and plot together.")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--pred_a", type=str, required=True)
    p.add_argument("--pred_b", type=str, required=True)
    p.add_argument("--label_a", type=str, default="A")
    p.add_argument("--label_b", type=str, default="B")
    p.add_argument("--out_dir", type=str, default="viz_compare")
    p.add_argument("--worst_k", type=int, default=12)
    args = p.parse_args()

    data = torch.load(args.data_path, map_location="cpu")
    a_test = data["a_test"]
    u_test = data["u_test"]

    pred_a = torch.load(args.pred_a, map_location="cpu")
    pred_b = torch.load(args.pred_b, map_location="cpu")
    if isinstance(pred_a, dict) and "pred" in pred_a:
        pred_a = pred_a["pred"]
    if isinstance(pred_b, dict) and "pred" in pred_b:
        pred_b = pred_b["pred"]

    if pred_a.shape != u_test.shape or pred_b.shape != u_test.shape:
        raise ValueError("pred shape mismatch with u_test")

    T, N, _, _ = u_test.shape
    geom = a_test[..., 0].numpy()
    bc = a_test[..., 1].numpy()
    gt = u_test[..., 0].numpy()
    pa = pred_a[..., 0].numpy()
    pb = pred_b[..., 0].numpy()

    # per-sample MSE
    mse_a = ((pa - gt) ** 2).mean(axis=(1, 2))
    mse_b = ((pb - gt) ** 2).mean(axis=(1, 2))

    order = np.argsort(-(0.5 * (mse_a + mse_b)))  # worst by average

    # distance curves
    bins = np.asarray([0, 1, 2, 4, 8, 16, 1e9], dtype=np.float64)
    curve_a = np.zeros(len(bins) - 1, dtype=np.float64)
    curve_b = np.zeros(len(bins) - 1, dtype=np.float64)
    cnt = np.zeros(len(bins) - 1, dtype=np.float64)
    xs = None
    for i in range(T):
        di = _distance_to_dirichlet(geom[i])
        err_a = np.abs(pa[i] - gt[i])
        err_b = np.abs(pb[i] - gt[i])
        xs_i, ya = _bin_curve(di, err_a, bins)
        _, yb = _bin_curve(di, err_b, bins)
        if xs is None:
            xs = xs_i
        m = ~np.isnan(ya) & ~np.isnan(yb)
        curve_a[m] += ya[m]
        curve_b[m] += yb[m]
        cnt[m] += 1.0
    curve_a = curve_a / np.maximum(cnt, 1.0)
    curve_b = curve_b / np.maximum(cnt, 1.0)

    os.makedirs(args.out_dir, exist_ok=True)

    import matplotlib.pyplot as plt

    # overlay err-vs-dist
    plt.figure(figsize=(6, 4))
    plt.plot(xs, curve_a, marker="o", label=args.label_a)
    plt.plot(xs, curve_b, marker="o", label=args.label_b)
    plt.xscale("log")
    plt.xlabel("distance to Dirichlet set (pixels, log)")
    plt.ylabel("mean |error|")
    plt.title("Error vs distance to Dirichlet (overlay)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "err_vs_dist_overlay.png"), dpi=160)
    plt.close()

    # ECDF overlay
    def ecdf(x):
        xs = np.sort(x)
        ys = np.linspace(0.0, 1.0, len(xs), endpoint=True)
        return xs, ys

    xa, ya = ecdf(mse_a)
    xb, yb = ecdf(mse_b)
    plt.figure(figsize=(6, 4))
    plt.plot(xa, ya, label=args.label_a)
    plt.plot(xb, yb, label=args.label_b)
    plt.xscale("log")
    plt.xlabel("per-sample MSE (log)")
    plt.ylabel("ECDF")
    plt.title("Per-sample MSE ECDF (overlay)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "mse_ecdf_overlay.png"), dpi=160)
    plt.close()

    # boxplot
    plt.figure(figsize=(5, 4))
    plt.boxplot([mse_a, mse_b], labels=[args.label_a, args.label_b], showfliers=False)
    plt.yscale("log")
    plt.ylabel("per-sample MSE (log)")
    plt.title("Per-sample MSE (boxplot)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "mse_boxplot.png"), dpi=160)
    plt.close()

    # worst-K side-by-side
    for k in range(int(args.worst_k)):
        j = int(order[k])
        _save_worst_pair(args.out_dir, k, geom[j], bc[j], gt[j], pa[j], pb[j], args.label_a, args.label_b)

    torch.save(
        {
            "mse_a": mse_a,
            "mse_b": mse_b,
            "bins": bins,
            "xs": xs,
            "curve_a": curve_a,
            "curve_b": curve_b,
            "label_a": args.label_a,
            "label_b": args.label_b,
        },
        os.path.join(args.out_dir, "compare_summary.pt"),
    )
    print("Saved comparison to:", args.out_dir)
    print(f"{args.label_a} mean MSE: {float(np.mean(mse_a)):.6e}")
    print(f"{args.label_b} mean MSE: {float(np.mean(mse_b)):.6e}")


if __name__ == "__main__":
    main()


