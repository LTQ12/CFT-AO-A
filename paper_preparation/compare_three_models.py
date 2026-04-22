"""
Compare THREE models on one dataset by overlay plots:
  - mean |err| vs distance-to-Dirichlet (3 curves)
  - per-sample MSE ECDF (3 curves)
  - per-sample MSE boxplot (3 boxes)
  - constraint violation ECDF (bdry/obs) for each model

Inputs:
  --data_path: dataset with a_test (geom, bc) and u_test
  --pred_plain: plain model predictions (raw)  (n_test,N,N,1)
  --pred_A1: A-model predictions (raw)
  --pred_A2: A-model predictions (raw)
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


def _ecdf(x: np.ndarray):
    xs = np.sort(x)
    ys = np.linspace(0.0, 1.0, len(xs), endpoint=True)
    return xs, ys


def main():
    p = argparse.ArgumentParser(description="Compare 3 models (plain vs two A models) on one dataset.")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--pred_plain", type=str, required=True)
    p.add_argument("--pred_A1", type=str, required=True)
    p.add_argument("--pred_A2", type=str, required=True)
    p.add_argument("--label_plain", type=str, default="FNO")
    p.add_argument("--label_A1", type=str, default="FNO+A")
    p.add_argument("--label_A2", type=str, default="CFT-AO+A")
    p.add_argument("--out_dir", type=str, default="viz/compare_three")
    args = p.parse_args()

    data = torch.load(args.data_path, map_location="cpu")
    a_test = data["a_test"]  # (T,N,N,2)
    u_test = data["u_test"]  # (T,N,N,1)

    pred_plain = torch.load(args.pred_plain, map_location="cpu")
    pred_A1 = torch.load(args.pred_A1, map_location="cpu")
    pred_A2 = torch.load(args.pred_A2, map_location="cpu")

    if pred_plain.shape != u_test.shape or pred_A1.shape != u_test.shape or pred_A2.shape != u_test.shape:
        raise ValueError("pred shape mismatch with u_test")

    T, N, _, _ = u_test.shape
    geom = a_test[..., 0].numpy()
    bc = a_test[..., 1].numpy()
    gt = u_test[..., 0].numpy()
    pp = pred_plain[..., 0].numpy()
    p1 = pred_A1[..., 0].numpy()
    p2 = pred_A2[..., 0].numpy()

    # per-sample MSE
    mse_plain = ((pp - gt) ** 2).mean(axis=(1, 2))
    mse_1 = ((p1 - gt) ** 2).mean(axis=(1, 2))
    mse_2 = ((p2 - gt) ** 2).mean(axis=(1, 2))

    # constraint violation (MSE to bc on boundary / on obstacle)
    bdry_mask = np.zeros((N, N), dtype=bool)
    bdry_mask[0, :] = True
    bdry_mask[-1, :] = True
    bdry_mask[:, 0] = True
    bdry_mask[:, -1] = True
    viol_bdry_plain = np.mean(((pp[:, bdry_mask] - bc[:, bdry_mask]) ** 2), axis=1)
    viol_bdry_1 = np.mean(((p1[:, bdry_mask] - bc[:, bdry_mask]) ** 2), axis=1)
    viol_bdry_2 = np.mean(((p2[:, bdry_mask] - bc[:, bdry_mask]) ** 2), axis=1)
    obs_mask = geom > 0.5
    # if some samples have no obstacle pixels, handle by nan then ignore in plot
    viol_obs_plain = np.array([np.mean(((pp[i][obs_mask[i]] - bc[i][obs_mask[i]]) ** 2)) if obs_mask[i].any() else np.nan for i in range(T)])
    viol_obs_1 = np.array([np.mean(((p1[i][obs_mask[i]] - bc[i][obs_mask[i]]) ** 2)) if obs_mask[i].any() else np.nan for i in range(T)])
    viol_obs_2 = np.array([np.mean(((p2[i][obs_mask[i]] - bc[i][obs_mask[i]]) ** 2)) if obs_mask[i].any() else np.nan for i in range(T)])

    # distance curves
    bins = np.asarray([0, 1, 2, 4, 8, 16, 1e9], dtype=np.float64)
    xs = None
    cur_p = np.zeros(len(bins) - 1); cur_1 = np.zeros(len(bins) - 1); cur_2 = np.zeros(len(bins) - 1); cnt = np.zeros(len(bins) - 1)
    for i in range(T):
        di = _distance_to_dirichlet(geom[i])
        e_p = np.abs(pp[i] - gt[i])
        e_1 = np.abs(p1[i] - gt[i])
        e_2 = np.abs(p2[i] - gt[i])
        xs_i, yp = _bin_curve(di, e_p, bins)
        _, y1 = _bin_curve(di, e_1, bins)
        _, y2 = _bin_curve(di, e_2, bins)
        if xs is None:
            xs = xs_i
        m = ~np.isnan(yp) & ~np.isnan(y1) & ~np.isnan(y2)
        cur_p[m] += yp[m]; cur_1[m] += y1[m]; cur_2[m] += y2[m]; cnt[m] += 1.0
    cur_p = cur_p / np.maximum(cnt, 1.0)
    cur_1 = cur_1 / np.maximum(cnt, 1.0)
    cur_2 = cur_2 / np.maximum(cnt, 1.0)

    os.makedirs(args.out_dir, exist_ok=True)
    import matplotlib.pyplot as plt

    # err-vs-dist overlay
    plt.figure(figsize=(6, 4))
    plt.plot(xs, cur_p, marker="o", label=args.label_plain)
    plt.plot(xs, cur_1, marker="o", label=args.label_A1)
    plt.plot(xs, cur_2, marker="o", label=args.label_A2)
    plt.xscale("log"); plt.xlabel("distance to Dirichlet set (pixels, log)")
    plt.ylabel("mean |error|"); plt.title("Error vs distance to Dirichlet (3-way)")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "err_vs_dist_3way.png"), dpi=160); plt.close()

    # MSE ECDF overlay
    xp, yp = _ecdf(mse_plain); x1, y1 = _ecdf(mse_1); x2, y2 = _ecdf(mse_2)
    plt.figure(figsize=(6, 4))
    plt.plot(xp, yp, label=args.label_plain)
    plt.plot(x1, y1, label=args.label_A1)
    plt.plot(x2, y2, label=args.label_A2)
    plt.xscale("log"); plt.xlabel("per-sample MSE (log)"); plt.ylabel("ECDF")
    plt.title("Per-sample MSE ECDF (3-way)")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "mse_ecdf_3way.png"), dpi=160); plt.close()

    # boxplot
    plt.figure(figsize=(6, 4))
    # Matplotlib>=3.9 renamed "labels" -> "tick_labels"
    try:
        plt.boxplot(
            [mse_plain, mse_1, mse_2],
            tick_labels=[args.label_plain, args.label_A1, args.label_A2],
            showfliers=False,
        )
    except TypeError:
        plt.boxplot(
            [mse_plain, mse_1, mse_2],
            labels=[args.label_plain, args.label_A1, args.label_A2],
            showfliers=False,
        )
    plt.yscale("log"); plt.ylabel("per-sample MSE (log)")
    plt.title("Per-sample MSE (boxplot)")
    plt.grid(True, axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "mse_boxplot_3way.png"), dpi=160); plt.close()

    # violation ECDF (boundary)
    xb0, yb0 = _ecdf(viol_bdry_plain); xb1, yb1 = _ecdf(viol_bdry_1); xb2, yb2 = _ecdf(viol_bdry_2)
    plt.figure(figsize=(6, 4))
    plt.plot(xb0, yb0, label=args.label_plain)
    plt.plot(xb1, yb1, label=args.label_A1)
    plt.plot(xb2, yb2, label=args.label_A2)
    plt.xscale("log"); plt.xlabel("boundary violation MSE (log)"); plt.ylabel("ECDF")
    plt.title("Boundary constraint violation ECDF")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "viol_bdry_ecdf.png"), dpi=160); plt.close()

    # violation ECDF (obstacle) - ignore nan samples
    v0 = viol_obs_plain[~np.isnan(viol_obs_plain)]
    v1 = viol_obs_1[~np.isnan(viol_obs_1)]
    v2 = viol_obs_2[~np.isnan(viol_obs_2)]
    xo0, yo0 = _ecdf(v0); xo1, yo1 = _ecdf(v1); xo2, yo2 = _ecdf(v2)
    plt.figure(figsize=(6, 4))
    plt.plot(xo0, yo0, label=args.label_plain)
    plt.plot(xo1, yo1, label=args.label_A1)
    plt.plot(xo2, yo2, label=args.label_A2)
    plt.xscale("log"); plt.xlabel("obstacle violation MSE (log)"); plt.ylabel("ECDF")
    plt.title("Obstacle constraint violation ECDF")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "viol_obs_ecdf.png"), dpi=160); plt.close()

    torch.save(
        {
            "mse": {args.label_plain: mse_plain, args.label_A1: mse_1, args.label_A2: mse_2},
            "viol_bdry": {args.label_plain: viol_bdry_plain, args.label_A1: viol_bdry_1, args.label_A2: viol_bdry_2},
            "viol_obs": {args.label_plain: viol_obs_plain, args.label_A1: viol_obs_1, args.label_A2: viol_obs_2},
        },
        os.path.join(args.out_dir, "summary.pt"),
    )
    print("Saved 3-way comparison to:", args.out_dir)


if __name__ == "__main__":
    main()


