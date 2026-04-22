"""
Evaluate a *plain* FNO2d model (no A-wrapper) on any dataset with:
  a_train/a_test: (B,N,N,C) with the first two channels = [geom, bc]
  u_train/u_test: (B,N,N,1)

Outputs:
  - pred tensor (n_test,N,N,1) in raw space
  - metrics: Raw MSE(px), Viol(bdry), Viol(obs) where violation is MSE(out, bc) on Dirichlet sets
  - plus worst-case Dirichlet violation stats: max_{Γ}|u-g| per sample (mean / p95 / max)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fourier_2d_baseline import FNO2d
from utilities3 import UnitGaussianNormalizer


def _dir_masks(geom_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    bdry = torch.zeros_like(geom_raw, dtype=torch.bool)
    bdry[:, 0, :, :] = True
    bdry[:, -1, :, :] = True
    bdry[:, :, 0, :] = True
    bdry[:, :, -1, :] = True
    obs = geom_raw > 0.5
    return bdry, obs


def _ecdf_p95_and_max(x: torch.Tensor) -> Tuple[float, float, float]:
    """
    x: (T,) non-negative
    returns: mean, p95, max
    """
    x = x.detach().float().flatten()
    if x.numel() == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(x.mean().item())
    p95 = float(torch.quantile(x, 0.95).item())
    mx = float(x.max().item())
    return mean, p95, mx


@torch.no_grad()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    data = torch.load(args.data_path, map_location="cpu")
    a_train_raw = data["a_train"]
    u_train_raw = data["u_train"]
    a_test_raw = data["a_test"]
    u_test_raw = data["u_test"]

    # rebuild normalizers like training
    a_normalizer = UnitGaussianNormalizer(a_train_raw).to(device)
    u_normalizer = UnitGaussianNormalizer(u_train_raw).to(device)

    a_test = a_normalizer.encode(a_test_raw.to(device))
    u_test = u_test_raw.to(device)

    in_channels = int(a_test.shape[-1])
    out_channels = int(u_test.shape[-1])

    model = FNO2d(
        modes1=args.modes,
        modes2=args.modes,
        width=args.width,
        in_channels=in_channels,
        out_channels=out_channels,
    )
    model.load_state_dict(torch.load(args.ckpt_path, map_location="cpu"))
    model.to(device).eval()

    N = int(a_test_raw.shape[1])
    num_pts = N * N
    n_test = int(a_test_raw.shape[0])

    pred_list = []
    raw_mse_sum = 0.0
    bdry_sum = 0.0
    obs_sum = 0.0
    bdry_cnt = 0.0
    obs_cnt = 0.0
    max_gamma_list = []

    a_test_raw_dev = a_test_raw.to(device)

    for i in range(0, n_test, args.batch_size):
        xb = a_test[i : i + args.batch_size]
        yb = u_test[i : i + args.batch_size]
        a_raw_b = a_test_raw_dev[i : i + args.batch_size]

        out_enc = model(xb)
        out = u_normalizer.decode(out_enc)

        raw_mse_sum += F.mse_loss(out, yb, reduction="sum").item()

        geom_raw = a_raw_b[..., 0:1]
        bc_raw = a_raw_b[..., 1:2]
        bdry, obs = _dir_masks(geom_raw)
        diff_bc2 = (out - bc_raw) ** 2
        bdry_sum += diff_bc2[bdry].sum().item()
        obs_sum += diff_bc2[obs].sum().item()
        bdry_cnt += float(bdry.sum().item())
        obs_cnt += float(obs.sum().item())

        pred_list.append(out.detach().cpu())

        # worst-case |u-g| on Γ for each sample in batch
        gamma = (bdry | obs)[..., 0]  # (B,N,N)
        abs_diff = (out - bc_raw).abs()[..., 0]  # (B,N,N)
        # set non-gamma to -inf then max
        masked = torch.where(gamma, abs_diff, torch.full_like(abs_diff, float("-inf")))
        max_gamma = masked.amax(dim=(1, 2))  # (B,)
        max_gamma_list.append(max_gamma.detach().cpu())

    pred = torch.cat(pred_list, dim=0)

    raw_mse = raw_mse_sum / (n_test * num_pts)
    viol_bdry = bdry_sum / max(bdry_cnt, 1.0)
    viol_obs = obs_sum / max(obs_cnt, 1.0)

    max_gamma_all = torch.cat(max_gamma_list, dim=0) if max_gamma_list else torch.empty(0)
    max_gamma_mean, max_gamma_p95, max_gamma_max = _ecdf_p95_and_max(max_gamma_all)

    out_dir = os.path.dirname(args.pred_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(pred, args.pred_out)
    torch.save(
        {
            "raw_mse": raw_mse,
            "viol_bdry": viol_bdry,
            "viol_obs": viol_obs,
            "max_gamma_mean": max_gamma_mean,
            "max_gamma_p95": max_gamma_p95,
            "max_gamma_max": max_gamma_max,
        },
        args.pred_out.replace(".pt", "_metrics.pt"),
    )

    print("=== Eval (plain FNO) ===")
    print(f"  Raw MSE(px):    {raw_mse:.6e}")
    print(f"  Viol(bdry) MSE: {viol_bdry:.6e}")
    print(f"  Viol(obs)  MSE: {viol_obs:.6e}")
    print(f"  max|u-g| on Γ (mean/p95/max): {max_gamma_mean:.3e} / {max_gamma_p95:.3e} / {max_gamma_max:.3e}")
    print(f"Saved pred to: {args.pred_out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Eval plain FNO2d and export predictions.")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--pred_out", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--modes", type=int, default=12)
    p.add_argument("--width", type=int, default=64)
    main(p.parse_args())


