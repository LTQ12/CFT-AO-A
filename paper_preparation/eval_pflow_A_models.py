"""
Evaluate Direction-A models on the potential-flow obstacle2d dataset.

It:
  - loads dataset
  - rebuilds a_normalizer/u_normalizer from train split (to match training)
  - loads a saved model state_dict (FNO+A or CFT-AO+A)
  - runs inference on test set
  - reports: Raw MSE(px), E-only MSE(px), MSE(bdry), MSE(obs)
  - saves predictions to a .pt file (Tensor shape: (n_test,N,N,1))

Example:
  python paper_preparation/eval_pflow_A_models.py \
    --data_path data/pflow_obstacle2d_N64.pt \
    --model_type fnoA \
    --ckpt_path models/fnoA_pflow_obstacle2d.pt \
    --pred_out preds/pflow_fnoA_pred.pt
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

from boundary_ext_residual_2d import ResidualOnDirichletExtension2D
from cft_ao_2d import CFT_AO_2D_Atlas
from fourier_2d_baseline import FNO2d
from utilities3 import UnitGaussianNormalizer


def _default_residual_clip(model_type: str) -> float:
    if model_type == "fnoA":
        return 3.0
    if model_type == "cftaoA":
        return 0.0
    raise ValueError(f"Unknown model_type={model_type}")


def _dir_masks(geom_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # geom_raw: (B,N,N,1)
    bdry = torch.zeros_like(geom_raw, dtype=torch.bool)
    bdry[:, 0, :, :] = True
    bdry[:, -1, :, :] = True
    bdry[:, :, 0, :] = True
    bdry[:, :, -1, :] = True
    obs = geom_raw > 0.5
    return bdry, obs


def build_model(args, *, in_channels: int, out_channels: int, y_mean: torch.Tensor, y_std: torch.Tensor):
    if args.model_type == "fnoA":
        backbone = FNO2d(
            modes1=args.modes,
            modes2=args.modes,
            width=args.width,
            in_channels=in_channels,
            out_channels=out_channels,
        )
    elif args.model_type == "cftaoA":
        backbone = CFT_AO_2D_Atlas(
            modes1=args.modes,
            modes2=args.modes,
            width=args.width,
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=args.n_layers,
            L_segments=args.cft_L,
            M_cheb=args.cft_M,
            L_segments_boundary=args.cft_L_boundary,
            M_cheb_boundary=args.cft_M_boundary,
            cft_res=args.cft_res,
            use_local=not args.no_local,
            rim_ratio=args.rim_ratio,
            cond_dim=2,
            inner_iters=args.inner_iters,
            n_bands=args.n_bands,
            n_sym_bases=args.n_sym_bases,
        )
    else:
        raise ValueError(f"Unknown model_type={args.model_type}")

    model = ResidualOnDirichletExtension2D(
        backbone,
        y_mean=y_mean,
        y_std=y_std,
        in_channels_norm=in_channels,
        delta=args.delta,
        res_scale_init=args.res_scale_init,
        res_scale_max=args.res_scale_max,
        ext_method=args.ext_method,
        ext_iters=args.ext_iters,
        poisson_src_hidden=args.poisson_src_hidden,
        poisson_src_scale_max=args.poisson_src_scale_max,
        residual_clip=args.residual_clip,
    )
    return model


@torch.no_grad()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    data = torch.load(args.data_path, map_location="cpu")
    a_train_raw = data["a_train"]
    u_train_raw = data["u_train"]
    a_test_raw = data["a_test"]
    u_test_raw = data["u_test"]

    a_normalizer = UnitGaussianNormalizer(a_train_raw)
    u_normalizer = UnitGaussianNormalizer(u_train_raw)
    a_normalizer.to(device)
    u_normalizer.to(device)

    # IMPORTANT: UnitGaussianNormalizer.encode() expects x on the same device as mean/std.
    a_test_raw_dev = a_test_raw.to(device)
    u_test = u_test_raw.to(device)
    a_test = a_normalizer.encode(a_test_raw_dev)
    # mixed input: [a_norm (all channels), geom_raw, bc_raw]
    # (keep geom_raw/bc_raw as the last two channels for A-wrap)
    geom_raw = a_test_raw_dev[..., 0:1]
    bc_raw = a_test_raw_dev[..., 1:2]
    x_test_mix = torch.cat([a_test, geom_raw, bc_raw], dim=-1)

    in_channels = int(a_test.shape[-1])
    out_channels = int(u_test.shape[-1])

    if args.residual_clip is None:
        args.residual_clip = _default_residual_clip(args.model_type)
    print(f"Using residual_clip={args.residual_clip} for model_type={args.model_type}")

    model = build_model(args, in_channels=in_channels, out_channels=out_channels, y_mean=u_normalizer.mean, y_std=u_normalizer.std)
    state = torch.load(args.ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARN] Missing keys: {missing}")
        print(f"[WARN] Unexpected keys: {unexpected}")
        raise RuntimeError("Checkpoint does not exactly match the rebuilt model. Refusing to continue with a partial load.")
    model.to(device)
    model.eval()

    N = int(a_test_raw.shape[1])
    num_pts = N * N
    n_test = int(a_test_raw.shape[0])

    pred_list = []
    raw_mse_sum = 0.0
    eonly_mse_sum = 0.0
    bdry_mse_sum = 0.0
    obs_mse_sum = 0.0
    bdry_cnt = 0.0
    obs_cnt = 0.0
    # per-sample worst-case Dirichlet discrepancy: max_{Γ}|u-g|
    max_gamma_list = []

    for i in range(0, n_test, args.batch_size):
        xb = x_test_mix[i : i + args.batch_size]
        yb = u_test[i : i + args.batch_size]

        out_enc = model(xb)
        out = u_normalizer.decode(out_enc)

        # E-only
        E_raw = model.build_extension_raw(xb)

        raw_mse_sum += F.mse_loss(out, yb, reduction="sum").item()
        eonly_mse_sum += F.mse_loss(E_raw, yb, reduction="sum").item()

        geom_raw = xb[..., in_channels : in_channels + 1]
        bc_raw = xb[..., in_channels + 1 : in_channels + 2]
        bdry, obs = _dir_masks(geom_raw)
        diff_bc2 = (out - bc_raw) ** 2
        bdry_mse_sum += diff_bc2[bdry].sum().item()
        obs_mse_sum += diff_bc2[obs].sum().item()
        bdry_cnt += float(bdry.sum().item())
        obs_cnt += float(obs.sum().item())

        # max_{Γ}|u-g| per sample (Γ = bdry ∪ obs)
        dir_mask = bdry | obs  # (B,N,N,1) bool
        abs_diff = (out - bc_raw).abs()
        # flatten each sample, select Γ nodes, take max
        B = int(abs_diff.shape[0])
        abs_diff_flat = abs_diff.view(B, -1)
        dir_flat = dir_mask.view(B, -1)
        # guard: Γ should be non-empty; if empty (unexpected), set 0
        for bi in range(B):
            if dir_flat[bi].any():
                max_gamma_list.append(float(abs_diff_flat[bi][dir_flat[bi]].max().detach().cpu()))
            else:
                max_gamma_list.append(0.0)

        pred_list.append(out.detach().cpu())

    pred = torch.cat(pred_list, dim=0)  # (n_test,N,N,1)

    raw_mse = raw_mse_sum / (n_test * num_pts)
    eonly_mse = eonly_mse_sum / (n_test * num_pts)
    viol_bdry = bdry_mse_sum / max(bdry_cnt, 1.0)
    viol_obs = obs_mse_sum / max(obs_cnt, 1.0)

    # aggregate max_{Γ}|u-g|
    if len(max_gamma_list) != n_test:
        # should not happen, but keep robust
        max_gamma_list = (max_gamma_list + [0.0] * n_test)[:n_test]
    max_gamma = torch.tensor(max_gamma_list, dtype=torch.float64)
    max_gamma_mean = float(max_gamma.mean().item())
    max_gamma_p95 = float(torch.quantile(max_gamma, 0.95).item())
    max_gamma_max = float(max_gamma.max().item())

    print("=== Eval (pflow, A-model) ===")
    print(f"  model_type: {args.model_type}")
    print(f"  Raw MSE(px):     {raw_mse:.6e}")
    print(f"  E-only MSE(px):  {eonly_mse:.6e}")
    print(f"  Viol(bdry) MSE:  {viol_bdry:.6e}")
    print(f"  Viol(obs)  MSE:  {viol_obs:.6e}")
    print(f"  max|u-g| on Γ (mean/p95/max): {max_gamma_mean:.3e} / {max_gamma_p95:.3e} / {max_gamma_max:.3e}")

    out_dir = os.path.dirname(args.pred_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(pred, args.pred_out)
    torch.save(
        {
            "raw_mse": raw_mse,
            "eonly_mse": eonly_mse,
            "viol_bdry": viol_bdry,
            "viol_obs": viol_obs,
            "max_gamma_mean": max_gamma_mean,
            "max_gamma_p95": max_gamma_p95,
            "max_gamma_max": max_gamma_max,
        },
        args.pred_out.replace(".pt", "_metrics.pt"),
    )
    print(f"Saved pred to: {args.pred_out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Eval A-models on pflow obstacle2d dataset.")
    p.add_argument("--data_path", type=str, default="data/pflow_obstacle2d_N64.pt")
    p.add_argument("--model_type", type=str, required=True, choices=["fnoA", "cftaoA"])
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--pred_out", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--cpu", action="store_true")

    # A-wrapper args
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--res_scale_init", type=float, default=0.02)
    p.add_argument("--res_scale_max", type=float, default=0.25)
    p.add_argument("--ext_method", type=str, default="harmonic", choices=["zero", "coons", "harmonic", "poisson", "poisson_learned"])
    p.add_argument("--ext_iters", type=int, default=80)
    p.add_argument("--poisson_src_hidden", type=int, default=32)
    p.add_argument("--poisson_src_scale_max", type=float, default=1.0)
    p.add_argument("--residual_clip", type=float, default=None, help="If omitted, auto-selects 3.0 for fnoA and 0.0 for cftaoA.")

    # backbone shared
    p.add_argument("--modes", type=int, default=12)
    p.add_argument("--width", type=int, default=64)

    # CFT-AO backbone args
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--cft_L", type=int, default=4)
    p.add_argument("--cft_M", type=int, default=4)
    p.add_argument("--cft_L_boundary", type=int, default=8)
    p.add_argument("--cft_M_boundary", type=int, default=8)
    p.add_argument("--cft_res", type=int, default=0)
    p.add_argument("--no_local", action="store_true")
    p.add_argument("--rim_ratio", type=float, default=0.15)
    p.add_argument("--inner_iters", type=int, default=2)
    p.add_argument("--n_bands", type=int, default=3)
    p.add_argument("--n_sym_bases", type=int, default=0)

    main(p.parse_args())


