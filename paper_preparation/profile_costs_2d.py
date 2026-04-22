"""
Profile parameter counts + inference latency + CUDA memory for 2D models in this repo.

It measures:
  - params (plain FNO, FNO+A, CFT-AO+A)
  - forward time per batch (ms)
  - peak CUDA memory during forward (MB)
  - A-wrap extension overhead: time(build_extension_raw) vs time(forward)

Example:
  python paper_preparation/profile_costs_2d.py \
    --data_path data/pflow_obstacle2d_N64.pt \
    --ckpt_plain models/new_fno.pt \
    --ckpt_fnoA models/new_fnoA.pt \
    --ckpt_cftaoA models/new_cftaoA.pt \
    --batch_size 16 --repeats 50 --warmup 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boundary_ext_residual_2d import ResidualOnDirichletExtension2D
from cft_ao_2d import CFT_AO_2D_Atlas
from fourier_2d_baseline import FNO2d
from uno_2d import UNO2d
from utilities3 import UnitGaussianNormalizer, count_params


def _default_residual_clip(model_type: str) -> float:
    if model_type == "fnoA":
        return 3.0
    if model_type in {"unoA", "cftaoA"}:
        return 0.0
    raise ValueError(model_type)


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def _time_forward(fn, device: torch.device, repeats: int, warmup: int) -> Tuple[float, float]:
    # warmup
    for _ in range(int(warmup)):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(int(repeats)):
        fn()
    _sync(device)
    t1 = time.perf_counter()
    total = (t1 - t0) * 1000.0  # ms
    per = total / max(int(repeats), 1)
    return per, total


def _peak_mem_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024**2))


def _build_plain(args, in_channels: int, out_channels: int) -> torch.nn.Module:
    m = FNO2d(modes1=args.modes, modes2=args.modes, width=args.width, in_channels=in_channels, out_channels=out_channels)
    m.load_state_dict(torch.load(args.ckpt_plain, map_location="cpu"))
    return m


def _build_A(args, *, model_type: str, in_channels: int, out_channels: int, y_mean: torch.Tensor, y_std: torch.Tensor) -> torch.nn.Module:
    if model_type == "fnoA":
        backbone = FNO2d(modes1=args.modes, modes2=args.modes, width=args.width, in_channels=in_channels, out_channels=out_channels)
        backbone.load_state_dict(torch.load(args.ckpt_fnoA, map_location="cpu"), strict=False)
        ckpt = args.ckpt_fnoA
    elif model_type == "unoA":
        backbone = UNO2d(
            in_channels=in_channels,
            out_channels=out_channels,
            width=args.width,
            modes1=args.modes,
            modes2=args.modes,
            pad=args.uno_pad,
            factor=args.uno_factor,
        )
        ckpt = args.ckpt_unoA
    elif model_type == "cftaoA":
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
        ckpt = args.ckpt_cftaoA
    else:
        raise ValueError(model_type)

    residual_clip = _default_residual_clip(model_type) if args.residual_clip is None else float(args.residual_clip)
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
        residual_clip=residual_clip,
    )
    state = torch.load(ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Profiling {model_type} with residual_clip={residual_clip}")
    if missing or unexpected:
        print(f"[WARN] Missing keys for {model_type}: {missing}")
        print(f"[WARN] Unexpected keys for {model_type}: {unexpected}")
        raise RuntimeError(f"Checkpoint for {model_type} does not exactly match the rebuilt model.")
    return model


def main():
    p = argparse.ArgumentParser(description="Profile cost/latency for 2D models.")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--ckpt_plain", type=str, required=True)
    p.add_argument("--ckpt_fnoA", type=str, required=True)
    p.add_argument("--ckpt_unoA", type=str, default="")
    p.add_argument("--ckpt_cftaoA", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--cpu", action="store_true")

    # backbone shared
    p.add_argument("--modes", type=int, default=12)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--uno_pad", type=int, default=8)
    p.add_argument("--uno_factor", type=int, default=1)

    # A-wrapper args (match eval scripts)
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--res_scale_init", type=float, default=0.02)
    p.add_argument("--res_scale_max", type=float, default=0.25)
    p.add_argument("--ext_method", type=str, default="harmonic", choices=["zero", "coons", "harmonic", "poisson", "poisson_learned"])
    p.add_argument("--ext_iters", type=int, default=80)
    p.add_argument("--poisson_src_hidden", type=int, default=32)
    p.add_argument("--poisson_src_scale_max", type=float, default=1.0)
    p.add_argument("--residual_clip", type=float, default=None, help="If omitted, auto-selects 3.0 for fnoA and 0.0 for cftaoA.")

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

    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Using device:", device)

    data = torch.load(args.data_path, map_location="cpu")
    a_train_raw = data["a_train"]
    u_train_raw = data["u_train"]
    a_test_raw = data["a_test"]

    a_norm = UnitGaussianNormalizer(a_train_raw).to(device)
    u_norm = UnitGaussianNormalizer(u_train_raw).to(device)

    # build a batch input (use real test samples)
    bs = min(int(args.batch_size), int(a_test_raw.shape[0]))
    a_raw_b = a_test_raw[:bs].to(device)
    a_b = a_norm.encode(a_raw_b)

    in_channels = int(a_b.shape[-1])
    out_channels = int(u_train_raw.shape[-1])

    # mixed input for A-wrap
    geom_raw = a_raw_b[..., 0:1]
    bc_raw = a_raw_b[..., 1:2]
    x_mix = torch.cat([a_b, geom_raw, bc_raw], dim=-1)

    results: Dict[str, Dict] = {}

    # plain
    plain = _build_plain(args, in_channels=in_channels, out_channels=out_channels).to(device).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    per_ms, _ = _time_forward(lambda: plain(a_b), device, args.repeats, args.warmup)
    mem = _peak_mem_mb(device)
    results["plain"] = {"params": int(count_params(plain)), "ms_per_batch": per_ms, "peak_mem_mb": mem}

    # A-models
    model_specs = [("fnoA", "fnoA")]
    if args.ckpt_unoA:
        model_specs.append(("unoA", "unoA"))
    model_specs.append(("cftaoA", "cftaoA"))

    for name, mt in model_specs:
        m = _build_A(args, model_type=mt, in_channels=in_channels, out_channels=out_channels, y_mean=u_norm.mean, y_std=u_norm.std).to(device).eval()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        per_ms_fwd, _ = _time_forward(lambda: m(x_mix), device, args.repeats, args.warmup)
        mem_fwd = _peak_mem_mb(device)
        # extension-only overhead
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        per_ms_ext, _ = _time_forward(lambda: m.build_extension_raw(x_mix), device, args.repeats, args.warmup)
        mem_ext = _peak_mem_mb(device)

        results[name] = {
            "params": int(count_params(m)),
            "ms_per_batch": per_ms_fwd,
            "peak_mem_mb": mem_fwd,
            "ms_ext_only": per_ms_ext,
            "peak_mem_ext_mb": mem_ext,
        }

    # Print markdown-ish table
    print("\n=== Cost table (per batch) ===")
    print("model | params | ms/batch | peak_mem(MB) | ext_ms/batch | ext_peak_mem(MB)")
    ordered_keys = ["plain", "fnoA"] + (["unoA"] if "unoA" in results else []) + ["cftaoA"]
    for k in ordered_keys:
        r = results[k]
        print(
            f"{k:>5s} | {r['params']:>7d} | {r['ms_per_batch']:.3f} | {r['peak_mem_mb']:.1f} | "
            f"{r.get('ms_ext_only', 0.0):.3f} | {r.get('peak_mem_ext_mb', 0.0):.1f}"
        )

    # Save json-like dict
    out = {
        "data_path": args.data_path,
        "batch_size": bs,
        "device": str(device),
        "results": results,
    }
    torch.save(out, "cost_profile_2d.pt")
    print("\nSaved: cost_profile_2d.pt")


if __name__ == "__main__":
    main()

