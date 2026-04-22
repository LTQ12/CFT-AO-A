from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_pflow_A_models import _dir_masks, build_model
from utilities3 import UnitGaussianNormalizer


def _parse_float_list(text: str) -> list[float]:
    vals = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if chunk:
            vals.append(float(chunk))
    return vals


@torch.no_grad()
def _evaluate(
    *,
    model,
    x_mix: torch.Tensor,
    u_raw: torch.Tensor,
    u_normalizer: UnitGaussianNormalizer,
    batch_size: int,
    in_channels: int,
) -> dict[str, float]:
    n_test = int(u_raw.shape[0])
    num_pts = int(u_raw.shape[1] * u_raw.shape[2])

    raw_mse_sum = 0.0
    eonly_mse_sum = 0.0
    bdry_mse_sum = 0.0
    obs_mse_sum = 0.0
    bdry_cnt = 0.0
    obs_cnt = 0.0
    max_gamma_list = []

    for i in range(0, n_test, batch_size):
        xb = x_mix[i : i + batch_size]
        yb = u_raw[i : i + batch_size]

        out_enc = model(xb)
        out = u_normalizer.decode(out_enc)
        e_raw = model.build_extension_raw(xb)

        geom_raw = xb[..., in_channels : in_channels + 1]
        bc_raw = xb[..., in_channels + 1 : in_channels + 2]
        bdry, obs = _dir_masks(geom_raw)
        diff_bc2 = (out - bc_raw) ** 2

        raw_mse_sum += F.mse_loss(out, yb, reduction="sum").item()
        eonly_mse_sum += F.mse_loss(e_raw, yb, reduction="sum").item()
        bdry_mse_sum += diff_bc2[bdry].sum().item()
        obs_mse_sum += diff_bc2[obs].sum().item()
        bdry_cnt += float(bdry.sum().item())
        obs_cnt += float(obs.sum().item())

        dir_mask = bdry | obs
        abs_diff = (out - bc_raw).abs()
        abs_diff_flat = abs_diff.view(int(abs_diff.shape[0]), -1)
        dir_flat = dir_mask.view(int(dir_mask.shape[0]), -1)
        for bi in range(int(abs_diff.shape[0])):
            if dir_flat[bi].any():
                max_gamma_list.append(float(abs_diff_flat[bi][dir_flat[bi]].max().detach().cpu()))
            else:
                max_gamma_list.append(0.0)

    raw_mse = raw_mse_sum / (n_test * num_pts)
    eonly_mse = eonly_mse_sum / (n_test * num_pts)
    max_gamma = torch.tensor(max_gamma_list, dtype=torch.float64)
    return {
        "raw_mse_px": raw_mse,
        "eonly_mse_px": eonly_mse,
        "gain_over_eonly_frac": 1.0 - raw_mse / eonly_mse if eonly_mse > 0 else 0.0,
        "viol_bdry_mse": bdry_mse_sum / max(bdry_cnt, 1.0),
        "viol_obs_mse": obs_mse_sum / max(obs_cnt, 1.0),
        "max_gamma_mean": float(max_gamma.mean().item()),
        "max_gamma_p95": float(torch.quantile(max_gamma, 0.95).item()),
        "max_gamma_max": float(max_gamma.max().item()),
    }


def _build_args(base_args, *, model_type: str, residual_clip: float):
    return SimpleNamespace(
        model_type=model_type,
        modes=base_args.modes,
        width=base_args.width,
        n_layers=base_args.n_layers,
        cft_L=base_args.cft_L,
        cft_M=base_args.cft_M,
        cft_L_boundary=base_args.cft_L_boundary,
        cft_M_boundary=base_args.cft_M_boundary,
        cft_res=base_args.cft_res,
        no_local=base_args.no_local,
        rim_ratio=base_args.rim_ratio,
        inner_iters=base_args.inner_iters,
        n_bands=base_args.n_bands,
        n_sym_bases=base_args.n_sym_bases,
        delta=base_args.delta,
        res_scale_init=base_args.res_scale_init,
        res_scale_max=base_args.res_scale_max,
        ext_method=base_args.ext_method,
        ext_iters=base_args.ext_iters,
        poisson_src_hidden=base_args.poisson_src_hidden,
        poisson_src_scale_max=base_args.poisson_src_scale_max,
        residual_clip=residual_clip,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Run pflow fairness ablation over residual_clip choices.")
    p.add_argument("--data_path", type=str, default="data/pflow_obstacle2d_N64.pt")
    p.add_argument("--ckpt_fnoA", type=str, default="")
    p.add_argument("--ckpt_cftaoA", type=str, default="")
    p.add_argument("--model_type", type=str, default="both", choices=["both", "fnoA", "cftaoA"])
    p.add_argument("--clips_fnoA", type=str, default="0.0,3.0")
    p.add_argument("--clips_cftaoA", type=str, default="0.0,3.0")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--json_out", type=str, default="")

    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--res_scale_init", type=float, default=0.02)
    p.add_argument("--res_scale_max", type=float, default=0.25)
    p.add_argument("--ext_method", type=str, default="harmonic", choices=["zero", "coons", "harmonic", "poisson", "poisson_learned"])
    p.add_argument("--ext_iters", type=int, default=80)
    p.add_argument("--poisson_src_hidden", type=int, default=32)
    p.add_argument("--poisson_src_scale_max", type=float, default=1.0)

    p.add_argument("--modes", type=int, default=12)
    p.add_argument("--width", type=int, default=64)
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
    data = torch.load(args.data_path, map_location="cpu")
    a_train_raw = data["a_train"]
    u_train_raw = data["u_train"]
    a_test_raw = data["a_test"]
    u_test_raw = data["u_test"]

    a_normalizer = UnitGaussianNormalizer(a_train_raw).to(device)
    u_normalizer = UnitGaussianNormalizer(u_train_raw).to(device)
    a_test_raw = a_test_raw.to(device)
    u_test_raw = u_test_raw.to(device)
    a_test = a_normalizer.encode(a_test_raw)
    x_mix = torch.cat([a_test, a_test_raw[..., 0:1], a_test_raw[..., 1:2]], dim=-1)
    in_channels = int(a_test.shape[-1])
    out_channels = int(u_test_raw.shape[-1])

    grids = {
        "fnoA": _parse_float_list(args.clips_fnoA),
        "cftaoA": _parse_float_list(args.clips_cftaoA),
    }
    ckpts = {"fnoA": args.ckpt_fnoA, "cftaoA": args.ckpt_cftaoA}
    model_names = ["fnoA", "cftaoA"] if args.model_type == "both" else [args.model_type]

    payload = {
        "data_path": str(Path(args.data_path)),
        "device": str(device),
        "protocol": {
            "delta": args.delta,
            "ext_method": args.ext_method,
            "ext_iters": args.ext_iters,
            "res_scale_init": args.res_scale_init,
            "res_scale_max": args.res_scale_max,
        },
        "results": {},
    }

    print("| model | residual_clip | raw_mse_px | eonly_mse_px | gain_over_eonly | viol_bdry | viol_obs | max_gamma_p95 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for model_name in model_names:
        ckpt_path = ckpts[model_name]
        if not ckpt_path:
            raise ValueError(f"Checkpoint path for {model_name} is required.")
        payload["results"][model_name] = {}
        for residual_clip in grids[model_name]:
            build_args = _build_args(args, model_type=model_name, residual_clip=residual_clip)
            model = build_model(
                build_args,
                in_channels=in_channels,
                out_channels=out_channels,
                y_mean=u_normalizer.mean,
                y_std=u_normalizer.std,
            )
            state = torch.load(ckpt_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"Checkpoint mismatch for {model_name} at residual_clip={residual_clip}: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            model.to(device).eval()
            metrics = _evaluate(
                model=model,
                x_mix=x_mix,
                u_raw=u_test_raw,
                u_normalizer=u_normalizer,
                batch_size=args.batch_size,
                in_channels=in_channels,
            )
            payload["results"][model_name][str(residual_clip)] = metrics
            print(
                f"| {model_name} | {residual_clip:.1f} | {metrics['raw_mse_px']:.6e} | {metrics['eonly_mse_px']:.6e} | "
                f"{metrics['gain_over_eonly_frac']:.4f} | {metrics['viol_bdry_mse']:.6e} | "
                f"{metrics['viol_obs_mse']:.6e} | {metrics['max_gamma_p95']:.3e} |"
            )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved JSON fairness report to: {out_path}")


if __name__ == "__main__":
    main()
