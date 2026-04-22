from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boundary_ext_residual_2d import ResidualOnDirichletExtension2D
from cft_ao_2d import CFT_AO_2D_Atlas
from fourier_2d_baseline import FNO2d
from utilities3 import UnitGaussianNormalizer, count_params


def _default_residual_clip(model_type: str) -> float:
    if model_type == "fnoA":
        return 3.0
    if model_type == "cftaoA":
        return 0.0
    raise ValueError(model_type)


def _dir_masks(geom_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bdry = torch.zeros_like(geom_raw, dtype=torch.bool)
    bdry[:, 0, :, :] = True
    bdry[:, -1, :, :] = True
    bdry[:, :, 0, :] = True
    bdry[:, :, -1, :] = True
    obs = geom_raw > 0.5
    fluid = ~(bdry | obs)
    return bdry, obs, fluid


def _masked_mse(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    mask_f = mask.to(x.dtype)
    denom = torch.sum(mask_f).clamp_min(1.0)
    return float((torch.sum(((x - y) ** 2) * mask_f) / denom).item())


def _build_fnoA(in_channels: int, out_channels: int, y_mean: torch.Tensor, y_std: torch.Tensor) -> ResidualOnDirichletExtension2D:
    backbone = FNO2d(
        modes1=12,
        modes2=12,
        width=64,
        in_channels=in_channels,
        out_channels=out_channels,
    )
    return ResidualOnDirichletExtension2D(
        backbone,
        y_mean=y_mean,
        y_std=y_std,
        in_channels_norm=in_channels,
        delta=0.05,
        res_scale_init=0.02,
        res_scale_max=0.25,
        ext_method="harmonic",
        ext_iters=80,
        poisson_src_hidden=32,
        poisson_src_scale_max=1.0,
        residual_clip=3.0,
    )


def _build_cftaoA(in_channels: int, out_channels: int, y_mean: torch.Tensor, y_std: torch.Tensor) -> ResidualOnDirichletExtension2D:
    backbone = CFT_AO_2D_Atlas(
        modes1=12,
        modes2=12,
        width=64,
        in_channels=in_channels,
        out_channels=out_channels,
        n_layers=4,
        L_segments=4,
        M_cheb=4,
        L_segments_boundary=8,
        M_cheb_boundary=8,
        cft_res=0,
        use_local=True,
        rim_ratio=0.15,
        cond_dim=2,
        inner_iters=2,
        n_bands=3,
        n_sym_bases=0,
    )
    return ResidualOnDirichletExtension2D(
        backbone,
        y_mean=y_mean,
        y_std=y_std,
        in_channels_norm=in_channels,
        delta=0.05,
        res_scale_init=0.02,
        res_scale_max=0.25,
        ext_method="harmonic",
        ext_iters=80,
        poisson_src_hidden=32,
        poisson_src_scale_max=1.0,
        residual_clip=0.0,
    )


def _build_model(model_type: str, in_channels: int, out_channels: int, y_mean: torch.Tensor, y_std: torch.Tensor):
    if model_type == "fnoA":
        return _build_fnoA(in_channels, out_channels, y_mean, y_std)
    if model_type == "cftaoA":
        return _build_cftaoA(in_channels, out_channels, y_mean, y_std)
    raise ValueError(model_type)


@torch.no_grad()
def _evaluate_split(
    *,
    model: ResidualOnDirichletExtension2D,
    a_raw: torch.Tensor,
    u_raw: torch.Tensor,
    a_normalizer: UnitGaussianNormalizer,
    u_normalizer: UnitGaussianNormalizer,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    a_raw = a_raw.to(device)
    u_raw = u_raw.to(device)
    a_norm = a_normalizer.encode(a_raw)
    x_mix = torch.cat([a_norm, a_raw[..., 0:1], a_raw[..., 1:2]], dim=-1)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_mix, u_raw),
        batch_size=batch_size,
        shuffle=False,
    )

    num_test = int(u_raw.shape[0])
    num_pts = int(u_raw.shape[1] * u_raw.shape[2])
    mse_raw = 0.0
    mse_eonly = 0.0
    mse_bdry = 0.0
    mse_obs = 0.0
    mse_target_corr_fluid = 0.0
    mse_pred_corr_fit_fluid = 0.0
    mse_pred_corr_energy_fluid = 0.0
    fluid_cnt = 0.0
    bdry_cnt = 0.0
    obs_cnt = 0.0

    model.eval()
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        out_enc = model(xb)
        out_raw = u_normalizer.decode(out_enc)
        e_raw = model.build_extension_raw(xb)

        geom = xb[..., model.in_channels_norm : model.in_channels_norm + 1]
        bdry, obs, fluid = _dir_masks(geom)
        target_corr = yb - e_raw
        pred_corr = out_raw - e_raw

        mse_raw += F.mse_loss(out_raw, yb, reduction="sum").item()
        mse_eonly += F.mse_loss(e_raw, yb, reduction="sum").item()
        mse_bdry += ((out_raw - xb[..., model.in_channels_norm + 1 : model.in_channels_norm + 2]) ** 2)[bdry].sum().item()
        mse_obs += ((out_raw - xb[..., model.in_channels_norm + 1 : model.in_channels_norm + 2]) ** 2)[obs].sum().item()
        mse_target_corr_fluid += ((target_corr ** 2) * fluid.to(target_corr.dtype)).sum().item()
        mse_pred_corr_fit_fluid += (((pred_corr - target_corr) ** 2) * fluid.to(pred_corr.dtype)).sum().item()
        mse_pred_corr_energy_fluid += ((pred_corr ** 2) * fluid.to(pred_corr.dtype)).sum().item()
        fluid_cnt += float(fluid.sum().item())
        bdry_cnt += float(bdry.sum().item())
        obs_cnt += float(obs.sum().item())

    raw_mse_px = mse_raw / (num_test * num_pts)
    eonly_mse_px = mse_eonly / (num_test * num_pts)
    target_corr_mse_fluid = mse_target_corr_fluid / max(fluid_cnt, 1.0)
    pred_corr_fit_mse_fluid = mse_pred_corr_fit_fluid / max(fluid_cnt, 1.0)
    pred_corr_energy_mse_fluid = mse_pred_corr_energy_fluid / max(fluid_cnt, 1.0)
    gain = 0.0
    if eonly_mse_px > 0:
        gain = 1.0 - raw_mse_px / eonly_mse_px

    return {
        "raw_mse_px": raw_mse_px,
        "eonly_mse_px": eonly_mse_px,
        "gain_over_eonly_frac": gain,
        "viol_bdry_mse": mse_bdry / max(bdry_cnt, 1.0),
        "viol_obs_mse": mse_obs / max(obs_cnt, 1.0),
        "target_corr_mse_fluid": target_corr_mse_fluid,
        "pred_corr_fit_mse_fluid": pred_corr_fit_mse_fluid,
        "pred_corr_energy_mse_fluid": pred_corr_energy_mse_fluid,
    }


def _pretty_print(results: dict) -> None:
    print("\n| model | split | raw_mse_px | eonly_mse_px | gain_over_eonly | target_corr_mse_fluid | pred_corr_fit_mse_fluid |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for model_name, model_res in results.items():
        for split_name, metrics in model_res.items():
            print(
                f"| {model_name} | {split_name} | {metrics['raw_mse_px']:.6e} | {metrics['eonly_mse_px']:.6e} | "
                f"{metrics['gain_over_eonly_frac']:.4f} | {metrics['target_corr_mse_fluid']:.6e} | "
                f"{metrics['pred_corr_fit_mse_fluid']:.6e} |"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze E-only and residual-correction load on pflow ID/OOD splits.")
    parser.add_argument("--id_data_path", type=str, required=True, help="In-distribution pflow dataset with train/test split.")
    parser.add_argument("--ood_data_path", type=str, required=True, help="OOD star-obstacle pflow dataset.")
    parser.add_argument("--ckpt_fnoA", type=str, default="")
    parser.add_argument("--ckpt_cftaoA", type=str, default="")
    parser.add_argument("--model_type", type=str, default="both", choices=["both", "fnoA", "cftaoA"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--json_out", type=str, default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    id_data = torch.load(args.id_data_path, map_location="cpu")
    ood_data = torch.load(args.ood_data_path, map_location="cpu")

    a_train = id_data["a_train"]
    u_train = id_data["u_train"]
    a_id = id_data["a_test"]
    u_id = id_data["u_test"]
    a_ood = ood_data["a_test"]
    u_ood = ood_data["u_test"]

    a_normalizer = UnitGaussianNormalizer(a_train).to(device)
    u_normalizer = UnitGaussianNormalizer(u_train).to(device)
    in_channels = int(a_train.shape[-1])
    out_channels = int(u_train.shape[-1])

    model_names = ["fnoA", "cftaoA"] if args.model_type == "both" else [args.model_type]
    ckpt_map = {"fnoA": args.ckpt_fnoA, "cftaoA": args.ckpt_cftaoA}
    results: dict[str, dict[str, dict[str, float]]] = {}
    meta: dict[str, dict[str, int]] = {}

    for model_name in model_names:
        ckpt_path = ckpt_map[model_name]
        if not ckpt_path:
            raise ValueError(f"Checkpoint path for {model_name} is required.")
        model = _build_model(model_name, in_channels, out_channels, u_normalizer.mean, u_normalizer.std).to(device)
        state = torch.load(ckpt_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded ckpt for {model_name}: {ckpt_path}")
        print(f"Using residual_clip={_default_residual_clip(model_name)} for {model_name}")
        if missing or unexpected:
            print(f"[WARN] Missing keys for {model_name}: {missing}")
            print(f"[WARN] Unexpected keys for {model_name}: {unexpected}")
            raise RuntimeError(f"Checkpoint for {model_name} does not exactly match the rebuilt model.")
        results[model_name] = {
            "id_test": _evaluate_split(
                model=model,
                a_raw=a_id,
                u_raw=u_id,
                a_normalizer=a_normalizer,
                u_normalizer=u_normalizer,
                batch_size=args.batch_size,
                device=device,
            ),
            "ood_test": _evaluate_split(
                model=model,
                a_raw=a_ood,
                u_raw=u_ood,
                a_normalizer=a_normalizer,
                u_normalizer=u_normalizer,
                batch_size=args.batch_size,
                device=device,
            ),
        }
        id_target = results[model_name]["id_test"]["target_corr_mse_fluid"]
        ood_target = results[model_name]["ood_test"]["target_corr_mse_fluid"]
        id_eonly = results[model_name]["id_test"]["eonly_mse_px"]
        ood_eonly = results[model_name]["ood_test"]["eonly_mse_px"]
        results[model_name]["load_shift"] = {
            "target_corr_fluid_ratio_ood_over_id": (ood_target / id_target) if id_target > 0 else 0.0,
            "eonly_ratio_ood_over_id": (ood_eonly / id_eonly) if id_eonly > 0 else 0.0,
        }
        meta[model_name] = {"params": int(count_params(model))}

    print(f"Using device: {device}")
    print(f"ID data:  {args.id_data_path}")
    print(f"OOD data: {args.ood_data_path}")
    for model_name, info in meta.items():
        print(f"{model_name} params: {info['params']}")
    _pretty_print({k: {"id_test": v["id_test"], "ood_test": v["ood_test"]} for k, v in results.items()})
    for model_name in model_names:
        shift = results[model_name]["load_shift"]
        print(
            f"\n[{model_name}] target_corr_fluid_ratio_ood_over_id={shift['target_corr_fluid_ratio_ood_over_id']:.4f} | "
            f"eonly_ratio_ood_over_id={shift['eonly_ratio_ood_over_id']:.4f}"
        )

    payload = {
        "id_data_path": str(Path(args.id_data_path)),
        "ood_data_path": str(Path(args.ood_data_path)),
        "device": str(device),
        "models": results,
        "meta": meta,
    }
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved analysis JSON to: {out_path}")


if __name__ == "__main__":
    main()
