from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import torch


TASK_DEFAULTS = {
    "diffusion": {"paper_iters": 800, "ref_iters": 4000},
    "pflow": {"paper_iters": 800, "ref_iters": 4000},
    "poisson": {"paper_iters": 1200, "ref_iters": 5000},
    "varcoeff": {"paper_iters": 1200, "ref_iters": 5000},
}


def _boundary_mask_like(x: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, 0, :, :] = True
    mask[:, -1, :, :] = True
    mask[:, :, 0, :] = True
    mask[:, :, -1, :] = True
    return mask


def _dirichlet_mask(geom: torch.Tensor) -> torch.Tensor:
    return _boundary_mask_like(geom) | (geom > 0.5)


def _fluid_mask(geom: torch.Tensor) -> torch.Tensor:
    return ~_dirichlet_mask(geom)


def _relative_l2(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    num = torch.sqrt(torch.sum(((pred - target) * mask) ** 2) + 1e-12)
    den = torch.sqrt(torch.sum((target * mask) ** 2) + 1e-12)
    return float((num / den).item())


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    mask_f = mask.to(pred.dtype)
    denom = torch.sum(mask_f).clamp_min(1.0)
    return float((torch.sum(((pred - target) ** 2) * mask_f) / denom).item())


def _laplace_step(u: torch.Tensor, dir_mask: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
    u_old = u
    u_e = torch.roll(u_old, shifts=-1, dims=1)
    u_w = torch.roll(u_old, shifts=1, dims=1)
    u_n = torch.roll(u_old, shifts=-1, dims=2)
    u_s = torch.roll(u_old, shifts=1, dims=2)
    u_new = 0.25 * (u_e + u_w + u_n + u_s)
    out = torch.where(~dir_mask, u_new, u_old)
    return torch.where(dir_mask, bc, out)


def _poisson_step(u: torch.Tensor, dir_mask: torch.Tensor, bc: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    n = int(u.shape[1])
    h = 1.0 / float(max(n - 1, 1))
    h2 = h * h
    u_old = u
    u_e = torch.roll(u_old, shifts=-1, dims=1)
    u_w = torch.roll(u_old, shifts=1, dims=1)
    u_n = torch.roll(u_old, shifts=-1, dims=2)
    u_s = torch.roll(u_old, shifts=1, dims=2)
    u_new = 0.25 * (u_e + u_w + u_n + u_s + h2 * src)
    out = torch.where(~dir_mask, u_new, u_old)
    return torch.where(dir_mask, bc, out)


def _harmonic_mean(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return (2.0 * a * b) / (a + b + eps)


def _varcoeff_faces(kappa: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ke = torch.zeros_like(kappa)
    kw = torch.zeros_like(kappa)
    kn = torch.zeros_like(kappa)
    ks = torch.zeros_like(kappa)
    ke[:, :-1, :, :] = _harmonic_mean(kappa[:, :-1, :, :], kappa[:, 1:, :, :])
    kw[:, 1:, :, :] = _harmonic_mean(kappa[:, 1:, :, :], kappa[:, :-1, :, :])
    kn[:, :, :-1, :] = _harmonic_mean(kappa[:, :, :-1, :], kappa[:, :, 1:, :])
    ks[:, :, 1:, :] = _harmonic_mean(kappa[:, :, 1:, :], kappa[:, :, :-1, :])
    return ke, kw, kn, ks


def _varcoeff_step(
    u: torch.Tensor,
    dir_mask: torch.Tensor,
    bc: torch.Tensor,
    kappa: torch.Tensor,
) -> torch.Tensor:
    ke, kw, kn, ks = _varcoeff_faces(torch.clamp(kappa, min=1e-4))
    denom = ke + kw + kn + ks + 1e-12
    u_old = u
    u_e = torch.roll(u_old, shifts=-1, dims=1)
    u_w = torch.roll(u_old, shifts=1, dims=1)
    u_n = torch.roll(u_old, shifts=-1, dims=2)
    u_s = torch.roll(u_old, shifts=1, dims=2)
    u_new = (ke * u_e + kw * u_w + kn * u_n + ks * u_s) / denom
    out = torch.where(~dir_mask, u_new, u_old)
    return torch.where(dir_mask, bc, out)


def _laplace_residual(u: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
    fluid = _fluid_mask(geom)
    u_e = torch.roll(u, shifts=-1, dims=1)
    u_w = torch.roll(u, shifts=1, dims=1)
    u_n = torch.roll(u, shifts=-1, dims=2)
    u_s = torch.roll(u, shifts=1, dims=2)
    res = 4.0 * u - (u_e + u_w + u_n + u_s)
    return torch.where(fluid, res, torch.zeros_like(res))


def _poisson_residual(u: torch.Tensor, geom: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    fluid = _fluid_mask(geom)
    n = int(u.shape[1])
    h = 1.0 / float(max(n - 1, 1))
    h2 = h * h
    u_e = torch.roll(u, shifts=-1, dims=1)
    u_w = torch.roll(u, shifts=1, dims=1)
    u_n = torch.roll(u, shifts=-1, dims=2)
    u_s = torch.roll(u, shifts=1, dims=2)
    res = 4.0 * u - (u_e + u_w + u_n + u_s + h2 * src)
    return torch.where(fluid, res, torch.zeros_like(res))


def _varcoeff_residual(u: torch.Tensor, geom: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    fluid = _fluid_mask(geom)
    ke, kw, kn, ks = _varcoeff_faces(torch.clamp(kappa, min=1e-4))
    u_e = torch.roll(u, shifts=-1, dims=1)
    u_w = torch.roll(u, shifts=1, dims=1)
    u_n = torch.roll(u, shifts=-1, dims=2)
    u_s = torch.roll(u, shifts=1, dims=2)
    res = ke * (u - u_e) + kw * (u - u_w) + kn * (u - u_n) + ks * (u - u_s)
    return torch.where(fluid, res, torch.zeros_like(res))


def _residual_stats(residual: torch.Tensor, fluid_mask: torch.Tensor) -> dict[str, float]:
    mask = fluid_mask.to(residual.dtype)
    denom = torch.sum(mask).clamp_min(1.0)
    abs_res = residual.abs()
    return {
        "res_l1_mean": float((torch.sum(abs_res * mask) / denom).item()),
        "res_l2_rms": float(torch.sqrt(torch.sum((residual ** 2) * mask) / denom + 1e-12).item()),
        "res_linf": float(torch.max(abs_res * mask).item()),
    }


def _parse_int_list(text: str) -> list[int]:
    vals = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if chunk:
            vals.append(int(chunk))
    return sorted(set(vals))


def _select_split(data: dict, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    prefix = "train" if split == "train" else "test"
    return data[f"a_{prefix}"], data[f"u_{prefix}"]


def _task_components(task: str, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    if task in {"diffusion", "pflow"}:
        return a[..., 1:2], None
    if task == "poisson":
        return a[..., 1:2], a[..., 2:3]
    if task == "varcoeff":
        return a[..., 1:2], a[..., 2:3]
    raise ValueError(f"Unsupported task: {task}")


def _iterate_to_checkpoints(
    *,
    task: str,
    geom: torch.Tensor,
    bc: torch.Tensor,
    aux: torch.Tensor | None,
    checkpoints: list[int],
) -> dict[int, torch.Tensor]:
    dir_mask = _dirichlet_mask(geom)
    u = torch.where(dir_mask, bc, torch.zeros_like(bc))
    out: dict[int, torch.Tensor] = {0: u.clone()}
    max_iter = max(checkpoints)
    for step in range(1, max_iter + 1):
        if task in {"diffusion", "pflow"}:
            u = _laplace_step(u, dir_mask, bc)
        elif task == "poisson":
            assert aux is not None
            u = _poisson_step(u, dir_mask, bc, aux)
        elif task == "varcoeff":
            assert aux is not None
            u = _varcoeff_step(u, dir_mask, bc, aux)
        else:
            raise ValueError(task)
        if step in checkpoints:
            out[step] = u.clone()
    return out


def _compute_residual(task: str, u: torch.Tensor, geom: torch.Tensor, aux: torch.Tensor | None) -> torch.Tensor:
    if task in {"diffusion", "pflow"}:
        return _laplace_residual(u, geom)
    if task == "poisson":
        assert aux is not None
        return _poisson_residual(u, geom, aux)
    if task == "varcoeff":
        assert aux is not None
        return _varcoeff_residual(u, geom, aux)
    raise ValueError(task)


def audit_sample(
    *,
    task: str,
    a_sample: torch.Tensor,
    paper_iters: int,
    ref_iters: int,
    checkpoints: list[int],
) -> dict:
    geom = a_sample[..., 0:1].unsqueeze(0)
    bc, aux = _task_components(task, a_sample.unsqueeze(0))
    fluid = _fluid_mask(geom)
    run_iters = sorted(set([0, paper_iters, ref_iters] + checkpoints))
    snapshots = _iterate_to_checkpoints(task=task, geom=geom, bc=bc, aux=aux, checkpoints=run_iters)
    u_ref = snapshots[ref_iters]

    rows = []
    for it in run_iters:
        u_it = snapshots[it]
        residual = _compute_residual(task, u_it, geom, aux)
        row = {
            "iters": int(it),
            "mse_full_vs_ref": float(torch.mean((u_it - u_ref) ** 2).item()),
            "mse_fluid_vs_ref": _masked_mse(u_it, u_ref, fluid),
            "rel_l2_fluid_vs_ref": _relative_l2(u_it, u_ref, fluid.to(u_it.dtype)),
        }
        row.update(_residual_stats(residual, fluid))
        rows.append(row)

    paper_row = next(row for row in rows if row["iters"] == paper_iters)
    return {
        "paper_iters": int(paper_iters),
        "ref_iters": int(ref_iters),
        "rows": rows,
        "paper_vs_ref_summary": {
            "mse_full_vs_ref": paper_row["mse_full_vs_ref"],
            "mse_fluid_vs_ref": paper_row["mse_fluid_vs_ref"],
            "rel_l2_fluid_vs_ref": paper_row["rel_l2_fluid_vs_ref"],
            "res_l2_rms": paper_row["res_l2_rms"],
            "res_linf": paper_row["res_linf"],
        },
    }


def _print_markdown(results: list[dict]) -> None:
    for item in results:
        print(f"\n## sample_idx={item['sample_idx']}")
        print("| iters | mse_full_vs_ref | mse_fluid_vs_ref | rel_l2_fluid_vs_ref | res_l2_rms | res_linf |")
        print("| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in item["audit"]["rows"]:
            print(
                f"| {row['iters']} | {row['mse_full_vs_ref']:.6e} | {row['mse_fluid_vs_ref']:.6e} | "
                f"{row['rel_l2_fluid_vs_ref']:.6e} | {row['res_l2_rms']:.6e} | {row['res_linf']:.6e} |"
            )


def _collect_paper_rows(results: list[dict]) -> list[dict]:
    rows = []
    for item in results:
        summary = item["audit"]["paper_vs_ref_summary"]
        rows.append(
            {
                "sample_idx": int(item["sample_idx"]),
                "mse_full_vs_ref": float(summary["mse_full_vs_ref"]),
                "mse_fluid_vs_ref": float(summary["mse_fluid_vs_ref"]),
                "rel_l2_fluid_vs_ref": float(summary["rel_l2_fluid_vs_ref"]),
                "res_l2_rms": float(summary["res_l2_rms"]),
                "res_linf": float(summary["res_linf"]),
            }
        )
    return rows


def _mean_max(values: list[float]) -> dict[str, float]:
    xs = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(xs.mean().item()),
        "max": float(xs.max().item()),
    }


def _build_summary(results: list[dict]) -> dict:
    paper_rows = _collect_paper_rows(results)
    return {
        "paper_rows": paper_rows,
        "aggregates": {
            "mse_full_vs_ref": _mean_max([row["mse_full_vs_ref"] for row in paper_rows]),
            "mse_fluid_vs_ref": _mean_max([row["mse_fluid_vs_ref"] for row in paper_rows]),
            "rel_l2_fluid_vs_ref": _mean_max([row["rel_l2_fluid_vs_ref"] for row in paper_rows]),
            "res_l2_rms": _mean_max([row["res_l2_rms"] for row in paper_rows]),
            "res_linf": _mean_max([row["res_linf"] for row in paper_rows]),
        },
    }


def _print_summary_markdown(*, task: str, paper_iters: int, ref_iters: int, summary: dict) -> None:
    print(f"\n## summary_at_{paper_iters}_vs_{ref_iters}")
    print("| sample_idx | mse_full_vs_ref | mse_fluid_vs_ref | rel_l2_fluid_vs_ref | res_l2_rms | res_linf |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in summary["paper_rows"]:
        print(
            f"| {row['sample_idx']} | {row['mse_full_vs_ref']:.6e} | {row['mse_fluid_vs_ref']:.6e} | "
            f"{row['rel_l2_fluid_vs_ref']:.6e} | {row['res_l2_rms']:.6e} | {row['res_linf']:.6e} |"
        )

    agg = summary["aggregates"]
    print("\n## aggregate_summary")
    print("| task | paper_iters | ref_iters | rel_l2_fluid_mean | rel_l2_fluid_max | res_l2_rms_mean | res_l2_rms_max |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    print(
        f"| {task} | {paper_iters} | {ref_iters} | "
        f"{agg['rel_l2_fluid_vs_ref']['mean']:.6e} | {agg['rel_l2_fluid_vs_ref']['max']:.6e} | "
        f"{agg['res_l2_rms']['mean']:.6e} | {agg['res_l2_rms']['max']:.6e} |"
    )


def _print_summary_latex(*, task: str, paper_iters: int, ref_iters: int, summary: dict) -> None:
    print("\n--- LaTeX summary table ---")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(
        rf"Sample & MSE(full, ${paper_iters}$ vs.\ ${ref_iters}$) & "
        rf"Rel.\ $L_2$(fluid, ${paper_iters}$ vs.\ ${ref_iters}$) & RMS residual at ${paper_iters}$ \\"
    )
    print(r"\midrule")
    for row in summary["paper_rows"]:
        print(
            rf"{row['sample_idx']} & {row['mse_full_vs_ref']:.3e} & "
            rf"{row['rel_l2_fluid_vs_ref']:.3e} & {row['res_l2_rms']:.3e} \\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")
    agg = summary["aggregates"]
    print(
        f"\n# Aggregate ({task}): rel_l2_fluid mean={agg['rel_l2_fluid_vs_ref']['mean']:.3e}, "
        f"max={agg['rel_l2_fluid_vs_ref']['max']:.3e}; res_l2_rms mean={agg['res_l2_rms']['mean']:.3e}, "
        f"max={agg['res_l2_rms']['max']:.3e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Jacobi reference-solver convergence on representative benchmark samples.")
    parser.add_argument("--task", type=str, required=True, choices=sorted(TASK_DEFAULTS.keys()))
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--sample_indices", type=str, default="0")
    parser.add_argument("--paper_iters", type=int, default=-1, help="Use task default when negative.")
    parser.add_argument("--ref_iters", type=int, default=-1, help="Use task default when negative.")
    parser.add_argument(
        "--check_iters",
        type=str,
        default="100,200,400,800,1200,2000,4000,5000",
        help="Comma-separated extra checkpoints.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--json_out", type=str, default="")
    args = parser.parse_args()

    defaults = TASK_DEFAULTS[args.task]
    paper_iters = defaults["paper_iters"] if args.paper_iters < 0 else int(args.paper_iters)
    ref_iters = defaults["ref_iters"] if args.ref_iters < 0 else int(args.ref_iters)
    checkpoints = [it for it in _parse_int_list(args.check_iters) if it <= ref_iters]

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    data = torch.load(args.data_path, map_location="cpu")
    a_split, _ = _select_split(data, args.split)
    a_split = a_split.to(device)

    sample_indices = _parse_int_list(args.sample_indices)
    results: list[dict] = []
    for idx in sample_indices:
        audit = audit_sample(
            task=args.task,
            a_sample=a_split[idx],
            paper_iters=paper_iters,
            ref_iters=ref_iters,
            checkpoints=checkpoints,
        )
        results.append({"sample_idx": int(idx), "audit": audit})

    payload = {
        "task": args.task,
        "data_path": str(Path(args.data_path)),
        "split": args.split,
        "sample_indices": sample_indices,
        "paper_iters": paper_iters,
        "ref_iters": ref_iters,
        "results": results,
    }
    payload["summary"] = _build_summary(results)

    print(
        f"Loaded {args.task} data from {args.data_path} | split={args.split} | "
        f"paper_iters={paper_iters} | ref_iters={ref_iters}"
    )
    _print_markdown(results)
    _print_summary_markdown(task=args.task, paper_iters=paper_iters, ref_iters=ref_iters, summary=payload["summary"])
    _print_summary_latex(task=args.task, paper_iters=paper_iters, ref_iters=ref_iters, summary=payload["summary"])

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved JSON audit to: {out_path}")


if __name__ == "__main__":
    main()
