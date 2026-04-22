"""
Aggregate multi-seed metrics (*.pt dicts) and print:
  - mean ± std for key metrics
  - a LaTeX table (GB-friendly numeric style) snippet

Metrics dict requirements:
  plain: {"raw_mse","viol_bdry","viol_obs", "max_gamma_mean","max_gamma_p95","max_gamma_max"}
  A-model: additionally has "eonly_mse"

Example:
  python paper_preparation/aggregate_multiseed_metrics.py \
    --metrics_plain "runs/pflow/seed*/plain_metrics.pt" \
    --metrics_fnoA  "runs/pflow/seed*/fnoA_metrics.pt" \
    --metrics_cftaoA "runs/pflow/seed*/cftaoA_metrics.pt" \
    --caption "Multi-seed results (5 seeds) on pflow_obstacle2d_N64." \
    --label "tab:multiseed_pflow"
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch


def _expand(pattern: str) -> List[str]:
    ps = sorted(glob.glob(pattern))
    if not ps:
        raise ValueError(f"No files matched: {pattern}")
    return ps


def _load_all(paths: List[str]) -> List[Dict]:
    out = []
    for p in paths:
        d = torch.load(p, map_location="cpu")
        if not isinstance(d, dict):
            raise ValueError(f"metrics must be dict: {p}")
        out.append(d)
    return out


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    x = np.array([float(v) for v in vals], dtype=np.float64)
    return float(x.mean()), float(x.std(ddof=1)) if x.size >= 2 else 0.0


def _fmt_pm(mean: float, std: float) -> str:
    # scientific 3 sig figs each
    return f"{mean:.3e} $\\pm$ {std:.1e}"


def _collect(ds: List[Dict], key: str) -> List[float]:
    if not ds:
        return []
    if key not in ds[0]:
        raise KeyError(f"Missing key {key} in metrics dict (example keys: {sorted(ds[0].keys())})")
    return [float(d[key]) for d in ds]


def _row(metrics: List[Dict], keys: List[str]) -> List[str]:
    cells = []
    for k in keys:
        m, s = _mean_std(_collect(metrics, k))
        cells.append(_fmt_pm(m, s))
    return cells


def _summary_dict(metrics: List[Dict], keys: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        mean, std = _mean_std(_collect(metrics, k))
        out[k] = {"mean": mean, "std": std}
    return out


def main():
    p = argparse.ArgumentParser(description="Aggregate multi-seed metrics and emit LaTeX table.")
    p.add_argument("--metrics_plain", type=str, required=True, help="Glob for plain FNO metrics.pt")
    p.add_argument("--metrics_fnoA", type=str, required=True, help="Glob for FNO+A metrics.pt")
    p.add_argument("--metrics_cftaoA", type=str, required=True, help="Glob for CFT-AO+A metrics.pt")
    p.add_argument("--metrics_unoA", type=str, default="", help="Optional glob for U-NO+A metrics.pt")
    p.add_argument("--caption", type=str, default="Multi-seed results.")
    p.add_argument("--label", type=str, default="tab:multiseed")
    p.add_argument("--json_out", type=str, default="", help="Optional JSON output with aggregated mean/std.")
    p.add_argument("--markdown", action="store_true", help="Also print a Markdown table.")
    args = p.parse_args()

    plain = _load_all(_expand(args.metrics_plain))
    fnoA = _load_all(_expand(args.metrics_fnoA))
    cftaoA = _load_all(_expand(args.metrics_cftaoA))
    unoA = _load_all(_expand(args.metrics_unoA)) if args.metrics_unoA else []

    # sanity: same #seeds
    n0, n1, n2 = len(plain), len(fnoA), len(cftaoA)
    if not (n0 == n1 == n2):
        raise ValueError(f"seed counts mismatch: plain={n0}, fnoA={n1}, cftaoA={n2}")
    if unoA and len(unoA) != n0:
        raise ValueError(f"seed counts mismatch: plain={n0}, unoA={len(unoA)}")

    keys_plain = ["raw_mse", "viol_bdry", "viol_obs", "max_gamma_p95"]
    keys_A = ["raw_mse", "eonly_mse", "viol_bdry", "viol_obs", "max_gamma_p95"]

    print(f"Loaded seeds: {n0}")
    print("Plain keys:", keys_plain)
    print("A keys    :", keys_A)

    # Print summary
    def show(name: str, ds: List[Dict], keys: List[str]):
        print(f"\n== {name} ==")
        for k in keys:
            m, s = _mean_std(_collect(ds, k))
            print(f"{k:>14s}: {m:.6e}  std={s:.2e}")

    show("FNO (plain)", plain, keys_plain)
    show("FNO + A", fnoA, keys_A)
    if unoA:
        show("U-NO + A", unoA, keys_A)
    show("CFT-AO + A", cftaoA, keys_A)

    payload = {
        "caption": args.caption,
        "label": args.label,
        "num_seeds": n0,
        "models": {
            "plain": _summary_dict(plain, keys_plain),
            "fnoA": _summary_dict(fnoA, keys_A),
            "cftaoA": _summary_dict(cftaoA, keys_A),
        },
    }
    if unoA:
        payload["models"]["unoA"] = _summary_dict(unoA, keys_A)

    if args.markdown:
        print("\n--- Markdown table ---")
        print("| Model | Raw MSE(px) | E-only MSE(px) | Viol(bdry) | Viol(obs) | max_Gamma|u-g| (p95) |")
        print("| --- | --- | --- | --- | --- | --- |")
        r0 = _row(plain, ["raw_mse", "viol_bdry", "viol_obs", "max_gamma_p95"])
        print(f"| FNO (plain) | {r0[0]} | -- | {r0[1]} | {r0[2]} | {r0[3]} |")
        r1 = _row(fnoA, ["raw_mse", "eonly_mse", "viol_bdry", "viol_obs", "max_gamma_p95"])
        print(f"| FNO + A | {r1[0]} | {r1[1]} | {r1[2]} | {r1[3]} | {r1[4]} |")
        if unoA:
            r_uno = _row(unoA, ["raw_mse", "eonly_mse", "viol_bdry", "viol_obs", "max_gamma_p95"])
            print(f"| U-NO + A | {r_uno[0]} | {r_uno[1]} | {r_uno[2]} | {r_uno[3]} | {r_uno[4]} |")
        r2 = _row(cftaoA, ["raw_mse", "eonly_mse", "viol_bdry", "viol_obs", "max_gamma_p95"])
        print(f"| CFT-AO + A | {r2[0]} | {r2[1]} | {r2[2]} | {r2[3]} | {r2[4]} |")

    # LaTeX table
    print("\n--- LaTeX table ---")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(rf"\caption{{{args.caption}}}")
    print(rf"\label{{{args.label}}}")
    print(r"\begin{tabular}{lccccc}")
    print(r"\toprule")
    print(r"Model & Raw MSE(px) & E-only MSE(px) & Viol(bdry) & Viol(obs) & max$_\Gamma|u-g|$ (p95) \\")
    print(r"\midrule")

    r0 = _row(plain, ["raw_mse", "viol_bdry", "viol_obs", "max_gamma_p95"])
    print(rf"FNO (plain) & {r0[0]} & -- & {r0[1]} & {r0[2]} & {r0[3]} \\")

    r1 = _row(fnoA, ["raw_mse", "eonly_mse", "viol_bdry", "viol_obs", "max_gamma_p95"])
    print(rf"FNO + A & {r1[0]} & {r1[1]} & {r1[2]} & {r1[3]} & {r1[4]} \\")

    if unoA:
        r_uno = _row(unoA, ["raw_mse", "eonly_mse", "viol_bdry", "viol_obs", "max_gamma_p95"])
        print(rf"U-NO + A & {r_uno[0]} & {r_uno[1]} & {r_uno[2]} & {r_uno[3]} & {r_uno[4]} \\")

    r2 = _row(cftaoA, ["raw_mse", "eonly_mse", "viol_bdry", "viol_obs", "max_gamma_p95"])
    print(rf"CFT-AO + A & {r2[0]} & {r2[1]} & {r2[2]} & {r2[3]} & {r2[4]} \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    if args.json_out:
        out_dir = os.path.dirname(args.json_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved JSON summary to: {args.json_out}")


if __name__ == "__main__":
    main()

