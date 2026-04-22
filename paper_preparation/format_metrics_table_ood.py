"""
Format evaluation metrics (saved as *_metrics.pt) into a small LaTeX table snippet.

Designed for the OOD geometry experiment where we evaluate:
  - plain FNO (unconstrained)
  - FNO + A
  - CFT-AO + A

Example:
  python paper_preparation/format_metrics_table_ood.py \
    --metrics_plain /content/preds_ood/fno_ood_pred_metrics.pt \
    --metrics_A1 /content/preds_ood/fnoA_ood_pred_metrics.pt \
    --metrics_A2 /content/preds_ood/cftaoA_ood_pred_metrics.pt \
    --caption "OOD star-shaped obstacles (train on circles, test on stars)." \
    --label "tab:ood_star"
"""

from __future__ import annotations

import argparse

import torch


def _load(path: str) -> dict:
    d = torch.load(path, map_location="cpu")
    if not isinstance(d, dict):
        raise ValueError(f"metrics file must be a dict: {path}")
    return d


def _fmt(x: float) -> str:
    # scientific, 3 sig figs
    return f"{float(x):.3e}"


def main():
    p = argparse.ArgumentParser(description="Format OOD metrics into LaTeX table.")
    p.add_argument("--metrics_plain", type=str, required=True)
    p.add_argument("--metrics_A1", type=str, required=True)
    p.add_argument("--metrics_A2", type=str, required=True)
    p.add_argument("--caption", type=str, default="OOD geometry test (star-shaped obstacles).")
    p.add_argument("--label", type=str, default="tab:ood")
    args = p.parse_args()

    m0 = _load(args.metrics_plain)
    m1 = _load(args.metrics_A1)
    m2 = _load(args.metrics_A2)

    # plain
    raw0 = _fmt(m0.get("raw_mse"))
    vb0 = _fmt(m0.get("viol_bdry"))
    vo0 = _fmt(m0.get("viol_obs"))

    # A-models
    raw1 = _fmt(m1.get("raw_mse"))
    e1 = _fmt(m1.get("eonly_mse"))
    vb1 = _fmt(m1.get("viol_bdry"))
    vo1 = _fmt(m1.get("viol_obs"))

    raw2 = _fmt(m2.get("raw_mse"))
    e2 = _fmt(m2.get("eonly_mse"))
    vb2 = _fmt(m2.get("viol_bdry"))
    vo2 = _fmt(m2.get("viol_obs"))

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(rf"\caption{{{args.caption}}}")
    print(rf"\label{{{args.label}}}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Model & Raw MSE(px) & E-only MSE(px) & Viol(bdry) & Viol(obs) \\")
    print(r"\midrule")
    print(rf"FNO (plain) & {raw0} & -- & {vb0} & {vo0} \\")
    print(rf"FNO + A     & {raw1} & {e1} & {vb1} & {vo1} \\")
    print(rf"CFT-AO + A  & {raw2} & {e2} & {vb2} & {vo2} \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()

