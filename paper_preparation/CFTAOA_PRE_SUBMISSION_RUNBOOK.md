# CFT-AO+A Pre-Submission Runbook

This note collects the exact commands for the pre-submission strengthening pass:

1. reference-solver audits,
2. residual-clip fairness checks,
3. multi-seed aggregation for ID/OOD tables,
4. one extra open-source baseline probe (`U-NO + A`) on the diffusion benchmark.

All commands assume you run from the repository root:

```bash
path/to/CFT-AO-A
```

## 1. Reference-solver audits

### pflow

```bash
python3 paper_preparation/audit_reference_jacobi_convergence.py \
  --task pflow \
  --data_path data/pflow_obstacle2d_N64.pt \
  --split test \
  --sample_indices 0,1,2 \
  --json_out paper_preparation/results/pflow_jacobi_audit.json
```

### diffusion

```bash
python3 paper_preparation/audit_reference_jacobi_convergence.py \
  --task diffusion \
  --data_path data/diffusion_obstacle2d_N64.pt \
  --split test \
  --sample_indices 0,1,2 \
  --json_out paper_preparation/results/diffusion_jacobi_audit.json
```

### poisson

```bash
python3 paper_preparation/audit_reference_jacobi_convergence.py \
  --task poisson \
  --data_path data/poisson_src_obstacle2d_N64.pt \
  --split test \
  --sample_indices 0,1,2 \
  --json_out paper_preparation/results/poisson_jacobi_audit.json
```

### varcoeff

```bash
python3 paper_preparation/audit_reference_jacobi_convergence.py \
  --task varcoeff \
  --data_path data/varcoeff_diffusion_obstacle2d_N64.pt \
  --split test \
  --sample_indices 0,1,2 \
  --json_out paper_preparation/results/varcoeff_jacobi_audit.json
```

The audit script now prints:

- per-checkpoint Markdown tables,
- an aggregate summary at `paper_iters`,
- an appendix-ready LaTeX table,
- and a JSON payload with `summary`.

## 2. Residual-clip fairness ablation

Run the pflow fairness matrix on the archived checkpoints:

```bash
python3 paper_preparation/run_pflow_fairness_ablation.py \
  --data_path data/pflow_obstacle2d_N64.pt \
  --ckpt_fnoA models/fnoA_pflow_obstacle2d.pt \
  --ckpt_cftaoA models/cftaoA_pflow_obstacle2d.pt \
  --clips_fnoA 0.0,3.0 \
  --clips_cftaoA 0.0,3.0 \
  --json_out paper_preparation/results/pflow_fairness_ablation.json
```

This prints a Markdown table with:

- `raw_mse_px`,
- `eonly_mse_px`,
- `gain_over_eonly_frac`,
- `viol_bdry_mse`,
- `viol_obs_mse`,
- `max_gamma_p95`.

## 3. Multi-seed ID table

Example for pflow with saved metrics per seed:

```bash
python3 paper_preparation/aggregate_multiseed_metrics.py \
  --metrics_plain "paper_preparation/results/pflow/seed*/plain_metrics.pt" \
  --metrics_fnoA "paper_preparation/results/pflow/seed*/fnoA_metrics.pt" \
  --metrics_cftaoA "paper_preparation/results/pflow/seed*/cftaoA_metrics.pt" \
  --caption "Multi-seed results on pflow_obstacle2d_N64." \
  --label "tab:multiseed_pflow" \
  --markdown \
  --json_out paper_preparation/results/pflow_multiseed_summary.json
```

The aggregator now emits:

- scalar mean/std summaries,
- a Markdown table,
- a LaTeX table,
- and a JSON summary.

## 3b. Diffusion 3-seed refresh

If you want to strengthen the diffusion evidence with the same protocol as the
updated pflow section, run the full 3-seed pipeline below.

### Train 3 seeds

Run these from the project root or in a Colab `%%bash` cell. Do not prefix the
commands with `!` inside a bash loop.

```bash
mkdir -p paper_preparation/results/diffusion_multiseed
mkdir -p models/diffusion_multiseed

for SEED in 0 1 2; do
  python3 train_fno_diff2d_obstacle.py \
    --data_path data/diffusion_obstacle2d_N64.pt \
    --model_save_path models/diffusion_multiseed/plain_seed${SEED}.pt \
    --seed ${SEED} \
    --epochs 60 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --weight_decay 1e-4 \
    --modes 12 \
    --width 64
done
```

```bash
for SEED in 0 1 2; do
  python3 train_fnoA_diff2d_obstacle.py \
    --data_path data/diffusion_obstacle2d_N64.pt \
    --model_save_path models/diffusion_multiseed/fnoA_seed${SEED}.pt \
    --seed ${SEED} \
    --epochs 60 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --weight_decay 1e-4 \
    --modes 12 \
    --width 64 \
    --delta 0.05 \
    --res_scale_init 0.02 \
    --res_scale_max 0.25 \
    --res_reg 1e-4 \
    --ext_method harmonic \
    --ext_iters 80 \
    --residual_clip 3.0
done
```

```bash
for SEED in 0 1 2; do
  python3 train_cftaoA_diff2d_obstacle.py \
    --data_path data/diffusion_obstacle2d_N64.pt \
    --model_save_path models/diffusion_multiseed/cftaoA_seed${SEED}.pt \
    --seed ${SEED} \
    --epochs 60 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --weight_decay 1e-4 \
    --modes 12 \
    --width 64 \
    --n_layers 4 \
    --cft_L 4 \
    --cft_M 4 \
    --cft_L_boundary 8 \
    --cft_M_boundary 8 \
    --cft_res 0 \
    --rim_ratio 0.15 \
    --inner_iters 2 \
    --n_bands 3 \
    --n_sym_bases 0 \
    --delta 0.05 \
    --res_scale_init 0.02 \
    --res_scale_max 0.25 \
    --res_reg 1e-4 \
    --ext_method coons \
    --ext_iters 50 \
    --residual_clip 0.0
done
```

### Export predictions and metrics

```bash
for SEED in 0 1 2; do
  python3 paper_preparation/eval_plain_fno2d.py \
    --data_path data/diffusion_obstacle2d_N64.pt \
    --ckpt_path models/diffusion_multiseed/plain_seed${SEED}.pt \
    --pred_out paper_preparation/results/diffusion_multiseed/plain_seed${SEED}.pt \
    --batch_size 32 \
    --modes 12 \
    --width 64
done
```

```bash
for SEED in 0 1 2; do
  python3 paper_preparation/eval_A_models_2d.py \
    --data_path data/diffusion_obstacle2d_N64.pt \
    --model_type fnoA \
    --ckpt_path models/diffusion_multiseed/fnoA_seed${SEED}.pt \
    --pred_out paper_preparation/results/diffusion_multiseed/fnoA_seed${SEED}.pt \
    --batch_size 32 \
    --modes 12 \
    --width 64 \
    --delta 0.05 \
    --res_scale_init 0.02 \
    --res_scale_max 0.25 \
    --ext_method harmonic \
    --ext_iters 80 \
    --residual_clip 3.0
done
```

```bash
for SEED in 0 1 2; do
  python3 paper_preparation/eval_A_models_2d.py \
    --data_path data/diffusion_obstacle2d_N64.pt \
    --model_type cftaoA \
    --ckpt_path models/diffusion_multiseed/cftaoA_seed${SEED}.pt \
    --pred_out paper_preparation/results/diffusion_multiseed/cftaoA_seed${SEED}.pt \
    --batch_size 32 \
    --modes 12 \
    --width 64 \
    --n_layers 4 \
    --cft_L 4 \
    --cft_M 4 \
    --cft_L_boundary 8 \
    --cft_M_boundary 8 \
    --cft_res 0 \
    --rim_ratio 0.15 \
    --inner_iters 2 \
    --n_bands 3 \
    --n_sym_bases 0 \
    --delta 0.05 \
    --res_scale_init 0.02 \
    --res_scale_max 0.25 \
    --ext_method coons \
    --ext_iters 50 \
    --residual_clip 0.0
done
```

### Aggregate the 3-seed table

```bash
python3 paper_preparation/aggregate_multiseed_metrics.py \
  --metrics_plain "paper_preparation/results/diffusion_multiseed/plain_seed*_metrics.pt" \
  --metrics_fnoA "paper_preparation/results/diffusion_multiseed/fnoA_seed*_metrics.pt" \
  --metrics_cftaoA "paper_preparation/results/diffusion_multiseed/cftaoA_seed*_metrics.pt" \
  --caption "Multi-seed results (3 seeds) on diffusion_obstacle2d_N64." \
  --label "tab:multiseed_diffusion_3seeds" \
  --markdown \
  --json_out paper_preparation/results/diffusion_multiseed_summary.json
```

## 3c. U-NO + A diffusion probe

This is the lowest-risk additional open-source baseline to add under the same
hard-constraint protocol. It is a stronger generic backbone than plain FNO, but
still operates on the same regular grid and can therefore be wrapped by the
same A-wrap construction.

### Train 3 seeds

```bash
mkdir -p paper_preparation/results/diffusion_multiseed
mkdir -p models/diffusion_multiseed

for SEED in 0 1 2; do
  python3 train_unoA_diff2d_obstacle.py \
    --data_path data/diffusion_obstacle2d_N64.pt \
    --model_save_path models/diffusion_multiseed/unoA_seed${SEED}.pt \
    --seed ${SEED} \
    --epochs 60 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --weight_decay 1e-4 \
    --modes 12 \
    --width 64 \
    --pad 8 \
    --factor 1 \
    --delta 0.05 \
    --res_scale_init 0.02 \
    --res_scale_max 0.25 \
    --res_reg 1e-4 \
    --ext_method harmonic \
    --ext_iters 80 \
    --residual_clip 0.0
done
```

### Export predictions and metrics

```bash
for SEED in 0 1 2; do
  python3 paper_preparation/eval_A_models_2d.py \
    --data_path data/diffusion_obstacle2d_N64.pt \
    --model_type unoA \
    --ckpt_path models/diffusion_multiseed/unoA_seed${SEED}.pt \
    --pred_out paper_preparation/results/diffusion_multiseed/unoA_seed${SEED}.pt \
    --batch_size 32 \
    --modes 12 \
    --width 64 \
    --uno_pad 8 \
    --uno_factor 1 \
    --delta 0.05 \
    --res_scale_init 0.02 \
    --res_scale_max 0.25 \
    --ext_method harmonic \
    --ext_iters 80 \
    --residual_clip 0.0
done
```

### Aggregate with the existing diffusion table

```bash
python3 paper_preparation/aggregate_multiseed_metrics.py \
  --metrics_plain "paper_preparation/results/diffusion_multiseed/plain_seed*_metrics.pt" \
  --metrics_fnoA "paper_preparation/results/diffusion_multiseed/fnoA_seed*_metrics.pt" \
  --metrics_unoA "paper_preparation/results/diffusion_multiseed/unoA_seed*_metrics.pt" \
  --metrics_cftaoA "paper_preparation/results/diffusion_multiseed/cftaoA_seed*_metrics.pt" \
  --caption "Multi-seed results (3 seeds) on diffusion_obstacle2d_N64." \
  --label "tab:multiseed_diffusion_3seeds" \
  --markdown \
  --json_out paper_preparation/results/diffusion_multiseed_summary_with_uno.json
```

### Optional: measure parameter count and latency

```bash
python3 paper_preparation/profile_costs_2d.py \
  --data_path data/diffusion_obstacle2d_N64.pt \
  --ckpt_plain models/diffusion_multiseed/plain_seed0.pt \
  --ckpt_fnoA models/diffusion_multiseed/fnoA_seed0.pt \
  --ckpt_unoA models/diffusion_multiseed/unoA_seed0.pt \
  --ckpt_cftaoA models/diffusion_multiseed/cftaoA_seed0.pt \
  --batch_size 16 \
  --repeats 50 \
  --warmup 10 \
  --modes 12 \
  --width 64 \
  --uno_pad 8 \
  --uno_factor 1 \
  --delta 0.05 \
  --res_scale_init 0.02 \
  --res_scale_max 0.25 \
  --ext_method harmonic \
  --ext_iters 80
```

Interpretation:

- if `U-NO + A` remains clearly above `CFT-AO + A` after 3 seeds, it strengthens
  the claim that the observed gain is not merely from using any stronger generic
  operator backbone;
- if it also has much larger parameter count or latency, that supports a
  parameter-efficiency argument for `CFT-AO + A`;
- if it unexpectedly matches or beats `CFT-AO + A`, move it into the main text
  and narrow the novelty claim accordingly.

### Qualitative figure for diffusion

The fastest useful figure is an A-vs-A comparison between `FNO+A` and
`CFT-AO+A`, using one representative seed (for example `seed0`).

```bash
python3 paper_preparation/compare_A_diff2d_errors_pair.py \
  --data_path data/diffusion_obstacle2d_N64.pt \
  --pred_a paper_preparation/results/diffusion_multiseed/fnoA_seed0.pt \
  --pred_b paper_preparation/results/diffusion_multiseed/cftaoA_seed0.pt \
  --label_a "FNO+A" \
  --label_b "CFT-AO+A" \
  --out_dir paper_preparation/results/diffusion_compare_seed0 \
  --worst_k 8
```

This writes:

- `err_vs_dist_overlay.png` for boundary-to-interior error decay,
- `mse_ecdf_overlay.png` for per-sample error distribution,
- `mse_boxplot.png` for compact summary,
- and `worst_pair_*.png` for qualitative panels.

For the main paper, the lowest-cost choice is usually either
`mse_ecdf_overlay.png` or one selected `worst_pair_*.png`.

## 4. OOD table formatting

Once the OOD metrics are exported:

```bash
python3 paper_preparation/format_metrics_table_ood.py \
  --metrics_plain paper_preparation/results/pflow_ood/plain_metrics.pt \
  --metrics_A1 paper_preparation/results/pflow_ood/fnoA_metrics.pt \
  --metrics_A2 paper_preparation/results/pflow_ood/cftaoA_metrics.pt \
  --caption "OOD geometry: star-shaped obstacles (train on circles, test on stars)." \
  --label "tab:ood_star"
```

## 5. Notes

- `paper_preparation/eval_pflow_A_models.py`,
  `paper_preparation/analyze_pflow_eonly_ood.py`,
  `paper_preparation/profile_costs_2d.py`,
  `paper_preparation/eval_A_models_2d.py`,
  and `paper_preparation/eval_plain_fno2d.py`
  now inject the repo root into `sys.path`, so they run directly from the project root.
- Several dataset generators were also made compatible with Python environments that do not accept `X | None` annotations unless `from __future__ import annotations` is enabled.
