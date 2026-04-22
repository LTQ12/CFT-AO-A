# CFT-AO+A

Standalone code package for `CFT-AO+A`, a feasibility-by-construction neural operator setup for 2D obstacle problems on masked grids.

This repository was extracted from a larger research workspace and keeps only the files needed for the `CFT-AO+A` training, evaluation, and reproducibility pipeline around the core 2D obstacle benchmarks.

## What is included

- Core `CFT-AO+A` modules: `cft_ao_2d.py`, `fourier_2d_cft_residual.py`, `chebyshev.py`
- A-wrap / hard-feasibility wrapper: `boundary_ext_residual_2d.py`
- Utility modules used by the training scripts: `Adam.py`, `utilities3.py`
- Baselines used in the comparison pipeline: `fourier_2d_baseline.py`, `uno_2d.py`
- Training scripts for `CFT-AO+A`, `FNO+A`, `U-NO+A`, and plain `FNO`
- Evaluation, aggregation, profiling, and plotting scripts under `paper_preparation/`
- Data generation scripts for the obstacle/diffusion/potential-flow tasks under `data_generation/`
- OOD geometry-atlas utilities under `the_third_paper/`

## What is intentionally excluded

- Paper sources, submission materials, PDFs, and LaTeX cache files
- Unrelated 3D experiments and legacy project branches
- Large datasets, checkpoints, prediction dumps, and temporary results

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

## Quick start

Run commands from the repository root so the current import layout works as expected.

```bash
python train_cftaoA_diff2d_obstacle.py --help
python paper_preparation/eval_A_models_2d.py --help
python paper_preparation/profile_costs_2d.py --help
```

## Reproducibility entry points

- `paper_preparation/CFTAOA_PRE_SUBMISSION_RUNBOOK.md` collects the main command chains used for the paper-strengthening experiments
- `paper_preparation/eval_A_models_2d.py` evaluates `FNO+A`, `U-NO+A`, and `CFT-AO+A` under a unified protocol
- `paper_preparation/aggregate_multiseed_metrics.py` summarizes multi-seed results
- `paper_preparation/profile_costs_2d.py` profiles parameters, runtime, and memory

## Repository layout

- Root directory: core models, wrappers, and training scripts
- `data_generation/`: data generation scripts for the main 2D tasks
- `paper_preparation/`: evaluation, aggregation, profiling, and plotting utilities
- `the_third_paper/`: supplementary OOD geometry-atlas scripts

## Data and checkpoints

Datasets and model checkpoints are not tracked in this repository. Use your own local paths via command-line arguments such as `--data_path`, `--ckpt_path`, `--fno_model`, and `--cft_model`.

## Notes

- The repository preserves the original flat module layout to minimize breakage during extraction from the larger workspace.
- `MANIFEST_INCLUDED_FILES.txt` lists the exact files included in this standalone release.
