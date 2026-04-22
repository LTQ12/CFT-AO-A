"""
Main method: Train CFT-AO + Direction-A wrapper on potential-flow obstacle2d dataset.
Hard constraints are enforced by construction through E[g] + w*r.
"""

import argparse

from train_cftaoA_diff2d_obstacle import main as _main_cftaoA


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train CFT-AO + A on potential-flow obstacle2d dataset.")
    # reuse same arguments as diffusion-obstacle CFT-AO+A trainer
    p.add_argument("--data_path", type=str, default="data/pflow_obstacle2d_N64.pt")
    p.add_argument("--model_save_path", type=str, default="models/cftaoA_pflow_obstacle2d.pt")
    p.add_argument("--init_ckpt_path", type=str, default="", help="Optional warm-start checkpoint (wrapped model state_dict).")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--modes", type=int, default=12)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)

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

    p.add_argument("--delta", type=float, default=0.05)
    # IMPORTANT: avoid getting stuck at "E-only" due to near-zero initial res_scale
    p.add_argument("--res_scale_init", type=float, default=0.02)
    p.add_argument("--res_scale_max", type=float, default=0.25)
    p.add_argument("--res_reg", type=float, default=1e-4)
    p.add_argument("--ext_method", type=str, default="harmonic", choices=["zero", "coons", "harmonic", "poisson", "poisson_learned"])
    # Match FNO+A default for fair comparison
    p.add_argument("--ext_iters", type=int, default=80)
    p.add_argument("--poisson_src_hidden", type=int, default=32)
    p.add_argument("--poisson_src_scale_max", type=float, default=1.0)
    p.add_argument("--residual_clip", type=float, default=0.0)

    # optional multiscale switches (keep default off)
    p.add_argument("--multiscale", action="store_true")
    p.add_argument("--coarse_factor", type=int, default=2)
    p.add_argument("--coarse_modes", type=int, default=8)
    p.add_argument("--coarse_width", type=int, default=32)
    p.add_argument("--coarse_layers", type=int, default=2)
    p.add_argument("--coarse_scale_init", type=float, default=0.02)
    p.add_argument("--coarse_scale_max", type=float, default=0.25)
    p.add_argument("--gamma_hidden", type=int, default=32)
    p.add_argument("--gamma_max", type=float, default=1.0)
    p.add_argument("--coarse_interior_eps", type=float, default=0.5)

    _main_cftaoA(p.parse_args())


