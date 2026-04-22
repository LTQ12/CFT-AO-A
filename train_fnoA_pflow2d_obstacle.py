"""
Fair baseline: Train FNO2d + Direction-A wrapper on potential-flow obstacle2d dataset.
Hard constraints are enforced by construction through E[g] + w*r.
"""

import argparse

from train_fnoA_diff2d_obstacle import main as _main_fnoA


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train FNO + A on potential-flow obstacle2d dataset.")
    # reuse the same arguments as diffusion-obstacle A trainer
    p.add_argument("--data_path", type=str, default="data/pflow_obstacle2d_N64.pt")
    p.add_argument("--model_save_path", type=str, default="models/fnoA_pflow_obstacle2d.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--modes", type=int, default=12)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--res_scale_init", type=float, default=0.02)
    p.add_argument("--res_scale_max", type=float, default=0.25)
    p.add_argument("--res_reg", type=float, default=1e-4)
    p.add_argument("--ext_method", type=str, default="harmonic", choices=["coons", "harmonic", "poisson", "poisson_learned"])
    p.add_argument("--ext_iters", type=int, default=80)
    p.add_argument("--poisson_src_hidden", type=int, default=32)
    p.add_argument("--poisson_src_scale_max", type=float, default=1.0)
    p.add_argument("--residual_clip", type=float, default=3.0)
    _main_fnoA(p.parse_args())


