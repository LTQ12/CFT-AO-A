"""
Generate an OOD-split version of the multi-obstacle diffusion dataset.

Goal: evaluate geometric OOD generalization (not just IID fitting).

We generate:
  - Train distribution: fewer obstacles + wider bars (easier connectivity)
  - Test  distribution: more obstacles + thinner bars (harder, narrow channels)

Output format matches existing training scripts:
    {
      'a_train': (n_train, N, N, 2),  # [geom, bc]
      'u_train': (n_train, N, N, 1),
      'a_test' : (n_test,  N, N, 2),
      'u_test' : (n_test,  N, N, 1),
    }
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from data_generation.generate_diffusion_multiobstacle2d import (
    build_boundary_profile_complex,
    solve_diffusion_dirichlet_masked,
)


def build_multi_obstacle_mask_param(
    N: int,
    *,
    n_obs_min: int,
    n_obs_max: int,
    circle_r_min: float,
    circle_r_max: float,
    bar_thick_min: float,
    bar_thick_max: float,
    seed: int | None = None,
) -> np.ndarray:
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    xs = np.linspace(0.0, 1.0, N, dtype=np.float64)
    ys = np.linspace(0.0, 1.0, N, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    mask = np.zeros((N, N), dtype=np.float64)
    n_obs = int(rng.integers(int(n_obs_min), int(n_obs_max) + 1))

    for _ in range(n_obs):
        shape_type = rng.choice(["circle", "bar"])
        if shape_type == "circle":
            margin = 0.15
            cx = float(rng.uniform(margin, 1.0 - margin))
            cy = float(rng.uniform(margin, 1.0 - margin))
            r = float(rng.uniform(circle_r_min, circle_r_max))
            dist2 = (X - cx) ** 2 + (Y - cy) ** 2
            mask[dist2 <= r**2] = 1.0
        else:
            orient = rng.choice(["h", "v"])
            if orient == "h":
                y0 = float(rng.uniform(0.2, 0.8))
                height = float(rng.uniform(bar_thick_min, bar_thick_max))
                x0 = float(rng.uniform(0.05, 0.25))
                x1 = float(rng.uniform(0.75, 0.95))
                cond = (np.abs(Y - y0) <= height) & (X >= x0) & (X <= x1)
                mask[cond] = 1.0
            else:
                x0 = float(rng.uniform(0.2, 0.8))
                width = float(rng.uniform(bar_thick_min, bar_thick_max))
                y0 = float(rng.uniform(0.05, 0.25))
                y1 = float(rng.uniform(0.75, 0.95))
                cond = (np.abs(X - x0) <= width) & (Y >= y0) & (Y <= y1)
                mask[cond] = 1.0

    return mask


def generate_split(
    *,
    n_train: int,
    n_test: int,
    N: int,
    n_iter: int,
    seed: int,
    # train distro
    train_n_obs_min: int,
    train_n_obs_max: int,
    train_bar_thick_min: float,
    train_bar_thick_max: float,
    # test distro
    test_n_obs_min: int,
    test_n_obs_max: int,
    test_bar_thick_min: float,
    test_bar_thick_max: float,
) -> dict:
    rng = np.random.default_rng(int(seed))

    def _one_sample(n_obs_min: int, n_obs_max: int, bar_tmin: float, bar_tmax: float):
        geom = build_multi_obstacle_mask_param(
            N,
            n_obs_min=n_obs_min,
            n_obs_max=n_obs_max,
            circle_r_min=0.05,
            circle_r_max=0.15,
            bar_thick_min=bar_tmin,
            bar_thick_max=bar_tmax,
            seed=int(rng.integers(0, 10**9)),
        )
        g_outer = build_boundary_profile_complex(N=N, rng=rng, n_modes=4)
        bc = np.zeros((N, N), dtype=np.float64)
        bc[0, :] = g_outer[0, :]
        bc[-1, :] = g_outer[-1, :]
        bc[:, 0] = g_outer[:, 0]
        bc[:, -1] = g_outer[:, -1]
        u = solve_diffusion_dirichlet_masked(geom=geom, bc=bc, n_iter=n_iter, kappa=1.0)
        a = np.stack([geom, bc], axis=-1)  # (N,N,2)
        return a.astype(np.float32), u[..., None].astype(np.float32)

    a_tr, u_tr = [], []
    for _ in range(int(n_train)):
        a, u = _one_sample(train_n_obs_min, train_n_obs_max, train_bar_thick_min, train_bar_thick_max)
        a_tr.append(a)
        u_tr.append(u)

    a_te, u_te = [], []
    for _ in range(int(n_test)):
        a, u = _one_sample(test_n_obs_min, test_n_obs_max, test_bar_thick_min, test_bar_thick_max)
        a_te.append(a)
        u_te.append(u)

    return {
        "a_train": torch.from_numpy(np.stack(a_tr, axis=0)),
        "u_train": torch.from_numpy(np.stack(u_tr, axis=0)),
        "a_test": torch.from_numpy(np.stack(a_te, axis=0)),
        "u_test": torch.from_numpy(np.stack(u_te, axis=0)),
        "meta": {
            "train_n_obs": [int(train_n_obs_min), int(train_n_obs_max)],
            "test_n_obs": [int(test_n_obs_min), int(test_n_obs_max)],
            "train_bar_thick": [float(train_bar_thick_min), float(train_bar_thick_max)],
            "test_bar_thick": [float(test_bar_thick_min), float(test_bar_thick_max)],
            "N": int(N),
            "n_iter": int(n_iter),
            "seed": int(seed),
        },
    }


def main():
    p = argparse.ArgumentParser(description="Generate OOD split diffusion-multiobstacle dataset.")
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--n_iter", type=int, default=1200)
    p.add_argument("--n_train", type=int, default=1200)
    p.add_argument("--n_test", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_path", type=str, default="data/diffusion_multiobstacle2d_ood_N64.pt")

    # OOD knobs
    p.add_argument("--train_n_obs_min", type=int, default=2)
    p.add_argument("--train_n_obs_max", type=int, default=3)
    p.add_argument("--test_n_obs_min", type=int, default=4)
    p.add_argument("--test_n_obs_max", type=int, default=5)

    p.add_argument("--train_bar_thick_min", type=float, default=0.04)
    p.add_argument("--train_bar_thick_max", type=float, default=0.08)
    p.add_argument("--test_bar_thick_min", type=float, default=0.01)
    p.add_argument("--test_bar_thick_max", type=float, default=0.04)

    args = p.parse_args()

    print(
        f"Generating OOD multiobstacle: N={args.N}, train={args.n_train}, test={args.n_test}, "
        f"train_obs=[{args.train_n_obs_min},{args.train_n_obs_max}], "
        f"test_obs=[{args.test_n_obs_min},{args.test_n_obs_max}]"
    )
    data = generate_split(
        n_train=args.n_train,
        n_test=args.n_test,
        N=args.N,
        n_iter=args.n_iter,
        seed=args.seed,
        train_n_obs_min=args.train_n_obs_min,
        train_n_obs_max=args.train_n_obs_max,
        train_bar_thick_min=args.train_bar_thick_min,
        train_bar_thick_max=args.train_bar_thick_max,
        test_n_obs_min=args.test_n_obs_min,
        test_n_obs_max=args.test_n_obs_max,
        test_bar_thick_min=args.test_bar_thick_min,
        test_bar_thick_max=args.test_bar_thick_max,
    )

    out_dir = os.path.dirname(args.out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    torch.save(data, args.out_path)
    print(f"Saved OOD dataset to {args.out_path}")


if __name__ == "__main__":
    main()


