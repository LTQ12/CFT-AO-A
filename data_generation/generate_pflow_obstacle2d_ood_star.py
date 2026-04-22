"""
Generate an OOD *geometry* test set for the potential-flow (streamfunction Laplace) obstacle dataset.

Goal
-----
Keep the *training distribution* identical to Dataset 2 (pflow_obstacle2d_N64.pt),
but replace the test split with *non-circular* obstacles (star-shaped).

This is designed for reviewers: train on circles, test on a clearly different geometry family.

Output format (compatible with existing trainers/evaluators):
  {
    "a_train": (n_train,N,N,2),  # copied from base dataset
    "u_train": (n_train,N,N,1),  # copied from base dataset
    "a_test" : (n_test, N,N,2),  # OOD star obstacles
    "u_test" : (n_test, N,N,1),
  }

Example:
  python data_generation/generate_pflow_obstacle2d_ood_star.py \
    --base_path data/pflow_obstacle2d_N64.pt \
    --n_test 200 --N 64 --n_iter 800 --seed 123 \
    --out_path data/pflow_obstacle2d_N64_ood_star.pt
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import torch

# Reuse solver + boundary generator from the in-distribution generator
from data_generation.generate_potential_flow_obstacle2d import (
    build_outer_streamfunction_bc,
    solve_laplace_dirichlet_masked,
)


def build_star_obstacle_mask_with_centers(
    N: int,
    *,
    n_obs: int = 1,
    r_min: float = 0.06,
    r_max: float = 0.16,
    arms: int = 5,
    amp_min: float = 0.25,
    amp_max: float = 0.50,
    seed: int | None = None,
) -> Tuple[np.ndarray, list[tuple[float, float]]]:
    """
    Build a binary obstacle mask with star-shaped obstacles.

    Star boundary (polar, centered at (cx,cy)):
        r(θ) = r0 * (1 + a * cos(arms*(θ - φ)))

    A point is inside obstacle if ||x-c|| <= r(θ).
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    xs = np.linspace(0.0, 1.0, N, dtype=np.float64)
    ys = np.linspace(0.0, 1.0, N, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    mask = np.zeros((N, N), dtype=np.float64)
    centers: list[tuple[float, float]] = []

    # keep obstacles away from the outer boundary
    margin = 0.18

    for _ in range(int(n_obs)):
        cx = float(rng.uniform(margin, 1.0 - margin))
        cy = float(rng.uniform(margin, 1.0 - margin))
        r0 = float(rng.uniform(r_min, r_max))
        a = float(rng.uniform(amp_min, amp_max))
        phi = float(rng.uniform(0.0, 2.0 * np.pi))

        dx = X - cx
        dy = Y - cy
        rr = np.sqrt(dx * dx + dy * dy)
        theta = np.arctan2(dy, dx)

        # star radial function; clamp to avoid non-positive radii
        r_theta = r0 * (1.0 + a * np.cos(int(arms) * (theta - phi)))
        r_theta = np.maximum(r_theta, 0.10 * r0)

        mask[rr <= r_theta] = 1.0
        centers.append((cx, cy))

    return mask, centers


def generate_ood_test(
    *,
    n_test: int,
    N: int,
    n_iter: int,
    seed: int,
    # star params
    arms: int = 5,
    n_obs: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)

    geom_list = []
    bc_list = []
    psi_list = []

    for _ in range(int(n_test)):
        geom, centers = build_star_obstacle_mask_with_centers(
            N,
            n_obs=n_obs,
            r_min=0.06,
            r_max=0.16,
            arms=arms,
            amp_min=0.25,
            amp_max=0.50,
            seed=int(rng.integers(0, 10**9)),
        )

        amp = float(rng.uniform(0.5, 2.0))
        bc = build_outer_streamfunction_bc(N, rng, n_modes=4, amp=amp)

        # set obstacle Dirichlet value(s): constant.
        # Keep consistent with the base generator's "coarse" assignment.
        for (_, cy) in centers:
            j = int(np.clip(round(cy * (N - 1)), 0, N - 1))
            c_obs = float(bc[0, j])  # g(y)
            bc[geom >= 0.5] = c_obs

        psi = solve_laplace_dirichlet_masked(geom=geom, bc=bc, n_iter=n_iter)

        geom_list.append(geom[..., None])
        bc_list.append(bc[..., None])
        psi_list.append(psi[..., None])

    geom_t = torch.from_numpy(np.stack(geom_list, axis=0)).float()
    bc_t = torch.from_numpy(np.stack(bc_list, axis=0)).float()
    psi_t = torch.from_numpy(np.stack(psi_list, axis=0)).float()

    a_test = torch.cat([geom_t, bc_t], dim=-1)
    u_test = psi_t
    return a_test, u_test


def main():
    p = argparse.ArgumentParser(description="Generate OOD (star-shaped) test set for pflow obstacle2d.")
    p.add_argument("--base_path", type=str, default="data/pflow_obstacle2d_N64.pt")
    p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--n_iter", type=int, default=800)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--arms", type=int, default=5)
    p.add_argument("--n_obs", type=int, default=1, help="number of star obstacles (use 1 for clean OOD)")
    p.add_argument("--out_path", type=str, default="data/pflow_obstacle2d_N64_ood_star.pt")
    args = p.parse_args()

    base = torch.load(args.base_path, map_location="cpu")
    a_train = base["a_train"]
    u_train = base["u_train"]

    print(f"Loaded base train split from: {args.base_path}")
    print(f"  a_train: {tuple(a_train.shape)}  u_train: {tuple(u_train.shape)}")
    print(f"Generating OOD star test split: N={args.N}, n_test={args.n_test}, iter={args.n_iter}, arms={args.arms}")

    a_test, u_test = generate_ood_test(n_test=args.n_test, N=args.N, n_iter=args.n_iter, seed=args.seed, arms=args.arms, n_obs=args.n_obs)

    out = {"a_train": a_train, "u_train": u_train, "a_test": a_test, "u_test": u_test}

    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(out, args.out_path)
    print(f"Saved OOD dataset to: {args.out_path}")
    print(f"  a_test:  {tuple(a_test.shape)}  u_test: {tuple(u_test.shape)}")


if __name__ == "__main__":
    main()

