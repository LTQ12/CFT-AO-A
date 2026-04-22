"""
Generate a 2D *engineering* dataset with hard constraints:

Incompressible, irrotational potential flow expressed via a stream function psi:

    Δ psi = 0  in fluid region Ω_f
    psi = g     on outer boundary ∂Ω
    psi = c_i   inside each obstacle (conductors / streamlines)  (Dirichlet)

This is a standard engineering surrogate for inviscid incompressible flow around obstacles.
The *hard constraint* is Dirichlet satisfaction on outer walls and obstacles; violating it
corresponds to violating impermeability (streamlines).

We solve the discrete Laplace equation with Jacobi iterations on a uniform N×N grid.

Saved dataset format (compatible with existing trainers):
    {
      'a_train': (n_train, N, N, 2),  # [geom, bc]   (bc includes obstacle Dirichlet values)
      'u_train': (n_train, N, N, 1),  # psi
      'a_test' : (n_test,  N, N, 2),
      'u_test' : (n_test,  N, N, 1),
    }
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import torch


def build_obstacle_mask_with_centers(
    N: int,
    *,
    n_obs_min: int = 1,
    n_obs_max: int = 3,
    r_min: float = 0.06,
    r_max: float = 0.16,
    seed: int | None = None,
) -> Tuple[np.ndarray, list[tuple[float, float]]]:
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    xs = np.linspace(0.0, 1.0, N, dtype=np.float64)
    ys = np.linspace(0.0, 1.0, N, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    mask = np.zeros((N, N), dtype=np.float64)
    centers: list[tuple[float, float]] = []

    n_obs = int(rng.integers(n_obs_min, n_obs_max + 1))
    for _ in range(n_obs):
        margin = 0.15
        cx = float(rng.uniform(margin, 1.0 - margin))
        cy = float(rng.uniform(margin, 1.0 - margin))
        r = float(rng.uniform(r_min, r_max))
        dist2 = (X - cx) ** 2 + (Y - cy) ** 2
        mask[dist2 <= r**2] = 1.0
        centers.append((cx, cy))

    return mask, centers


def build_outer_streamfunction_bc(
    N: int,
    rng: np.random.Generator,
    *,
    n_modes: int = 4,
    amp: float = 1.0,
) -> np.ndarray:
    """
    Build a consistent Dirichlet psi on the *entire* outer boundary.

    We set left/right edges to the same profile g(y), and top/bottom to constants
    g(1), g(0) to keep corner consistency.
    """
    ys = np.linspace(0.0, 1.0, N, dtype=np.float64)
    profile = np.zeros_like(ys)
    for k in range(1, n_modes + 1):
        ak = rng.normal(loc=0.0, scale=1.0 / k)
        profile += ak * np.sin(k * np.pi * ys)
    # normalize and scale
    profile = profile / (np.max(np.abs(profile)) + 1e-8)
    profile = float(amp) * profile

    bc = np.zeros((N, N), dtype=np.float64)
    # left/right
    bc[0, :] = profile
    bc[-1, :] = profile
    # bottom/top constants (match corners)
    bc[:, 0] = profile[0]
    bc[:, -1] = profile[-1]
    return bc


def solve_laplace_dirichlet_masked(
    geom: np.ndarray,
    bc: np.ndarray,
    *,
    n_iter: int = 800,
) -> np.ndarray:
    N = geom.shape[0]
    assert geom.shape == (N, N)
    assert bc.shape == (N, N)

    dirichlet = np.zeros((N, N), dtype=bool)
    dirichlet[0, :] = True
    dirichlet[-1, :] = True
    dirichlet[:, 0] = True
    dirichlet[:, -1] = True
    dirichlet = np.logical_or(dirichlet, geom >= 0.5)

    u = np.zeros((N, N), dtype=np.float64)
    u[dirichlet] = bc[dirichlet]

    for _ in range(int(n_iter)):
        u_old = u.copy()
        cn = u_old[2:, 1:-1]
        cs = u_old[:-2, 1:-1]
        ce = u_old[1:-1, 2:]
        cw = u_old[1:-1, :-2]
        u_new = 0.25 * (cn + cs + ce + cw)
        mask_inner = ~dirichlet[1:-1, 1:-1]
        u[1:-1, 1:-1][mask_inner] = u_new[mask_inner]
        u[dirichlet] = bc[dirichlet]

    return u


def generate_dataset(
    *,
    n_samples: int,
    N: int,
    n_iter: int,
    seed: int | None = None,
    n_obs_min: int = 1,
    n_obs_max: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    geom_list = []
    bc_list = []
    psi_list = []

    ys = np.linspace(0.0, 1.0, N, dtype=np.float64)

    for _ in range(int(n_samples)):
        geom, centers = build_obstacle_mask_with_centers(
            N,
            n_obs_min=n_obs_min,
            n_obs_max=n_obs_max,
            r_min=0.06,
            r_max=0.16,
            seed=int(rng.integers(0, 10**9)),
        )

        amp = float(rng.uniform(0.5, 2.0))
        bc = build_outer_streamfunction_bc(N, rng, n_modes=4, amp=amp)

        # set obstacle Dirichlet values: constant per obstacle.
        # Use outer profile evaluated at obstacle center y as a physically plausible streamline value.
        for (_, cy) in centers:
            j = int(np.clip(round(cy * (N - 1)), 0, N - 1))
            c_obs = float(bc[0, j])  # g(y)
            bc[geom >= 0.5] = c_obs  # (coarse, but keeps obstacle region constant)

        psi = solve_laplace_dirichlet_masked(geom=geom, bc=bc, n_iter=n_iter)

        geom_list.append(geom[..., None])
        bc_list.append(bc[..., None])
        psi_list.append(psi[..., None])

    geom_t = torch.from_numpy(np.stack(geom_list, axis=0)).float()
    bc_t = torch.from_numpy(np.stack(bc_list, axis=0)).float()
    psi_t = torch.from_numpy(np.stack(psi_list, axis=0)).float()
    return geom_t, bc_t, psi_t


def main():
    p = argparse.ArgumentParser(description="Generate 2D potential-flow (streamfunction Laplace) obstacle dataset.")
    p.add_argument("--n_samples", type=int, default=1200)
    p.add_argument("--n_train", type=int, default=1000)
    p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--n_iter", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_obs_min", type=int, default=1)
    p.add_argument("--n_obs_max", type=int, default=3)
    p.add_argument("--out_path", type=str, default="data/pflow_obstacle2d_N64.pt")
    args = p.parse_args()

    assert args.n_samples >= args.n_train + args.n_test

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Generating potential-flow obstacle2d: N={args.N}, samples={args.n_samples}, iter={args.n_iter}")
    geom_all, bc_all, psi_all = generate_dataset(
        n_samples=args.n_samples,
        N=args.N,
        n_iter=args.n_iter,
        seed=args.seed,
        n_obs_min=args.n_obs_min,
        n_obs_max=args.n_obs_max,
    )

    a_all = torch.cat([geom_all, bc_all], dim=-1)
    a_train = a_all[: args.n_train]
    u_train = psi_all[: args.n_train]
    a_test = a_all[args.n_train : args.n_train + args.n_test]
    u_test = psi_all[args.n_train : args.n_train + args.n_test]

    data = {"a_train": a_train, "u_train": u_train, "a_test": a_test, "u_test": u_test}

    out_dir = os.path.dirname(args.out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    torch.save(data, args.out_path)
    print(f"Saved dataset to {args.out_path}")
    print(f"  train: a {tuple(a_train.shape)}, u {tuple(u_train.shape)}")
    print(f"  test : a {tuple(a_test.shape)}, u {tuple(u_test.shape)}")


if __name__ == "__main__":
    main()


