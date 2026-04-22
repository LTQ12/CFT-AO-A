"""
Generate 2D *variable-coefficient* diffusion dataset with internal obstacles (hard Dirichlet BC).

PDE (variable-coefficient steady diffusion):
    -∇ · (k(x) ∇u(x)) = 0    in Ω_f
    u(x) = g(x)             on ∂Ω_f (outer walls + obstacle regions)

We discretize on a uniform N×N grid on [0,1]^2 and solve by Jacobi iterations
on the *fluid* nodes, treating outer boundary and obstacle nodes as Dirichlet.

Input channels (kept compatible with existing evaluators):
  a[...,0] = geom  (1=obstacle, 0=fluid)
  a[...,1] = bc    (Dirichlet values on all nodes; interior fluid nodes set to 0)
  a[...,2] = kappa (variable coefficient field, positive)

Saved dataset:
  {
    'a_train': (n_train, N, N, 3),
    'u_train': (n_train, N, N, 1),
    'a_test' : (n_test,  N, N, 3),
    'u_test' : (n_test,  N, N, 1),
  }

Example:
  python data_generation/generate_varcoeff_diffusion_obstacle2d.py \
    --n_samples 1200 --n_train 1000 --n_test 200 \
    --N 64 --n_iter 1200 --seed 0 \
    --out_path data/varcoeff_diffusion_obstacle2d_N64.pt
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import torch


def build_obstacle_mask(
    N: int,
    *,
    n_obs_min: int = 1,
    n_obs_max: int = 2,
    r_min: float = 0.08,
    r_max: float = 0.18,
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
    n_obs = int(rng.integers(n_obs_min, n_obs_max + 1))
    for _ in range(n_obs):
        margin = 0.15
        cx = float(rng.uniform(margin, 1.0 - margin))
        cy = float(rng.uniform(margin, 1.0 - margin))
        r = float(rng.uniform(r_min, r_max))
        dist2 = (X - cx) ** 2 + (Y - cy) ** 2
        mask[dist2 <= r * r] = 1.0
    return mask


def build_boundary_profile(
    N: int,
    rng: np.random.Generator,
    *,
    n_modes: int = 3,
) -> np.ndarray:
    ys = np.linspace(0.0, 1.0, N, dtype=np.float64)
    profile = np.zeros_like(ys)
    for k in range(1, int(n_modes) + 1):
        ak = rng.normal(loc=0.0, scale=1.0 / k)
        profile += ak * np.sin(k * np.pi * ys)
    profile = profile / (np.max(np.abs(profile)) + 1e-8)

    g = np.zeros((N, N), dtype=np.float64)
    g[0, :] = profile
    g[-1, :] = profile
    g[:, 0] = profile[0]
    g[:, -1] = profile[-1]
    return g


def build_kappa_field(
    N: int,
    rng: np.random.Generator,
    *,
    n_modes: int = 4,
    log_kappa_std: float = 0.8,
    kappa_min: float = 0.1,
    kappa_max: float = 10.0,
) -> np.ndarray:
    """
    Smooth random positive coefficient field via low-frequency sine basis on [0,1]^2.
    """
    xs = np.linspace(0.0, 1.0, N, dtype=np.float64)
    ys = np.linspace(0.0, 1.0, N, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    logk = np.zeros((N, N), dtype=np.float64)
    for i in range(1, int(n_modes) + 1):
        for j in range(1, int(n_modes) + 1):
            aij = rng.normal(0.0, log_kappa_std / (i * j))
            logk += aij * np.sin(i * np.pi * X) * np.sin(j * np.pi * Y)

    # shift/scale then exponentiate
    logk = logk - float(np.mean(logk))
    kappa = np.exp(logk)

    # clamp to a reasonable range
    kappa = np.clip(kappa, float(kappa_min), float(kappa_max))
    return kappa


def _harmonic_mean(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return (2.0 * a * b) / (a + b + eps)


def solve_varcoeff_diffusion_dirichlet_masked(
    *,
    geom: np.ndarray,
    bc: np.ndarray,
    kappa: np.ndarray,
    n_iter: int = 1200,
) -> np.ndarray:
    """
    Jacobi solver for -∇·(k ∇u)=0 with Dirichlet on outer boundary and obstacles.
    """
    N = geom.shape[0]
    assert geom.shape == (N, N)
    assert bc.shape == (N, N)
    assert kappa.shape == (N, N)

    dirichlet = np.zeros((N, N), dtype=bool)
    dirichlet[0, :] = True
    dirichlet[-1, :] = True
    dirichlet[:, 0] = True
    dirichlet[:, -1] = True
    dirichlet = np.logical_or(dirichlet, geom >= 0.5)

    # face conductances via harmonic mean
    ke = np.zeros((N, N), dtype=np.float64)
    kw = np.zeros((N, N), dtype=np.float64)
    kn = np.zeros((N, N), dtype=np.float64)
    ks = np.zeros((N, N), dtype=np.float64)

    ke[:-1, :] = _harmonic_mean(kappa[:-1, :], kappa[1:, :])
    kw[1:, :] = _harmonic_mean(kappa[1:, :], kappa[:-1, :])
    kn[:, :-1] = _harmonic_mean(kappa[:, :-1], kappa[:, 1:])
    ks[:, 1:] = _harmonic_mean(kappa[:, 1:], kappa[:, :-1])

    denom = ke + kw + kn + ks
    denom = denom + 1e-12

    u = np.zeros((N, N), dtype=np.float64)
    u[dirichlet] = bc[dirichlet]

    for _ in range(int(n_iter)):
        uE = np.roll(u, -1, axis=0)
        uW = np.roll(u, 1, axis=0)
        uN = np.roll(u, -1, axis=1)
        uS = np.roll(u, 1, axis=1)
        u_new = (ke * uE + kw * uW + kn * uN + ks * uS) / denom
        u_new[dirichlet] = bc[dirichlet]
        u = u_new

    return u


def generate_dataset(
    *,
    n_samples: int,
    N: int,
    n_iter: int,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    geom_list = []
    bc_list = []
    kappa_list = []
    u_list = []

    for _ in range(int(n_samples)):
        geom = build_obstacle_mask(
            N,
            n_obs_min=1,
            n_obs_max=2,
            r_min=0.08,
            r_max=0.18,
            seed=int(rng.integers(0, 10**9)),
        )
        g_outer = build_boundary_profile(N, rng, n_modes=3)

        bc = np.zeros((N, N), dtype=np.float64)
        bc[0, :] = g_outer[0, :]
        bc[-1, :] = g_outer[-1, :]
        bc[:, 0] = g_outer[:, 0]
        bc[:, -1] = g_outer[:, -1]
        # obstacle Dirichlet = 0 (already 0)

        kappa = build_kappa_field(N, rng, n_modes=4)

        u = solve_varcoeff_diffusion_dirichlet_masked(geom=geom, bc=bc, kappa=kappa, n_iter=n_iter)

        geom_list.append(geom[..., None])
        bc_list.append(bc[..., None])
        kappa_list.append(kappa[..., None])
        u_list.append(u[..., None])

    geom_t = torch.from_numpy(np.stack(geom_list, axis=0)).float()
    bc_t = torch.from_numpy(np.stack(bc_list, axis=0)).float()
    kappa_t = torch.from_numpy(np.stack(kappa_list, axis=0)).float()
    u_t = torch.from_numpy(np.stack(u_list, axis=0)).float()
    a_t = torch.cat([geom_t, bc_t, kappa_t], dim=-1)
    return a_t, u_t


def main():
    p = argparse.ArgumentParser(description="Generate variable-coefficient diffusion obstacle2d dataset.")
    p.add_argument("--n_samples", type=int, default=1200)
    p.add_argument("--n_train", type=int, default=1000)
    p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--n_iter", type=int, default=1200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_path", type=str, default="data/varcoeff_diffusion_obstacle2d_N64.pt")
    args = p.parse_args()

    assert args.n_samples >= args.n_train + args.n_test
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    print(f"Generating varcoeff diffusion obstacle2d: N={args.N}, samples={args.n_samples}, iter={args.n_iter}")
    a_all, u_all = generate_dataset(n_samples=args.n_samples, N=args.N, n_iter=args.n_iter, seed=args.seed)
    a_train = a_all[: args.n_train]
    u_train = u_all[: args.n_train]
    a_test = a_all[args.n_train : args.n_train + args.n_test]
    u_test = u_all[args.n_train : args.n_train + args.n_test]

    out = {"a_train": a_train, "u_train": u_train, "a_test": a_test, "u_test": u_test}
    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(out, args.out_path)
    print(f"Saved dataset to {args.out_path}")
    print(f"  train: a {tuple(a_train.shape)}, u {tuple(u_train.shape)}")
    print(f"  test : a {tuple(a_test.shape)}, u {tuple(u_test.shape)}")


if __name__ == "__main__":
    main()

