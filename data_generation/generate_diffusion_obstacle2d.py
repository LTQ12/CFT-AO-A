"""
Generate 2D steady diffusion dataset with internal obstacles (Dirichlet BC).

PDE (steady diffusion):
    -kappa * Δc(x) = 0    in fluid region Ω_f
    c(x) = g(x)          on ∂Ω_f (outer walls + obstacle boundaries)

We work on a fixed rectangular domain [0,1] x [0,1] with a uniform N x N grid.
Internal obstacles are represented as circular inclusions; they and the outer
boundary are treated as Dirichlet nodes.

For each sample:
  - randomly place 1–2 circular obstacles;
  - construct a Dirichlet boundary profile g(x) on all boundaries:
      * left wall: random combination of low-frequency sines in y
      * other walls and obstacles: 0
  - solve Laplace equation with Jacobi iterations on the fluid nodes;
  - save (geom, bc, c) as a torch .pt file, where:
      geom: obstacle mask (1=obstacle, 0=fluid)
      bc  : Dirichlet values on all nodes (0 for interior fluid nodes)
      c   : steady-state solution.

Saved dataset (for compatibility with FNO/CFT-AO trainers):
    {
      'a_train': (n_train, N, N, 2),  # geom, bc
      'u_train': (n_train, N, N, 1),
      'a_test' : (n_test,  N, N, 2),
      'u_test' : (n_test,  N, N, 1),
    }

Typical usage (from project root):
    python data_generation/generate_diffusion_obstacle2d.py \\
        --n_samples 1200 --n_train 1000 --n_test 200 \\
        --N 64 --n_iter 800 \\
        --out_path data/diffusion_obstacle2d_N64.pt
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch


def build_obstacle_mask(
    N: int,
    n_obs_min: int = 1,
    n_obs_max: int = 2,
    r_min: float = 0.08,
    r_max: float = 0.18,
    seed: int | None = None,
) -> np.ndarray:
    """
    Build a binary obstacle mask on an N x N grid.
    Obstacles are random circles inside (0,1)^2 with radius in [r_min,r_max].
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    xs = np.linspace(0.0, 1.0, N, dtype=np.float64)
    ys = np.linspace(0.0, 1.0, N, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    mask = np.zeros((N, N), dtype=np.float64)

    n_obs = rng.integers(n_obs_min, n_obs_max + 1)
    for _ in range(n_obs):
        # avoid placing centers too close to outer boundary
        margin = 0.15
        cx = rng.uniform(margin, 1.0 - margin)
        cy = rng.uniform(margin, 1.0 - margin)
        r = rng.uniform(r_min, r_max)
        dist2 = (X - cx) ** 2 + (Y - cy) ** 2
        mask[dist2 <= r**2] = 1.0

    return mask


def build_boundary_profile(
    N: int,
    rng: np.random.Generator,
    n_modes: int = 3,
) -> np.ndarray:
    """
    Build a random Dirichlet boundary profile g(x,y) on the outer boundary.
    - Left wall: random low-frequency sine series in y.
    - Other walls: 0.
    Obstacles will also be Dirichlet=0.
    """
    xs = np.linspace(0.0, 1.0, N, dtype=np.float64)
    ys = np.linspace(0.0, 1.0, N, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    g = np.zeros((N, N), dtype=np.float64)

    # left wall at x=0 → i=0
    y = ys
    profile = np.zeros_like(y)
    for k in range(1, n_modes + 1):
        ak = rng.normal(loc=0.0, scale=1.0 / k)
        profile += ak * np.sin(k * np.pi * y)
    # normalize profile to [-1,1]
    max_abs = np.max(np.abs(profile)) + 1e-8
    profile = profile / max_abs
    # assign to left wall
    g[0, :] = profile

    # other walls default 0
    return g


def solve_diffusion_dirichlet_masked(
    geom: np.ndarray,
    bc: np.ndarray,
    n_iter: int = 800,
    kappa: float = 1.0,
) -> np.ndarray:
    """
    Solve -kappa Δc = 0 with Dirichlet BC on:
      - outer boundary
      - obstacle boundary (geom=1)

    geom: (N,N) obstacle mask (1=obstacle, 0=fluid)
    bc  : (N,N) Dirichlet values on all Dirichlet nodes (outer boundary + obstacles)
    Returns:
        c: (N,N) steady-state solution.
    """
    N = geom.shape[0]
    assert geom.shape == (N, N)
    assert bc.shape == (N, N)

    # Dirichlet nodes: outer boundary + obstacle interior
    dirichlet = np.zeros((N, N), dtype=bool)
    dirichlet[0, :] = True
    dirichlet[-1, :] = True
    dirichlet[:, 0] = True
    dirichlet[:, -1] = True
    dirichlet = np.logical_or(dirichlet, geom >= 0.5)

    # initialize solution with BC where Dirichlet, zeros elsewhere
    c = np.zeros((N, N), dtype=np.float64)
    c[dirichlet] = bc[dirichlet]

    # 5-point Laplacian, Jacobi iterations on non-Dirichlet nodes
    for _ in range(n_iter):
        c_old = c.copy()
        # interior indices
        i = slice(1, N - 1)
        j = slice(1, N - 1)

        # neighbors
        cn = c_old[2:, 1:-1]
        cs = c_old[:-2, 1:-1]
        ce = c_old[1:-1, 2:]
        cw = c_old[1:-1, :-2]

        c_new = 0.25 * (cn + cs + ce + cw)

        # update only fluid interior nodes (not Dirichlet)
        mask_inner = ~dirichlet[1:-1, 1:-1]
        c[i, j][mask_inner] = c_new[mask_inner]
        # Dirichlet nodes remain fixed from bc
        c[dirichlet] = bc[dirichlet]

    return c


def generate_dataset(
    n_samples: int,
    N: int,
    n_iter: int,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate (geom, bc, c) triples.

    Returns:
        geom_t: (n_samples, N, N, 1)
        bc_t  : (n_samples, N, N, 1)
        c_t   : (n_samples, N, N, 1)
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    geom_list = []
    bc_list = []
    c_list = []

    for _ in range(n_samples):
        geom = build_obstacle_mask(
            N=N,
            n_obs_min=1,
            n_obs_max=2,
            r_min=0.08,
            r_max=0.18,
            seed=rng.integers(0, 10**9),
        )

        # boundary profile on outer walls
        g_outer = build_boundary_profile(N=N, rng=rng, n_modes=3)

        # full bc: obstacles and outer boundary = g; fluid interior = 0
        bc = np.zeros((N, N), dtype=np.float64)
        bc[0, :] = g_outer[0, :]
        bc[-1, :] = g_outer[-1, :]
        bc[:, 0] = g_outer[:, 0]
        bc[:, -1] = g_outer[:, -1]
        # obstacles Dirichlet = 0 (already 0)

        c = solve_diffusion_dirichlet_masked(geom=geom, bc=bc, n_iter=n_iter, kappa=1.0)

        geom_list.append(geom[..., None])  # (N,N,1)
        bc_list.append(bc[..., None])
        c_list.append(c[..., None])

    geom_arr = np.stack(geom_list, axis=0)
    bc_arr = np.stack(bc_list, axis=0)
    c_arr = np.stack(c_list, axis=0)

    geom_t = torch.from_numpy(geom_arr).float()
    bc_t = torch.from_numpy(bc_arr).float()
    c_t = torch.from_numpy(c_arr).float()
    return geom_t, bc_t, c_t


def main():
    parser = argparse.ArgumentParser(
        description="Generate 2D steady diffusion dataset with internal obstacles (Dirichlet BC)."
    )
    parser.add_argument("--n_samples", type=int, default=1200, help="total number of samples (train+test)")
    parser.add_argument("--n_train", type=int, default=1000, help="number of training samples")
    parser.add_argument("--n_test", type=int, default=200, help="number of test samples")
    parser.add_argument("--N", type=int, default=64, help="spatial resolution N x N")
    parser.add_argument("--n_iter", type=int, default=800, help="Jacobi iterations per sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_path", type=str, default="data/diffusion_obstacle2d_N64.pt")
    args = parser.parse_args()

    assert args.n_samples >= args.n_train + args.n_test, "n_samples must be >= n_train + n_test"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(
        f"Generating 2D diffusion-obstacle dataset: N={args.N}, samples={args.n_samples}, "
        f"iter={args.n_iter}"
    )
    geom_all, bc_all, c_all = generate_dataset(
        n_samples=args.n_samples,
        N=args.N,
        n_iter=args.n_iter,
        seed=args.seed,
    )

    # pack geom & bc into input "a"
    a_all = torch.cat([geom_all, bc_all], dim=-1)  # (S,N,N,2)
    u_all = c_all  # (S,N,N,1)

    # split train/test
    a_train = a_all[: args.n_train]
    u_train = u_all[: args.n_train]
    a_test = a_all[args.n_train : args.n_train + args.n_test]
    u_test = u_all[args.n_train : args.n_train + args.n_test]

    data = {
        "a_train": a_train,
        "u_train": u_train,
        "a_test": a_test,
        "u_test": u_test,
    }

    out_dir = os.path.dirname(args.out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    torch.save(data, args.out_path)
    print(f"Saved diffusion-obstacle 2D dataset to {args.out_path}")
    print(f"  train: a {tuple(a_train.shape)}, u {tuple(u_train.shape)}")
    print(f"  test : a {tuple(a_test.shape)}, u {tuple(u_test.shape)}")


if __name__ == "__main__":
    main()


