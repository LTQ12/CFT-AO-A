"""
Generate 2D Poisson-with-source dataset with internal obstacles (hard Dirichlet BC).

PDE:
    -Δu(x) = s(x)     in Ω_f
    u(x) = g(x)       on ∂Ω_f (outer walls + obstacle regions)

Discretization on uniform N×N grid on [0,1]^2 and Jacobi iterations on fluid nodes.

Input channels (kept compatible with existing evaluators):
  a[...,0] = geom  (1=obstacle, 0=fluid)
  a[...,1] = bc    (Dirichlet values on all nodes; interior fluid nodes set to 0)
  a[...,2] = src   (source term s(x) on all nodes; obstacle nodes ignored)

Saved dataset:
  {
    'a_train': (n_train, N, N, 3),
    'u_train': (n_train, N, N, 1),
    'a_test' : (n_test,  N, N, 3),
    'u_test' : (n_test,  N, N, 1),
  }
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch


def build_obstacle_mask(N: int, *, seed: int | None = None) -> np.ndarray:
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    xs = np.linspace(0.0, 1.0, N, dtype=np.float64)
    ys = np.linspace(0.0, 1.0, N, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    mask = np.zeros((N, N), dtype=np.float64)
    n_obs = int(rng.integers(1, 3))
    for _ in range(n_obs):
        margin = 0.15
        cx = float(rng.uniform(margin, 1.0 - margin))
        cy = float(rng.uniform(margin, 1.0 - margin))
        r = float(rng.uniform(0.08, 0.18))
        dist2 = (X - cx) ** 2 + (Y - cy) ** 2
        mask[dist2 <= r * r] = 1.0
    return mask


def build_boundary_profile(N: int, rng: np.random.Generator, *, n_modes: int = 3) -> np.ndarray:
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


def build_source_field(
    N: int,
    rng: np.random.Generator,
    *,
    n_modes: int = 5,
    amp: float = 1.0,
) -> np.ndarray:
    xs = np.linspace(0.0, 1.0, N, dtype=np.float64)
    ys = np.linspace(0.0, 1.0, N, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    s = np.zeros((N, N), dtype=np.float64)
    for i in range(1, int(n_modes) + 1):
        for j in range(1, int(n_modes) + 1):
            aij = rng.normal(0.0, 1.0 / (i * j))
            s += aij * np.sin(i * np.pi * X) * np.sin(j * np.pi * Y)
    s = s / (np.max(np.abs(s)) + 1e-8)
    s = float(amp) * s
    return s


def solve_poisson_dirichlet_masked(
    *,
    geom: np.ndarray,
    bc: np.ndarray,
    src: np.ndarray,
    n_iter: int = 1200,
) -> np.ndarray:
    N = geom.shape[0]
    assert geom.shape == (N, N)
    assert bc.shape == (N, N)
    assert src.shape == (N, N)

    dirichlet = np.zeros((N, N), dtype=bool)
    dirichlet[0, :] = True
    dirichlet[-1, :] = True
    dirichlet[:, 0] = True
    dirichlet[:, -1] = True
    dirichlet = np.logical_or(dirichlet, geom >= 0.5)

    h = 1.0 / max(N - 1, 1)
    h2 = h * h

    u = np.zeros((N, N), dtype=np.float64)
    u[dirichlet] = bc[dirichlet]

    for _ in range(int(n_iter)):
        uE = np.roll(u, -1, axis=0)
        uW = np.roll(u, 1, axis=0)
        uN = np.roll(u, -1, axis=1)
        uS = np.roll(u, 1, axis=1)
        u_new = (uE + uW + uN + uS + h2 * src) / 4.0
        u_new[dirichlet] = bc[dirichlet]
        u = u_new

    return u


def generate_dataset(*, n_samples: int, N: int, n_iter: int, seed: int | None = None):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    geom_list, bc_list, src_list, u_list = [], [], [], []

    for _ in range(int(n_samples)):
        geom = build_obstacle_mask(N, seed=int(rng.integers(0, 10**9)))
        g_outer = build_boundary_profile(N, rng, n_modes=3)

        bc = np.zeros((N, N), dtype=np.float64)
        bc[0, :] = g_outer[0, :]
        bc[-1, :] = g_outer[-1, :]
        bc[:, 0] = g_outer[:, 0]
        bc[:, -1] = g_outer[:, -1]
        # obstacle Dirichlet = 0 (already 0)

        amp = float(rng.uniform(0.5, 2.0))
        src = build_source_field(N, rng, n_modes=5, amp=amp)

        u = solve_poisson_dirichlet_masked(geom=geom, bc=bc, src=src, n_iter=n_iter)

        geom_list.append(geom[..., None])
        bc_list.append(bc[..., None])
        src_list.append(src[..., None])
        u_list.append(u[..., None])

    geom_t = torch.from_numpy(np.stack(geom_list, axis=0)).float()
    bc_t = torch.from_numpy(np.stack(bc_list, axis=0)).float()
    src_t = torch.from_numpy(np.stack(src_list, axis=0)).float()
    u_t = torch.from_numpy(np.stack(u_list, axis=0)).float()
    a_t = torch.cat([geom_t, bc_t, src_t], dim=-1)
    return a_t, u_t


def main():
    p = argparse.ArgumentParser(description="Generate Poisson-with-source obstacle2d dataset.")
    p.add_argument("--n_samples", type=int, default=1200)
    p.add_argument("--n_train", type=int, default=1000)
    p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--n_iter", type=int, default=1200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_path", type=str, default="data/poisson_src_obstacle2d_N64.pt")
    args = p.parse_args()

    assert args.n_samples >= args.n_train + args.n_test
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    print(f"Generating Poisson(src) obstacle2d: N={args.N}, samples={args.n_samples}, iter={args.n_iter}")
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

