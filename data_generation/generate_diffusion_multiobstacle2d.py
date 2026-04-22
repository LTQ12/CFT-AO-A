"""
Generate a harder 2D steady diffusion dataset with multiple complex obstacles (Dirichlet BC).

相比原来的 diffusion_obstacle2d：
- 分辨率更高 (默认 N=128)；
- 每个样本有 2~5 个障碍物，形状包括圆形和细长矩形（类似窄缝/挡板），
  随机位置+大小，形成狭窄通道、近接触几何；
- 外边界给定非平凡的 Dirichlet 型边界条件，内部为纯扩散：

    -kappa * Δc(x) = 0    in fluid region Ω_f
     c(x) = g(x)         on ∂Ω_f ∪ Γ_obstacle

几何编码：
    geom: obstacle mask (1=obstacle, 0=fluid)
边界编码：
    bc  : Dirichlet values on all nodes (0 for interior fluid nodes)

输出数据格式（兼容现有 FNO/CFT 脚本）：
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


def build_multi_obstacle_mask(
    N: int,
    n_obs_min: int = 2,
    n_obs_max: int = 5,
    seed: int | None = None,
) -> np.ndarray:
    """
    在 N x N 网格上生成多个障碍物：
    - 圆形: (x-cx)^2 + (y-cy)^2 <= r^2
    - 细长矩形条: 水平或垂直，宽度较小，长度覆盖比较大区域
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
        shape_type = rng.choice(["circle", "bar"])
        if shape_type == "circle":
            # 避免太靠边界，半径也随机
            margin = 0.15
            cx = rng.uniform(margin, 1.0 - margin)
            cy = rng.uniform(margin, 1.0 - margin)
            r = rng.uniform(0.05, 0.15)
            dist2 = (X - cx) ** 2 + (Y - cy) ** 2
            mask[dist2 <= r**2] = 1.0
        else:
            # 细长矩形条：水平或垂直
            orient = rng.choice(["h", "v"])
            if orient == "h":
                # 水平条：高度较窄，长度较长
                y0 = rng.uniform(0.2, 0.8)
                height = rng.uniform(0.03, 0.08)
                x0 = rng.uniform(0.05, 0.25)
                x1 = rng.uniform(0.75, 0.95)
                cond = (np.abs(Y - y0) <= height) & (X >= x0) & (X <= x1)
                mask[cond] = 1.0
            else:
                # 垂直条
                x0 = rng.uniform(0.2, 0.8)
                width = rng.uniform(0.03, 0.08)
                y0 = rng.uniform(0.05, 0.25)
                y1 = rng.uniform(0.75, 0.95)
                cond = (np.abs(X - x0) <= width) & (Y >= y0) & (Y <= y1)
                mask[cond] = 1.0

    return mask


def build_boundary_profile_complex(
    N: int,
    rng: np.random.Generator,
    n_modes: int = 4,
) -> np.ndarray:
    """
    构造一个稍复杂的 Dirichlet 外边界：
    - 左/右边：不同相位/频率的正弦叠加；
    - 上/下边：幅度缩小的正弦；
    """
    xs = np.linspace(0.0, 1.0, N, dtype=np.float64)
    ys = np.linspace(0.0, 1.0, N, dtype=np.float64)

    g = np.zeros((N, N), dtype=np.float64)

    # 左右边：随机低频和稍高频的组合
    y = ys
    prof_left = np.zeros_like(y)
    prof_right = np.zeros_like(y)
    for k in range(1, n_modes + 1):
        aL = rng.normal(loc=0.0, scale=1.0 / k)
        aR = rng.normal(loc=0.0, scale=1.0 / k)
        prof_left += aL * np.sin(k * np.pi * y)
        prof_right += aR * np.sin(k * np.pi * y + 0.5 * np.pi)
    # 归一化到 [-1,1] 附近
    for prof in (prof_left, prof_right):
        max_abs = np.max(np.abs(prof)) + 1e-8
        prof /= max_abs

    g[0, :] = prof_left
    g[-1, :] = prof_right

    # 上下边：幅度缩小的正弦
    x = xs
    prof_bottom = np.zeros_like(x)
    prof_top = np.zeros_like(x)
    for k in range(1, n_modes + 1):
        bB = rng.normal(loc=0.0, scale=0.5 / k)
        bT = rng.normal(loc=0.0, scale=0.5 / k)
        prof_bottom += bB * np.sin(k * np.pi * x)
        prof_top += bT * np.sin(k * np.pi * x + 0.25 * np.pi)
    for prof in (prof_bottom, prof_top):
        max_abs = np.max(np.abs(prof)) + 1e-8
        prof /= max_abs

    g[:, 0] = prof_bottom
    g[:, -1] = prof_top

    return g


def solve_diffusion_dirichlet_masked(
    geom: np.ndarray,
    bc: np.ndarray,
    n_iter: int = 1200,
    kappa: float = 1.0,
) -> np.ndarray:
    """
    与之前 generate_diffusion_obstacle2d.py 类似：
      -kappa Δc = 0 with Dirichlet BC on:
        - outer boundary
        - obstacle region (geom=1)
    使用 Jacobi 迭代求稳态解。
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

    # 初始化解
    c = np.zeros((N, N), dtype=np.float64)
    c[dirichlet] = bc[dirichlet]

    for _ in range(n_iter):
        c_old = c.copy()
        i = slice(1, N - 1)
        j = slice(1, N - 1)

        cn = c_old[2:, 1:-1]
        cs = c_old[:-2, 1:-1]
        ce = c_old[1:-1, 2:]
        cw = c_old[1:-1, :-2]

        c_new = 0.25 * (cn + cs + ce + cw)

        mask_inner = ~dirichlet[1:-1, 1:-1]
        c[i, j][mask_inner] = c_new[mask_inner]
        c[dirichlet] = bc[dirichlet]

    return c


def generate_dataset(
    n_samples: int,
    N: int,
    n_iter: int,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    生成 (geom, bc, c) 样本。
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    geom_list, bc_list, c_list = [], [], []

    for _ in range(n_samples):
        geom = build_multi_obstacle_mask(
            N=N,
            n_obs_min=2,
            n_obs_max=5,
            seed=rng.integers(0, 10**9),
        )

        g_outer = build_boundary_profile_complex(N=N, rng=rng, n_modes=4)

        bc = np.zeros((N, N), dtype=np.float64)
        # 外边界
        bc[0, :] = g_outer[0, :]
        bc[-1, :] = g_outer[-1, :]
        bc[:, 0] = g_outer[:, 0]
        bc[:, -1] = g_outer[:, -1]
        # 障碍物内部 Dirichlet = 0（已经是 0）

        c = solve_diffusion_dirichlet_masked(geom=geom, bc=bc, n_iter=n_iter, kappa=1.0)

        geom_list.append(geom[..., None])
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
        description="Generate harder 2D diffusion dataset with multiple complex obstacles (Dirichlet BC)."
    )
    parser.add_argument("--n_samples", type=int, default=1500, help="total number of samples (train+test)")
    parser.add_argument("--n_train", type=int, default=1200, help="number of training samples")
    parser.add_argument("--n_test", type=int, default=300, help="number of test samples")
    parser.add_argument("--N", type=int, default=128, help="spatial resolution N x N")
    parser.add_argument("--n_iter", type=int, default=1200, help="Jacobi iterations per sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out_path", type=str, default="data/diffusion_multiobstacle2d_N128.pt", help="output .pt path"
    )
    args = parser.parse_args()

    assert args.n_samples >= args.n_train + args.n_test, "n_samples must be >= n_train + n_test"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(
        f"Generating 2D diffusion-multiobstacle dataset: N={args.N}, samples={args.n_samples}, "
        f"iter={args.n_iter}"
    )
    geom_all, bc_all, c_all = generate_dataset(
        n_samples=args.n_samples,
        N=args.N,
        n_iter=args.n_iter,
        seed=args.seed,
    )

    a_all = torch.cat([geom_all, bc_all], dim=-1)
    u_all = c_all

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
    print(f"Saved diffusion-multiobstacle 2D dataset to {args.out_path}")
    print(f"  train: a {tuple(a_train.shape)}, u {tuple(u_train.shape)}")
    print(f"  test : a {tuple(a_test.shape)}, u {tuple(u_test.shape)}")


if __name__ == "__main__":
    main()



