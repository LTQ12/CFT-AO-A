"""
可视化几何 OOD multi-obstacle 样本：
  - 几何掩码 geom
  - 真解 u_true
  - FNO 预测 u_fno
  - CFT-AO+A（几何-atlas）预测 u_cft
  - |误差| 图：|u_fno - u_true|, |u_cft - u_true|

默认路径按你当前 Colab/Drive 布局写好，用法（在 Colab）：

    cd /content
    python plot_multiobstacle_ood_samples.py

生成的图会保存在当前目录下，例如：
    ood_sample_0.png, ood_sample_1.png, ...
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from boundary_ext_residual_2d import ResidualOnDirichletExtension2D
from cft_ao_2d import CFT_AO_2D_Atlas
from fourier_2d_baseline import FNO2d
from utilities3 import UnitGaussianNormalizer


def build_cftaoA_model(
    *,
    in_channels_norm: int,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    device: torch.device,
) -> ResidualOnDirichletExtension2D:
    modes = 12
    width = 64
    n_layers = 4
    cft_L = 4
    cft_M = 4
    cft_L_boundary = 8
    cft_M_boundary = 8
    cft_res = 0
    use_local = True
    rim_ratio = 0.15
    cond_dim = 4
    inner_iters = 2
    n_bands = 3
    n_sym_bases = 0

    backbone = CFT_AO_2D_Atlas(
        modes1=modes,
        modes2=modes,
        width=width,
        in_channels=in_channels_norm,
        out_channels=1,
        n_layers=n_layers,
        L_segments=cft_L,
        M_cheb=cft_M,
        L_segments_boundary=cft_L_boundary,
        M_cheb_boundary=cft_M_boundary,
        cft_res=cft_res,
        use_local=use_local,
        rim_ratio=rim_ratio,
        cond_dim=cond_dim,
        inner_iters=inner_iters,
        n_bands=n_bands,
        n_sym_bases=n_sym_bases,
    ).to(device)

    model = ResidualOnDirichletExtension2D(
        backbone,
        y_mean=y_mean,
        y_std=y_std,
        in_channels_norm=in_channels_norm,
        delta=0.05,
        res_scale_init=0.02,
        res_scale_max=0.25,
        ext_method="harmonic",
        ext_iters=40,
        poisson_src_hidden=32,
        poisson_src_scale_max=1.0,
        residual_clip=0.0,
    ).to(device)
    return model


def build_fno_model(
    *,
    in_channels: int,
    out_channels: int,
    state_dict: dict,
    device: torch.device,
) -> FNO2d:
    modes = 12
    # 从 checkpoint 推断宽度
    if "fc0.weight" in state_dict:
        width = int(state_dict["fc0.weight"].shape[0])
    else:
        width = 64

    model = FNO2d(
        modes1=modes,
        modes2=modes,
        width=width,
        in_channels=in_channels,
        out_channels=out_channels,
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    return model


@torch.no_grad()
def plot_samples(
    *,
    a_ood_raw: torch.Tensor,
    u_ood_raw: torch.Tensor,
    a_normalizer: UnitGaussianNormalizer,
    u_normalizer: UnitGaussianNormalizer,
    fno_model: FNO2d,
    cft_model: ResidualOnDirichletExtension2D,
    indices: list[int],
    out_dir: str,
    device: torch.device,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    a_ood_raw = a_ood_raw.to(device)
    u_ood_raw = u_ood_raw.to(device)

    a_ood = a_normalizer.encode(a_ood_raw)

    geom = a_ood_raw[..., 0]  # (B,N,N)

    in_channels_norm = cft_model.in_channels_norm

    for idx in indices:
        a_sample = a_ood[idx : idx + 1]          # (1,N,N,2)
        a_raw_sample = a_ood_raw[idx : idx + 1]  # (1,N,N,2)
        u_true = u_ood_raw[idx : idx + 1]        # (1,N,N,1)

        # FNO 预测：输入为归一化后的 a
        u_true_enc = u_normalizer.encode(u_true)
        u_fno_enc = fno_model(a_sample)
        u_fno = u_normalizer.decode(u_fno_enc)

        # CFT 预测：输入为 [a_norm, geom_raw, bc_raw]
        geom_raw = a_raw_sample[..., 0:1]
        bc_raw = a_raw_sample[..., 1:2]
        x_mix = torch.cat([a_sample, geom_raw, bc_raw], dim=-1)
        u_cft_enc = cft_model(x_mix)
        u_cft = u_normalizer.decode(u_cft_enc)

        g = geom[idx].cpu().numpy()
        u_t = u_true[0, ..., 0].cpu().numpy()
        u_f = u_fno[0, ..., 0].cpu().numpy()
        u_c = u_cft[0, ..., 0].cpu().numpy()

        err_f = np.abs(u_f - u_t)
        err_c = np.abs(u_c - u_t)

        # 为了解耦尺度影响：
        # - u_true 与 u_CFT-atlas 共用一个较窄色标（根据两者范围）
        # - u_FNO 单独使用自己的色标（可能非常大）
        # - 误差图 |u_FNO-u_true| / |u_CFT-u_true| 共用同一色标以便直接比较
        vmin_tc = min(u_t.min(), u_c.min())
        vmax_tc = max(u_t.max(), u_c.max())
        vmin_f = u_f.min()
        vmax_f = u_f.max()

        err_max = max(err_f.max(), err_c.max())
        err_vmin, err_vmax = 0.0, float(err_max)

        fig, axes = plt.subplots(2, 3, figsize=(10, 6))

        im0 = axes[0, 0].imshow(g, origin="lower", cmap="gray")
        axes[0, 0].set_title("Geom mask")
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

        im1 = axes[0, 1].imshow(u_t, origin="lower", cmap="viridis", vmin=vmin_tc, vmax=vmax_tc)
        axes[0, 1].set_title("u_true")
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

        im2 = axes[0, 2].imshow(u_f, origin="lower", cmap="viridis", vmin=vmin_f, vmax=vmax_f)
        axes[0, 2].set_title("u_FNO")
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

        im3 = axes[1, 0].imshow(u_c, origin="lower", cmap="viridis", vmin=vmin_tc, vmax=vmax_tc)
        axes[1, 0].set_title("u_CFT-atlas")
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

        im4 = axes[1, 1].imshow(err_f, origin="lower", cmap="magma", vmin=err_vmin, vmax=err_vmax)
        axes[1, 1].set_title("|u_FNO - u_true|")
        plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

        im5 = axes[1, 2].imshow(err_c, origin="lower", cmap="magma", vmin=err_vmin, vmax=err_vmax)
        axes[1, 2].set_title("|u_CFT - u_true|")
        plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)

        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        out_path = os.path.join(out_dir, f"ood_sample_{idx}.png")
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved {out_path}")


def main():
    p = argparse.ArgumentParser(description="Plot OOD multi-obstacle samples: FNO vs geometry-atlas CFT-AO+A.")
    p.add_argument(
        "--data_path_train",
        type=str,
        default="/content/drive/MyDrive/diffusion_multiobstacle2d_N128.pt",
    )
    p.add_argument(
        "--data_path_ood",
        type=str,
        default="/content/drive/MyDrive/diffusion_multiobstacle2d_ood_N64.pt",
    )
    p.add_argument(
        "--fno_model",
        type=str,
        default="/content/drive/MyDrive/fno_diff2d_multiobs_best.pt",
    )
    p.add_argument(
        "--cft_model",
        type=str,
        default="/content/cftaoA_geomaware_multiobstacle.pt",
    )
    p.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Indices of OOD samples to plot.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="ood_plots",
    )
    p.add_argument("--device", type=str, default="cuda")

    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) 载入 ID 数据，重建 normalizer
    print(f"Loading in-distribution dataset from {args.data_path_train} ...")
    data_train = torch.load(args.data_path_train, map_location="cpu")
    a_train_raw = data_train["a_train"]
    u_train_raw = data_train["u_train"]

    a_normalizer = UnitGaussianNormalizer(a_train_raw)
    u_normalizer = UnitGaussianNormalizer(u_train_raw)
    a_normalizer.to(device)
    u_normalizer.to(device)

    in_channels_norm = int(a_train_raw.shape[-1])

    # 2) 载入 OOD 数据，插值到 ID 分辨率
    print(f"Loading OOD dataset from {args.data_path_ood} ...")
    data_ood = torch.load(args.data_path_ood, map_location="cpu")
    a_ood_raw = data_ood["a_test"]
    u_ood_raw = data_ood["u_test"]

    N_id = int(a_train_raw.shape[1])
    N_ood = int(a_ood_raw.shape[1])
    if N_ood != N_id:
        print(f"Resizing OOD from N={N_ood} to N={N_id} ...")
        a_ood_raw = F.interpolate(
            a_ood_raw.permute(0, 3, 1, 2), size=(N_id, N_id), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        u_ood_raw = F.interpolate(
            u_ood_raw.permute(0, 3, 1, 2), size=(N_id, N_id), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)

    # 3) 构建并加载 FNO / CFT 模型
    state_fno = torch.load(args.fno_model, map_location="cpu")
    fno_model = build_fno_model(
        in_channels=in_channels_norm,
        out_channels=int(u_train_raw.shape[-1]),
        state_dict=state_fno,
        device=device,
    )

    y_mean = u_normalizer.mean.to(device)
    y_std = u_normalizer.std.to(device)
    cft_model = build_cftaoA_model(
        in_channels_norm=in_channels_norm,
        y_mean=y_mean,
        y_std=y_std,
        device=device,
    )
    state_cft = torch.load(args.cft_model, map_location="cpu")
    cft_model.load_state_dict(state_cft, strict=False)

    # 4) 画图
    plot_samples(
        a_ood_raw=a_ood_raw,
        u_ood_raw=u_ood_raw,
        a_normalizer=a_normalizer,
        u_normalizer=u_normalizer,
        fno_model=fno_model,
        cft_model=cft_model,
        indices=args.indices,
        out_dir=args.out_dir,
        device=device,
    )


if __name__ == "__main__":
    main()


