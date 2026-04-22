"""
Evaluate FNO vs geometry-atlas CFT-AO+A on geometric OOD multi-obstacle diffusion dataset.

默认路径按你当前的 Colab/Drive 布局写好，只需要：

    cd /content
    python eval_cftaoA_geomatlas_ood.py

即可直接跑出在 OOD 数据集上的对比结果。
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boundary_ext_residual_2d import ResidualOnDirichletExtension2D
from cft_ao_2d import CFT_AO_2D_Atlas
from fourier_2d_baseline import FNO2d
from utilities3 import LpLoss, UnitGaussianNormalizer, count_params


def build_cftaoA_model(
    *,
    in_channels_norm: int,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    device: torch.device,
) -> ResidualOnDirichletExtension2D:
    """
    重建几何-atlas 版本的 CFT_AO_2D_Atlas + A-wrap，
    超参数与 train_cftaoA_diff2d_multiobstacle.py 保持一致。
    """
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
    cond_dim = 4  # geometry-aware conditioning
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
    width: int,
    device: torch.device,
) -> FNO2d:
    """
    重建 vanilla FNO2d baseline。
    - modes 与训练脚本一致 (12)
    - width 从 checkpoint 中自动推断 (避免尺寸不匹配)
    """
    modes = 12

    model = FNO2d(
        modes1=modes,
        modes2=modes,
        width=width,
        in_channels=in_channels,
        out_channels=out_channels,
    ).to(device)
    return model


@torch.no_grad()
def evaluate_fno(
    model: FNO2d,
    *,
    a_ood: torch.Tensor,
    a_ood_raw: torch.Tensor,
    u_ood_raw: torch.Tensor,
    u_normalizer: UnitGaussianNormalizer,
    batch_size: int,
    device: torch.device,
) -> dict:
    """
    在 OOD 数据集上评估 FNO baseline。
    """
    model.eval()
    loss_func = LpLoss(size_average=False)

    dataset = torch.utils.data.TensorDataset(a_ood, u_ood_raw)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    N = int(a_ood_raw.shape[1])
    num_test = int(a_ood_raw.shape[0])
    num_pts = N * N

    test_l2_enc = 0.0
    test_l2_raw = 0.0
    mse_raw = 0.0
    mse_bdry = 0.0
    mse_obs = 0.0

    # 构造 Dirichlet / 障碍掩码（来自原始几何通道）
    geom = a_ood_raw[..., 0:1]
    bdry = torch.zeros_like(geom, dtype=torch.bool)
    bdry[:, 0, :, :] = True
    bdry[:, -1, :, :] = True
    bdry[:, :, 0, :] = True
    bdry[:, :, -1, :] = True
    obs_mask = geom > 0.5
    dir_mask = bdry | obs_mask
    cnt_bdry = float(dir_mask.sum().item())
    cnt_obs = float(obs_mask.sum().item())

    offset = 0
    for a, y_raw in loader:
        a = a.to(device)
        y_raw = y_raw.to(device)

        y_enc = u_normalizer.encode(y_raw)
        out_enc = model(a)

        test_l2_enc += loss_func(out_enc, y_enc).item()

        out_raw = u_normalizer.decode(out_enc)
        test_l2_raw += loss_func(out_raw, y_raw).item()

        mse_raw += F.mse_loss(out_raw, y_raw, reduction="sum").item()

        bs = int(out_raw.shape[0])
        dm = dir_mask[offset : offset + bs].to(device)
        om = obs_mask[offset : offset + bs].to(device)
        diff2 = (out_raw - y_raw) ** 2
        mse_bdry += diff2[dm].sum().item()
        mse_obs += diff2[om].sum().item()
        offset += bs

    metrics = {
        "Test_L2_enc": test_l2_enc / num_test,
        "Test_L2_raw": test_l2_raw / num_test,
        "MSE_px": mse_raw / (num_test * num_pts),
        "MSE_bdry": mse_bdry / max(cnt_bdry, 1.0),
        "MSE_obs": mse_obs / max(cnt_obs, 1.0),
    }
    return metrics


@torch.no_grad()
def evaluate_cft(
    model: ResidualOnDirichletExtension2D,
    *,
    x_mix_ood: torch.Tensor,
    u_ood_raw: torch.Tensor,
    u_normalizer: UnitGaussianNormalizer,
    batch_size: int,
    device: torch.device,
) -> dict:
    """
    在 OOD 数据集上评估几何-atlas CFT-AO+A。
    """
    model.eval()
    loss_func = LpLoss(size_average=False)

    dataset = torch.utils.data.TensorDataset(x_mix_ood, u_ood_raw)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    N = int(u_ood_raw.shape[1])
    num_test = int(u_ood_raw.shape[0])
    num_pts = N * N

    test_l2_enc = 0.0
    test_l2_raw = 0.0
    mse_raw = 0.0
    mse_bdry = 0.0
    mse_obs = 0.0
    mse_Eonly = 0.0
    cnt_bdry = 0.0
    cnt_obs = 0.0

    in_channels_norm = model.in_channels_norm

    for x, y_raw in loader:
        x = x.to(device)
        y_raw = y_raw.to(device)

        out_enc = model(x)
        y_enc = (y_raw - model.y_mean) / model.y_std.clamp(min=model.eps)
        out_raw = u_normalizer.decode(out_enc)

        test_l2_enc += loss_func(out_enc, y_enc).item()
        test_l2_raw += loss_func(out_raw, y_raw).item()

        mse_raw += F.mse_loss(out_raw, y_raw, reduction="sum").item()

        geom_raw = x[..., in_channels_norm : in_channels_norm + 1]
        bdry = torch.zeros_like(geom_raw, dtype=torch.bool)
        bdry[:, 0, :, :] = True
        bdry[:, -1, :, :] = True
        bdry[:, :, 0, :] = True
        bdry[:, :, -1, :] = True
        obs = geom_raw > 0.5

        diff2 = (out_raw - y_raw) ** 2
        mse_bdry += diff2[bdry].sum().item()
        mse_obs += diff2[obs].sum().item()
        cnt_bdry += float(bdry.sum().item())
        cnt_obs += float(obs.sum().item())

        E_raw = model.build_extension_raw(x)
        mse_Eonly += F.mse_loss(E_raw, y_raw, reduction="sum").item()

    metrics = {
        "Test_L2_enc": test_l2_enc / num_test,
        "Test_L2_raw": test_l2_raw / num_test,
        "MSE_px": mse_raw / (num_test * num_pts),
        "MSE_Eonly_px": mse_Eonly / (num_test * num_pts),
        "MSE_bdry": mse_bdry / max(cnt_bdry, 1.0),
        "MSE_obs": mse_obs / max(cnt_obs, 1.0),
    }
    return metrics


def main():
    p = argparse.ArgumentParser(description="Evaluate FNO vs geometry-atlas CFT-AO+A on OOD multi-obstacle data.")
    p.add_argument(
        "--data_path_train",
        type=str,
        default="data/diffusion_multiobstacle2d_N128.pt",
        help="In-distribution training dataset (.pt).",
    )
    p.add_argument(
        "--data_path_ood",
        type=str,
        default="data/diffusion_multiobstacle2d_ood_N64.pt",
        help="Geometric OOD dataset (.pt).",
    )
    p.add_argument(
        "--fno_model",
        type=str,
        default="models/fno_diff2d_multiobs_best.pt",
        help="Checkpoint of FNO baseline.",
    )
    p.add_argument(
        "--cft_model",
        type=str,
        default="models/cftaoA_geomaware_multiobstacle.pt",
        help="Checkpoint of geometry-atlas CFT-AO+A.",
    )
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")

    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 载入 ID 数据，重建 normalizer
    print(f"Loading in-distribution dataset from {args.data_path_train} ...")
    data_train = torch.load(args.data_path_train, map_location="cpu")
    a_train_raw = data_train["a_train"]
    u_train_raw = data_train["u_train"]

    print(
        f"ID shapes: a_train={tuple(a_train_raw.shape)}, "
        f"u_train={tuple(u_train_raw.shape)}"
    )

    a_normalizer = UnitGaussianNormalizer(a_train_raw)
    u_normalizer = UnitGaussianNormalizer(u_train_raw)
    a_normalizer.to(device)
    u_normalizer.to(device)

    in_channels_norm = a_train_raw.shape[-1]

    # 2. 载入 OOD 数据
    print(f"\nLoading OOD dataset from {args.data_path_ood} ...")
    data_ood = torch.load(args.data_path_ood, map_location="cpu")
    a_ood_raw = data_ood["a_test"]
    u_ood_raw = data_ood["u_test"]

    print(
        f"OOD shapes: a_test={tuple(a_ood_raw.shape)}, "
        f"u_test={tuple(u_ood_raw.shape)}"
    )

    # 如果 OOD 分辨率与训练集不同（例如 64 vs 128），先双线性插值到训练分辨率，
    # 这样可以复用同一个 normalizer 和同一套模型权重。
    N_id = int(a_train_raw.shape[1])
    N_ood = int(a_ood_raw.shape[1])
    if N_ood != N_id:
        print(f"Resizing OOD fields from N={N_ood} to N={N_id} for fair comparison ...")
        a_ood_raw = F.interpolate(
            a_ood_raw.permute(0, 3, 1, 2), size=(N_id, N_id), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        u_ood_raw = F.interpolate(
            u_ood_raw.permute(0, 3, 1, 2), size=(N_id, N_id), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        print(
            f"Resized OOD shapes: a_test={tuple(a_ood_raw.shape)}, "
            f"u_test={tuple(u_ood_raw.shape)}"
        )

    # 将 OOD 张量移到与 normalizer 相同的设备，再做 encode
    a_ood_raw = a_ood_raw.to(a_normalizer.mean.device)
    u_ood_raw = u_ood_raw.to(u_normalizer.mean.device)

    a_ood = a_normalizer.encode(a_ood_raw)
    geom_ood_raw = a_ood_raw[..., 0:1]
    bc_ood_raw = a_ood_raw[..., 1:2]
    x_ood_mix = torch.cat([a_ood, geom_ood_raw, bc_ood_raw], dim=-1)

    # 3. 构建并加载 FNO baseline
    print("\nBuilding FNO2d baseline and loading checkpoint ...")
    state_fno = torch.load(args.fno_model, map_location="cpu")
    # 从 checkpoint 推断宽度：fc0.weight 形状为 (width, in_channels+2)
    width_ckpt = None
    if "fc0.weight" in state_fno:
        width_ckpt = int(state_fno["fc0.weight"].shape[0])
        print(f"  Detected FNO width from checkpoint: {width_ckpt}")
    else:
        width_ckpt = 64
        print("  Warning: fc0.weight not found in checkpoint, fallback to width=64.")

    fno_model = build_fno_model(
        in_channels=in_channels_norm,
        out_channels=int(u_train_raw.shape[-1]),
        width=width_ckpt,
        device=device,
    )
    print(f"  Parameters (FNO): {count_params(fno_model)}")
    missing_fno, unexpected_fno = fno_model.load_state_dict(state_fno, strict=False)
    if missing_fno or unexpected_fno:
        print(f"  [FNO] Missing keys: {missing_fno}")
        print(f"  [FNO] Unexpected keys: {unexpected_fno}")

    # 4. 构建并加载几何-atlas CFT-AO+A
    print("\nBuilding geometry-atlas CFT-AO+A model and loading checkpoint ...")
    y_mean = u_normalizer.mean.to(device)
    y_std = u_normalizer.std.to(device)
    cft_model = build_cftaoA_model(
        in_channels_norm=in_channels_norm,
        y_mean=y_mean,
        y_std=y_std,
        device=device,
    )
    print(f"  Backbone params (CFT): {count_params(cft_model.backbone)}")
    print(f"  Total params   (CFT): {count_params(cft_model)}")
    state_cft = torch.load(args.cft_model, map_location="cpu")
    missing_cft, unexpected_cft = cft_model.load_state_dict(state_cft, strict=False)
    if missing_cft or unexpected_cft:
        print(f"  [CFT] Missing keys: {missing_cft}")
        print(f"  [CFT] Unexpected keys: {unexpected_cft}")

    # 5. 评估
    print("\n--- Evaluating FNO baseline on OOD ---")
    metrics_fno = evaluate_fno(
        fno_model,
        a_ood=a_ood.to(device),
        a_ood_raw=a_ood_raw,
        u_ood_raw=u_ood_raw,
        u_normalizer=u_normalizer,
        batch_size=args.batch_size,
        device=device,
    )
    for k, v in metrics_fno.items():
        print(f"[FNO] {k}: {v:.6e}")

    print("\n--- Evaluating geometry-atlas CFT-AO+A on OOD ---")
    metrics_cft = evaluate_cft(
        cft_model,
        x_mix_ood=x_ood_mix,
        u_ood_raw=u_ood_raw,
        u_normalizer=u_normalizer,
        batch_size=args.batch_size,
        device=device,
    )
    for k, v in metrics_cft.items():
        print(f"[CFT] {k}: {v:.6e}")


if __name__ == "__main__":
    main()



