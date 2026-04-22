"""
Train U-NO + A on obstacle2d-style datasets with [geom, bc] inputs.

This mirrors train_fnoA_diff2d_obstacle.py so the comparison remains within
the same A-wrap protocol.
"""

from __future__ import annotations

import argparse
import os
from timeit import default_timer

import numpy as np
import torch
import torch.nn.functional as F

from Adam import Adam
from boundary_ext_residual_2d import ResidualOnDirichletExtension2D
from uno_2d import UNO2d
from utilities3 import LpLoss, UnitGaussianNormalizer, count_params


def main(args):
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Seed: {int(args.seed)}")

    print(f"Loading dataset from {args.data_path} ...")
    data = torch.load(args.data_path, map_location="cpu")
    a_train_raw = data["a_train"]
    u_train_raw = data["u_train"]
    a_test_raw = data["a_test"]
    u_test_raw = data["u_test"]

    print(
        f"Shapes: a_train={tuple(a_train_raw.shape)}, u_train={tuple(u_train_raw.shape)}, "
        f"a_test={tuple(a_test_raw.shape)}, u_test={tuple(u_test_raw.shape)}"
    )

    a_normalizer = UnitGaussianNormalizer(a_train_raw)
    a_train = a_normalizer.encode(a_train_raw)
    a_test = a_normalizer.encode(a_test_raw)

    u_normalizer = UnitGaussianNormalizer(u_train_raw)
    u_train = u_normalizer.encode(u_train_raw)
    u_test_enc = u_normalizer.encode(u_test_raw)

    geom_train_raw = a_train_raw[..., 0:1]
    bc_train_raw = a_train_raw[..., 1:2]
    geom_test_raw = a_test_raw[..., 0:1]
    bc_test_raw = a_test_raw[..., 1:2]
    x_train_mix = torch.cat([a_train, geom_train_raw, bc_train_raw], dim=-1)
    x_test_mix = torch.cat([a_test, geom_test_raw, bc_test_raw], dim=-1)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train_mix, u_train),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test_mix, u_test_enc),
        batch_size=args.batch_size,
        shuffle=False,
    )

    a_normalizer.to(device)
    u_normalizer.to(device)

    in_channels = a_train.shape[-1]
    out_channels = u_train.shape[-1]

    backbone = UNO2d(
        in_channels=in_channels,
        out_channels=out_channels,
        width=args.width,
        modes1=args.modes,
        modes2=args.modes,
        pad=args.pad,
        factor=args.factor,
    ).to(device)

    model = ResidualOnDirichletExtension2D(
        backbone,
        y_mean=u_normalizer.mean,
        y_std=u_normalizer.std,
        in_channels_norm=in_channels,
        delta=args.delta,
        res_scale_init=args.res_scale_init,
        res_scale_max=args.res_scale_max,
        ext_method=args.ext_method,
        ext_iters=args.ext_iters,
        poisson_src_hidden=args.poisson_src_hidden,
        poisson_src_scale_max=args.poisson_src_scale_max,
        residual_clip=args.residual_clip,
    ).to(device)

    print("\nModel: U-NO + A(residual-on-extension) (obstacle2d)")
    print(f"Backbone parameters: {count_params(backbone)}")
    print(f"Total parameters: {count_params(model)}")
    print(
        f"Hyperparameters: Modes={args.modes}, Width={args.width}, Pad={args.pad}, Factor={args.factor}, "
        f"LR={args.learning_rate}, WeightDecay={args.weight_decay}, delta={args.delta}, "
        f"res_scale_init={args.res_scale_init}, res_scale_max={args.res_scale_max}, res_reg={args.res_reg}, "
        f"ext_method={args.ext_method}, ext_iters={args.ext_iters}, residual_clip={args.residual_clip}"
    )

    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_func = LpLoss(size_average=False)

    n = int(a_train_raw.shape[1])
    num_pts = n * n
    best_test = float("inf")
    print("\n--- Starting Training (U-NO + A) ---")

    for ep in range(args.epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_func(out, y)

            if args.res_reg > 0:
                e_raw = model.build_extension_raw(x)
                e_enc = (e_raw - u_normalizer.mean) / (u_normalizer.std + 1e-12)
                res = out - e_enc
                loss = loss + float(args.res_reg) * (res**2).mean()

            loss.backward()
            optimizer.step()
            train_l2 += loss.item()

        scheduler.step()

        model.eval()
        test_l2 = 0.0
        raw_test_l2 = 0.0
        mse_raw = 0.0
        mse_bdry = 0.0
        mse_obs = 0.0
        mse_eonly = 0.0
        cnt_bdry = 0.0
        cnt_obs = 0.0

        with torch.no_grad():
            for x, y_enc in test_loader:
                x, y_enc = x.to(device), y_enc.to(device)
                out_enc = model(x)
                test_l2 += loss_func(out_enc, y_enc).item()
                out = u_normalizer.decode(out_enc)
                y = u_normalizer.decode(y_enc)
                raw_test_l2 += loss_func(out, y).item()
                mse_raw += F.mse_loss(out, y, reduction="sum").item()

                geom_raw = x[..., in_channels : in_channels + 1]
                bdry = torch.zeros_like(geom_raw, dtype=torch.bool)
                bdry[:, 0, :, :] = True
                bdry[:, -1, :, :] = True
                bdry[:, :, 0, :] = True
                bdry[:, :, -1, :] = True
                obs = geom_raw > 0.5
                diff2 = (out - y) ** 2
                mse_bdry += diff2[bdry].sum().item()
                mse_obs += diff2[obs].sum().item()
                cnt_bdry += float(bdry.sum().item())
                cnt_obs += float(obs.sum().item())

                e_raw = model.build_extension_raw(x)
                mse_eonly += F.mse_loss(e_raw, y, reduction="sum").item()

        train_l2 /= a_train_raw.shape[0]
        test_l2 /= a_test_raw.shape[0]
        raw_test_l2 /= a_test_raw.shape[0]
        num_test = int(a_test_raw.shape[0])
        mse_raw /= (num_test * num_pts)
        mse_eonly /= (num_test * num_pts)
        mse_bdry = mse_bdry / max(cnt_bdry, 1.0)
        mse_obs = mse_obs / max(cnt_obs, 1.0)
        t2 = default_timer()

        if (ep + 1) % 10 == 0 or ep == 0:
            print(
                f"Epoch {ep+1}/{args.epochs} | Time: {t2-t1:.2f}s | "
                f"Train L2: {train_l2:.6f} | Test L2: {test_l2:.6f} | Raw Test L2: {raw_test_l2:.6f} | "
                f"Raw MSE(px): {mse_raw:.6e} | E-only MSE(px): {mse_eonly:.6e} | "
                f"MSE(bdry): {mse_bdry:.6e} | MSE(obs): {mse_obs:.6e} | "
                f"res_scale: {float(model.get_res_scale().detach().cpu()):.4f}"
            )

        if test_l2 < best_test - 1e-8:
            best_test = test_l2
            if args.model_save_path:
                best_path = args.model_save_path.replace(".pt", "_best.pt")
                best_dir = os.path.dirname(best_path)
                if best_dir and not os.path.exists(best_dir):
                    os.makedirs(best_dir, exist_ok=True)
                torch.save(model.state_dict(), best_path)

    print("--- Training Finished (U-NO + A) ---")
    if args.model_save_path:
        model_dir = os.path.dirname(args.model_save_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), args.model_save_path)
        yn_stats = {"mean": u_normalizer.mean.detach().cpu(), "std": u_normalizer.std.detach().cpu()}
        yn_path = os.path.join(model_dir, "unoA_y_normalizer.pt")
        torch.save(yn_stats, yn_path)
        print(f"Model saved to {args.model_save_path}\nSaved u_normalizer stats to {yn_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train U-NO + A on obstacle2d datasets.")
    p.add_argument("--data_path", type=str, default="data/diffusion_obstacle2d_N64.pt")
    p.add_argument("--model_save_path", type=str, default="models/unoA_diffusion_obstacle2d.pt")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--modes", type=int, default=12)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--pad", type=int, default=8)
    p.add_argument("--factor", type=int, default=1)
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
    p.add_argument("--residual_clip", type=float, default=0.0)

    main(p.parse_args())
