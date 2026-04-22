"""
U-NO style 2D backbone for regular-grid operator learning.

This implementation follows the public U-NO design at a high level:
  - add coordinate channels,
  - lift to a channel space,
  - alternate spectral operator blocks with domain contraction/expansion,
  - use skip connections in a U-shaped layout,
  - project back to the target channel dimension.

The module is written to match this repository's NHWC convention so it can be
used directly as a backbone inside ResidualOnDirichletExtension2D.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2dUNO(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        scale = (1.0 / max(2 * self.in_channels, 1)) ** 0.5
        self.weights_pos = nn.Parameter(
            scale * torch.randn(self.in_channels, self.out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights_neg = nn.Parameter(
            scale * torch.randn(self.in_channels, self.out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def compl_mul2d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor, out_h: int | None = None, out_w: int | None = None) -> torch.Tensor:
        batch_size, _, in_h, in_w = x.shape
        out_h = int(in_h if out_h is None else out_h)
        out_w = int(in_w if out_w is None else out_w)

        x_ft = torch.fft.rfft2(x, norm="forward")
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            out_h,
            out_w // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        m1 = min(self.modes1, in_h // 2, out_h // 2)
        m2 = min(self.modes2, x_ft.shape[-1], out_ft.shape[-1])
        if m1 > 0 and m2 > 0:
            out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], self.weights_pos[:, :, :m1, :m2])
            out_ft[:, :, -m1:, :m2] = self.compl_mul2d(x_ft[:, :, -m1:, :m2], self.weights_neg[:, :, :m1, :m2])

        return torch.fft.irfft2(out_ft, s=(out_h, out_w), norm="forward")


class PointwiseOp2dUNO(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1)

    def forward(self, x: torch.Tensor, out_h: int | None = None, out_w: int | None = None) -> torch.Tensor:
        y = self.proj(x)
        if out_h is not None and out_w is not None and (y.shape[-2] != out_h or y.shape[-1] != out_w):
            y = F.interpolate(y, size=(out_h, out_w), mode="bicubic", align_corners=False)
        return y


class OperatorBlock2dUNO(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        *,
        normalize: bool = False,
        nonlin: bool = True,
    ):
        super().__init__()
        self.spec = SpectralConv2dUNO(in_channels, out_channels, modes1, modes2)
        self.point = PointwiseOp2dUNO(in_channels, out_channels)
        self.normalize = nn.InstanceNorm2d(int(out_channels), affine=True) if normalize else None
        self.nonlin = bool(nonlin)

    def forward(self, x: torch.Tensor, out_h: int | None = None, out_w: int | None = None) -> torch.Tensor:
        y = self.spec(x, out_h, out_w) + self.point(x, out_h, out_w)
        if self.normalize is not None:
            y = self.normalize(y)
        if self.nonlin:
            y = F.gelu(y)
        return y


class UNO2d(nn.Module):
    """
    NHWC in / NHWC out U-NO style backbone for 2D scalar-field prediction.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        width: int = 64,
        modes1: int = 12,
        modes2: int = 12,
        pad: int = 8,
        factor: int = 1,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.width = int(width)
        self.pad = int(pad)
        self.factor = int(factor)

        w = self.width
        f = self.factor

        self.fc_n1 = nn.Linear(self.in_channels + 2, max(w // 2, 16))
        self.fc0 = nn.Linear(max(w // 2, 16), w)

        self.block0 = OperatorBlock2dUNO(w, 2 * f * w, modes1, modes2, normalize=False, nonlin=True)
        self.block1 = OperatorBlock2dUNO(2 * f * w, 4 * f * w, max(modes1 // 2, 4), max(modes2 // 2, 4), normalize=True, nonlin=True)
        self.block2 = OperatorBlock2dUNO(4 * f * w, 4 * f * w, max(modes1 // 2, 4), max(modes2 // 2, 4), normalize=False, nonlin=True)
        self.block3 = OperatorBlock2dUNO(4 * f * w, 2 * f * w, max(modes1 // 2, 4), max(modes2 // 2, 4), normalize=True, nonlin=True)
        self.block4 = OperatorBlock2dUNO(4 * f * w, w, modes1, modes2, normalize=False, nonlin=True)

        self.fc1 = nn.Linear(2 * w, w)
        self.fc2 = nn.Linear(w, self.out_channels)

        nn.init.xavier_uniform_(self.fc_n1.weight)
        nn.init.zeros_(self.fc_n1.bias)
        nn.init.xavier_uniform_(self.fc0.weight)
        nn.init.zeros_(self.fc0.bias)

    @staticmethod
    def get_grid(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        batch_size, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float32, device=device)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat(batch_size, 1, size_y, 1)
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float32, device=device)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat(batch_size, size_x, 1, 1)
        return torch.cat((gridx, gridy), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grid = self.get_grid(tuple(x.shape), x.device)
        x = torch.cat((x, grid), dim=-1)

        x = F.gelu(self.fc_n1(x))
        x = F.gelu(self.fc0(x))
        x0 = x.permute(0, 3, 1, 2)

        if self.pad > 0:
            x0 = F.pad(x0, [0, self.pad, 0, self.pad])

        h, w = x0.shape[-2], x0.shape[-1]
        h2, w2 = max(h // 2, 4), max(w // 2, 4)
        h4, w4 = max(h // 4, 4), max(w // 4, 4)

        x1 = self.block0(x0, h2, w2)
        x2 = self.block1(x1, h4, w4)
        x3 = self.block2(x2, h4, w4)
        x4 = self.block3(x3, h2, w2)
        x4 = torch.cat([x4, x1], dim=1)
        x5 = self.block4(x4, h, w)
        x5 = torch.cat([x5, x0], dim=1)

        if self.pad > 0:
            x5 = x5[..., : -self.pad, : -self.pad]

        x5 = x5.permute(0, 2, 3, 1)
        x5 = F.gelu(self.fc1(x5))
        return self.fc2(x5)
