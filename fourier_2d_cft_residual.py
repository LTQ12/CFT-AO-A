from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter

from utilities3 import *
from chebyshev import vectorized_batched_cft

import operator
from functools import reduce
from functools import partial

torch.manual_seed(0)
np.random.seed(0)

################################################################
# 2d cft (Re-used) with optional segment-wise coordinate mapping
################################################################
def _apply_segment_power_map(t_coords: torch.Tensor, L_segments: int, seg_pow_h: torch.Tensor | None):
    """
    Piecewise power mapping on [0,1]:
    For each segment l, u in [0,1] within the segment is mapped to u^{p_l},
    where p_l = softplus(h_l) + 1 to ensure monotone increasing and >=1.
    If seg_pow_h is None, identity is returned.
    """
    if (seg_pow_h is None) or (L_segments is None) or (L_segments <= 0):
        return t_coords
    device = t_coords.device
    dtype = t_coords.dtype
    L = int(L_segments)
    # ensure shape (L,)
    h = seg_pow_h.to(device=device, dtype=dtype).view(-1)
    if h.numel() != L:
        return t_coords
    p = torch.nn.functional.softplus(h) + 1.0  # (L,)
    # compute segment index for each coordinate
    seg_idx = torch.clamp((t_coords * L).floor().long(), 0, L - 1)  # same shape as t_coords
    seg_start = seg_idx.to(dtype) / float(L)
    seg_len = 1.0 / float(L)
    u = (t_coords - seg_start) / seg_len
    # gather p for each position
    p_g = p[seg_idx]
    u_mapped = torch.pow(torch.clamp(u, min=0.0, max=1.0), p_g)
    t_mapped = seg_start + seg_len * u_mapped
    return t_mapped

def cft2d(x, modes1, modes2, L_segments=10, M_cheb=10,
          segmap_h: torch.Tensor | None = None,
          segmap_w: torch.Tensor | None = None):
    B, C, H, W = x.shape
    device = x.device
    
    t_coords_w = torch.linspace(0, 1, W, device=device, dtype=x.dtype)
    t_coords_w = _apply_segment_power_map(t_coords_w, L_segments, segmap_w)
    f_points_w = torch.fft.rfftfreq(W, d=1.0/W)[:modes2].to(device)
    x_reshaped_w = x.permute(0, 2, 1, 3).reshape(B * H, C, W)
    cft_w_complex = vectorized_batched_cft(x_reshaped_w, t_coords_w, f_points_w, L_segments, M_cheb)
    cft_w_complex = cft_w_complex.view(B, H, C, modes2).permute(0, 2, 1, 3)

    t_coords_h = torch.linspace(0, 1, H, device=device, dtype=x.dtype)
    t_coords_h = _apply_segment_power_map(t_coords_h, L_segments, segmap_h)
    f_points_h = torch.fft.fftfreq(H, d=1.0/H).to(device)
    h_indices = torch.cat((torch.arange(0, modes1//2), torch.arange(H-(modes1-modes1//2), H))).to(device)
    f_points_h_selected = f_points_h[h_indices]
    x_reshaped_h = cft_w_complex.permute(0, 3, 1, 2).reshape(B * modes2, C, H)

    cft_h_real = vectorized_batched_cft(x_reshaped_h.real, t_coords_h, f_points_h_selected, L_segments, M_cheb)
    cft_h_imag = vectorized_batched_cft(x_reshaped_h.imag, t_coords_h, f_points_h_selected, L_segments, M_cheb)
    cft_hw_complex = cft_h_real + 1j * cft_h_imag
    
    cft_hw_complex = cft_hw_complex.view(B, modes2, C, modes1).permute(0, 2, 3, 1)
    return cft_hw_complex

################################################################
# The final architecture: FNO with CFT-based Residual Correction
################################################################
class SpectralConv2d_RC(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, L_segments=4, M_cheb=8):
        super(SpectralConv2d_RC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        # 1. Standard FNO learnable weights (the "main path")
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

        # 2. Lightweight network to generate the "residual correction" from CFT features
        self.cft_modes1 = modes1 // 4
        self.cft_modes2 = modes2 // 4
        self.L_segments = L_segments
        self.M_cheb = M_cheb
        cft_flat_dim = self.in_channels * self.cft_modes1 * self.cft_modes2 * 2 # Real/Imag

        self.correction_generator = nn.Sequential(
            nn.Linear(cft_flat_dim, (self.in_channels + self.out_channels) * 2),
            nn.GELU(),
            nn.Linear((self.in_channels + self.out_channels) * 2, self.out_channels)
        )
        # CRITICAL: Initialize last layer to zeros for stability.
        # This makes the correction path an identity function at the start of training.
        nn.init.zeros_(self.correction_generator[-1].weight)
        nn.init.zeros_(self.correction_generator[-1].bias)


    def forward(self, x):
        B, C, H, W = x.shape

        # --- Main FNO Path ---
        x_ft = torch.fft.rfft2(x, s=(H, W))
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = \
            torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        x_fno = torch.fft.irfft2(out_ft, s=(H, W))

        # --- CFT Residual Correction Path ---
        cft_coeffs = cft2d(x, self.cft_modes1, self.cft_modes2, self.L_segments, self.M_cheb)
        cft_flat = torch.view_as_real(cft_coeffs).flatten(1)
        correction = self.correction_generator(cft_flat) # (B, out_channels)
        
        # Reshape correction to be broadcastable with the spatial dimensions
        correction = correction.view(B, self.out_channels, 1, 1)

        # Add the correction to the main path output
        return x_fno + correction

class FNO_RC(nn.Module):
    def __init__(self, modes1, modes2, width, in_channels, out_channels):
        super(FNO_RC, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9
        self.fc0 = nn.Linear(in_channels + 2, self.width)

        self.conv0 = SpectralConv2d_RC(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_RC(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_RC(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_RC(self.width, self.width, self.modes1, self.modes2)
        
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device) 