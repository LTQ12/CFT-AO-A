from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from fourier_2d_cft_residual import cft2d


class CFTAOBlock2D(nn.Module):
    """
    CFT-Analytic Operator block (2D, single chart, low-parameter PDE symbol).

    - 输入:  (B, C, H, W)
    - CFT:   x -> coeffs in (k_y, k_x) with learnable segment-wise power map
    - 符号核: A(|k|; alpha, nu, c) * exp(i Phi(|k|; omega))
    - 逆变换: coeffs -> rFFT 网格 -> irfft2
    - 薄局部残差: 1x1 conv + InstanceNorm + GELU
    - 输出: x + residual
    """

    def __init__(
        self,
        width: int,
        modes1: int,
        modes2: int,
        L_segments: int = 4,
        M_cheb: int = 4,
        cft_res: int = 0,
        use_local: bool = True,
        radial_K: int = 4,
        anisotropic: bool = False,
        cond_dim: int = 0,
        n_iter: int = 1,
        n_bands: int = 3,
        n_sym_bases: int = 0,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        self.L_segments = int(L_segments)
        self.M_cheb = int(M_cheb)
        self.cft_res = int(cft_res)
        self.radial_K = int(radial_K)
        self.anisotropic = bool(anisotropic)
        self.cond_dim = int(cond_dim)
        self.n_iter = max(int(n_iter), 1)
        self.n_bands = max(int(n_bands), 0)
        self.n_sym_bases = max(int(n_sym_bases), 0)

        # 段映射参数：seg_h_h / seg_h_w，控制 H/W 方向的 power-map，近似 conformal 压缩/展开
        init_h = -4.0  # softplus(-4)≈0.018 => p≈1.018 ~ 接近恒等
        self.seg_h_h = nn.Parameter(torch.full((self.L_segments,), init_h, dtype=torch.float32))
        self.seg_h_w = nn.Parameter(torch.full((self.L_segments,), init_h, dtype=torch.float32))

        # 低参数 PDE 符号核：A(|k|) = exp(-nu |k|^2) + c / (1 + alpha |k|^2)
        # 相位 Phi(|k|) = omega_y * ky + omega_x * kx
        self.nu_log = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))     # softplus -> 正的扩散系数
        self.alpha_log = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))  # 正的 1/(1+alpha|k|^2) 权重
        self.c_log = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))      # 正的 c
        self.omega_y = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.omega_x = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # 各向异性符号核参数（仅在 anisotropic=True 时起主导作用）
        self.nu_y_log = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        self.nu_x_log = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        self.alpha_y_log = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        self.alpha_x_log = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))

        # 符号基库: 额外的各向同性 PDE 符号基 (仅在 isotropic 分支中使用)
        # 总的各向同性基数 = 1(main) + n_sym_bases(extra)
        if self.n_sym_bases > 0:
            self.sym_nu_log = nn.Parameter(torch.full((self.n_sym_bases,), -2.0, dtype=torch.float32))
            self.sym_alpha_log = nn.Parameter(torch.full((self.n_sym_bases,), -2.0, dtype=torch.float32))
            self.sym_c_log = nn.Parameter(torch.full((self.n_sym_bases,), -2.0, dtype=torch.float32))
            # 每个通道对各个符号基的线性权重 (包含 main 基在内, 共 1+n_sym_bases 个)
            self.sym_weights = nn.Parameter(torch.zeros(self.width, 1 + self.n_sym_bases, dtype=torch.float32))
        else:
            self.sym_nu_log = None
            self.sym_alpha_log = None
            self.sym_c_log = None
            self.sym_weights = None

        # 通道相关的径向谱头：对每个输出通道引入一个小多项式调制
        # amp_c(r) = 1 + sum_k a_ck * r^k,  phase_c(r) = sum_k b_ck * r^k
        self.amp_coef = nn.Parameter(torch.zeros(self.width, self.radial_K, dtype=torch.float32))
        self.phase_coef = nn.Parameter(torch.zeros(self.width, self.radial_K, dtype=torch.float32))

        # 频带图册: 对低/中/高三段频率的幅度做全局可学习增益
        if self.n_bands > 0:
            self.band_gain = nn.Parameter(torch.zeros(self.n_bands, dtype=torch.float32))
        else:
            self.band_gain = None

        # 条件化符号核: 根据输入场的全局统计量调节各通道径向增益
        if self.cond_dim > 0:
            self.cond_mlp = nn.Sequential(
                nn.Linear(self.cond_dim, self.width),
                nn.Tanh(),
            )
        else:
            self.cond_mlp = None

        # 薄局部残差
        self.use_local = bool(use_local)
        if self.use_local:
            self.local = nn.Conv2d(self.width, self.width, kernel_size=1)
            # 极薄 3x3 depthwise 残差, 提供局部各向异性非线性
            self.local_dw3 = nn.Conv2d(self.width, self.width, kernel_size=3, padding=1, groups=self.width)
            # 通道间非线性 MLP, 仅在通道维度混合
            self.channel_mlp = nn.Sequential(
                nn.Conv2d(self.width, self.width * 2, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(self.width * 2, self.width, kernel_size=1),
            )
            self.norm = nn.InstanceNorm2d(self.width, affine=True)
        else:
            self.local = None
            self.local_dw3 = None
            self.channel_mlp = None
            self.norm = None

        # runtime 可调缩放（由训练脚本写入）
        self.local_scale = 0.3
        self.spatial_scale = 0.15
        self.spec_scale = 1.0

        # 轻量自由谱核残差: 允许在 CFT 先验核之外做小的频域修正，提升可学习性
        # we_param: (C,m1,m2) 复数权重, 初始为 0, 只在 coeffs 上做 (1 + eps * we_param) 的乘法
        self.free_eps = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.free_kernel = nn.Parameter(
            torch.zeros(self.width, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    def _maybe_resize(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        B, C, H, W = x.shape
        if self.cft_res > 0 and (H != self.cft_res or W != self.cft_res):
            Hr = self.cft_res
            Wr = self.cft_res
            x_rs = F.interpolate(x, size=(Hr, Wr), mode="bilinear", align_corners=False)
            return x_rs, H, W
        return x, H, W

    def _inv_resize(self, y: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # y: (B,C,Hc,Wc) -> (B,C,H,W)
        Hc, Wc = y.shape[-2], y.shape[-1]
        if Hc == H and Wc == W:
            return y
        y_rs = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
        return y_rs

    def _single_step(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        # 单步谱+局部更新: x -> x + F(x; theta, cond)
        B, C, H_in, W_in = x.shape
        x_cft, Hc, Wc = self._maybe_resize(x)

        # CFT: (B,C,Hc,Wc) -> (B,C,m1,m2)
        coeffs = cft2d(
            x_cft,
            self.modes1,
            self.modes2,
            L_segments=self.L_segments,
            M_cheb=self.M_cheb,
            segmap_h=self.seg_h_h,
            segmap_w=self.seg_h_w,
        )  # (B,C,m1,m2)
        Bc, Cc, m1, m2 = coeffs.shape

        # 构造 |k| 与 |k|^2 及 (ky,kx) 网格
        device = coeffs.device
        dtype = coeffs.real.dtype
        ky = torch.linspace(0.0, 1.0, m1, device=device, dtype=dtype)
        kx = torch.linspace(0.0, 1.0, m2, device=device, dtype=dtype)
        Ky, Kx = torch.meshgrid(ky, kx, indexing="ij")  # (m1,m2)
        r2 = Ky * Ky + Kx * Kx
        r = torch.sqrt(r2 + 1e-12)

        # 全局幅度 A(k): 可选各向同性 / 各向异性 + 各向同性符号基库
        c_amp_main = F.softplus(self.c_log)
        if self.anisotropic:
            ky2 = Ky * Ky
            kx2 = Kx * Kx
            nu_y = F.softplus(self.nu_y_log)
            nu_x = F.softplus(self.nu_x_log)
            alpha_y = F.softplus(self.alpha_y_log)
            alpha_x = F.softplus(self.alpha_x_log)
            amp_base = torch.exp(-(nu_y * ky2 + nu_x * kx2)) + c_amp_main / (
                1.0 + alpha_y * ky2 + alpha_x * kx2 + 1e-6
            )
        else:
            # main 各向同性基
            nu0 = F.softplus(self.nu_log)
            alpha0 = F.softplus(self.alpha_log)
            base0 = torch.exp(-nu0 * r2) + c_amp_main / (1.0 + alpha0 * r2 + 1e-6)  # (m1,m2)
            if self.n_sym_bases > 0 and self.sym_nu_log is not None:
                # 额外各向同性符号基
                nus = F.softplus(self.sym_nu_log)      # (K,)
                alphas = F.softplus(self.sym_alpha_log)
                cs = F.softplus(self.sym_c_log)
                # (K, m1, m2)
                r2_expanded = r2.unsqueeze(0)  # (1,m1,m2)
                bases_extra = torch.exp(-nus.view(-1, 1, 1) * r2_expanded) + cs.view(-1, 1, 1) / (
                    1.0 + alphas.view(-1, 1, 1) * r2_expanded + 1e-6
                )
                # 拼接 main+extra -> (1+K, m1, m2)
                bases_all = torch.cat([base0.unsqueeze(0), bases_extra], dim=0)
                # 每通道对各符号基的线性组合
                theta = self.sym_weights  # (C, 1+K)
                amp_mix = torch.einsum("ck,khw->chw", theta, bases_all)  # (C,m1,m2)
                amp_base = amp_mix  # (C,m1,m2)
            else:
                amp_base = base0  # (m1,m2)

        # 频带图册: 对低/中/高频带做全局增益, 体现符号在不同频率段的结构差异
        if self.band_gain is not None:
            # 简单三段: [0,t1], (t1,t2], (t2,1]
            t1, t2 = 0.33, 0.66
            w0 = (r <= t1).to(amp_base.dtype)
            w2 = (r >= t2).to(amp_base.dtype)
            w1 = (1.0 - w0 - w2).clamp(min=0.0)
            gains = F.softplus(self.band_gain)  # (n_bands,)
            # 兼容 n_bands = 1,2,3+ 的情况
            if gains.numel() >= 3:
                g0, g1, g2 = gains[0], gains[1], gains[2]
            elif gains.numel() == 2:
                g0, g1, g2 = gains[0], gains[1], gains[1]
            elif gains.numel() == 1:
                g0 = g1 = g2 = gains[0]
            else:
                g0 = g1 = g2 = amp_base.new_tensor(0.0)
            scale = (1.0 + g0) * w0 + (1.0 + g1) * w1 + (1.0 + g2) * w2
            amp_base = amp_base * scale

        # 全局相位 Phi(|k|)
        phi_base = self.omega_y * Ky + self.omega_x * Kx

        # 通道相关径向调制
        # 构造多项式基 B_k(r) = r^k, k=0..K-1
        basis = [torch.ones_like(r)]
        for k in range(1, self.radial_K):
            basis.append(basis[-1] * r)
        B_rad = torch.stack(basis, dim=0)  # (K, m1, m2)

        # (C,K) x (K,m1,m2) -> (C,m1,m2)
        amp_delta = torch.einsum("ck,khw->chw", F.softplus(self.amp_coef), B_rad)
        phase_delta = torch.einsum("ck,khw->chw", self.phase_coef, B_rad)

        # 条件化调制: 使用输入场的全局统计量调节各通道的径向增益
        if self.cond_mlp is not None and cond is not None:
            # cond_embed: (B,width) -> 每个样本、每个通道一个 gate
            cond_embed = self.cond_mlp(cond)  # (B,C)
            cond_gate = (1.0 + 0.5 * cond_embed).view(B, Cc, 1, 1)  # (B,C,1,1)

            amp_delta_bc = amp_delta.unsqueeze(0) * cond_gate  # (B,C,Hk,Wk)
            phase_delta_bc = phase_delta.unsqueeze(0)          # (B,C,Hk,Wk)

            # amp_base 可能是 (C,m1,m2) 或 (m1,m2)
            if amp_base.dim() == 3:
                amp_base_bc = amp_base.unsqueeze(0)  # (1,C,m1,m2)
            else:
                amp_base_bc = amp_base.view(1, 1, m1, m2)
            amp_full = amp_base_bc * (1.0 + amp_delta_bc.clamp(min=0.0))  # (B,C,m1,m2)
            phi_full = phi_base.view(1, 1, m1, m2) + phase_delta_bc                       # (B,C,m1,m2)

            kernel = torch.complex(torch.cos(phi_full) * amp_full, torch.sin(phi_full) * amp_full)  # (B,C,m1,m2)
            coeffs = coeffs * kernel * self.spec_scale
        else:
            if amp_base.dim() == 3:
                amp_full = amp_base * (1.0 + amp_delta.clamp(min=0.0))  # (C,m1,m2)
            else:
                amp_full = amp_base.view(1, m1, m2) * (1.0 + amp_delta.clamp(min=0.0))  # (1,m1,m2)
            phi_full = phi_base.view(1, m1, m2) + phase_delta                       # (C,m1,m2)
            kernel = torch.complex(torch.cos(phi_full) * amp_full, torch.sin(phi_full) * amp_full)  # (C,m1,m2)
            coeffs = coeffs * kernel.view(1, Cc, m1, m2) * self.spec_scale

        # 自由谱核残差: 在 CFT 先验核之外添加一个小的、可学习的频域修正
        if self.free_kernel is not None:
            # 截断到当前有效模式数，防止 m1/m2 变化时越界
            m1_eff = min(self.modes1, m1)
            m2_eff = min(self.modes2, m2)
            if m1_eff > 0 and m2_eff > 0:
                fk = self.free_kernel[:, :m1_eff, :m2_eff]  # (C,m1_eff,m2_eff)
                # 扩展到 batch 维并在高频区用 1 填充
                free_mask = torch.ones(Bc, Cc, m1, m2, dtype=torch.cfloat, device=device)
                free_mask[:, :, :m1_eff, :m2_eff] = 1.0 + self.free_eps * fk.unsqueeze(0)
                coeffs = coeffs * free_mask

        # 嵌入 rFFT 网格并 irfft2 回到物理域
        out_ft = torch.zeros(Bc, Cc, Hc, Wc // 2 + 1, dtype=torch.cfloat, device=device)
        half = m1 // 2
        neg = m1 - half
        pos_idx = torch.arange(0, half, device=device)
        neg_idx = torch.arange(Hc - neg, Hc, device=device)
        if half > 0:
            out_ft[:, :, pos_idx, :m2] = coeffs[:, :, :half, :]
        if neg > 0:
            out_ft[:, :, neg_idx, :m2] = coeffs[:, :, half:, :]

        y_spec = torch.fft.irfft2(out_ft, s=(Hc, Wc))  # (B,C,Hc,Wc)
        y_spec = self._inv_resize(y_spec, H_in, W_in)

        if self.use_local and self.local is not None and self.norm is not None:
            y_loc1 = self.local(x) * self.local_scale
            if self.local_dw3 is not None:
                y_loc2 = self.local_dw3(x) * self.spatial_scale
                y = y_spec + y_loc1 + y_loc2
            else:
                y = y_spec + y_loc1
            y = self.norm(F.gelu(y))
            if self.channel_mlp is not None:
                y = y + self.channel_mlp(y) * 0.1
        else:
            y = y_spec

        return x + y

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B,C,H,W), cond: (B,cond_dim) or None
        out = x
        for _ in range(self.n_iter):
            out = self._single_step(out, cond)
        return out


class CFT_AO_2D(nn.Module):
    """
    CFT-Analytic Operator (2D) network.

    - 输入:  (B, H, W, C_in)   (a(x), x, y)
    - 输出:  (B, H, W, C_out) (u(x))
    - 结构: fc0 抬升 -> K 层 CFTAOBlock2D -> fc1/fc2 投影
    """

    def __init__(
        self,
        modes1: int,
        modes2: int,
        width: int = 32,
        in_channels: int = 1,
        out_channels: int = 1,
        n_layers: int = 3,
        L_segments: int = 4,
        M_cheb: int = 4,
        cft_res: int = 0,
        use_local: bool = True,
        cond_dim: int = 2,
        inner_iters: int = 2,
        n_bands: int = 3,
    ) -> None:
        super().__init__()
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        self.width = int(width)
        self.padding = 9  # 非周期边界时做零填充，模仿 FNO2d 习惯

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_layers = int(n_layers)
        self.cond_dim = int(cond_dim)
        self.inner_iters = max(int(inner_iters), 1)
        self.n_bands = max(int(n_bands), 0)

        # 输入抬升：拼接坐标 (x,y)
        self.fc0 = nn.Linear(self.in_channels + 2, self.width)

        blocks = []
        for _ in range(self.n_layers):
            blk = CFTAOBlock2D(
                width=self.width,
                modes1=self.modes1,
                modes2=self.modes2,
                L_segments=L_segments,
                M_cheb=M_cheb,
                cft_res=cft_res,
                use_local=use_local,
                cond_dim=self.cond_dim,
                n_iter=self.inner_iters,
                n_bands=self.n_bands,
            )
            blocks.append(blk)
        self.blocks = nn.ModuleList(blocks)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.out_channels)

    def get_grid(self, shape, device):
        B, H, W, _ = shape
        gridx = torch.tensor(np.linspace(0, 1, H), dtype=torch.float32, device=device)
        gridx = gridx.reshape(1, H, 1, 1).repeat(B, 1, W, 1)
        gridy = torch.tensor(np.linspace(0, 1, W), dtype=torch.float32, device=device)
        gridy = gridy.reshape(1, 1, W, 1).repeat(B, H, 1, 1)
        return torch.cat((gridx, gridy), dim=-1)

    @staticmethod
    def get_dirichlet_window(H: int, W: int, device: torch.device) -> torch.Tensor:
        """
        构造满足零 Dirichlet 边界的固定权重:
            w(x,y) = x(1-x) y(1-y),   x,y in [0,1]
        在边界处 w=0, 内部 >0.
        """
        xs = torch.linspace(0.0, 1.0, H, device=device, dtype=torch.float32)
        ys = torch.linspace(0.0, 1.0, W, device=device, dtype=torch.float32)
        X, Y = torch.meshgrid(xs, ys, indexing="ij")
        w = (X * (1.0 - X)) * (Y * (1.0 - Y))  # (H,W)
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,H,W,C_in)
        B, H, W, Cin = x.shape
        device = x.device

        # 使用输入场的全局统计量 (均值/标准差) 作为符号核的条件信息
        a_field = x[..., 0]  # (B,H,W) 假定第一个通道是 Darcy 系数场 a(x)
        mean_a = a_field.mean(dim=(1, 2), keepdim=False)  # (B,)
        std_a = a_field.std(dim=(1, 2), unbiased=False)   # (B,)
        cond = torch.stack((mean_a, std_a), dim=-1)       # (B,2)

        grid = self.get_grid(x.shape, device)
        x = torch.cat((x, grid), dim=-1)  # (B,H,W,Cin+2)
        x = self.fc0(x)                   # (B,H,W,width)
        x = x.permute(0, 3, 1, 2)        # (B,width,H,W)

        # padding for non-periodic boundary
        x = F.pad(x, [0, self.padding, 0, self.padding])  # pad W,H

        for blk in self.blocks:
            x = blk(x, cond)

        # remove padding
        x = x[:, :, :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)  # (B,H,W,width)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)            # (B,H,W,Cout)

        # 物理先验：Darcy2D 数据集采用 u=0 的 Dirichlet 边界条件
        # 在网络输出端显式嵌入边界层权重，使预测在边界严格为 0
        w = self.get_dirichlet_window(H, W, device)
        x = x * w.view(1, H, W, 1)
        return x


class CFTAtlasBlock2D(nn.Module):
    """
    CFT-AO 2D 的 boundary-atlas 版本:
    - 一条 core 图册: 侧重内部区域 (远离边界)
    - 一条 rim 图册 : 侧重边界附近, 可用不同的 L/M 与符号核
    通过几何先验的软权重 (基于到边界的距离) 做融合.
    """

    def __init__(
        self,
        width: int,
        modes1: int,
        modes2: int,
        L_segments_core: int = 4,
        M_cheb_core: int = 4,
        L_segments_rim: int = 6,
        M_cheb_rim: int = 6,
        cft_res: int = 0,
        use_local: bool = True,
        rim_ratio: float = 0.15,
        cond_dim: int = 0,
        n_iter: int = 1,
        n_bands: int = 3,
        n_sym_bases: int = 2,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        self.rim_ratio = float(rim_ratio)
        self.n_iter = max(int(n_iter), 1)
        self.n_bands = max(int(n_bands), 0)
        self.n_sym_bases = max(int(n_sym_bases), 0)
        # core 区域再细分的子图册个数 (1 表示单一 core, >1 表示多 core 图册)
        self.n_core_charts = 2  # 这里先固定为 2 个 core 图册: 左/右 或 上/下

        # 内部/边界各自一套 CFT-AO block, 拥有独立的符号核与映射参数
        # core: 多图册版本, 先实现 2 个子图册
        core_blocks = []
        for _ in range(self.n_core_charts):
            core_blocks.append(
                CFTAOBlock2D(
                    width=self.width,
                    modes1=self.modes1,
                    modes2=self.modes2,
                    L_segments=L_segments_core,
                    M_cheb=M_cheb_core,
                    cft_res=cft_res,
                    use_local=use_local,
                    anisotropic=False,
                    cond_dim=cond_dim,
                    n_iter=self.n_iter,
                    n_bands=self.n_bands,
                    n_sym_bases=self.n_sym_bases,
                )
            )
        self.core_blocks = nn.ModuleList(core_blocks)
        # 边界图册采用各向异性符号核, 更贴合 Darcy 通道方向
        self.rim_block = CFTAOBlock2D(
            width=self.width,
            modes1=self.modes1,
            modes2=self.modes2,
            L_segments=L_segments_rim,
            M_cheb=M_cheb_rim,
            cft_res=cft_res,
            use_local=use_local,
            anisotropic=True,
            cond_dim=cond_dim,
            n_iter=self.n_iter,
            n_bands=self.n_bands,
            n_sym_bases=self.n_sym_bases,
        )

    @staticmethod
    def boundary_distance(H: int, W: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        几何先验: 到边界的距离 d(x,y) = min{x,1-x,y,1-y}, 归一化到 [0,0.5].
        """
        xs = torch.linspace(0.0, 1.0, H, device=device, dtype=torch.float32)
        ys = torch.linspace(0.0, 1.0, W, device=device, dtype=torch.float32)
        X, Y = torch.meshgrid(xs, ys, indexing="ij")
        d = torch.minimum(
            torch.minimum(X, 1.0 - X),
            torch.minimum(Y, 1.0 - Y),
        )  # (H,W)
        return d, X, Y

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B,C,H,W), cond: (B,cond_dim) or None
        B, C, H, W = x.shape
        device = x.device

        d, X, Y = self.boundary_distance(H, W, device)  # (H,W), (H,W), (H,W)
        # 边界图册的软权重: 距离越近权重越大
        # 使用指数衰减, rim_ratio 控制边界层厚度
        tau = max(self.rim_ratio, 1e-3)
        w_rim = torch.exp(-d / tau)  # (H,W)
        w_rim = w_rim / (w_rim.max() + 1e-6)
        w_core = 1.0 - w_rim

        w_rim = w_rim.view(1, 1, H, W)
        w_core = w_core.view(1, 1, H, W)

        x_rim_in = x * w_rim

        # core 路径: 多图册版本 (当前实现为 2 个子图册, 按 X 方向软划分)
        # 使用平滑的 sigmoid 在 X=0.5 处划分左右区域
        if self.n_core_charts > 1:
            sharp = 20.0
            gate_right = torch.sigmoid((X - 0.5) * sharp)  # (H,W), 左~0, 右~1
            gate_left = 1.0 - gate_right
            gate_left = gate_left.view(1, 1, H, W)
            gate_right = gate_right.view(1, 1, H, W)

            y_core_total = 0.0
            # 左侧 core 图册
            x_core_left = x * (w_core * gate_left)
            y_core_left = self.core_blocks[0](x_core_left, cond)
            y_core_total = y_core_total + y_core_left
            # 右侧 core 图册
            if self.n_core_charts > 1:
                x_core_right = x * (w_core * gate_right)
                y_core_right = self.core_blocks[1](x_core_right, cond)
                y_core_total = y_core_total + y_core_right
            y_core = y_core_total
        else:
            x_core_in = x * w_core
            y_core = self.core_blocks[0](x_core_in, cond)

        y_rim = self.rim_block(x_rim_in, cond)

        # 图册融合: 在空间上按权重拼接两个谱算子
        y = w_core * y_core + w_rim * y_rim
        return x + y


class CFT_AO_2D_Atlas(nn.Module):
    """
    CFT-Analytic Operator (2D) with boundary atlas:
    - 前若干层: 单图册 CFTAOBlock2D
    - 最后一层: CFTAtlasBlock2D (core+rim 双图册)
    """

    def __init__(
        self,
        modes1: int,
        modes2: int,
        width: int = 32,
        in_channels: int = 1,
        out_channels: int = 1,
        n_layers: int = 3,
        L_segments: int = 4,
        M_cheb: int = 4,
        L_segments_boundary: int = 6,
        M_cheb_boundary: int = 6,
        cft_res: int = 0,
        use_local: bool = True,
        rim_ratio: float = 0.15,
        cond_dim: int = 2,
        inner_iters: int = 2,
        n_bands: int = 3,
        n_sym_bases: int = 2,
    ) -> None:
        super().__init__()
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        self.width = int(width)
        self.padding = 9  # 与 CFT_AO_2D 一致

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_layers = int(n_layers)
        self.cond_dim = int(cond_dim)
        self.inner_iters = max(int(inner_iters), 1)
        self.n_bands = max(int(n_bands), 0)
        self.n_sym_bases = max(int(n_sym_bases), 0)

        # 输入抬升：拼接坐标 (x,y)
        self.fc0 = nn.Linear(self.in_channels + 2, self.width)

        blocks = []
        for li in range(self.n_layers):
            is_last = li == self.n_layers - 1
            if is_last:
                blk = CFTAtlasBlock2D(
                    width=self.width,
                    modes1=self.modes1,
                    modes2=self.modes2,
                    L_segments_core=L_segments,
                    M_cheb_core=M_cheb,
                    L_segments_rim=L_segments_boundary,
                    M_cheb_rim=M_cheb_boundary,
                    cft_res=cft_res,
                    use_local=use_local,
                    rim_ratio=rim_ratio,
                    cond_dim=self.cond_dim,
                    n_iter=self.inner_iters,
                    n_bands=self.n_bands,
                    n_sym_bases=self.n_sym_bases,
                )
            else:
                blk = CFTAOBlock2D(
                    width=self.width,
                    modes1=self.modes1,
                    modes2=self.modes2,
                    L_segments=L_segments,
                    M_cheb=M_cheb,
                    cft_res=cft_res,
                    use_local=use_local,
                    cond_dim=self.cond_dim,
                    n_iter=self.inner_iters,
                    n_bands=self.n_bands,
                    n_sym_bases=self.n_sym_bases,
                )
            blocks.append(blk)
        self.blocks = nn.ModuleList(blocks)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.out_channels)

    def get_grid(self, shape, device):
        B, H, W, _ = shape
        gridx = torch.tensor(np.linspace(0, 1, H), dtype=torch.float32, device=device)
        gridx = gridx.reshape(1, H, 1, 1).repeat(B, 1, W, 1)
        gridy = torch.tensor(np.linspace(0, 1, W), dtype=torch.float32, device=device)
        gridy = gridy.reshape(1, 1, W, 1).repeat(B, H, 1, 1)
        return torch.cat((gridx, gridy), dim=-1)

    @staticmethod
    def get_dirichlet_window(H: int, W: int, device: torch.device) -> torch.Tensor:
        xs = torch.linspace(0.0, 1.0, H, device=device, dtype=torch.float32)
        ys = torch.linspace(0.0, 1.0, W, device=device, dtype=torch.float32)
        X, Y = torch.meshgrid(xs, ys, indexing="ij")
        w = (X * (1.0 - X)) * (Y * (1.0 - Y))
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,H,W,C_in)
        B, H, W, Cin = x.shape
        device = x.device

        # 几何条件编码：
        # - 默认 (cond_dim=2): 使用第一个通道的 mean/std（与旧版保持兼容）
        # - 若 cond_dim=4 且输入通道数≥4: 使用几何 mask 与 Dirichlet 边界的 mean/std 作为条件，
        #   用于在 multi-obstacle 等几何问题中显式感知几何与边界强度。
        if self.cond_dim == 4 and Cin >= 4:
            geom_chan = x[..., 2]
            bc_chan = x[..., 3]
            mean_geom = geom_chan.mean(dim=(1, 2), keepdim=False)
            std_geom = geom_chan.std(dim=(1, 2), unbiased=False)
            mean_bc = bc_chan.mean(dim=(1, 2), keepdim=False)
            std_bc = bc_chan.std(dim=(1, 2), unbiased=False)
            cond = torch.stack((mean_geom, std_geom, mean_bc, std_bc), dim=-1)  # (B,4)
        else:
            a_field = x[..., 0]
            mean_a = a_field.mean(dim=(1, 2), keepdim=False)
            std_a = a_field.std(dim=(1, 2), unbiased=False)
            cond = torch.stack((mean_a, std_a), dim=-1)  # (B,2)
            if self.cond_dim > 2:
                pad = torch.zeros(B, self.cond_dim - 2, device=device, dtype=cond.dtype)
                cond = torch.cat([cond, pad], dim=-1)

        grid = self.get_grid(x.shape, device)
        x = torch.cat((x, grid), dim=-1)  # (B,H,W,Cin+2)
        x = self.fc0(x)                   # (B,H,W,width)
        x = x.permute(0, 3, 1, 2)        # (B,width,H,W)

        # padding for non-periodic boundary
        x = F.pad(x, [0, self.padding, 0, self.padding])

        for blk in self.blocks:
            x = blk(x, cond)

        # remove padding
        x = x[:, :, :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)  # (B,H,W,width)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)            # (B,H,W,Cout)

        # Dirichlet 边界先验
        w = self.get_dirichlet_window(H, W, device)
        x = x * w.view(1, H, W, 1)
        return x


