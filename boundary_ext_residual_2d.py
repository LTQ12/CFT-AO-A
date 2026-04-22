import torch
import torch.nn as nn
import torch.nn.functional as F


def _coons_patch_extension_from_bc(bc_raw: torch.Tensor) -> torch.Tensor:
    """
    Coons patch / transfinite interpolation extension from outer boundary values.

    bc_raw: (B, N, N, 1) with Dirichlet values on boundary nodes (interior may be 0).
    Returns:
        E_raw: (B, N, N, 1) that matches bc_raw on the four outer edges.
    """
    if bc_raw.ndim != 4 or bc_raw.shape[-1] != 1:
        raise ValueError(f"bc_raw must have shape (B,N,N,1), got {tuple(bc_raw.shape)}")

    B, N, _, _ = bc_raw.shape
    device = bc_raw.device
    dtype = bc_raw.dtype

    # boundary traces
    gL = bc_raw[:, 0, :, 0]      # (B, N)
    gR = bc_raw[:, -1, :, 0]     # (B, N)
    gB = bc_raw[:, :, 0, 0]      # (B, N)
    gT = bc_raw[:, :, -1, 0]     # (B, N)

    # corners
    c00 = bc_raw[:, 0, 0, 0]     # (B,)
    c01 = bc_raw[:, 0, -1, 0]
    c10 = bc_raw[:, -1, 0, 0]
    c11 = bc_raw[:, -1, -1, 0]

    xs = torch.linspace(0.0, 1.0, N, device=device, dtype=dtype).view(1, N, 1)  # (1,N,1)
    ys = torch.linspace(0.0, 1.0, N, device=device, dtype=dtype).view(1, 1, N)  # (1,1,N)

    # broadcast traces to (B,N,N)
    gL_y = gL[:, None, :]  # (B,1,N)
    gR_y = gR[:, None, :]
    gB_x = gB[:, :, None]  # (B,N,1)
    gT_x = gT[:, :, None]

    # basic transfinite blend
    E = (1.0 - xs) * gL_y + xs * gR_y + (1.0 - ys) * gB_x + ys * gT_x  # (B,N,N)

    # corner correction (avoid double counting)
    corr = (
        (1.0 - xs) * (1.0 - ys) * c00[:, None, None]
        + (1.0 - xs) * ys * c01[:, None, None]
        + xs * (1.0 - ys) * c10[:, None, None]
        + xs * ys * c11[:, None, None]
    )
    E = E - corr
    return E[..., None]  # (B,N,N,1)


def _harmonic_extension_jacobi(
    geom_raw: torch.Tensor,
    bc_raw: torch.Tensor,
    *,
    n_iter: int = 50,
) -> torch.Tensor:
    """
    Batched Jacobi harmonic extension for Laplace equation with Dirichlet BC:
      - Outer boundary: bc_raw values
      - Obstacles (geom_raw==1): Dirichlet 0 (by default in dataset)

    This is a cheap physics-inspired extension E[g] that is much closer to the
    diffusion solution than pure linear interpolation.

    geom_raw: (B,N,N,1) in {0,1}
    bc_raw  : (B,N,N,1)
    Returns:
      E_raw: (B,N,N,1)
    """
    if geom_raw.shape != bc_raw.shape:
        raise ValueError(f"geom_raw and bc_raw must have same shape, got {geom_raw.shape} vs {bc_raw.shape}")
    if geom_raw.ndim != 4 or geom_raw.shape[-1] != 1:
        raise ValueError(f"geom_raw must have shape (B,N,N,1), got {tuple(geom_raw.shape)}")

    B, N, _, _ = geom_raw.shape
    device = geom_raw.device
    dtype = geom_raw.dtype

    # Dirichlet nodes: outer boundary OR obstacle
    bdry = torch.zeros((B, N, N, 1), device=device, dtype=torch.bool)
    bdry[:, 0, :, :] = True
    bdry[:, -1, :, :] = True
    bdry[:, :, 0, :] = True
    bdry[:, :, -1, :] = True
    obs = geom_raw > 0.5
    dir_mask = bdry | obs
    fluid_mask = ~dir_mask

    # initialize with bc on Dirichlet nodes, zeros elsewhere
    u = torch.zeros((B, N, N, 1), device=device, dtype=dtype)
    u = torch.where(dir_mask, bc_raw, u)

    # Jacobi iterations on fluid nodes, 5-point stencil
    for _ in range(int(n_iter)):
        u_old = u
        # neighbors (pad with replication; boundary will be overwritten anyway)
        up = torch.roll(u_old, shifts=-1, dims=1)
        down = torch.roll(u_old, shifts=1, dims=1)
        right = torch.roll(u_old, shifts=-1, dims=2)
        left = torch.roll(u_old, shifts=1, dims=2)
        u_new = 0.25 * (up + down + left + right)
        u = torch.where(fluid_mask, u_new, u_old)
        # enforce Dirichlet exactly
        u = torch.where(dir_mask, bc_raw, u)

    return u


def _poisson_extension_jacobi(
    geom_raw: torch.Tensor,
    bc_raw: torch.Tensor,
    src_raw: torch.Tensor,
    *,
    n_iter: int = 50,
) -> torch.Tensor:
    """
    Batched Jacobi Poisson extension for:

        Δu = src   in fluid region
        u = g      on outer boundary
        u = 0      inside obstacles (geom_raw==1)

    This generalizes harmonic extension (src=0). It is a "learned lifting" hook:
    if src_raw is produced by a small network, E[g] can absorb more of the low-frequency
    interior structure, making the residual operator easier to learn.

    geom_raw: (B,N,N,1) in {0,1}
    bc_raw  : (B,N,N,1) outer boundary Dirichlet values (interior may be 0)
    src_raw : (B,N,N,1) source term on the grid (will be ignored on Dirichlet nodes)
    """
    if geom_raw.shape != bc_raw.shape or geom_raw.shape != src_raw.shape:
        raise ValueError(
            f"geom_raw/bc_raw/src_raw must have same shape, got {geom_raw.shape} vs {bc_raw.shape} vs {src_raw.shape}"
        )
    if geom_raw.ndim != 4 or geom_raw.shape[-1] != 1:
        raise ValueError(f"geom_raw must have shape (B,N,N,1), got {tuple(geom_raw.shape)}")

    B, N, _, _ = geom_raw.shape
    device = geom_raw.device
    dtype = geom_raw.dtype

    # Dirichlet nodes: outer boundary OR obstacle
    bdry = torch.zeros((B, N, N, 1), device=device, dtype=torch.bool)
    bdry[:, 0, :, :] = True
    bdry[:, -1, :, :] = True
    bdry[:, :, 0, :] = True
    bdry[:, :, -1, :] = True
    obs = geom_raw > 0.5
    dir_mask = bdry | obs
    fluid_mask = ~dir_mask

    # initialize u with Dirichlet values (outer boundary + obstacles), zero elsewhere
    u = torch.zeros((B, N, N, 1), device=device, dtype=dtype)
    u = torch.where(dir_mask, bc_raw, u)

    # grid spacing (domain is (0,1)^2)
    dx = 1.0 / float(max(N - 1, 1))
    dx2 = dx * dx

    # src should not act on Dirichlet nodes
    src = torch.where(fluid_mask, src_raw, torch.zeros_like(src_raw))

    for _ in range(int(n_iter)):
        u_old = u
        up = torch.roll(u_old, shifts=-1, dims=1)
        down = torch.roll(u_old, shifts=1, dims=1)
        right = torch.roll(u_old, shifts=-1, dims=2)
        left = torch.roll(u_old, shifts=1, dims=2)
        # Jacobi update for Δu = src: u = (neighbors - dx^2 * src)/4
        u_new = 0.25 * (up + down + left + right - dx2 * src)
        u = torch.where(fluid_mask, u_new, u_old)
        # enforce Dirichlet exactly
        u = torch.where(dir_mask, bc_raw, u)

    return u


class _PoissonSrcNet2D(nn.Module):
    """
    Very thin source predictor s(x) for Poisson lifting, used only when ext_method == 'poisson_learned'.
    Input is NHWC; internally converts to NCHW convs and returns NHWC (B,N,N,1).
    """

    def __init__(self, in_ch: int, hidden: int = 32):
        super().__init__()
        self.in_ch = int(in_ch)
        self.hidden = int(hidden)
        self.conv1 = nn.Conv2d(self.in_ch, self.hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.hidden, self.hidden, kernel_size=3, padding=1)
        self.proj = nn.Conv2d(self.hidden, 1, kernel_size=1)

        torch.nn.init.kaiming_uniform_(self.conv1.weight, a=0.0, nonlinearity="relu")
        torch.nn.init.zeros_(self.conv1.bias)
        torch.nn.init.kaiming_uniform_(self.conv2.weight, a=0.0, nonlinearity="relu")
        torch.nn.init.zeros_(self.conv2.bias)
        # Important: avoid "stuck at zero" source by using a tiny non-zero init.
        torch.nn.init.normal_(self.proj.weight, mean=0.0, std=1e-3)
        torch.nn.init.zeros_(self.proj.bias)

    def forward(self, x_nhwc: torch.Tensor) -> torch.Tensor:
        x = x_nhwc.permute(0, 3, 1, 2)  # NCHW
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = self.proj(x)  # (B,1,H,W)
        return x.permute(0, 2, 3, 1)  # NHWC


class ResidualOnDirichletExtension2D(nn.Module):
    """
    A-wrap: enforce Dirichlet boundary by construction, learn only a residual:

        u_enc(x) = E_enc[g](x) + w(x) * r_enc_theta(x)

    - E is built from the *raw* bc channel via Coons patch extension (outer boundary),
      then masked by (1-geom) to enforce obstacle interior Dirichlet=0.
    - w(x) vanishes on outer boundary and obstacle nodes, so the residual cannot
      alter Dirichlet nodes.

    Input convention:
      x_mix: (B,N,N, C_norm + 2) = [x_norm, geom_raw, bc_raw]
        - x_norm: normalized input for the backbone (e.g., [geom_norm, bc_norm])
        - geom_raw: raw obstacle mask (1=obstacle, 0=fluid) as last-2 channel
        - bc_raw  : raw Dirichlet values as last-1 channel

    Output:
      u_enc: (B,N,N,1) encoded with (y_mean, y_std) to match UnitGaussianNormalizer.encode(u).
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        y_mean: torch.Tensor,
        y_std: torch.Tensor,
        in_channels_norm: int,
        delta: float = 0.05,
        res_scale_init: float = 0.0,
        res_scale_max: float = 0.25,
        ext_method: str = "coons",
        ext_iters: int = 50,
        poisson_src_hidden: int = 32,
        poisson_src_scale_max: float = 1.0,
        residual_clip: float = 0.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.backbone = backbone
        self.in_channels_norm = int(in_channels_norm)
        self.delta = float(delta)
        # IMPORTANT: must match UnitGaussianNormalizer.encode() behavior, which clamps std by ~1e-5
        self.eps = float(eps)

        # store output normalizer stats as buffers so they move with .to(device)
        self.register_buffer("y_mean", y_mean.detach().clone())
        self.register_buffer("y_std", y_std.detach().clone())

        # residual scaling gate: bounded non-negative in [0, res_scale_max]
        self.res_scale_max = float(res_scale_max)
        init = float(res_scale_init)
        init = max(0.0, min(init, self.res_scale_max))
        # convert init scale -> logit in (0,1)
        p = init / max(self.res_scale_max, 1e-12)
        p = float(min(max(p, 1e-6), 1.0 - 1e-6))
        logit = float(torch.log(torch.tensor(p / (1.0 - p))))
        self.res_scale_logit = nn.Parameter(torch.tensor(logit))

        self.ext_method = str(ext_method)
        self.ext_iters = int(ext_iters)
        self.residual_clip = float(residual_clip)

        # optional learned source term for Poisson lifting
        self.poisson_src_hidden = int(poisson_src_hidden)
        self.poisson_src_scale_max = float(poisson_src_scale_max)
        if self.ext_method == "poisson_learned":
            # input uses x_norm + geom_raw + bc_raw (NHWC)
            src_in = self.in_channels_norm + 2
            self.src_net = _PoissonSrcNet2D(src_in, hidden=self.poisson_src_hidden)
            # bounded non-negative scale to keep src magnitude controlled
            self.src_scale_logit = nn.Parameter(torch.tensor(0.0))
        else:
            self.src_net = None
            self.src_scale_logit = None

        # last-batch diagnostics (useful for debugging poisson_learned)
        self.register_buffer("_last_src_scale", torch.tensor(0.0), persistent=False)
        self.register_buffer("_last_src_rms", torch.tensor(0.0), persistent=False)

        # cache for boundary-distance window
        self.register_buffer("_cached_N", torch.tensor(-1, dtype=torch.long), persistent=False)
        self.register_buffer("_cached_w_outer", torch.empty(0), persistent=False)

    def _ensure_w_outer(self, N: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if int(self._cached_N.item()) == int(N) and self._cached_w_outer.numel() > 0:
            return self._cached_w_outer

        xs = torch.linspace(0.0, 1.0, N, device=device, dtype=dtype).view(N, 1)  # (N,1)
        ys = torch.linspace(0.0, 1.0, N, device=device, dtype=dtype).view(1, N)  # (1,N)
        d_outer = torch.minimum(torch.minimum(xs, 1.0 - xs), torch.minimum(ys, 1.0 - ys))  # (N,N)
        w_outer = torch.clamp(d_outer / max(self.delta, 1e-6), 0.0, 1.0)[None, :, :, None]  # (1,N,N,1)

        self._cached_N = torch.tensor(int(N), device=device, dtype=torch.long)
        self._cached_w_outer = w_outer
        return w_outer

    def _encode_y(self, y_raw: torch.Tensor) -> torch.Tensor:
        return (y_raw - self.y_mean) / self.y_std.clamp(min=self.eps)

    def get_res_scale(self) -> torch.Tensor:
        return torch.sigmoid(self.res_scale_logit) * self.res_scale_max

    def build_extension_raw(self, x_mix: torch.Tensor) -> torch.Tensor:
        """
        Build the raw extension E_raw (NHWC, 1 channel) from the mixed input.
        This exposes the same E[g] used in forward(), so training scripts can compute:
          - E-only MSE
          - residual energy regularization based on (out - E_enc)
        in a way that stays consistent across ext_method variants (including poisson_learned).
        """
        if x_mix.ndim != 4:
            raise ValueError(f"x_mix must have shape (B,N,N,C), got {tuple(x_mix.shape)}")
        if x_mix.shape[-1] < self.in_channels_norm + 2:
            raise ValueError(
                f"x_mix last dim must be >= in_channels_norm+2 = {self.in_channels_norm+2}, got {x_mix.shape[-1]}"
            )
        x_norm = x_mix[..., : self.in_channels_norm]
        geom_raw = x_mix[..., self.in_channels_norm : self.in_channels_norm + 1]
        bc_raw = x_mix[..., self.in_channels_norm + 1 : self.in_channels_norm + 2]

        if self.ext_method == "zero":
            E_raw = torch.zeros_like(bc_raw)
        elif self.ext_method == "coons":
            E_raw = _coons_patch_extension_from_bc(bc_raw)
        elif self.ext_method == "harmonic":
            E_raw = _harmonic_extension_jacobi(geom_raw, bc_raw, n_iter=self.ext_iters)
        elif self.ext_method == "poisson":
            src = torch.zeros_like(bc_raw)
            E_raw = _poisson_extension_jacobi(geom_raw, bc_raw, src, n_iter=self.ext_iters)
        elif self.ext_method == "poisson_learned":
            if self.src_net is None or self.src_scale_logit is None:
                raise RuntimeError("poisson_learned selected but src_net is not initialized.")
            src_in = torch.cat([x_norm, geom_raw, bc_raw], dim=-1)
            src = self.src_net(src_in)
            src_scale = torch.sigmoid(self.src_scale_logit) * self.poisson_src_scale_max
            src = src_scale * src
            # diagnostics (detach to avoid holding graph)
            self._last_src_scale = src_scale.detach()
            self._last_src_rms = torch.sqrt(torch.mean(src.detach() ** 2) + 1e-12)
            E_raw = _poisson_extension_jacobi(geom_raw, bc_raw, src, n_iter=self.ext_iters)
        else:
            raise ValueError(
                f"Unknown ext_method={self.ext_method}. Use 'zero', 'coons', 'harmonic', 'poisson', or 'poisson_learned'."
            )

        # enforce obstacle Dirichlet values from bc_raw on obstacle nodes
        # (backward compatible: diffusion datasets use bc_raw=0 on obstacles)
        obs = geom_raw > 0.5
        E_raw = torch.where(obs, bc_raw, E_raw)
        return E_raw

    def get_last_src_scale(self) -> torch.Tensor:
        return self._last_src_scale

    def get_last_src_rms(self) -> torch.Tensor:
        return self._last_src_rms

    def forward(self, x_mix: torch.Tensor) -> torch.Tensor:
        if x_mix.ndim != 4:
            raise ValueError(f"x_mix must have shape (B,N,N,C), got {tuple(x_mix.shape)}")
        if x_mix.shape[-1] < self.in_channels_norm + 2:
            raise ValueError(
                f"x_mix last dim must be >= in_channels_norm+2 = {self.in_channels_norm+2}, "
                f"got {x_mix.shape[-1]}"
            )

        x_norm = x_mix[..., : self.in_channels_norm]
        geom_raw = x_mix[..., self.in_channels_norm : self.in_channels_norm + 1]
        bc_raw = x_mix[..., self.in_channels_norm + 1 : self.in_channels_norm + 2]

        # build extension (raw) using the same logic exposed by build_extension_raw()
        E_raw = self.build_extension_raw(x_mix)
        E_enc = self._encode_y(E_raw)

        # boundary window, enforce residual=0 on Dirichlet nodes (outer boundary + obstacles)
        B, N, _, _ = x_mix.shape
        w_outer = self._ensure_w_outer(N, x_mix.device, x_mix.dtype)  # (1,N,N,1)
        w = w_outer * (1.0 - geom_raw)

        # residual in encoded space
        r_enc = self.backbone(x_norm)
        # optional safety clip to prevent rare blow-ups (useful for high-capacity backbones)
        if self.residual_clip > 0:
            c = float(self.residual_clip)
            r_enc = torch.tanh(r_enc / c) * c
        res_scale = self.get_res_scale()
        return E_enc + w * res_scale * r_enc


class _GammaFieldNet2D(nn.Module):
    """
    Low-res gating field gamma(x) for multiscale residual fusion.
    Input is NHWC; internally does NCHW convs; output NHWC (B,H,W,1).
    """

    def __init__(self, in_ch: int, hidden: int = 32):
        super().__init__()
        self.in_ch = int(in_ch)
        self.hidden = int(hidden)
        self.conv1 = nn.Conv2d(self.in_ch, self.hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.hidden, self.hidden, kernel_size=3, padding=1)
        self.proj = nn.Conv2d(self.hidden, 1, kernel_size=1)

        torch.nn.init.kaiming_uniform_(self.conv1.weight, a=0.0, nonlinearity="relu")
        torch.nn.init.zeros_(self.conv1.bias)
        torch.nn.init.kaiming_uniform_(self.conv2.weight, a=0.0, nonlinearity="relu")
        torch.nn.init.zeros_(self.conv2.bias)
        torch.nn.init.zeros_(self.proj.weight)
        torch.nn.init.zeros_(self.proj.bias)

    def forward(self, x_nhwc: torch.Tensor) -> torch.Tensor:
        x = x_nhwc.permute(0, 3, 1, 2)  # NCHW
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = self.proj(x)
        return x.permute(0, 2, 3, 1)  # NHWC


class ResidualOnDirichletExtension2D_MultiScale(ResidualOnDirichletExtension2D):
    """
    Multiscale A-wrap (principle-level upgrade):

        u_enc = E_enc[g] + w * ( s_f * r_fine + s_c * gamma(x) * up(r_coarse) )

    where:
    - coarse branch is evaluated on a downsampled grid, improving long-range propagation;
    - gamma(x) is a *field* (not scalar), predicted at low-res then upsampled;
    - coarse contribution is masked away near Dirichlet nodes by an interior indicator (w > eps).
    """

    def __init__(
        self,
        fine_backbone: nn.Module,
        coarse_backbone: nn.Module,
        *,
        y_mean: torch.Tensor,
        y_std: torch.Tensor,
        in_channels_norm: int,
        delta: float = 0.05,
        # fine scale gate
        res_scale_init: float = 0.0,
        res_scale_max: float = 0.25,
        # coarse scale gate
        coarse_scale_init: float = 0.0,
        coarse_scale_max: float = 0.25,
        # extension configs (inherited)
        ext_method: str = "coons",
        ext_iters: int = 50,
        poisson_src_hidden: int = 32,
        poisson_src_scale_max: float = 1.0,
        residual_clip: float = 0.0,
        # multiscale configs
        coarse_factor: int = 2,
        gamma_hidden: int = 32,
        gamma_max: float = 1.0,
        coarse_interior_eps: float = 0.5,
        eps: float = 1e-12,
    ):
        super().__init__(
            fine_backbone,
            y_mean=y_mean,
            y_std=y_std,
            in_channels_norm=in_channels_norm,
            delta=delta,
            res_scale_init=res_scale_init,
            res_scale_max=res_scale_max,
            ext_method=ext_method,
            ext_iters=ext_iters,
            poisson_src_hidden=poisson_src_hidden,
            poisson_src_scale_max=poisson_src_scale_max,
            residual_clip=residual_clip,
            eps=eps,
        )

        self.coarse_backbone = coarse_backbone
        self.coarse_factor = int(max(1, coarse_factor))

        # separate bounded gate for coarse branch
        self.coarse_scale_max = float(coarse_scale_max)
        init = float(coarse_scale_init)
        init = max(0.0, min(init, self.coarse_scale_max))
        p = init / max(self.coarse_scale_max, 1e-12)
        p = float(min(max(p, 1e-6), 1.0 - 1e-6))
        logit = float(torch.log(torch.tensor(p / (1.0 - p))))
        self.coarse_scale_logit = nn.Parameter(torch.tensor(logit))

        # gamma(x): low-res gating field, upsampled to full res
        self.gamma_hidden = int(gamma_hidden)
        self.gamma_max = float(gamma_max)
        gamma_in = int(in_channels_norm) + 1  # [x_norm_lr, geom_lr]
        self.gamma_net = _GammaFieldNet2D(gamma_in, hidden=self.gamma_hidden)

        # only let coarse act on interior: mask = 1[w > eps]
        self.coarse_interior_eps = float(coarse_interior_eps)

        # diagnostics
        self.register_buffer("_last_gamma_mean", torch.tensor(0.0), persistent=False)

    def get_coarse_scale(self) -> torch.Tensor:
        return torch.sigmoid(self.coarse_scale_logit) * self.coarse_scale_max

    def get_last_gamma_mean(self) -> torch.Tensor:
        return self._last_gamma_mean

    def _downsample_nhwc(self, x: torch.Tensor, factor: int, *, mode: str) -> torch.Tensor:
        # NHWC -> NCHW -> downsample -> NHWC
        x_nchw = x.permute(0, 3, 1, 2)
        if factor <= 1:
            y = x_nchw
        else:
            y = F.interpolate(x_nchw, scale_factor=1.0 / float(factor), mode=mode)
        return y.permute(0, 2, 3, 1)

    def forward(self, x_mix: torch.Tensor) -> torch.Tensor:
        if x_mix.ndim != 4:
            raise ValueError(f"x_mix must have shape (B,N,N,C), got {tuple(x_mix.shape)}")
        if x_mix.shape[-1] < self.in_channels_norm + 2:
            raise ValueError(
                f"x_mix last dim must be >= in_channels_norm+2 = {self.in_channels_norm+2}, got {x_mix.shape[-1]}"
            )

        x_norm = x_mix[..., : self.in_channels_norm]  # (B,N,N,Cn)
        geom_raw = x_mix[..., self.in_channels_norm : self.in_channels_norm + 1]  # (B,N,N,1)

        # extension + window (same as base)
        E_raw = self.build_extension_raw(x_mix)
        E_enc = self._encode_y(E_raw)

        B, N, _, _ = x_mix.shape
        w_outer = self._ensure_w_outer(N, x_mix.device, x_mix.dtype)  # (1,N,N,1)
        w = w_outer * (1.0 - geom_raw)

        # fine residual (encoded)
        r_fine = self.backbone(x_norm)
        if self.residual_clip > 0:
            c = float(self.residual_clip)
            r_fine = torch.tanh(r_fine / c) * c

        # coarse branch on downsampled grid
        f = self.coarse_factor
        x_norm_lr = self._downsample_nhwc(x_norm, f, mode="area")
        # obstacle mask should be downsampled with max pooling to preserve obstacles
        geom_lr = self._downsample_nhwc(geom_raw, f, mode="nearest")
        r_coarse_lr = self.coarse_backbone(x_norm_lr)  # encoded, (B,Nc,Nc,1)

        # upsample coarse residual to full res
        r_coarse_up = F.interpolate(
            r_coarse_lr.permute(0, 3, 1, 2),
            size=(N, N),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)

        # gamma field predicted at low-res from [x_norm_lr, geom_lr], then upsample
        gamma_lr_logits = self.gamma_net(torch.cat([x_norm_lr, geom_lr], dim=-1))  # (B,Nc,Nc,1)
        gamma_up = F.interpolate(
            gamma_lr_logits.permute(0, 3, 1, 2),
            size=(N, N),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        gamma = torch.sigmoid(gamma_up) * self.gamma_max  # (B,N,N,1)
        self._last_gamma_mean = gamma.detach().mean()

        # coarse only acts on interior (avoid boundary-adjacent pollution)
        interior = (w > self.coarse_interior_eps).to(w.dtype)

        s_f = self.get_res_scale()
        s_c = self.get_coarse_scale()
        return E_enc + w * (s_f * r_fine + interior * s_c * gamma * r_coarse_up)

