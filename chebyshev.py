"""
This module provides a high-performance, vectorized implementation of the
Chebyshev Fourier Transform (CFT) and its inverse, optimized for PyTorch
and `torch.compile`.

The core `vectorized_batched_cft` function is implemented using a
real-decomposed approach, avoiding complex number operations in the
performance-critical `einsum` operations to maximize compatibility with
JIT compilers like Triton.
"""
import torch

def vectorized_batched_cft(signals, t_coords, f_points, L_segments, M_cheb, is_inverse=False):
    """
    Computes the batch CFT or its inverse using a fully vectorized, real-decomposed method.

    This function serves as the high-performance core for all CFT operations.

    Args:
        signals (torch.Tensor): Real-valued input signals.
                                Shape: [batch, channels, n_samples].
        t_coords (torch.Tensor): Time coordinates for the signals. Shape: [n_samples].
        f_points (torch.Tensor): Frequency points for the transform. Shape: [n_freqs].
        L_segments (int): Number of segments for the integration approximation.
        M_cheb (int): Number of Chebyshev nodes per segment.
        is_inverse (bool): If True, computes the inverse CFT.

    Returns:
        torch.Tensor: The complex-valued transform coefficients.
                      Shape: [batch, channels, n_freqs].
    """
    if signals.is_complex():
        raise NotImplementedError("This engine currently only supports real-valued input signals.")

    device = signals.device
    batch_size, in_channels, n_samples = signals.shape
    n_freqs = f_points.shape[0]

    # Precompute constants
    segment_len = 1.0 / L_segments
    segment_starts = torch.linspace(0, 1, L_segments + 1, device=device)[:-1]

    k = torch.arange(M_cheb, device=device)
    cheb_nodes_ref = -torch.cos((2 * k + 1) * torch.pi / (2 * M_cheb))  # on [-1, 1]
    T_k_at_nodes = torch.cos(k.unsqueeze(1) * torch.acos(cheb_nodes_ref.unsqueeze(0)))  # T_k(x_m)

    # --- Real-Decomposed Quadrature Weight Calculation ---
    # This is the core of the JIT-friendly optimization.
    freq_factor = 2 * torch.pi * (segment_len / 2)
    w_prime = f_points.unsqueeze(0) * freq_factor

    cheb_nodes_grid, w_prime_grid = torch.meshgrid(cheb_nodes_ref, w_prime.squeeze(0), indexing='ij')
    angle_quad = cheb_nodes_grid * w_prime_grid
    
    # exp(-j*w'*x) for forward, exp(j*w'*x) for inverse
    sign = -1.0 if not is_inverse else 1.0
    exp_term_real = torch.cos(angle_quad)
    exp_term_imag = sign * torch.sin(angle_quad)
    
    # Integral of T_k(x) * exp(+/-j*w'*x)
    quad_weights_real = torch.einsum("km,mf->kf", T_k_at_nodes, exp_term_real) * (segment_len / 2)
    quad_weights_imag = torch.einsum("km,mf->kf", T_k_at_nodes, exp_term_imag) * (segment_len / 2)
    
    # Initialize real and imaginary parts of the final spectrum
    total_spectrum_real = torch.zeros(batch_size, in_channels, n_freqs, device=device, dtype=torch.float32)
    total_spectrum_imag = torch.zeros(batch_size, in_channels, n_freqs, device=device, dtype=torch.float32)

    for i in range(L_segments):
        a = segment_starts[i]
        t_segment = a + (segment_len / 2) * (cheb_nodes_ref + 1)

        # --- Vectorized Linear Interpolation (on real signal) ---
        right_indices = torch.searchsorted(t_coords, t_segment).clamp(max=n_samples - 1)
        left_indices = (right_indices - 1).clamp(min=0)
        t_left, t_right = t_coords[left_indices], t_coords[right_indices]
        
        denom = t_right - t_left
        denom[denom < 1e-8] = 1.0 # Avoid division by zero
        
        w_right = (t_segment - t_left) / denom
        w_left = 1.0 - w_right
        
        signal_segments = w_left * signals[..., left_indices] + w_right * signals[..., right_indices]

        # --- Apply decomposed quadrature weights to real signal ---
        spectrum_segment_real = torch.einsum("bcm,mf->bcf", signal_segments, quad_weights_real)
        spectrum_segment_imag = torch.einsum("bcm,mf->bcf", signal_segments, quad_weights_imag)

        # --- Decompose phase shift and add to total ---
        # exp(-/+ 2j*pi*f*a)
        angle_shift = sign * 2 * torch.pi * f_points * a
        exp_shift_real = torch.cos(angle_shift)
        exp_shift_imag = torch.sin(angle_shift) # sin is odd, sign is already in angle_shift

        # Complex multiplication in real-decomposed form: (A+iB)*(C+iD) = (AC-BD) + i(AD+BC)
        total_spectrum_real += (spectrum_segment_real * exp_shift_real) - (spectrum_segment_imag * exp_shift_imag)
        total_spectrum_imag += (spectrum_segment_real * exp_shift_imag) + (spectrum_segment_imag * exp_shift_real)

    return torch.complex(total_spectrum_real, total_spectrum_imag)

# `transform_nd` and `_integral_core` are now deprecated and replaced by the single,
# optimized `vectorized_batched_cft` function. They are removed to avoid confusion. 