
import torch
import triton
import triton.language as tl

@triton.jit
def protocol_c_widely_linear_kernel(
    x_real_ptr, x_imag_ptr, 
    u_bits_ptr, w_bits_ptr, 
    y_real_ptr, y_imag_ptr,
    n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    xr = tl.load(x_real_ptr + offsets, mask=mask)
    xi = tl.load(x_imag_ptr + offsets, mask=mask)
    u_bits = tl.load(u_bits_ptr + offsets, mask=mask)
    w_bits = tl.load(w_bits_ptr + offsets, mask=mask)

    # --- U * x logic (Inline) ---
    # bit[1] >= 2: rotation by i (+i or -i)
    # bit[0] % 2 == 1: sign flip
    u_swapped_r = tl.where(u_bits >= 2, -xi, xr)
    u_swapped_i = tl.where(u_bits >= 2, xr, xi)
    
    ur = tl.where(u_bits % 2 == 1, -u_swapped_r, u_swapped_r)
    ui = tl.where(u_bits % 2 == 1, -u_swapped_i, u_swapped_i)

    # --- W * conj(x) logic (Inline) ---
    # conj(x) = xr - i*xi
    conj_xi = -xi
    
    w_swapped_r = tl.where(w_bits >= 2, -conj_xi, xr)
    w_swapped_i = tl.where(w_bits >= 2, xr, conj_xi)
    
    wr = tl.where(w_bits % 2 == 1, -w_swapped_r, w_swapped_r)
    wi = tl.where(w_bits % 2 == 1, -w_swapped_i, w_swapped_i)

    # --- Final Summation ---
    tl.store(y_real_ptr + offsets, ur + wr, mask=mask)
    tl.store(y_imag_ptr + offsets, ui + wi, mask=mask)
