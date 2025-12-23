
import torch
import triton
import triton.language as tl

# LUTs
GRADIENT_LUT_4BIT = [
    0.0143, 0.0164, 0.0699, 0.1923, 0.3752, 0.5000, 
    0.6248, 0.8130, 0.9647, 1.0532, 1.0817, 1.0283, 
    1.0089, 1.0012, 1.0001, 1.0000
]
X_BINS_4BIT = [-4.0, -2.5, -1.5, -0.8, -0.3, 0.0, 0.3, 0.7, 1.2, 1.8, 2.5, 3.5, 4.5, 6.0, 8.0, 10.0]

@triton.jit
def _pack_2bit_silu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE * 4
    offsets = block_start + tl.arange(0, BLOCK_SIZE) * 4
    packed_val = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    for i in range(4):
        curr_idx = offsets + i
        mask = curr_idx < n_elements
        x = tl.load(x_ptr + curr_idx, mask=mask, other=0.0)
        bit_0 = (x >= 0).to(tl.int32)
        bit_1 = (tl.abs(x) >= 2.0).to(tl.int32)
        two_bits = (bit_1 << 1) | bit_0
        packed_val = packed_val | (two_bits << (i * 2))
    out_idx = block_start // 4 + tl.arange(0, BLOCK_SIZE)
    out_mask = out_idx < (n_elements + 3) // 4
    tl.store(out_ptr + out_idx, packed_val.to(tl.int8), mask=out_mask)

@triton.jit
def _unpack_2bit_backward_kernel(grad_output_ptr, packed_activation_ptr, grad_input_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE * 4
    offsets = block_start + tl.arange(0, BLOCK_SIZE) * 4
    for i in range(4):
        curr_idx = offsets + i
        mask = curr_idx < n_elements
        grad_out = tl.load(grad_output_ptr + curr_idx, mask=mask, other=0.0)
        byte_idx = curr_idx // 4
        shift = (curr_idx % 4) * 2
        packed_byte = tl.load(packed_activation_ptr + byte_idx, mask=mask, other=0).to(tl.int32)
        two_bits = (packed_byte >> shift) & 0b11
        grad_coeff = tl.where(two_bits == 0, 0.0, tl.where(two_bits == 1, 0.5, 1.0))
        tl.store(grad_input_ptr + curr_idx, grad_out * grad_coeff, mask=mask)

@triton.jit
def _pack_4bit_silu_kernel(activation_ptr, packed_output_ptr, threshold_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pair_idx = pid * (BLOCK_SIZE // 2) + tl.arange(0, BLOCK_SIZE // 2)
    idx_low = pair_idx * 2
    idx_high = pair_idx * 2 + 1
    mask_low = idx_low < n_elements
    mask_high = idx_high < n_elements
    val_low = tl.load(activation_ptr + idx_low, mask=mask_low, other=-99.0)
    val_high = tl.load(activation_ptr + idx_high, mask=mask_high, other=-99.0)
    nibble_low = tl.zeros([BLOCK_SIZE // 2], dtype=tl.uint8)
    nibble_high = tl.zeros([BLOCK_SIZE // 2], dtype=tl.uint8)
    for i in range(16):
        threshold = tl.load(threshold_ptr + i)
        nibble_low += (val_low >= threshold).to(tl.uint8)
        nibble_high += (val_high >= threshold).to(tl.uint8)
    packed_byte = (tl.minimum(nibble_low, 15)) | (tl.minimum(nibble_high, 15) << 4)
    tl.store(packed_output_ptr + pair_idx, packed_byte, mask=mask_low)

@triton.jit
def _unpack_4bit_backward_kernel(grad_output_ptr, packed_activation_ptr, grad_input_ptr, gradient_lut_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    grad_out = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0)
    packed_byte = tl.load(packed_activation_ptr + (offsets >> 1), mask=mask, other=0)
    nibble = (packed_byte >> tl.where(offsets & 1, 4, 0)) & 0x0F
    grad_coeff = tl.load(gradient_lut_ptr + nibble, mask=mask, other=1.0)
    tl.store(grad_input_ptr + offsets, grad_out * grad_coeff, mask=mask)
