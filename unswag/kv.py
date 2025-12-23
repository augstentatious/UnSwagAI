
import torch
import triton
import triton.language as tl

@triton.jit
def _pack_4bit_kv_kernel(
    x_ptr,          # Input FP16 [N]
    scale_ptr,      # Input FP16 Scales [N / GroupSize]
    out_ptr,        # Output INT8 Packed [N / 2]
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE * 2 + tl.arange(0, BLOCK_SIZE * 2)
    mask = offsets < n_elements
    val = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    scale_idx = offsets // GROUP_SIZE
    scale = tl.load(scale_ptr + scale_idx, mask=mask, other=1.0)
    x_norm = val / (scale + 1e-6)
    x_quant = (x_norm + 1.0) * 7.5
    x_quant = tl.math.floor(x_quant + 0.5) # Corrected rounding
    x_quant = tl.minimum(tl.maximum(x_quant, 0.0), 15.0).to(tl.int8)

    val_low_ptr = x_ptr + (pid * BLOCK_SIZE * 2 + 2 * tl.arange(0, BLOCK_SIZE))
    val_high_ptr = x_ptr + (pid * BLOCK_SIZE * 2 + 2 * tl.arange(0, BLOCK_SIZE) + 1)
    mask_low = (pid * BLOCK_SIZE * 2 + 2 * tl.arange(0, BLOCK_SIZE)) < n_elements
    mask_high = (pid * BLOCK_SIZE * 2 + 2 * tl.arange(0, BLOCK_SIZE) + 1) < n_elements

    val_low = tl.load(val_low_ptr, mask=mask_low, other=0.0)
    val_high = tl.load(val_high_ptr, mask=mask_high, other=0.0)

    scale_low_idx = (pid * BLOCK_SIZE * 2 + 2 * tl.arange(0, BLOCK_SIZE)) // GROUP_SIZE
    scale_high_idx = (pid * BLOCK_SIZE * 2 + 2 * tl.arange(0, BLOCK_SIZE) + 1) // GROUP_SIZE

    scale_low = tl.load(scale_ptr + scale_low_idx, mask=scale_low_idx < (n_elements // GROUP_SIZE), other=1.0)
    scale_high = tl.load(scale_ptr + scale_high_idx, mask=scale_high_idx < (n_elements // GROUP_SIZE), other=1.0)

    q_low = ((val_low / (scale_low + 1e-6)) + 1.0) * 7.5
    q_low = tl.math.floor(q_low + 0.5) # Corrected rounding
    q_low = tl.minimum(tl.maximum(q_low, 0.0), 15.0).to(tl.int8)

    q_high = ((val_high / (scale_high + 1e-6)) + 1.0) * 7.5
    q_high = tl.math.floor(q_high + 0.5) # Corrected rounding
    q_high = tl.minimum(tl.maximum(q_high, 0.0), 15.0).to(tl.int8)

    packed_byte = q_low | (q_high << 4)

    out_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(out_ptr + out_idx, packed_byte, mask=out_idx < (n_elements // 2))

@triton.jit
def _unpack_4bit_kv_kernel(
    packed_ptr,
    scale_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE * 2 + tl.arange(0, BLOCK_SIZE * 2)
    mask = offsets < n_elements
    byte_offsets = offsets // 2
    packed_byte = tl.load(packed_ptr + byte_offsets, mask=mask, other=0)
    shift = (offsets % 2) * 4
    nibble = (packed_byte >> shift) & 0x0F
    scale_idx = offsets // GROUP_SIZE
    scale = tl.load(scale_ptr + scale_idx, mask=mask, other=1.0)
    val = (nibble.to(tl.float32) / 7.5) - 1.0
    val = val * scale
    tl.store(out_ptr + offsets, val.to(tl.float16), mask=mask)

class UnSwagKV:
    @staticmethod
    def pack(x, group_size=32):
        original_shape = x.shape
        n_elements = x.numel()
        x_flat = x.flatten()

        pad_len = (group_size - (n_elements % group_size)) % group_size
        if pad_len > 0:
            x_padded = torch.nn.functional.pad(x_flat, (0, pad_len))
        else:
            x_padded = x_flat

        x_groups = x_padded.view(-1, group_size)
        scales = x_groups.abs().max(dim=1).values
        packed = torch.empty(n_elements // 2, dtype=torch.uint8, device=x.device)
        grid = lambda meta: (triton.cdiv(n_elements // 2, meta['BLOCK_SIZE']),)

        _pack_4bit_kv_kernel[grid](
            x_flat,
            scales,
            packed,
            n_elements,
            BLOCK_SIZE=1024,
            GROUP_SIZE=group_size
        )
        return packed, scales, original_shape

    @staticmethod
    def unpack(packed, scales, original_shape, group_size=32):
        n_elements = packed.numel() * 2
        out = torch.empty(n_elements, dtype=torch.float16, device=packed.device)
        grid = lambda meta: (triton.cdiv(packed.numel(), meta['BLOCK_SIZE']),)
        _unpack_4bit_kv_kernel[grid](
            packed,
            scales,
            out,
            n_elements,
            BLOCK_SIZE=1024,
            GROUP_SIZE=group_size
        )
        return out.view(original_shape)
