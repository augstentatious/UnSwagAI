
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
from .kernels.triton_ops import (
    _pack_2bit_silu_kernel, _unpack_2bit_backward_kernel,
    _pack_4bit_silu_kernel, _unpack_4bit_backward_kernel,
    GRADIENT_LUT_4BIT, X_BINS_4BIT
)

class UnSwagSiLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        n = x.numel()
        out_size = (n + 3) // 4
        packed = torch.empty(out_size, dtype=torch.int8, device=x.device)
        grid = lambda m: (triton.cdiv(n // 4, m['BLOCK_SIZE']),)
        _pack_2bit_silu_kernel[grid](x, packed, n, BLOCK_SIZE=1024)
        ctx.save_for_backward(packed)
        ctx.n = n
        return F.silu(x)

    @staticmethod
    def backward(ctx, grad_out):
        packed, = ctx.saved_tensors
        grad_in = torch.empty_like(grad_out)
        grid = lambda m: (triton.cdiv(ctx.n // 4, m['BLOCK_SIZE']),)
        _unpack_2bit_backward_kernel[grid](grad_out, packed, grad_in, ctx.n, BLOCK_SIZE=1024)
        return grad_in

class UnSwagSiLU4BitFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        n = x.numel()
        packed = torch.empty((n + 1) // 2, dtype=torch.uint8, device=x.device)
        thresh = torch.tensor(X_BINS_4BIT, dtype=torch.float32, device=x.device)
        grid = lambda m: (triton.cdiv(n, m['BLOCK_SIZE']),)
        _pack_4bit_silu_kernel[grid](x.flatten(), packed, thresh, n, BLOCK_SIZE=1024)
        ctx.save_for_backward(packed)
        ctx.shape, ctx.n = x.shape, n
        return F.silu(x)

    @staticmethod
    def backward(ctx, grad_out):
        packed, = ctx.saved_tensors
        grad_in = torch.empty(ctx.n, dtype=torch.float32, device=grad_out.device)
        lut = torch.tensor(GRADIENT_LUT_4BIT, dtype=torch.float32, device=grad_out.device)
        grid = lambda m: (triton.cdiv(ctx.n, m['BLOCK_SIZE']),)
        _unpack_4bit_backward_kernel[grid](grad_out.flatten(), packed, grad_in, lut, ctx.n, BLOCK_SIZE=1024)
        return grad_in.reshape(ctx.shape)

class UnSwagSiLU(nn.Module):
    def __init__(self, mode="4bit"):
        super().__init__()
        self.mode = mode
    def forward(self, x):
        if self.mode == "4bit": return UnSwagSiLU4BitFunction.apply(x)
        return UnSwagSiLUFunction.apply(x)
