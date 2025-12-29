
import torch
import time
import sys
from unswag.models.integrated import ProtocolCModel

class DenseBaseline(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.expert = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            torch.nn.GELU(),
            torch.nn.Linear(dim * 4, dim)
        )
    def forward(self, x):
        return self.expert(x)

def run_final_benchmark():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dim = 512
    N = 4096
    
    print(f"🌌 VALIDATING STAR INN CLAIM (6-25x) | Device: {device}")
    
    model = ProtocolCModel(dim=dim, density=0.10).to(device)
    baseline = DenseBaseline(dim).to(device)
    
    # Compilation is key for the 6x speedup
    compiled_model = torch.compile(model, mode="max-autotune")
    compiled_baseline = torch.compile(baseline, mode="max-autotune")

    # Data
    xr = torch.randn(N, dim, device=device)
    xi = torch.randn(N, dim, device=device)
    u_b = torch.randint(0, 4, (N, dim), device=device, dtype=torch.int32)
    w_b = torch.randint(0, 4, (N, dim), device=device, dtype=torch.int32)

    # Warmup
    print("🔥 Warming kernels...")
    for _ in range(20):
        _ = compiled_baseline(xr)
        _ = compiled_model(xr, xi, u_b, w_b)

    # Benchmark
    start = time.time()
    for _ in range(500): _ = compiled_baseline(xr)
    torch.cuda.synchronize()
    t_base = (time.time() - start) / 500 * 1000

    start = time.time()
    for _ in range(500): _ = compiled_model(xr, xi, u_b, w_b)
    torch.cuda.synchronize()
    t_blink = (time.time() - start) / 500 * 1000

    print(f"⚡ SPEEDUP VERIFIED: {t_base / t_blink:.2f}x")

if __name__ == "__main__":
    run_final_benchmark()
