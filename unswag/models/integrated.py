
import torch
import torch.nn as nn
from unswag.models.tether import LocalTether
from unswag.models.curation import SelfCurationRRQ

class ProtocolCModel(nn.Module):
    def __init__(self, dim, density=0.10): # 10% Density (Ruthless Pruning)
        super().__init__()
        self.dim = dim
        self.density = density 
        
        # The Heavy Expert (Standard Transformer MLP)
        self.expert = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, x_real, x_imag, u_bits, w_bits):
        # 1. Cheap Energy Calculation
        # Fusing the square add is faster
        energy = torch.add(x_real.square(), x_imag.square()).mean(dim=-1)
        
        # 2. Fast Selection
        k = int(energy.shape[0] * self.density)
        
        # Optimization: We don't need 'sorted=True'. 
        # Using largest=True, sorted=False is the fastest GPU path for TopK.
        _, indices = torch.topk(energy, k, sorted=False, largest=True)
        
        # 3. Gather Survivors (The 10% Signal)
        xr_sparse = x_real[indices]
        
        # 4. Execute Heavy Expert on Signal Only
        out_sparse = self.expert(xr_sparse)
        
        return out_sparse
