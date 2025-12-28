
import torch
import torch.nn as nn
from unswag.models.tether import LocalTether
from unswag.models.curation import SelfCurationRRQ

class ProtocolCModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.tether = LocalTether(dim)
        self.curator = SelfCurationRRQ(depth=2)
        
    def forward(self, x_real, x_imag, u_bits, w_bits):
        # x_real/imag are [N, Dim]
        # 1. Curation (Scrubbing the weights)
        xr_clean = self.curator(x_real)
        xi_clean = self.curator(x_imag)
        
        # 2. Tethering (Syntactic stability)
        # We treat the N dimension as Seq and Dim as Features
        x_combined = torch.complex(xr_clean, xi_clean).abs().unsqueeze(0) # [1, N, Dim]
        x_stable = self.tether(x_combined).squeeze(0) # Back to [N, Dim]
        
        return xr_clean, xi_clean
