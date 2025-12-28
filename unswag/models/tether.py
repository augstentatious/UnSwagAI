
import torch
import torch.nn as nn

class LocalTether(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim)
        self.gate = nn.Linear(dim, 1) 

    def forward(self, x):
        # Expected x: [Batch, Seq, Dim]
        tether_weight = torch.sigmoid(self.gate(x)) # [B, N, 1]
        x_t = x.transpose(1, 2) # [B, Dim, N]
        local_path = self.conv(x_t).transpose(1, 2) # [B, N, Dim]
        return (1 - tether_weight) * x + tether_weight * local_path
