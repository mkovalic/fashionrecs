import torch
import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=512, hidden=512, out_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        z = self.net(x)
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-12)
        return z