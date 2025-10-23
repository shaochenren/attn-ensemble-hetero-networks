
import torch, torch.nn as nn

class MLPAE(nn.Module):
    def __init__(self, input_dim, hidden=64, bottleneck=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, bottleneck))
        self.dec = nn.Sequential(nn.Linear(bottleneck, hidden), nn.ReLU(), nn.Linear(hidden, input_dim))

    def forward(self, x): # x: (B,T,F) -> flatten time
        B,T,F = x.shape
        xf = x.reshape(B, T*F)
        z = self.enc(xf)
        recon = self.dec(z).reshape(B, T, F)
        return recon, z

    def score(self, x, recon):
        return ((x - recon)**2).mean(dim=(1,2))
