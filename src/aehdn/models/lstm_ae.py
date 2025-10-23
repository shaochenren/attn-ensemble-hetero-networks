
import torch, torch.nn as nn

class LSTMAE(nn.Module):
    def __init__(self, input_dim, hidden=64, bottleneck=32):
        super().__init__()
        self.enc = nn.LSTM(input_dim, hidden, batch_first=True)
        self.to_b = nn.Linear(hidden, bottleneck)
        self.dec = nn.LSTM(bottleneck, hidden, batch_first=True)
        self.out = nn.Linear(hidden, input_dim)

    def forward(self, x):
        h,_ = self.enc(x)
        z = self.to_b(h[:,-1,:]).unsqueeze(1).expand(-1, x.size(1), -1)
        y,_ = self.dec(z)
        recon = self.out(y)
        return recon, z[:,0,:]

    def score(self, x, recon):
        return ((x - recon)**2).mean(dim=(1,2))
