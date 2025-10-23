
import torch, torch.nn as nn, torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=2, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.lin = nn.Linear(in_dim, out_dim*heads, bias=False)
        self.attn_l = nn.Parameter(torch.randn(heads, out_dim))
        self.attn_r = nn.Parameter(torch.randn(heads, out_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        N = x.size(0)
        h = self.lin(x).view(N, self.heads, -1)  # (N,H,D)
        el = (h * self.attn_l).sum(-1)
        er = (h * self.attn_r).sum(-1)
        e = el.unsqueeze(1) + er.unsqueeze(0)     # (N,N,H)
        e = e.masked_fill(adj.unsqueeze(-1)==0, float('-inf'))
        a = torch.softmax(e, dim=1)
        a = self.dropout(a)
        out = torch.einsum("ijh,jhd->ihd", a, h)
        return out.reshape(N, -1)

class GATEncoder(nn.Module):
    def __init__(self, in_dim=16, hidden=32, heads=2):
        super().__init__()
        self.g1 = GATLayer(in_dim, hidden, heads=heads)
        self.proj = nn.Linear(hidden*heads, hidden)

    def forward(self, x, adj):
        h = self.g1(x, adj)
        return self.proj(h)
