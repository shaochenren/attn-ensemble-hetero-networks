
import torch, torch.nn as nn

class AttentionFusion(nn.Module):
    def __init__(self, in_features=4, ctx_dim=32, hidden=32):
        super().__init__()
        self.ctx = nn.Sequential(nn.Linear(ctx_dim, hidden), nn.ReLU())
        self.attn = nn.Linear(hidden, in_features)

    def forward(self, scores, context):
        h = self.ctx(context)            # (B, hidden)
        w = torch.softmax(self.attn(h), dim=-1)  # (B,K)
        fused = (scores * w).sum(-1)
        return fused, w
