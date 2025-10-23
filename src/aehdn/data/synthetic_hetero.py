
import numpy as np, torch, networkx as nx
from torch.utils.data import Dataset

class HeteroNetDataset(Dataset):
    def __init__(self, nodes=40, edges=80, T=240, seq_len=32, feature_dim=8, anomaly_rate=0.08, seed=13):
        rng = np.random.default_rng(seed)
        self.G = nx.gnm_random_graph(nodes, edges, seed=seed, directed=False)
        self.adj = nx.to_numpy_array(self.G).astype(np.float32)
        # node metrics time-series
        t = np.linspace(0, 20, T)
        base = np.sin(t)[:,None] + 0.05*rng.normal(size=(T,1))
        X = base + 0.05*rng.normal(size=(T, feature_dim))
        # replicate per-node + small node-specific deviation
        self.X_nodes = np.stack([X + 0.03*rng.normal(size=X.shape) for _ in range(nodes)], axis=0) # (N,T,F)
        labels = np.zeros((nodes, T), dtype=int)
        for n in range(nodes):
            idx = rng.choice(T, size=max(1,int(T*anomaly_rate)), replace=False)
            labels[n, idx] = 1
        self.labels = labels
        self.seq_len = seq_len
        self.feature_dim = feature_dim

        # windows per node
        self.windows = []
        for n in range(nodes):
            for i in range(T-seq_len):
                xw = self.X_nodes[n, i:i+seq_len, :]
                y = labels[n, i+seq_len-1]
                self.windows.append((n, xw.astype(np.float32), float(y)))
        self.N = nodes

    def __len__(self): return len(self.windows)
    def __getitem__(self, idx):
        n, xw, y = self.windows[idx]
        return n, torch.tensor(xw), torch.tensor(y, dtype=torch.float32)

    def adjacency(self):
        import torch
        return torch.tensor(self.adj, dtype=torch.float32)
