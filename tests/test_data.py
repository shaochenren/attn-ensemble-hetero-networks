
from aehdn.data.synthetic_hetero import HeteroNetDataset

def test_len():
    ds = HeteroNetDataset(nodes=10, edges=15, T=60, seq_len=8, feature_dim=4, anomaly_rate=0.1, seed=2)
    assert len(ds) == 10*(60-8)
