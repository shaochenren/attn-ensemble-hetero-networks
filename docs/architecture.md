
# Architecture
- **Data**: synthetic heterogeneous generator yields (graph, per-node TS windows, labels).
- **Base detectors**:
  - `models/ae.py`: simple MLP AE (per-window)
  - `models/lstm_ae.py`: temporal AE
  - `models/gat.py`: light Graph Attention block to produce node embeddings
  - `ensemble/iforest.py`: scikit-learn IsolationForest wrapper
- **Fusion**: `ensemble/attn_fusion.py` learns attention weights over base scores conditioned on node & context features.
- **Drift**: `drift/calibration.py` PSI/KL drift signals → EMA reweighting.
- **Eval**: AUROC/AP/F1 and threshold selection.
