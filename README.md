
# attn-ensemble-hetero-networks

Attention-Driven Ensemble Models for Adaptive Anomaly Detection in Heterogeneous Distributed Networks (time-series + topology; AE/LSTM/IForest/GAT; attention meta-learner with domain shift adaptation).

## ✨ Key Features
- **Heterogeneous signals**: node metrics (time series), edges (graph topology), and optional event logs.
- **Ensemble base detectors**:
  - **Autoencoder (AE)** for per-node metrics
  - **LSTM-AE** for temporal dynamics
  - **Isolation Forest** for tabular snapshots
  - **Graph Attention Network (GAT)** for topology-aware embeddings
- **Attention meta-learner** that fuses per-detector scores into adaptive anomaly scores conditioned on node context.
- **Domain shift adaptation** via online EMA calibration and drift detectors (PSI/KL) to reweight ensemble.
- **Reproducible experiments**: configs, seeds, and synthetic heterogeneous generator.
- **Docker & CI**; **tests** for core components.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .   # or: pip install -r requirements.txt
python scripts/run_demo.py --config configs/demo.yaml
```
This runs: data synth → train base detectors → fit attention fusion → evaluate AUROC/AP/F1.
