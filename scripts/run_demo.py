
import argparse, yaml, numpy as np, torch
from torch.utils.data import DataLoader, random_split
from aehdn.utils.common import set_seed
from aehdn.data.synthetic_hetero import HeteroNetDataset
from aehdn.models.ae import MLPAE
from aehdn.models.lstm_ae import LSTMAE
from aehdn.models.gat import GATEncoder
from aehdn.ensemble.iforest import IForestWrapper
from aehdn.ensemble.attn_fusion import AttentionFusion
from aehdn.eval.metrics import evaluate_scores

def to_batches(ds, batch_size):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    for n, xw, y in loader:
        yield n, xw, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    set_seed(cfg.get("seed", 13))
    device = cfg.get("device","cpu")

    ds = HeteroNetDataset(nodes=cfg["data"]["nodes"], edges=cfg["data"]["edges"],
                          T=cfg["data"]["T"], seq_len=cfg["data"]["seq_len"],
                          feature_dim=cfg["data"]["feature_dim"], anomaly_rate=cfg["data"]["anomaly_rate"],
                          seed=cfg.get("seed",13))
    n = len(ds)
    n_train = int(0.6*n); n_val = int(0.2*n)
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n-n_train-n_val],
                                             generator=torch.Generator().manual_seed(cfg.get("seed",13)))

    adj = ds.adjacency().to(device)
    N = ds.N
    # Node context features: simple degree + random init
    degrees = torch.tensor(adj.sum(dim=1, keepdim=True), device=device)
    node_init = torch.cat([degrees, torch.randn(N, 15, device=device)], dim=1) # (N,16)

    # Models
    ae = MLPAE(input_dim=cfg["data"]["seq_len"]*cfg["data"]["feature_dim"],
               hidden=cfg["model"]["ae_hidden"], bottleneck=cfg["model"]["ae_bottleneck"]).to(device)
    lstm = LSTMAE(input_dim=cfg["data"]["feature_dim"],
                  hidden=cfg["model"]["lstm_hidden"], bottleneck=cfg["model"]["lstm_bottleneck"]).to(device)
    gat = GATEncoder(in_dim=node_init.shape[1], hidden=cfg["model"]["gat_hidden"],
                     heads=cfg["model"]["gat_heads"]).to(device)
    attn = AttentionFusion(in_features=4, ctx_dim=cfg["model"]["gat_hidden"], hidden=cfg["model"]["attn_hidden"]).to(device)
    iforest = IForestWrapper(seed=cfg.get("seed",13))

    # Optimizers
    opt_ae = torch.optim.Adam(ae.parameters(), lr=cfg["train"]["lr"])
    opt_lstm = torch.optim.Adam(lstm.parameters(), lr=cfg["train"]["lr"])
    opt_attn = torch.optim.Adam(attn.parameters(), lr=cfg["train"]["lr"])

    # Fit Isolation Forest on flattened windows from train set
    X_if = []
    for n_idx, xw, y in to_batches(train_ds, cfg["train"]["batch_size"]):
        X_if.append(xw.reshape(xw.size(0), -1).numpy())
    X_if = np.concatenate(X_if, axis=0)
    iforest.fit(X_if)

    # Train AE & LSTM
    for ep in range(cfg["train"]["epochs"]):
        losses = []
        for n_idx, xw, y in to_batches(train_ds, cfg["train"]["batch_size"]):
            xw = xw.to(device)
            # AE
            recon_ae, _ = ae(xw)
            loss_ae = ((xw - recon_ae)**2).mean()
            opt_ae.zero_grad(); loss_ae.backward(); opt_ae.step()
            # LSTM
            recon_lstm, _ = lstm(xw)
            loss_lstm = ((xw - recon_lstm)**2).mean()
            opt_lstm.zero_grad(); loss_lstm.backward(); opt_lstm.step()
            losses.append((loss_ae.item()+loss_lstm.item())/2)
        print(f"[train] epoch {ep} recon_loss={np.mean(losses):.4f}")

    with torch.no_grad():
        H = gat(node_init, adj)  # (N, hidden)

    # Prepare attention training on validation (weakly supervised using labels)
    all_scores, all_ctx, all_labels = [], [], []
    val_loader = DataLoader(val_ds, batch_size=cfg['train']['batch_size'], shuffle=False)
    for n_idx, xw, y in val_loader:
        xw = xw.to(device); y = y.to(device)
        recon_ae, _ = ae(xw)
        recon_lstm, _ = lstm(xw)
        s_ae = ((xw - recon_ae)**2).mean(dim=(1,2))
        s_lstm = ((xw - recon_lstm)**2).mean(dim=(1,2))
        import numpy as np
        from torch import tensor
        s_if = tensor(iforest.score(xw.reshape(xw.size(0), -1).cpu().numpy()), device=device)
        s_diff = (xw[:,1:,:] - xw[:,:-1,:]).abs().mean(dim=(1,2))
        S = torch.stack([s_ae, s_lstm, s_if, s_diff], dim=1)  # (B,4)
        ctx = H[n_idx]  # (B, hidden)
        all_scores.append(S); all_ctx.append(ctx); all_labels.append(y)
    all_scores = torch.cat(all_scores, dim=0); all_ctx = torch.cat(all_ctx, dim=0); all_labels = torch.cat(all_labels, dim=0)

    for ep in range(2):
        fused, w = attn(all_scores, all_ctx)
        fs = (fused - fused.mean()) / (fused.std()+1e-6)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(fs, all_labels)
        opt_attn.zero_grad(); loss.backward(); opt_attn.step()
        print(f"[attn] epoch {ep} loss={loss.item():.4f}")

    # Evaluate on test
    scores, labels = [], []
    test_loader = DataLoader(test_ds, batch_size=cfg['train']['batch_size'], shuffle=False)
    with torch.no_grad():
        for n_idx, xw, y in test_loader:
            xw = xw.to(device); y = y.to(device)
            recon_ae, _ = ae(xw)
            recon_lstm, _ = lstm(xw)
            s_ae = ((xw - recon_ae)**2).mean(dim=(1,2))
            s_lstm = ((xw - recon_lstm)**2).mean(dim=(1,2))
            from torch import tensor
            s_if = tensor(iforest.score(xw.reshape(xw.size(0), -1).cpu().numpy()), device=device)
            s_diff = (xw[:,1:,:] - xw[:,:-1,:]).abs().mean(dim=(1,2))
            S = torch.stack([s_ae, s_lstm, s_if, s_diff], dim=1)
            ctx = H[n_idx]
            fused, w = attn(S, ctx)
            scores.extend(fused.detach().cpu().numpy().tolist())
            labels.extend(y.detach().cpu().numpy().tolist())

    metrics = evaluate_scores(scores, labels, threshold=cfg["eval"]["threshold"])
    print("Evaluation:", metrics)

if __name__ == "__main__":
    main()
