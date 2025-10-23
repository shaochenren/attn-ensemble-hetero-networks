
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve

def evaluate_scores(scores, labels, threshold='auto'):
    scores = np.asarray(scores).reshape(-1)
    labels = np.asarray(labels).astype(int).reshape(-1)
    auroc = roc_auc_score(labels, scores) if len(np.unique(labels))>1 else float("nan")
    ap = average_precision_score(labels, scores) if len(np.unique(labels))>1 else float("nan")
    if threshold == 'auto':
        P,R,T = precision_recall_curve(labels, scores)
        f1s = 2*P*R/(P+R+1e-12)
        t = T[np.nanargmax(f1s[:-1])] if len(T) else np.percentile(scores,95)
    else:
        t = float(threshold)
    preds = (scores >= t).astype(int)
    f1 = f1_score(labels, preds) if len(np.unique(labels))>1 else float("nan")
    return {"auroc": auroc, "ap": ap, "f1": f1, "threshold": t}
