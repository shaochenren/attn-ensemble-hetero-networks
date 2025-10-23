
import numpy as np

def population_stability_index(expected, actual, bins=10):
    qs = np.quantile(np.concatenate([expected, actual]), np.linspace(0,1,bins+1))
    e_hist, _ = np.histogram(expected, bins=qs)
    a_hist, _ = np.histogram(actual, bins=qs)
    e = e_hist / max(1, e_hist.sum())
    a = a_hist / max(1, a_hist.sum())
    eps = 1e-12
    psi = np.sum((a - e) * np.log((a+eps)/(e+eps)))
    return psi

class EMAWeight:
    def __init__(self, k, alpha=0.9):
        self.w = np.ones(k)/k
        self.alpha = alpha
    def update(self, perf):
        p = np.maximum(perf, 1e-6)
        p = p / p.sum()
        self.w = self.alpha*self.w + (1-self.alpha)*p
        return self.w
