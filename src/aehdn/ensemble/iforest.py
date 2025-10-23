
import numpy as np
from sklearn.ensemble import IsolationForest

class IForestWrapper:
    def __init__(self, **kwargs):
        self.clf = IsolationForest(random_state=kwargs.get("seed", 42), n_estimators=200, contamination="auto")

    def fit(self, X):
        self.clf.fit(X)
        return self

    def score(self, X):
        s = -self.clf.score_samples(X)
        return s
