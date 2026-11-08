# -*- coding: utf-8 -*-
"""Two-stage (hierarchical) probability wrapper: P(apple) = P(orchard) * P(apple | orchard). Used by 39_retrain_expanded.py."""
import numpy as np


class HierProba:
    def __init__(self, c1, c2):
        self.c1, self.c2 = c1, c2

    def predict_proba(self, X):
        p = self.c1.predict_proba(X)[:, 1] * self.c2.predict_proba(X)[:, 1]
        return np.c_[1 - p, p]
