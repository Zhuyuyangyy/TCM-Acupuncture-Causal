"""Causal Forest for heterogeneous treatment effects."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class CausalForest:
    """因果森林 — 估计个体化治疗效应"""
    def __init__(self, n_estimators=100, max_depth=5):
        self.treated_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
        self.control_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    
    def fit(self, X, treatment, y):
        t_mask = treatment == 1
        self.treated_model.fit(X[t_mask], y[t_mask])
        self.control_model.fit(X[~t_mask], y[~t_mask])
        return self
    
    def predict_ite(self, X):
        return self.treated_model.predict(X) - self.control_model.predict(X)
