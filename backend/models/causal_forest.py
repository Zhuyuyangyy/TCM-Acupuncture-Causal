"""Causal Forest for heterogeneous treatment effects (T-learner based)."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor


class CausalForest:
    """因果森林 — 估计个体化治疗效应 (Individualized Treatment Effects).

    Uses a T-learner approach: fits separate models for treated and control
    groups, then the ITE / CATE is the difference in predictions.

    Supports bootstrap-based confidence intervals for CATE estimates.
    """

    def __init__(self, n_estimators=100, max_depth=5, n_bootstrap=100,
                 ci_level=0.95, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_bootstrap = n_bootstrap
        self.ci_level = ci_level
        self.random_state = random_state

        # Primary models (full-data fits)
        self.treated_model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state)
        self.control_model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state)

        # Storage for bootstrap replicates
        self._bootstrap_treated = []
        self._bootstrap_control = []
        self._fitted = False

    def fit(self, X, treatment, y):
        """Fit causal forest on observed data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        treatment : array-like of shape (n_samples,) — binary indicator (1=treated)
        y : array-like of shape (n_samples,) — outcome
        """
        X = np.asarray(X)
        treatment = np.asarray(treatment).ravel()
        y = np.asarray(y).ravel()

        t_mask = treatment == 1

        # Fit primary models on full data
        self.treated_model.fit(X[t_mask], y[t_mask])
        self.control_model.fit(X[~t_mask], y[~t_mask])

        # Bootstrap for confidence intervals
        rng = np.random.RandomState(self.random_state)
        self._bootstrap_treated = []
        self._bootstrap_control = []

        n_t = int(t_mask.sum())
        n_c = int((~t_mask).sum())

        for b in range(self.n_bootstrap):
            # Resample within each arm
            idx_t = rng.choice(np.where(t_mask)[0], size=n_t, replace=True)
            idx_c = rng.choice(np.where(~t_mask)[0], size=n_c, replace=True)

            m_t = RandomForestRegressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                random_state=rng.randint(0, 2**31))
            m_c = RandomForestRegressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                random_state=rng.randint(0, 2**31))

            m_t.fit(X[idx_t], y[idx_t])
            m_c.fit(X[idx_c], y[idx_c])

            self._bootstrap_treated.append(m_t)
            self._bootstrap_control.append(m_c)

        self._fitted = True
        return self

    def predict_ite(self, X):
        """Return point estimates of ITE / CATE for each row of X."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_ite()")
        X = np.asarray(X)
        return self.treated_model.predict(X) - self.control_model.predict(X)

    def predict_ite_ci(self, X):
        """Return CATE estimates with bootstrap confidence intervals.

        Returns
        -------
        result : dict with keys
            'ite'       : ndarray of point CATE estimates
            'ci_lower'  : lower bound of confidence interval
            'ci_upper'  : upper bound of confidence interval
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_ite_ci()")

        X = np.asarray(X)
        point = self.predict_ite(X)

        # Collect bootstrap ITE predictions
        boot_ites = np.empty((self.n_bootstrap, len(X)))
        for b in range(self.n_bootstrap):
            boot_ites[b] = (self._bootstrap_treated[b].predict(X)
                            - self._bootstrap_control[b].predict(X))

        alpha = (1.0 - self.ci_level) / 2.0
        ci_lower = np.percentile(boot_ites, 100 * alpha, axis=0)
        ci_upper = np.percentile(boot_ites, 100 * (1 - alpha), axis=0)

        return {
            'ite': point,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
        }

    def ate(self, X):
        """Average Treatment Effect over population X."""
        return np.mean(self.predict_ite(X))

    def ate_ci(self, X):
        """ATE with bootstrap confidence interval.

        Returns
        -------
        result : dict with keys 'ate', 'ci_lower', 'ci_upper'
        """
        res = self.predict_ite_ci(X)
        alpha = (1.0 - self.ci_level) / 2.0
        boot_ates = []
        for b in range(self.n_bootstrap):
            boot_ates.append(np.mean(
                self._bootstrap_treated[b].predict(X)
                - self._bootstrap_control[b].predict(X)))

        boot_ates = np.array(boot_ates)
        return {
            'ate': np.mean(res['ite']),
            'ci_lower': np.percentile(boot_ates, 100 * alpha),
            'ci_upper': np.percentile(boot_ates, 100 * (1 - alpha)),
        }
