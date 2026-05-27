"""Uplift modeling for acupuncture treatment effects using Two-Model approach."""
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score


class TwoModelUplift:
    """Two-Model (T-learner) uplift estimator.

    Fits separate outcome models for treatment and control arms, then
    estimates individual treatment effects (ITE) as the difference in
    predicted outcomes.  Optionally uses cross-validation to select
    the best base learner.

    This is complementary to CausalForest — it offers gradient-boosted
    trees as an alternative base learner and adds Qini-style evaluation.
    """

    def __init__(
        self,
        learner: str = "gbm",
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        random_state: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        learner : 'gbm' for GradientBoostingRegressor, 'rf' for RandomForestRegressor
        """
        self.learner = learner
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

        self._treated_model = None
        self._control_model = None
        self._fitted = False

    def _make_estimator(self):
        if self.learner == "gbm":
            return GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
            )
        elif self.learner == "rf":
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unknown learner: {self.learner!r}. Use 'gbm' or 'rf'.")

    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> "TwoModelUplift":
        """Fit separate models on treated and control groups.

        Parameters
        ----------
        X         : feature matrix (n_samples, n_features)
        treatment : binary indicator (1 = treated, 0 = control)
        y         : outcome (continuous)
        """
        X = np.asarray(X)
        treatment = np.asarray(treatment).ravel()
        y = np.asarray(y).ravel()

        t_mask = treatment == 1

        self._treated_model = self._make_estimator()
        self._control_model = self._make_estimator()

        self._treated_model.fit(X[t_mask], y[t_mask])
        self._control_model.fit(X[~t_mask], y[~t_mask])
        self._fitted = True
        return self

    def predict_ite(self, X: np.ndarray) -> np.ndarray:
        """Estimate Individual Treatment Effect for each sample."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_ite()")
        X = np.asarray(X)
        return self._treated_model.predict(X) - self._control_model.predict(X)

    def predict_potential_outcomes(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Return both potential outcome predictions.

        Returns
        -------
        dict with keys 'y_treated', 'y_control', 'ite'
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_potential_outcomes()")
        X = np.asarray(X)
        y_t = self._treated_model.predict(X)
        y_c = self._control_model.predict(X)
        return {
            "y_treated": y_t,
            "y_control": y_c,
            "ite": y_t - y_c,
        }

    def qini_curve(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
        n_bins: int = 10,
    ) -> Dict[str, np.ndarray]:
        """Compute Qini curve for uplift evaluation.

        Returns
        -------
        dict with 'proportion', 'uplift_gain', 'random_gain'
        """
        ite = self.predict_ite(X)
        order = np.argsort(-ite)

        n = len(X)
        proportions = np.linspace(0, 1, n_bins + 1)[1:]
        uplift_gains = []
        random_gains = []

        treatment = np.asarray(treatment).ravel()
        y = np.asarray(y).ravel()

        for p in proportions:
            k = int(n * p)
            top_k = order[:k]

            # Uplift gain: treated outcome in top-k vs control outcome in top-k
            t_in_top = treatment[top_k] == 1
            c_in_top = treatment[top_k] == 0

            y_t_mean = y[top_k][t_in_top].mean() if t_in_top.any() else 0
            y_c_mean = y[top_k][c_in_top].mean() if c_in_top.any() else 0

            n_t = t_in_top.sum()
            n_c = c_in_top.sum()

            gain = (n_t / k) * y_t_mean - (n_c / k) * y_c_mean if k > 0 else 0
            uplift_gains.append(gain)

            # Random baseline
            rand_gain = p * (y[treatment == 1].mean() - y[treatment == 0].mean())
            random_gains.append(rand_gain)

        return {
            "proportion": proportions,
            "uplift_gain": np.array(uplift_gains),
            "random_gain": np.array(random_gains),
        }

    def ate(self, X: np.ndarray) -> float:
        """Average Treatment Effect over population X."""
        return float(np.mean(self.predict_ite(X)))
