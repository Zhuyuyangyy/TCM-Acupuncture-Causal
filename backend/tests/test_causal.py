"""Tests for CausalForest using synthetic data with known treatment effects."""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from backend.models.causal_forest import CausalForest


def _make_synthetic(n=600, seed=42):
    """Generate synthetic data with a known heterogeneous treatment effect.

    The true CATE is: 2.0 + 1.5 * x0  (depends on first covariate).
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 4)
    treatment = rng.binomial(1, 0.5, size=n)
    # Outcome: baseline + covariate effect + treatment effect + noise
    cate_true = 2.0 + 1.5 * X[:, 0]
    y = 1.0 + 0.5 * X[:, 1] + cate_true * treatment + rng.randn(n) * 0.5
    return X, treatment, y, cate_true


class TestCausalForest:
    """Test suite for CausalForest."""

    def test_fit_and_predict_ite(self):
        X, t, y, _ = _make_synthetic()
        cf = CausalForest(n_estimators=50, max_depth=5, random_state=0)
        cf.fit(X, t, y)
        ite = cf.predict_ite(X)
        assert ite.shape == (len(X),), "ITE shape mismatch"
        assert not np.any(np.isnan(ite)), "ITE contains NaN"

    def test_ite_correlates_with_true_cate(self):
        X, t, y, cate_true = _make_synthetic(n=800, seed=7)
        cf = CausalForest(n_estimators=80, max_depth=6, random_state=0)
        cf.fit(X, t, y)
        ite = cf.predict_ite(X)
        corr = np.corrcoef(ite, cate_true)[0, 1]
        assert corr > 0.5, f"ITE-CATE correlation too low: {corr:.3f}"
        print(f"  ITE-true CATE correlation: {corr:.3f}")

    def test_ate_recovers_ground_truth(self):
        X, t, y, cate_true = _make_synthetic(n=800, seed=99)
        true_ate = np.mean(cate_true)
        cf = CausalForest(n_estimators=80, max_depth=6, random_state=0)
        cf.fit(X, t, y)
        estimated_ate = cf.ate(X)
        # Should be within reasonable range of true ATE
        assert abs(estimated_ate - true_ate) < 1.5,             f"ATE estimate {estimated_ate:.2f} far from true {true_ate:.2f}"
        print(f"  True ATE={true_ate:.3f}, Estimated ATE={estimated_ate:.3f}")

    def test_confidence_interval_contains_true_ite(self):
        """Bootstrap CI should contain the true CATE for a majority of samples."""
        X, t, y, cate_true = _make_synthetic(n=800, seed=123)
        cf = CausalForest(
            n_estimators=60, max_depth=5, n_bootstrap=50,
            ci_level=0.90, random_state=42)
        cf.fit(X, t, y)
        res = cf.predict_ite_ci(X)

        assert 'ite' in res
        assert 'ci_lower' in res
        assert 'ci_upper' in res
        assert res['ci_lower'].shape == res['ite'].shape

        # CI width should be positive
        widths = res['ci_upper'] - res['ci_lower']
        assert np.all(widths >= 0), "CI upper must be >= CI lower"

        # At least 70% of true CATEs should fall within 90% CI
        covered = np.mean(
            (cate_true >= res['ci_lower']) & (cate_true <= res['ci_upper']))
        assert covered > 0.50,             f"Only {covered:.1%} of true CATEs in 90% CI (expected >50%)"
        print(f"  90% CI coverage: {covered:.1%}")

    def test_ate_ci(self):
        X, t, y, cate_true = _make_synthetic(n=600, seed=55)
        cf = CausalForest(
            n_estimators=50, max_depth=5, n_bootstrap=40,
            ci_level=0.95, random_state=7)
        cf.fit(X, t, y)
        ate_res = cf.ate_ci(X)
        true_ate = np.mean(cate_true)
        assert 'ate' in ate_res
        assert 'ci_lower' in ate_res
        assert 'ci_upper' in ate_res
        print(f"  ATE CI: [{ate_res['ci_lower']:.3f}, {ate_res['ci_upper']:.3f}], "
              f"true={true_ate:.3f}")

    def test_fit_before_predict_raises(self):
        cf = CausalForest()
        with pytest.raises(RuntimeError):
            cf.predict_ite(np.zeros((5, 2)))

    def test_fit_before_ci_raises(self):
        cf = CausalForest()
        with pytest.raises(RuntimeError):
            cf.predict_ite_ci(np.zeros((5, 2)))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
