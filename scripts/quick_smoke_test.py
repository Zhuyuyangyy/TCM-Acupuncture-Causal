#!/usr/bin/env python3
"""Quick smoke test: generate synthetic acupuncture data, fit CausalForest, print ITE."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from backend.models.causal_forest import CausalForest


def main():
    print("=" * 60)
    print("TCM Acupuncture Causal Forest — Quick Smoke Test")
    print("=" * 60)

    rng = np.random.RandomState(42)
    n = 500

    # --- Synthetic acupuncture data ---
    # Features: age, pain_severity, chronicity_years, sessions
    age = rng.uniform(25, 75, n)
    pain = rng.uniform(3, 10, n)
    chronicity = rng.uniform(0.5, 15, n)
    sessions = rng.randint(5, 20, n).astype(float)
    X = np.column_stack([age, pain, chronicity, sessions])

    # Treatment: received acupuncture (1) vs control (0)
    treatment = rng.binomial(1, 0.5, n)

    # True CATE: older patients and higher pain respond better
    cate_true = 1.0 + 0.03 * age + 0.2 * pain - 0.05 * chronicity

    # Outcome: pain reduction
    baseline = 7.0 - 0.02 * age + 0.3 * pain
    y = baseline + cate_true * treatment + rng.randn(n) * 0.8

    true_ate = np.mean(cate_true)
    print(f"\nDataset: {n} patients, 4 features")
    print(f"True ATE (mean pain reduction from acupuncture): {true_ate:.3f}")

    # --- Fit Causal Forest ---
    print("\nFitting CausalForest (n_estimators=80, n_bootstrap=50)...")
    cf = CausalForest(
        n_estimators=80, max_depth=5,
        n_bootstrap=50, ci_level=0.95, random_state=42)
    cf.fit(X, treatment, y)

    # --- Point ITE estimates ---
    ite = cf.predict_ite(X)
    print(f"\nITE estimates — mean: {np.mean(ite):.3f}, "
          f"std: {np.std(ite):.3f}, "
          f"min: {np.min(ite):.3f}, max: {np.max(ite):.3f}")

    # --- CATE with confidence intervals ---
    res = cf.predict_ite_ci(X)
    ci_widths = res['ci_upper'] - res['ci_lower']
    print(f"\nCI width — mean: {np.mean(ci_widths):.3f}, "
          f"std: {np.std(ci_widths):.3f}")

    # Coverage
    covered = np.mean(
        (cate_true >= res['ci_lower']) & (cate_true <= res['ci_upper']))
    print(f"95% CI coverage of true CATE: {covered:.1%}")

    # --- ATE with CI ---
    ate_res = cf.ate_ci(X)
    print(f"\nATE estimate: {ate_res['ate']:.3f}")
    print(f"ATE 95% CI:  [{ate_res['ci_lower']:.3f}, {ate_res['ci_upper']:.3f}]")
    print(f"True ATE:     {true_ate:.3f}")

    # --- Top responders ---
    top5 = np.argsort(ite)[-5:][::-1]
    print(f"\nTop 5 predicted responders (index -> ITE):")
    for idx in top5:
        print(f"  Patient {idx:3d}: age={age[idx]:.0f}, pain={pain[idx]:.1f}, "
              f"ITE={ite[idx]:.3f}  "
              f"[{res['ci_lower'][idx]:.3f}, {res['ci_upper'][idx]:.3f}]")

    print("\n" + "=" * 60)
    print("Smoke test PASSED")
    print("=" * 60)


if __name__ == '__main__':
    main()
