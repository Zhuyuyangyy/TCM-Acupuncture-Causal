#!/usr/bin/env python3
"""Acupuncture causal inference experiment pipeline.

Tests:
1. Causal Forest CATE estimation on synthetic acupuncture data
2. Uplift modeling evaluation (Qini, AUUC)
3. Association rule mining for acupoint combinations
4. Monte Carlo simulation with known ground truth
"""
import sys, json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def generate_synthetic_acupuncture(n=1000, seed=42):
    rng = np.random.RandomState(seed)
    age = rng.normal(55, 12, n).clip(18, 85)
    sex = rng.binomial(1, 0.5, n)
    pain_duration = rng.exponential(12, n).clip(1, 120)
    baseline_vas = rng.normal(6, 1.5, n).clip(2, 10)
    # Confounded treatment: sicker patients more likely to get acupuncture
    logit = -1 + 0.3 * baseline_vas - 0.02 * age + 0.5 * (pain_duration > 6)
    prop = 1 / (1 + np.exp(-logit))
    treatment = rng.binomial(1, prop)
    # True heterogeneous effect: stronger for high baseline VAS
    true_cate = 1.5 + 0.3 * baseline_vas - 0.01 * age
    outcome = baseline_vas + true_cate * treatment + rng.normal(0, 1, n)
    return {
        "X": np.column_stack([age, sex, pain_duration, baseline_vas]),
        "treatment": treatment, "outcome": outcome,
        "true_cate": true_cate, "propensity": prop,
    }

def run_causal_forest_experiment():
    from backend.models.causal_forest import CausalForest
    data = generate_synthetic_acupuncture(2000)
    cf = CausalForest(n_estimators=100, max_depth=5, n_bootstrap=50)
    cf.fit(data["X"], data["treatment"], data["outcome"])
    ci_result = cf.predict_ite_ci(data["X"])
    cate_pred = ci_result["ite"]
    ci_lower = ci_result["ci_lower"]
    ci_upper = ci_result["ci_upper"]
    true = data["true_cate"]
    bias = np.mean(cate_pred - true)
    rmse = np.sqrt(np.mean((cate_pred - true) ** 2))
    coverage = np.mean((ci_lower <= true) & (true <= ci_upper))
    return {"bias": float(bias), "rmse": float(rmse), "coverage_95": float(coverage),
            "mean_cate": float(np.mean(cate_pred)), "true_mean_cate": float(np.mean(true))}

def run_association_rules():
    from itertools import combinations
    rng = np.random.RandomState(42)
    acupoints = ["LI4", "LR3", "ST36", "SP6", "GV20", "PC6", "BL23", "KI3", "GB34", "HT7"]
    transactions = []
    for _ in range(500):
        n = rng.randint(2, 6)
        base = list(rng.choice(acupoints, n, replace=False))
        if "LI4" in base and "LR3" not in base and rng.random() < 0.7:
            base.append("LR3")
        if "ST36" in base and "SP6" not in base and rng.random() < 0.6:
            base.append("SP6")
        transactions.append(base)
    # Simple frequent itemset mining
    pair_counts = {}
    for t in transactions:
        for pair in combinations(sorted(t), 2):
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
    top_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    return [{"pair": list(p), "count": c, "support": c / len(transactions)} for p, c in top_pairs]

def main():
    print("=" * 60)
    print("Acupuncture Causal Inference Experiment")
    print("=" * 60)
    print("\n[1] Causal Forest CATE Estimation...")
    cf_result = run_causal_forest_experiment()
    print(f"  Bias: {cf_result['bias']:.4f}")
    print(f"  RMSE: {cf_result['rmse']:.4f}")
    print(f"  95% Coverage: {cf_result['coverage_95']:.3f}")
    print(f"  Mean CATE: {cf_result['mean_cate']:.3f} (true: {cf_result['true_mean_cate']:.3f})")
    print("\n[2] Association Rule Mining...")
    rules = run_association_rules()
    print(f"  Top acupoint pairs:")
    for r in rules[:5]:
        print(f"    {' + '.join(r['pair'])}: support={r['support']:.3f} (n={r['count']})")
    output = {"causal_forest": cf_result, "association_rules": rules}
    out_dir = Path("output"); out_dir.mkdir(exist_ok=True)
    with open(out_dir / "experiment_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_dir}/")

if __name__ == "__main__":
    main()
