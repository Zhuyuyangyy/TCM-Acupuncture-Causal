"""Acupoint combination mining with association rule support."""
from collections import Counter
from typing import List, Dict, Tuple
from itertools import combinations


class AcupointMiner:
    """穴位组合挖掘器 — supports frequent itemset mining AND association rules.

    Capabilities
    ------------
    1. Frequent pair / itemset mining with configurable min_support
    2. Association rule generation (support, confidence, lift)
    3. Multi-itemset support (pairs, triples, k-itemsets)
    """

    def __init__(self):
        self.combinations: Counter = Counter()
        self._n_prescriptions: int = 0
        self._item_counts: Counter = Counter()

    # ------------------------------------------------------------------
    # Frequent itemset mining
    # ------------------------------------------------------------------

    def mine_frequent(
        self,
        prescriptions: List[List[str]],
        min_support: int = 5,
        max_k: int = 2,
    ) -> Dict[Tuple[str, ...], int]:
        """Mine frequent itemsets up to size *max_k*.

        Parameters
        ----------
        prescriptions : list of lists of acupoint codes
        min_support   : minimum absolute count to be considered frequent
        max_k         : maximum itemset size (2 = pairs, 3 = triples, …)

        Returns
        -------
        dict mapping itemset tuple -> count
        """
        self._n_prescriptions = len(prescriptions)
        self._item_counts.clear()
        self.combinations.clear()

        for rx in prescriptions:
            unique = sorted(set(rx))
            # track single-item counts
            for pt in unique:
                self._item_counts[pt] += 1
            # generate k-itemsets
            for k in range(2, max_k + 1):
                for combo in combinations(unique, k):
                    self.combinations[combo] += 1

        return {k: v for k, v in self.combinations.items() if v >= min_support}

    def mine_frequent_pairs(
        self,
        prescriptions: List[List[str]],
        min_support: int = 5,
    ) -> Dict[Tuple[str, str], int]:
        """Convenience: mine only pairs (k=2)."""
        return self.mine_frequent(prescriptions, min_support=min_support, max_k=2)

    # ------------------------------------------------------------------
    # Association rule generation
    # ------------------------------------------------------------------

    def generate_rules(
        self,
        min_support: float = 0.01,
        min_confidence: float = 0.1,
        min_lift: float = 1.0,
    ) -> List[Dict]:
        """Generate association rules from previously mined itemsets.

        Requires that mine_frequent() has been called first.

        Parameters
        ----------
        min_support    : minimum *relative* support (count / n_prescriptions)
        min_confidence : minimum confidence P(B|A)
        min_lift       : minimum lift P(B|A) / P(B)

        Returns
        -------
        List of dicts, each with keys:
            antecedent, consequent, support, confidence, lift, count
        """
        if self._n_prescriptions == 0:
            raise RuntimeError("Call mine_frequent() before generate_rules()")

        n = self._n_prescriptions
        rules: List[Dict] = []

        for itemset, count in self.combinations.items():
            if len(itemset) < 2:
                continue
            rel_support = count / n

            # generate all non-empty proper subsets as antecedents
            for k in range(1, len(itemset)):
                for antecedent_tuple in combinations(itemset, k):
                    antecedent = tuple(sorted(antecedent_tuple))
                    consequent = tuple(sorted(set(itemset) - set(antecedent)))

                    # P(A) = count(A) / n
                    ant_count = self._item_counts.get(antecedent[0], 0) if len(antecedent) == 1 else self.combinations.get(antecedent, 0)
                    # For single-item antecedents, use item_counts directly
                    if len(antecedent) == 1:
                        ant_count = self._item_counts.get(antecedent[0], 0)
                    else:
                        ant_count = self.combinations.get(antecedent, 0)

                    if ant_count == 0:
                        continue

                    confidence = count / ant_count  # P(B|A)

                    # P(B) for lift calculation
                    if len(consequent) == 1:
                        con_count = self._item_counts.get(consequent[0], 0)
                    else:
                        con_count = self.combinations.get(consequent, 0)

                    p_b = con_count / n if con_count > 0 else 0
                    lift = confidence / p_b if p_b > 0 else 0.0

                    if (rel_support >= min_support
                            and confidence >= min_confidence
                            and lift >= min_lift):
                        rules.append({
                            "antecedent": list(antecedent),
                            "consequent": list(consequent),
                            "support": round(rel_support, 6),
                            "confidence": round(confidence, 6),
                            "lift": round(lift, 4),
                            "count": count,
                        })

        rules.sort(key=lambda r: (-r["lift"], -r["confidence"]))
        return rules

    def top_rules(
        self,
        prescriptions: List[List[str]],
        n_rules: int = 20,
        min_support: float = 0.01,
        min_confidence: float = 0.1,
        min_lift: float = 1.0,
    ) -> List[Dict]:
        """One-shot: mine frequent itemsets then return top association rules."""
        self.mine_frequent(prescriptions, min_support=1, max_k=3)
        return self.generate_rules(
            min_support=min_support,
            min_confidence=min_confidence,
            min_lift=min_lift,
        )[:n_rules]
