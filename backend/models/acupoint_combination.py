"""Acupoint combination mining."""
from collections import Counter
from typing import List, Dict

class AcupointMiner:
    """穴位组合挖掘器"""
    def __init__(self):
        self.combinations = Counter()
    
    def mine_frequent(self, prescriptions: List[List[str]], min_support=5):
        for rx in prescriptions:
            for i in range(len(rx)):
                for j in range(i+1, len(rx)):
                    pair = tuple(sorted([rx[i], rx[j]]))
                    self.combinations[pair] += 1
        return {k: v for k, v in self.combinations.items() if v >= min_support}
