from __future__ import annotations
from typing import List, Dict

class FinalSelector:
    """Ensure exact final_k chunk selection."""

    def select(self, items: List[Dict], k: int) -> List[Dict]:
        if len(items) >= k:
            return items[:k]
        if not items:
            return []
        pad = items[-1].copy()
        return items + [pad.copy() for _ in range(k - len(items))]
