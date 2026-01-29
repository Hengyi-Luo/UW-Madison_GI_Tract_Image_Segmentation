from __future__ import annotations

import random
from typing import Dict, List, Tuple


def split_case_days(case_day_keys: List[str], val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[str], List[str]]:
    """Split by case id to reduce leakage across days."""
    rng = random.Random(seed)

    case_to_days: Dict[str, List[str]] = {}
    for cd in case_day_keys:
        case = cd.split("_")[0]  # "case123"
        case_to_days.setdefault(case, []).append(cd)

    cases = sorted(case_to_days.keys())
    rng.shuffle(cases)

    n_val = max(1, int(len(cases) * float(val_ratio)))
    val_cases = set(cases[:n_val])

    train_days: List[str] = []
    val_days: List[str] = []
    for c, days in case_to_days.items():
        if c in val_cases:
            val_days.extend(days)
        else:
            train_days.extend(days)

    return train_days, val_days

