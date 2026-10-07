"""
Scoring of extracted values against ground truth. Pure Python, no extra deps.

Metrics (SQuAD-style, after lower-casing and stripping punctuation/articles):
  exact match   normalised strings are identical
  token F1      overlap between the two token bags; a match when F1 >= F1_THRESHOLD

Components:
  Evaluator.evaluate_field   one field -> {status, exact_match, f1_score, f1_match, symbol}
  Evaluator.evaluate_entity  all configured fields for one entity + a summary block
  Evaluator.evaluate_batch   many entities (used by scripts/compare_results.py)
"""

import re
import string
from collections import Counter
from typing import Dict, List, Optional

from core.config_helpers import get_entity_primary_value

F1_THRESHOLD = 0.5
MISSING_VALUES = {"", "unknown", "not available", "unable to retrieve"}


def normalize(text: str) -> str:
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, truth: str) -> float:
    pred, gold = normalize(prediction).split(), normalize(truth).split()
    common = sum((Counter(pred) & Counter(gold)).values())
    if not pred or not gold or common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(gold)
    return 2 * precision * recall / (precision + recall)


class Evaluator:
    def evaluate_field(self, field: str, extracted, truth: Optional[str]) -> Dict:
        result = {"field": field, "extracted": extracted, "ground_truth": truth}
        if truth is None or str(truth).strip() == "":
            return {**result, "status": "no_ground_truth", "exact_match": None, "f1_score": None, "symbol": "⚪"}
        if extracted is None or str(extracted).strip().lower() in MISSING_VALUES:
            return {**result, "status": "missing", "exact_match": False, "f1_score": 0.0, "f1_match": False, "symbol": "🔴"}

        exact = normalize(extracted) == normalize(truth)
        f1 = round(token_f1(str(extracted), str(truth)), 3)
        return {
            **result,
            "status": "evaluated",
            "exact_match": exact,
            "f1_score": f1,
            "f1_match": f1 >= F1_THRESHOLD,
            "symbol": "🟢" if exact else "🟡" if f1 >= F1_THRESHOLD else "🔴",
        }

    def evaluate_entity(self, entity_name: str, extracted: Dict, truth: Dict, fields: List[Dict]) -> Dict:
        summary = {"total_fields": 0, "exact_matches": 0, "semantic_matches": 0, "misses": 0,
                   "missing": 0, "no_ground_truth": 0}
        per_field = {}
        for f in fields:
            key = f["key"]
            r = self.evaluate_field(key, extracted.get(key), (truth or {}).get(key))
            per_field[key] = r
            summary["total_fields"] += 1
            if r["status"] == "no_ground_truth":
                summary["no_ground_truth"] += 1
            elif r["status"] == "missing":
                summary["missing"] += 1
            elif r["exact_match"]:
                summary["exact_matches"] += 1
            elif r["f1_match"]:
                summary["semantic_matches"] += 1
            else:
                summary["misses"] += 1

        scored = summary["total_fields"] - summary["no_ground_truth"]
        summary["exact_accuracy"] = summary["exact_matches"] / scored if scored else None
        summary["overall_accuracy"] = (summary["exact_matches"] + summary["semantic_matches"]) / scored if scored else None
        return {"entity": entity_name, "fields": per_field, "summary": summary}

    def evaluate_batch(self, results: List[Dict], ground_truth: Dict, fields: List[Dict]) -> Dict:
        entities, totals = [], Counter()
        for r in results:
            name = get_entity_primary_value(r.get("entity", {}))
            ev = self.evaluate_entity(name, r.get("extracted_info", {}), ground_truth.get(name), fields)
            entities.append(ev)
            totals.update({k: v for k, v in ev["summary"].items() if isinstance(v, int)})
        scored = totals["total_fields"] - totals["no_ground_truth"]
        overall = dict(totals)
        overall["exact_accuracy"] = totals["exact_matches"] / scored if scored else None
        overall["overall_accuracy"] = (totals["exact_matches"] + totals["semantic_matches"]) / scored if scored else None
        return {"entities": entities, "overall_summary": overall}
