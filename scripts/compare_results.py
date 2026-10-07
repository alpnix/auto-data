"""
Score a saved run against the GROUND_TRUTH in pipeline_config.py.

  python scripts/compare_results.py research_results/<session>/full_results.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config_helpers import load_config  # noqa: E402
from core.evaluation import Evaluator  # noqa: E402


def main(path: str):
    results = json.loads(Path(path).read_text())
    config = load_config()
    fields = getattr(config, "EXTRACTION_FIELDS", []) if config else []
    truth = getattr(config, "GROUND_TRUTH", {}) if config else {}
    report = Evaluator().evaluate_batch(results, truth, fields)

    for ev in report["entities"]:
        print(f"\n{ev['entity']}")
        for key, r in ev["fields"].items():
            f1 = f"  F1={r['f1_score']:.2f}" if r.get("f1_score") is not None else ""
            print(f"  {r['symbol']} {key:<16} {str(r['extracted'])[:40]:<42} expected: {r['ground_truth']}{f1}")

    o = report["overall_summary"]
    print(f"\nFields scored: {o['total_fields'] - o['no_ground_truth']}   "
          f"exact: {o['exact_matches']}   partial: {o['semantic_matches']}   "
          f"missed: {o['misses']}   missing: {o['missing']}")
    if o["overall_accuracy"] is not None:
        print(f"Exact accuracy: {o['exact_accuracy']:.1%}   Accuracy incl. partial: {o['overall_accuracy']:.1%}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
