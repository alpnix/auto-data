#!/usr/bin/env python3
"""
Compare extracted entity data with ground truth.
Reads field definitions from pipeline_config for dataset-agnostic comparison.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config_helpers import load_config, get_entity_primary_value, get_entity_display


def normalize_value(value):
    """Normalize values for comparison"""
    if not value or value == "Unknown" or value == "Not available":
        return None
    return str(value).strip().lower()


def compare_field(ground_truth_val, extracted_val):
    """Compare two field values"""
    gt = normalize_value(ground_truth_val)
    ex = normalize_value(extracted_val)

    if gt is None and ex is None:
        return "---", "Both unknown"
    elif gt is None:
        return "~", "Extracted but no ground truth"
    elif ex is None:
        return "X", "Missing from extraction"
    elif gt == ex:
        return "OK", "Exact match"
    elif gt in ex or ex in gt:
        return "~", "Partial match"
    else:
        return "X", "Mismatch"


def print_comparison(ground_truth_dict, extracted_list, fields_config):
    """Print comparison table using config-driven fields"""
    config = load_config()

    print("\n" + "="*80)
    print("DATA EXTRACTION RESULTS")
    print("="*80 + "\n")

    for i, result in enumerate(extracted_list):
        entity = result.get('entity', result.get('faculty', {}))
        ex = result.get('extracted_info', {})
        entity_display = get_entity_display(entity, config)
        entity_id = get_entity_primary_value(entity, config)

        gt = ground_truth_dict.get(entity_id, {}) if ground_truth_dict else {}

        print(f"\n{'='*80}")
        print(f"Entity #{i+1}: {entity_display}")
        print(f"{'='*80}")

        for field_cfg in fields_config:
            key = field_cfg['key']
            label = field_cfg['label']
            gt_val = gt.get(key, None)
            ex_val = ex.get(key, "Unknown")
            status, note = compare_field(gt_val, ex_val)

            print(f"\n{label}:")
            print(f"  Ground Truth: {gt_val or 'N/A'}")
            print(f"  Extracted:    {ex_val}")
            print(f"  Status:       [{status}] {note}")

    # Calculate accuracy
    print("\n" + "="*80)
    print("ACCURACY SUMMARY")
    print("="*80)

    total_fields = 0
    exact_matches = 0
    partial_matches = 0
    misses = 0

    for result in extracted_list:
        entity = result.get('entity', result.get('faculty', {}))
        ex = result.get('extracted_info', {})
        entity_id = get_entity_primary_value(entity, config)
        gt = ground_truth_dict.get(entity_id, {}) if ground_truth_dict else {}

        for field_cfg in fields_config:
            key = field_cfg['key']
            gt_val = gt.get(key, None)
            ex_val = ex.get(key, "Unknown")

            if gt_val is None:
                continue  # Skip fields without ground truth

            total_fields += 1
            status, _ = compare_field(gt_val, ex_val)

            if status == "OK":
                exact_matches += 1
            elif status == "~":
                partial_matches += 1
            else:
                misses += 1

    if total_fields > 0:
        accuracy = (exact_matches / total_fields) * 100
        partial_accuracy = ((exact_matches + partial_matches) / total_fields) * 100

        print(f"\nTotal Fields:     {total_fields}")
        print(f"Exact Matches:    {exact_matches} ({accuracy:.1f}%)")
        print(f"Partial Matches:  {partial_matches}")
        print(f"Misses/Errors:    {misses}")
        print(f"\nAccuracy (exact):         {accuracy:.1f}%")
        print(f"Accuracy (incl. partial): {partial_accuracy:.1f}%")
    else:
        print("\nNo ground truth available for comparison.")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    import json
    import sys

    config = load_config()

    if len(sys.argv) > 1:
        # Load results from file
        with open(sys.argv[1], 'r') as f:
            extracted_list = json.load(f)
    else:
        print("Usage: python scripts/compare_results.py <results.json>")
        print("\nOr run the pipeline first to generate results:")
        print("  python app.py")
        print("\nThis script compares extracted data against ground truth from pipeline_config.")
        sys.exit(1)

    # Load ground truth and fields from config
    ground_truth = getattr(config, 'GROUND_TRUTH', {}) if config else {}
    fields_config = getattr(config, 'EXTRACTION_FIELDS', []) if config else []

    if not fields_config:
        print("[WARN] No EXTRACTION_FIELDS found in pipeline_config. Using keys from extracted data.")
        # Infer fields from first result
        if extracted_list:
            ex = extracted_list[0].get('extracted_info', {})
            fields_config = [
                {'key': k, 'label': k.replace('_', ' ').title()}
                for k in ex.keys()
                if k not in ('error', 'raw_response')
            ]

    print_comparison(ground_truth, extracted_list, fields_config)
