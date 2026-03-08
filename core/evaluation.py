#!/usr/bin/env python3
"""
Evaluation Module - PEDANTS and other metrics for comparing extracted vs ground truth
"""

from typing import Dict, List, Optional, Tuple
import json
from core.config_helpers import get_entity_primary_value

# Try to import qa_metrics, gracefully handle if not installed
QA_METRICS_AVAILABLE = False
PEDANTS_AVAILABLE = False

try:
    from qa_metrics.em import em_match
    from qa_metrics.f1 import f1_match, f1_score_with_precision_recall
    QA_METRICS_AVAILABLE = True
    print("[OK] qa-metrics F1/EM loaded")
except ImportError:
    print("[WARN] qa-metrics not installed. Install with: pip install qa-metrics")

try:
    from qa_metrics.pedant import PEDANT
    # Test if PEDANT can be instantiated (may fail due to sklearn version)
    _test_pedant = PEDANT()
    PEDANTS_AVAILABLE = True
    print("[OK] PEDANTS loaded")
except Exception as e:
    print(f"[WARN] PEDANTS not available (sklearn version issue): {type(e).__name__}")


class Evaluator:
    """Evaluates extracted answers against ground truth using multiple metrics"""

    def __init__(self, use_pedants: bool = True, use_f1: bool = True, use_exact: bool = True):
        self.use_pedants = use_pedants and PEDANTS_AVAILABLE
        self.use_f1 = use_f1 and QA_METRICS_AVAILABLE
        self.use_exact = use_exact and QA_METRICS_AVAILABLE

        if self.use_pedants:
            self.pedant = PEDANT()
        else:
            self.pedant = None

    def evaluate_field(
        self,
        field_name: str,
        extracted: str,
        ground_truth: str,
        context_question: Optional[str] = None
    ) -> Dict:
        """
        Evaluate a single field extraction against ground truth.

        Args:
            field_name: Name of the field (e.g., 'phd_institution')
            extracted: The extracted value
            ground_truth: The expected correct value
            context_question: Optional question for PEDANTS context

        Returns:
            Dict with evaluation metrics and match status
        """
        # Handle missing values
        if not extracted or extracted.lower() == 'unknown':
            return {
                'field': field_name,
                'extracted': extracted,
                'ground_truth': ground_truth,
                'status': 'missing',
                'exact_match': False,
                'f1_score': 0.0,
                'pedants_score': 0.0,
                'pedants_match': False,
                'overall_match': False
            }

        if not ground_truth:
            return {
                'field': field_name,
                'extracted': extracted,
                'ground_truth': ground_truth,
                'status': 'no_ground_truth',
                'exact_match': None,
                'f1_score': None,
                'pedants_score': None,
                'pedants_match': None,
                'overall_match': None
            }

        # Generate context question if not provided
        if not context_question:
            context_question = f"What is the {field_name.replace('_', ' ')} of this person?"

        result = {
            'field': field_name,
            'extracted': extracted,
            'ground_truth': ground_truth,
            'status': 'evaluated'
        }

        # Exact match
        if self.use_exact:
            result['exact_match'] = em_match([ground_truth], extracted)
        else:
            # Simple string comparison
            result['exact_match'] = extracted.lower().strip() == ground_truth.lower().strip()

        # F1 Score
        if self.use_f1:
            f1_stats = f1_score_with_precision_recall(ground_truth, extracted)
            result['f1_score'] = f1_stats['f1']
            result['f1_precision'] = f1_stats['precision']
            result['f1_recall'] = f1_stats['recall']
            result['f1_match'] = f1_match([ground_truth], extracted, threshold=0.5)
        else:
            result['f1_score'] = None
            result['f1_match'] = None

        # PEDANTS (semantic matching)
        if self.use_pedants and self.pedant:
            try:
                scores = self.pedant.get_scores([ground_truth], extracted, context_question)
                result['pedants_score'] = max(scores) if scores else 0.0
                result['pedants_match'] = self.pedant.evaluate([ground_truth], extracted, context_question)
            except Exception as e:
                print(f"[WARN] PEDANTS evaluation failed for {field_name}: {e}")
                result['pedants_score'] = None
                result['pedants_match'] = None
        else:
            result['pedants_score'] = None
            result['pedants_match'] = None

        # Overall match (any method matches)
        matches = [
            result.get('exact_match'),
            result.get('f1_match'),
            result.get('pedants_match')
        ]
        result['overall_match'] = any(m for m in matches if m is not None)

        # Determine status symbol
        if result['exact_match']:
            result['symbol'] = '🟢'  # Exact match
        elif result.get('pedants_match') or result.get('f1_match'):
            result['symbol'] = '🟡'  # Semantic/partial match
        else:
            result['symbol'] = '🔴'  # No match

        return result

    def evaluate_entity(
        self,
        entity_name: str,
        extracted_data: Dict,
        ground_truth: Dict,
        fields_config: List[Dict]
    ) -> Dict:
        """
        Evaluate all fields for an entity.

        Args:
            entity_name: Name of the entity (e.g., 'Francis Couvares')
            extracted_data: Dict of extracted field values
            ground_truth: Dict of ground truth values
            fields_config: List of field configurations

        Returns:
            Dict with per-field results and summary statistics
        """
        results = {
            'entity': entity_name,
            'fields': {},
            'summary': {
                'total_fields': 0,
                'exact_matches': 0,
                'semantic_matches': 0,
                'misses': 0,
                'missing': 0,
                'no_ground_truth': 0
            }
        }

        for field_cfg in fields_config:
            field_key = field_cfg['key']
            extracted_value = extracted_data.get(field_key, 'Unknown')
            truth_value = ground_truth.get(field_key) if ground_truth else None

            # Create context question from field description
            context_q = f"What is the {field_cfg['label'].lower()} of {entity_name}?"

            eval_result = self.evaluate_field(
                field_name=field_key,
                extracted=extracted_value,
                ground_truth=truth_value,
                context_question=context_q
            )

            results['fields'][field_key] = eval_result
            results['summary']['total_fields'] += 1

            # Update summary stats
            if eval_result['status'] == 'missing':
                results['summary']['missing'] += 1
            elif eval_result['status'] == 'no_ground_truth':
                results['summary']['no_ground_truth'] += 1
            elif eval_result.get('exact_match'):
                results['summary']['exact_matches'] += 1
            elif eval_result.get('overall_match'):
                results['summary']['semantic_matches'] += 1
            else:
                results['summary']['misses'] += 1

        # Calculate accuracy
        evaluated = results['summary']['total_fields'] - results['summary']['no_ground_truth']
        if evaluated > 0:
            results['summary']['exact_accuracy'] = results['summary']['exact_matches'] / evaluated
            results['summary']['overall_accuracy'] = (
                results['summary']['exact_matches'] + results['summary']['semantic_matches']
            ) / evaluated
        else:
            results['summary']['exact_accuracy'] = None
            results['summary']['overall_accuracy'] = None

        return results

    def evaluate_batch(
        self,
        results_list: List[Dict],
        ground_truth_dict: Dict,
        fields_config: List[Dict]
    ) -> Dict:
        """
        Evaluate a batch of entity results.

        Args:
            results_list: List of result dicts (each with 'faculty' and 'extracted_info')
            ground_truth_dict: Dict mapping entity names to ground truth
            fields_config: List of field configurations

        Returns:
            Dict with per-entity results and overall summary
        """
        batch_results = {
            'entities': [],
            'overall_summary': {
                'total_entities': 0,
                'total_fields': 0,
                'exact_matches': 0,
                'semantic_matches': 0,
                'misses': 0,
                'missing': 0
            }
        }

        for result in results_list:
            entity = result.get('entity', result.get('faculty', {}))
            entity_name = get_entity_primary_value(entity)
            extracted = result.get('extracted_info', {})
            truth = ground_truth_dict.get(entity_name) if ground_truth_dict else None

            entity_eval = self.evaluate_entity(
                entity_name=entity_name,
                extracted_data=extracted,
                ground_truth=truth,
                fields_config=fields_config
            )

            batch_results['entities'].append(entity_eval)
            batch_results['overall_summary']['total_entities'] += 1

            # Aggregate stats
            for key in ['total_fields', 'exact_matches', 'semantic_matches', 'misses', 'missing']:
                batch_results['overall_summary'][key] += entity_eval['summary'][key]

        # Calculate overall accuracy
        total = batch_results['overall_summary']['total_fields']
        if total > 0:
            batch_results['overall_summary']['exact_accuracy'] = (
                batch_results['overall_summary']['exact_matches'] / total
            )
            batch_results['overall_summary']['overall_accuracy'] = (
                (batch_results['overall_summary']['exact_matches'] +
                 batch_results['overall_summary']['semantic_matches']) / total
            )

        return batch_results


def format_evaluation_report(eval_results: Dict) -> str:
    """Format evaluation results as a readable report"""
    lines = []
    lines.append("=" * 80)
    lines.append("EVALUATION REPORT")
    lines.append("=" * 80)

    for entity_eval in eval_results['entities']:
        lines.append(f"\n{entity_eval['entity']}")
        lines.append("-" * 80)

        for field_key, field_eval in entity_eval['fields'].items():
            symbol = field_eval.get('symbol', '⚪')
            extracted = field_eval['extracted']
            truth = field_eval['ground_truth']

            lines.append(f"{symbol} {field_key}:")
            lines.append(f"   Extracted: {extracted}")
            lines.append(f"   Expected:  {truth}")

            if field_eval.get('pedants_score') is not None:
                lines.append(f"   PEDANTS:   {field_eval['pedants_score']:.3f}")
            if field_eval.get('f1_score') is not None:
                lines.append(f"   F1 Score:  {field_eval['f1_score']:.3f}")

        # Entity summary
        s = entity_eval['summary']
        lines.append(f"\n   Summary: {s['exact_matches']} exact, {s['semantic_matches']} semantic, "
                    f"{s['misses']} missed, {s['missing']} unknown")
        if s.get('overall_accuracy') is not None:
            lines.append(f"   Accuracy: {s['overall_accuracy']*100:.1f}%")

    # Overall summary
    os = eval_results['overall_summary']
    lines.append("\n" + "=" * 80)
    lines.append("OVERALL SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Total Entities: {os['total_entities']}")
    lines.append(f"Total Fields:   {os['total_fields']}")
    lines.append(f"Exact Matches:  {os['exact_matches']}")
    lines.append(f"Semantic:       {os['semantic_matches']}")
    lines.append(f"Misses:         {os['misses']}")
    lines.append(f"Missing:        {os['missing']}")
    if os.get('exact_accuracy') is not None:
        lines.append(f"\nExact Accuracy:   {os['exact_accuracy']*100:.1f}%")
    if os.get('overall_accuracy') is not None:
        lines.append(f"Overall Accuracy: {os['overall_accuracy']*100:.1f}%")

    return "\n".join(lines)


if __name__ == "__main__":
    # Test the evaluator
    print("Testing Evaluator...")
    print(f"QA Metrics Available: {QA_METRICS_AVAILABLE}")

    if QA_METRICS_AVAILABLE:
        evaluator = Evaluator()

        # Test single field
        result = evaluator.evaluate_field(
            field_name="phd_institution",
            extracted="University of Michigan",
            ground_truth="University of Michigan, Ann Arbor",
            context_question="Where did this professor get their PhD?"
        )

        print(f"\nTest Result:")
        print(json.dumps(result, indent=2))
    else:
        print("\nInstall qa-metrics to test: pip install qa-metrics==0.2.29")
