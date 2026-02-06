#!/usr/bin/env python3
"""
Debug script to see what sources were actually collected
"""

import json
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python debug_sources.py <session_folder>")
    print("\nExample: python debug_sources.py 20260117_180828")
    sys.exit(1)

session_folder = Path("research_results") / sys.argv[1]
results_file = session_folder / "full_results.json"

if not results_file.exists():
    print(f"Error: {results_file} not found")
    sys.exit(1)

with open(results_file, 'r') as f:
    results = json.load(f)

print("="*80)
print("SOURCE DOCUMENTS DEBUG")
print("="*80)

for result in results:
    entity = result.get('entity', result.get('faculty', {}))
    display = " - ".join(str(v) for v in entity.values() if v)
    print(f"\n{display}")
    print("-"*80)

    sources = result.get('sources', [])
    print(f"Total documents: {len(sources)}\n")

    for i, source in enumerate(sources, 1):
        query_type = source.get('query_type', 'unknown')
        url = source.get('url', 'N/A')
        text = source.get('text', '')

        print(f"{i}. [{query_type}] {url}")
        print(f"   Text length: {len(text)} chars")
        print(f"   Preview: {text[:200]}...")
        print()

print("="*80)
