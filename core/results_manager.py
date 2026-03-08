#!/usr/bin/env python3
"""
Results Manager - Save and organize research results
Saves results with timestamps for tracking different test runs.
Dataset-agnostic: reads field definitions from pipeline_config.py.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from core.config_helpers import (
    load_config, get_entity_schema, get_entity_primary_value,
    get_entity_display, get_entity_fields
)


class ResultsManager:
    """Manages saving and organizing research results"""

    def __init__(self, base_dir: str = "research_results"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

    def create_session_folder(self) -> Path:
        """Create a timestamped folder for this research session"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_folder = self.base_dir / timestamp
        session_folder.mkdir(exist_ok=True)
        return session_folder

    def _get_entity(self, result: Dict) -> Dict:
        """Get entity dict from result, handling both old and new key names."""
        return result.get('entity', result.get('faculty', {}))

    def save_results(self, results: List[Dict], session_folder: Path) -> Dict[str, str]:
        """
        Save research results to multiple formats.
        Returns dict with paths to saved files.
        """
        saved_files = {}

        # 1. Save full results as JSON
        results_file = session_folder / "full_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        saved_files['full_results'] = str(results_file)

        # 2. Save extracted info only (clean format)
        extracted_file = session_folder / "extracted_data.json"
        extracted_data = [
            {
                "entity": self._get_entity(r),
                "extracted_info": r['extracted_info'],
                "documents_collected": r['documents_collected']
            }
            for r in results
        ]
        with open(extracted_file, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, indent=2, ensure_ascii=False)
        saved_files['extracted_data'] = str(extracted_file)

        # 3. Save as CSV for easy viewing
        csv_file = session_folder / "entity_data.csv"
        self._save_as_csv(results, csv_file)
        saved_files['csv'] = str(csv_file)

        # 4. Save sources/URLs collected
        sources_file = session_folder / "sources.txt"
        self._save_sources(results, sources_file)
        saved_files['sources'] = str(sources_file)

        return saved_files

    def save_stats(self, stats: Dict, session_folder: Path) -> str:
        """Save session statistics"""
        stats_file = session_folder / "stats.json"

        # Add timestamp to stats
        stats['timestamp'] = datetime.now().isoformat()
        stats['session_folder'] = str(session_folder)

        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

        return str(stats_file)

    def save_session_log(self, log_entries: List[str], session_folder: Path) -> str:
        """Save processing log"""
        log_file = session_folder / "process_log.txt"

        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Research Session Log\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            f.write("\n".join(log_entries))

        return str(log_file)

    def _save_as_csv(self, results: List[Dict], csv_file: Path):
        """Save results as CSV format, using config-defined fields"""
        import csv

        config = load_config()
        entity_schema = get_entity_schema(config)
        extraction_fields = getattr(config, 'EXTRACTION_FIELDS', []) if config else []

        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Build header dynamically from entity fields + extraction fields
            headers = [field.replace('_', ' ').title() for field in entity_schema['fields']]
            headers += [ef['label'] for ef in extraction_fields]
            headers.append('Documents Collected')
            writer.writerow(headers)

            # Data rows
            for result in results:
                entity = self._get_entity(result)
                info = result.get('extracted_info', {})

                row = [entity.get(field, 'Unknown') for field in entity_schema['fields']]
                row += [info.get(ef['key'], 'Unknown') for ef in extraction_fields]
                row.append(result.get('documents_collected', 0))
                writer.writerow(row)

    def _save_sources(self, results: List[Dict], sources_file: Path):
        """Save all collected sources/URLs"""
        config = load_config()

        with open(sources_file, 'w', encoding='utf-8') as f:
            f.write("Research - Collected Sources\n")
            f.write("=" * 80 + "\n\n")

            for result in results:
                entity = self._get_entity(result)
                display = get_entity_display(entity, config)
                f.write(f"\n{display}\n")
                f.write("-" * 80 + "\n")

                sources = result.get('sources', [])
                seen_urls = set()

                for source in sources:
                    url = source.get('url', 'Unknown')
                    if url not in seen_urls:
                        seen_urls.add(url)
                        query_type = source.get('query_type', 'unknown')
                        f.write(f"[{query_type}] {url}\n")

                f.write(f"\nTotal unique sources: {len(seen_urls)}\n\n")

    def create_summary_report(self, results: List[Dict], stats: Dict,
                            session_folder: Path) -> str:
        """Create a human-readable summary report"""
        report_file = session_folder / "SUMMARY.txt"

        config = load_config()
        extraction_fields = getattr(config, 'EXTRACTION_FIELDS', []) if config else []
        dataset_name = getattr(config, 'DATASET_NAME', 'Research') if config else 'Research'

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"RESEARCH SESSION SUMMARY - {dataset_name}\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Session Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Session Folder: {session_folder.name}\n\n")

            # Overall stats
            f.write("-" * 80 + "\n")
            f.write("STATISTICS\n")
            f.write("-" * 80 + "\n")
            entity_count = stats.get('total_entities', stats.get('total_faculty', 0))
            f.write(f"Total Entities Processed: {entity_count}\n")
            f.write(f"Total Documents Collected: {stats.get('total_documents', 0)}\n")
            f.write(f"Unique Sources: {stats.get('unique_sources', 0)}\n")
            f.write(f"Google Searches Used: {stats.get('total_searches', 0)}\n")
            f.write(f"Processing Time: {stats.get('processing_time', 'N/A')}\n\n")

            # Per-entity results
            f.write("-" * 80 + "\n")
            f.write("ENTITY RESULTS\n")
            f.write("-" * 80 + "\n\n")

            for i, result in enumerate(results, 1):
                entity = self._get_entity(result)
                info = result.get('extracted_info', {})
                display = get_entity_display(entity, config)

                f.write(f"{i}. {display}\n")
                f.write(f"   Documents: {result.get('documents_collected', 0)}\n")

                # Write extraction fields dynamically
                for ef in extraction_fields:
                    value = info.get(ef['key'], 'Unknown')
                    f.write(f"   {ef['label']}: {value}\n")

                f.write("\n")

            # Files created
            f.write("-" * 80 + "\n")
            f.write("FILES CREATED\n")
            f.write("-" * 80 + "\n")
            f.write("- full_results.json      (Complete results with sources)\n")
            f.write("- extracted_data.json    (Extracted entity information)\n")
            f.write("- entity_data.csv        (Spreadsheet format)\n")
            f.write("- sources.txt            (All collected URLs)\n")
            f.write("- stats.json             (Session statistics)\n")
            f.write("- process_log.txt        (Processing log)\n")
            f.write("- SUMMARY.txt            (This file)\n\n")

        return str(report_file)

    def list_sessions(self) -> List[Dict]:
        """List all saved research sessions"""
        sessions = []

        for folder in sorted(self.base_dir.iterdir(), reverse=True):
            if folder.is_dir():
                stats_file = folder / "stats.json"
                if stats_file.exists():
                    with open(stats_file, 'r') as f:
                        stats = json.load(f)
                else:
                    stats = {}

                sessions.append({
                    'folder': folder.name,
                    'path': str(folder),
                    'timestamp': stats.get('timestamp', 'Unknown'),
                    'entity_count': stats.get('total_entities', stats.get('total_faculty', 'Unknown')),
                    'documents': stats.get('total_documents', 'Unknown')
                })

        return sessions

    def get_latest_session(self) -> Path:
        """Get the most recent session folder"""
        sessions = sorted(self.base_dir.iterdir(), reverse=True)
        if sessions:
            return sessions[0]
        return None


def print_session_summary(session_folder: Path):
    """Print a quick summary of a session"""
    summary_file = session_folder / "SUMMARY.txt"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            print(f.read())
    else:
        print(f"No summary found in {session_folder}")


if __name__ == "__main__":
    # Demo usage
    manager = ResultsManager()

    print("=" * 80)
    print("Results Manager - Test")
    print("=" * 80)
    print()

    # List existing sessions
    sessions = manager.list_sessions()
    if sessions:
        print(f"Found {len(sessions)} previous session(s):\n")
        for session in sessions:
            print(f"  {session['folder']}")
            print(f"     Entities: {session['entity_count']}, "
                  f"Documents: {session['documents']}")
            print()
    else:
        print("No previous sessions found.")
        print()
        print("Sessions will be saved to: research_results/")
        print()

    print("=" * 80)
