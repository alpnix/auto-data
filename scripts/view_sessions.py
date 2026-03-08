#!/usr/bin/env python3
"""
View and compare saved research sessions
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from core.results_manager import ResultsManager


def list_all_sessions():
    """List all saved sessions"""
    root = Path(__file__).resolve().parent.parent
    manager = ResultsManager(base_dir=str(root / "research_results"))
    sessions = manager.list_sessions()

    print("=" * 80)
    print("SAVED RESEARCH SESSIONS")
    print("=" * 80)
    print()

    if not sessions:
        print("No sessions found.")
        print()
        print("Run the pipeline to create a session:")
        print("  python app.py")
        return

    print(f"Found {len(sessions)} session(s):\n")

    for i, session in enumerate(sessions, 1):
        print(f"{i}. Session: {session['folder']}")
        print(f"   Time: {session['timestamp']}")
        print(f"   Entities: {session['entity_count']}")
        print(f"   Documents: {session['documents']}")
        print(f"   Path: {session['path']}")
        print()

    print("=" * 80)
    print()
    print("To view a session summary:")
    print(f"  python scripts/view_sessions.py {sessions[0]['folder']}")
    print()


def view_session(session_name: str):
    """View detailed session information"""
    root = Path(__file__).resolve().parent.parent
    manager = ResultsManager(base_dir=str(root / "research_results"))
    session_path = Path(manager.base_dir) / session_name

    if not session_path.exists():
        print(f"Error: Session '{session_name}' not found")
        return

    # Read and display summary
    summary_file = session_path / "SUMMARY.txt"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            print(f.read())
    else:
        print(f"No summary file found in {session_path}")

    # Show available files
    print("\n" + "=" * 80)
    print("AVAILABLE FILES")
    print("=" * 80)

    files = [
        ("full_results.json", "Complete results with all sources"),
        ("extracted_data.json", "Extracted entity information"),
        ("entity_data.csv", "Spreadsheet format"),
        ("sources.txt", "All collected URLs"),
        ("stats.json", "Session statistics"),
        ("process_log.txt", "Processing log"),
        ("SUMMARY.txt", "This summary")
    ]

    for filename, description in files:
        file_path = session_path / filename
        exists = "Y" if file_path.exists() else "N"
        size = f"({file_path.stat().st_size} bytes)" if file_path.exists() else ""
        print(f"  {exists} {filename:<25} - {description} {size}")

    print("\n" + "=" * 80)
    print()


def compare_sessions(session1_name: str, session2_name: str):
    """Compare two research sessions"""
    root = Path(__file__).resolve().parent.parent
    manager = ResultsManager(base_dir=str(root / "research_results"))
    session1_path = Path(manager.base_dir) / session1_name
    session2_path = Path(manager.base_dir) / session2_name

    if not session1_path.exists() or not session2_path.exists():
        print("Error: One or both sessions not found")
        return

    # Load stats
    stats1_file = session1_path / "stats.json"
    stats2_file = session2_path / "stats.json"

    with open(stats1_file, 'r') as f:
        stats1 = json.load(f)

    with open(stats2_file, 'r') as f:
        stats2 = json.load(f)

    print("=" * 80)
    print("SESSION COMPARISON")
    print("=" * 80)
    print()

    print(f"Session 1: {session1_name}")
    print(f"Session 2: {session2_name}")
    print()

    metrics = [
        ('Total Entities', 'total_entities', 'total_faculty'),
        ('Total Documents', 'total_documents', None),
        ('Unique Sources', 'unique_sources', None),
        ('Google Searches', 'total_searches', None),
        ('Processing Time', 'processing_time', None),
        ('Avg Time/Entity', 'avg_time_per_entity', 'avg_time_per_faculty')
    ]

    print(f"{'Metric':<25} {'Session 1':<20} {'Session 2':<20} {'Difference'}")
    print("-" * 80)

    for entry in metrics:
        label = entry[0]
        key = entry[1]
        fallback_key = entry[2] if len(entry) > 2 else None

        val1 = stats1.get(key, stats1.get(fallback_key, 'N/A') if fallback_key else 'N/A')
        val2 = stats2.get(key, stats2.get(fallback_key, 'N/A') if fallback_key else 'N/A')

        # Calculate difference for numeric values
        diff = ""
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            diff_val = val2 - val1
            diff = f"{'+' if diff_val > 0 else ''}{diff_val}"

        print(f"{label:<25} {str(val1):<20} {str(val2):<20} {diff}")

    print()
    print("=" * 80)
    print()


def export_csv_summary():
    """Export summary of all sessions to CSV"""
    root = Path(__file__).resolve().parent.parent
    manager = ResultsManager(base_dir=str(root / "research_results"))
    sessions = manager.list_sessions()

    if not sessions:
        print("No sessions to export")
        return

    output_file = Path(manager.base_dir) / "all_sessions_summary.csv"

    import csv
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Session', 'Timestamp', 'Entity Count', 'Total Documents',
                        'Unique Sources', 'Total Searches', 'Processing Time'])

        for session in sessions:
            stats_file = Path(manager.base_dir) / session['folder'] / 'stats.json'
            if stats_file.exists():
                with open(stats_file, 'r') as sf:
                    stats = json.load(sf)
                    writer.writerow([
                        session['folder'],
                        session['timestamp'],
                        stats.get('total_entities', stats.get('total_faculty', 'N/A')),
                        stats.get('total_documents', 'N/A'),
                        stats.get('unique_sources', 'N/A'),
                        stats.get('total_searches', 'N/A'),
                        stats.get('processing_time', 'N/A')
                    ])

    print(f"Summary exported to: {output_file}")


def main():
    if len(sys.argv) == 1:
        # No arguments - list all sessions
        list_all_sessions()

    elif len(sys.argv) == 2:
        if sys.argv[1] == '--export':
            # Export summary
            export_csv_summary()
        else:
            # View specific session
            view_session(sys.argv[1])

    elif len(sys.argv) == 3:
        # Compare two sessions
        compare_sessions(sys.argv[1], sys.argv[2])

    else:
        print("Usage:")
        print("  python scripts/view_sessions.py                    - List all sessions")
        print("  python scripts/view_sessions.py <session_name>     - View session details")
        print("  python scripts/view_sessions.py <s1> <s2>          - Compare two sessions")
        print("  python scripts/view_sessions.py --export           - Export summary CSV")


if __name__ == "__main__":
    main()
