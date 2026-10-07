"""
Inspect saved runs in research_results/.

  python scripts/view_sessions.py                 list all sessions
  python scripts/view_sessions.py <session>       print that session's SUMMARY.txt
  python scripts/view_sessions.py --export        write all_sessions_summary.csv
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.results_manager import ResultsManager  # noqa: E402

manager = ResultsManager(base_dir=str(Path(__file__).resolve().parent.parent / "research_results"))
COLUMNS = ["folder", "timestamp", "entity_count", "documents", "searches", "processing_time"]


def list_sessions():
    sessions = manager.list_sessions()
    if not sessions:
        print("No sessions yet. Run `python app.py` and start a research run.")
        return
    print(f"{'Session':<18}{'Entities':>9}{'Docs':>7}{'Searches':>10}  Time")
    for s in sessions:
        print(f"{s['folder']:<18}{s['entity_count']:>9}{s['documents']:>7}{s['searches']:>10}  {s['processing_time']}")


def show_session(name: str):
    summary = manager.base_dir / name / "SUMMARY.txt"
    print(summary.read_text() if summary.exists() else f"No session named {name}")


def export_csv():
    out = manager.base_dir / "all_sessions_summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(manager.list_sessions())
    print(f"Wrote {out}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    export_csv() if arg == "--export" else show_session(arg) if arg else list_sessions()
