"""
Persist one research run to research_results/<timestamp>/.

Files written per session:
  full_results.json    everything process_entity() returned, including sources
  extracted_data.json  entity + extracted fields only
  entity_data.csv      spreadsheet view: entity fields, extraction fields, doc count
  sources.txt          unique URLs per entity
  stats.json           run statistics (counts, searches, timing)
  SUMMARY.txt          human-readable recap of stats and per-entity values

Components:
  ResultsManager.create_session_folder  new timestamped directory
  ResultsManager.save_session           writes all files above; returns {name: path}
  ResultsManager.list_sessions          summaries of past runs for the UI / CLI
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from core.config_helpers import get_entity_display, get_entity_schema, load_config


class ResultsManager:
    def __init__(self, base_dir: str = "research_results"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

    def create_session_folder(self) -> Path:
        folder = self.base_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        folder.mkdir(exist_ok=True)
        return folder

    def save_session(self, results: List[Dict], stats: Dict, folder: Path) -> Dict[str, str]:
        config = load_config()
        fields = getattr(config, "EXTRACTION_FIELDS", []) or []
        entity_fields = get_entity_schema(config)["fields"]
        stats = {**stats, "timestamp": datetime.now().isoformat(), "session_folder": str(folder)}

        (folder / "full_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
        (folder / "extracted_data.json").write_text(json.dumps(
            [{"entity": r["entity"], "extracted_info": r["extracted_info"]} for r in results],
            indent=2, ensure_ascii=False))
        (folder / "stats.json").write_text(json.dumps(stats, indent=2))

        with open(folder / "entity_data.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([e.replace("_", " ").title() for e in entity_fields] + [x["label"] for x in fields] + ["Documents"])
            for r in results:
                w.writerow([r["entity"].get(e, "") for e in entity_fields]
                           + [r["extracted_info"].get(x["key"], "") for x in fields]
                           + [r["documents_collected"]])

        with open(folder / "sources.txt", "w", encoding="utf-8") as f:
            for r in results:
                f.write(f"\n{get_entity_display(r['entity'], config)}\n{'-' * 60}\n")
                for url in dict.fromkeys(s["url"] for s in r["sources"]):
                    f.write(f"{url}\n")

        with open(folder / "SUMMARY.txt", "w", encoding="utf-8") as f:
            f.write(f"RESEARCH SESSION — {getattr(config, 'DATASET_NAME', 'Research')}\n{'=' * 60}\n")
            for k, v in stats.items():
                f.write(f"{k}: {v}\n")
            for i, r in enumerate(results, 1):
                f.write(f"\n{i}. {get_entity_display(r['entity'], config)}  ({r['documents_collected']} docs)\n")
                for x in fields:
                    f.write(f"   {x['label']}: {r['extracted_info'].get(x['key'], '')}\n")

        return {p.stem: str(p) for p in folder.iterdir()}

    def list_sessions(self) -> List[Dict]:
        sessions = []
        for folder in sorted(self.base_dir.iterdir(), reverse=True):
            stats_file = folder / "stats.json"
            if not folder.is_dir() or not stats_file.exists():
                continue
            stats = json.loads(stats_file.read_text())
            sessions.append({
                "folder": folder.name,
                "path": str(folder),
                "timestamp": stats.get("timestamp"),
                "entity_count": stats.get("total_entities"),
                "documents": stats.get("total_documents"),
                "searches": stats.get("total_searches"),
                "processing_time": stats.get("processing_time"),
            })
        return sessions
