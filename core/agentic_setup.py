"""
Setup agent: turns a CSV + plain-English instructions into pipeline_config.py.

The agent may write exactly one file (pipeline_config.py in the project root);
everything else it produces is returned to the caller for review.

Components (in the order process() runs them):
  parse_csv_dataset   split CSV columns into entity-identifying fields vs. fields to extract,
                      and lift any filled-in target columns into ground truth
  parse_instructions  LLM summarises the instruction text into goal / hints / field guidance
  generate_config     LLM drafts the config module; the fixed parts are then injected by code
  _finalize_config    deterministic post-processing: entities, schema, fields, ground truth, braces
  verify_config       independent LLM review of the final config (results go to the log)
  write_config        syntax-check and save to the single allowed path
"""

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.llm import DEFAULT_MODEL, MODEL_CONFIGS, build_llm_client, chat_json, chat_text

CONFIG_FILENAME = "pipeline_config.py"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Column names containing one of these are treated as identifying the entity
# (what we search for) rather than as a value to extract.
ENTITY_HINTS = ("name", "institution", "university", "organization", "company", "state", "country", "id")


@dataclass
class DatasetSchema:
    entity_type: str            # "person", "institution", "company", ... (only used in prompts)
    entity_fields: List[str]    # original CSV headers that identify an entity
    target_fields: List[Dict]   # [{key, label}] columns the pipeline should fill in
    entities: List[Dict]        # one dict per selected row, keyed by snake_case entity field
    ground_truth: Dict          # {primary value: {field key: known value}} for evaluation


@dataclass
class Instructions:
    goal: str
    hints: List[str]
    field_descriptions: Dict[str, str]


def snake_case(text: str) -> str:
    return re.sub(r"\s+", "_", re.sub(r"[^\w\s]", "", text).strip()).lower()


class AgenticSetup:
    def __init__(self, model: str = None):
        self.model_id = model or DEFAULT_MODEL
        self.logs: List[str] = []
        try:
            self.client = build_llm_client(self.model_id)
        except ValueError as e:
            self.client = None
            print(f"[WARN] {e} — setup agent will fall back to rule-based parsing")

    def log(self, message: str, level: str = "INFO"):
        entry = f"[{level}] {message}"
        self.logs.append(entry)
        print(entry)

    # ── 1. inputs ────────────────────────────────────────────────────────────

    def parse_csv_dataset(self, csv_text: str, start_row: int = 0, end_row: int = None) -> DatasetSchema:
        rows = list(csv.DictReader(csv_text.strip().splitlines()))
        headers = list(rows[0].keys()) if rows else []
        rows = rows[start_row:end_row]

        entity_fields = [h for h in headers if any(k in h.lower() for k in ENTITY_HINTS)][:5]
        # The primary identifier should be the name column when there is one.
        entity_fields.sort(key=lambda h: "name" not in h.lower())
        target_fields = [{"key": snake_case(h), "label": h} for h in headers if h not in entity_fields]

        first = entity_fields[0].lower() if entity_fields else ""
        entity_type = ("person" if "name" in first else "institution" if "institution" in first or "university" in first
                       else "company" if "company" in first else "entity")

        entities, ground_truth = [], {}
        for row in rows:
            entities.append({snake_case(f): row.get(f, "") for f in entity_fields})
            key = row.get(entity_fields[0], "") if entity_fields else ""
            truth = {t["key"]: row[t["label"]] for t in target_fields if row.get(t["label"])}
            if key and truth:
                ground_truth[key] = truth

        self.log(f"CSV: {len(headers)} columns, using {len(rows)} rows "
                 f"(entity fields {entity_fields}, {len(target_fields)} target fields)")
        return DatasetSchema(entity_type, entity_fields, target_fields, entities, ground_truth)

    def parse_instructions(self, text: str) -> Instructions:
        fallback = Instructions(
            goal=text[:200].strip(),
            hints=[l.strip().lstrip("-•*").strip() for l in text.splitlines() if l.strip().startswith(("-", "•", "*"))][:10],
            field_descriptions={},
        )
        if not self.client:
            return fallback
        self.log("Parsing instructions with LLM...")
        prompt = (f"Analyze these research instructions.\n\n{text[:3000]}\n\n"
                  'Return JSON: {"goal": "...", "hints": ["..."], "field_descriptions": {"field": "guidance"}}')
        try:
            data = chat_json(self.client, self.model_id, "Return ONLY a raw JSON object.", prompt, 800)
        except Exception as e:
            self.log(f"Instruction parsing failed: {e} — using rule-based fallback", "WARN")
            return fallback
        if not data:
            return fallback
        return Instructions(data.get("goal", ""), data.get("hints", []), data.get("field_descriptions", {}))

    # ── 2. generation ────────────────────────────────────────────────────────

    def generate_config(self, schema: DatasetSchema, instr: Instructions, extra: str = "") -> Optional[str]:
        """Ask the LLM for the creative parts (descriptions, templates, prompt); inject the rest."""
        if not self.client:
            self.log("No LLM client — cannot generate config", "ERROR")
            return None
        keys = [snake_case(f) for f in schema.entity_fields]
        placeholders = ", ".join("{{%s}}" % k for k in keys)
        prompt = f"""Generate a Python configuration file for a research data pipeline.

DATASET
- Entity type: {schema.entity_type}
- Entity fields: {keys}
- Fields to extract: {[t['key'] for t in schema.target_fields]}
- Sample entities (3 of {len(schema.entities)}): {json.dumps(schema.entities[:3])}

RESEARCH GOAL: {instr.goal[:300]}
HINTS: {json.dumps(instr.hints[:5])}
FIELD GUIDANCE: {json.dumps(dict(list(instr.field_descriptions.items())[:8]))}
{f"EXTRA: {extra[:300]}" if extra else ""}

The file must define, in this order:
1. DATASET_NAME, DATASET_DESCRIPTION (strings)
2. ENTITY_SCHEMA = {{"primary_field": "{keys[0] if keys else 'name'}", "fields": {keys}}}
3. TARGET_ENTITIES = []  (left empty; filled in automatically)
4. EXTRACTION_FIELDS: list of {{"key", "label", "description", "example"}} using EXACTLY the field keys above
5. SEARCH_QUERY_TEMPLATES: 4-6 strings using double-brace placeholders: {placeholders}
6. def get_extraction_prompt(entity: dict, context: str) -> str — per-field guidance, asks for a JSON
   object whose keys are exactly the EXTRACTION_FIELDS keys, "Not available" when absent
7. GROUND_TRUTH = {{}}  (filled in automatically)

Return ONLY valid Python code."""
        self.log(f"Generating config with {MODEL_CONFIGS[self.model_id]['label']}...")
        try:
            code = chat_text(self.client, self.model_id,
                             "You are a Python developer. Return ONLY raw Python code, no fences, no prose.",
                             prompt, 3000)
        except Exception as e:
            self.log(f"Config generation failed: {e}", "ERROR")
            return None
        if not code.strip():
            self.log("LLM returned an empty config", "ERROR")
            return None
        return self._finalize_config(code, schema)

    def _finalize_config(self, code: str, schema: DatasetSchema) -> str:
        """Overwrite the data-bearing sections with values derived from the CSV itself."""
        keys = list(schema.entities[0].keys()) if schema.entities else []
        primary = next((k for k in keys if "name" in k.lower()), keys[0] if keys else "name")

        def set_section(name: str, body_pattern: str, value) -> None:
            """Replace `NAME = <literal>` with our value, or append it if the LLM left it out."""
            nonlocal code
            replacement = f"{name} = {json.dumps(value, indent=4)}"
            pattern = rf"{name}\s*=\s*(?:{body_pattern})"
            if re.search(pattern, code):
                code = re.sub(pattern, lambda _: replacement, code, count=1)
            else:
                code += f"\n\n{replacement}\n"
                self.log(f"{name} was missing from the generated config; appended it", "WARN")

        set_section("ENTITY_SCHEMA", r"\{[^}]*\}", {"primary_field": primary, "fields": keys})
        set_section("TARGET_ENTITIES", r"\[\s*\]|\[[\s\S]*?\n\]", schema.entities)
        if "EXTRACTION_FIELDS" not in code:
            set_section("EXTRACTION_FIELDS", r"(?!)", [
                {"key": t["key"], "label": t["label"], "description": t["label"], "example": ""}
                for t in schema.target_fields
            ])
        if schema.ground_truth:
            set_section("GROUND_TRUTH", r"\{\s*\}|\{[\s\S]*?\n\}", schema.ground_truth)

        # Normalise {field} -> {{field}} inside the templates list so substitution always works.
        code = re.sub(r"SEARCH_QUERY_TEMPLATES\s*=\s*\[[\s\S]*?\]",
                      lambda m: re.sub(r"(?<!\{)\{([a-zA-Z_]\w*)\}(?!\})", r"{{\1}}", m.group(0)), code, count=1)
        self.log(f"Injected {len(schema.entities)} entities and {len(schema.ground_truth)} ground-truth rows")
        return code

    # ── 3. review & save ─────────────────────────────────────────────────────

    def verify_config(self, code: str, schema: DatasetSchema) -> Dict:
        """Second LLM opinion on the generated file; findings are logged for the user."""
        if not self.client:
            return {"valid": True, "issues": []}
        prompt = f"""Review this pipeline configuration for bugs.

```python
{code[:6000]}
```

Expected: entity fields {schema.entity_fields}; extraction keys {[t['key'] for t in schema.target_fields]};
{len(schema.entities)} entities. Check that EXTRACTION_FIELDS keys match the JSON keys requested in
get_extraction_prompt, that templates use {{{{double_braces}}}}, and that the file is valid Python.

Return JSON: {{"valid": true/false, "issues": ["..."]}}"""
        try:
            result = chat_json(self.client, self.model_id, "You are a strict code reviewer. Return only JSON.", prompt, 800)
        except Exception as e:
            self.log(f"Verification skipped: {e}", "WARN")
            return {"valid": True, "issues": []}
        result = result or {"valid": True, "issues": []}
        for issue in result.get("issues", []):
            self.log(f"Review: {issue}", "WARN")
        return result

    @staticmethod
    def write_config(code: str) -> Tuple[bool, str]:
        """Compile-check then write pipeline_config.py. This is the agent's only write path."""
        try:
            compile(code, CONFIG_FILENAME, "exec")
        except SyntaxError as e:
            return False, f"Invalid Python syntax: {e}"
        (PROJECT_ROOT / CONFIG_FILENAME).write_text(code)
        return True, f"Wrote {CONFIG_FILENAME}"

    # ── workflow ─────────────────────────────────────────────────────────────

    def process(self, dataset_csv: str, instructions_text: str, extra: str = "",
                start_row: int = 0, end_row: int = None) -> Dict:
        """Parse -> generate -> verify. Returns {success, config, logs}; the caller decides when to save."""
        self.logs = []
        self.log(f"Starting setup with model {self.model_id}")
        try:
            schema = self.parse_csv_dataset(dataset_csv, start_row, end_row)
            instr = self.parse_instructions(instructions_text)
            code = self.generate_config(schema, instr, extra)
            if not code:
                return {"success": False, "error": "Config generation failed", "logs": self.logs}
            self.verify_config(code, schema)
            self.log("Setup complete")
            return {"success": True, "config": code, "logs": self.logs}
        except Exception as e:
            self.log(f"Setup failed: {e}", "ERROR")
            return {"success": False, "error": str(e), "logs": self.logs}
