"""
Read access to the generated pipeline_config.py.

The config file is rewritten by the setup agent while the server is running, so
every reader calls load_config() to get a freshly reloaded module instead of
caching it at import time.

Components:
  load_config               reload and return the pipeline_config module (None if missing/broken)
  get_entity_schema         {"primary_field": ..., "fields": [...]} with inference fallback
  get_entity_primary_value  the value that identifies one entity (e.g. the person's name)
  get_entity_display        human-readable "field1 - field2" label for logs and reports
"""

import importlib
from typing import Dict, Optional


def load_config():
    """Import (or re-import) pipeline_config from the project root."""
    try:
        import pipeline_config
        return importlib.reload(pipeline_config)
    except Exception:
        return None


def get_entity_schema(config=None) -> Dict:
    """Explicit ENTITY_SCHEMA if present, otherwise inferred from the first entity's keys."""
    config = config or load_config()
    schema = getattr(config, "ENTITY_SCHEMA", None)
    if schema:
        return schema
    entities = getattr(config, "TARGET_ENTITIES", [])
    if entities:
        keys = list(entities[0].keys())
        primary = next((k for k in keys if "name" in k.lower()), keys[0])
        return {"primary_field": primary, "fields": keys}
    return {"primary_field": "name", "fields": ["name"]}


def get_entity_primary_value(entity: Dict, config=None) -> str:
    return str(entity.get(get_entity_schema(config)["primary_field"], entity))


def get_entity_display(entity: Dict, config=None) -> str:
    parts = [str(entity[f]) for f in get_entity_schema(config)["fields"] if entity.get(f)]
    return " - ".join(parts) or str(entity)
