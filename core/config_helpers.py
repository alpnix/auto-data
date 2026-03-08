#!/usr/bin/env python3
"""
Config Helpers - Centralized access to pipeline_config with backward-compatible defaults.
All pipeline files import from here instead of duplicating fallback logic.
"""

import importlib
from typing import Dict, List, Optional


def load_config():
    """Load and reload pipeline_config, returning the module or None."""
    try:
        import pipeline_config
        importlib.reload(pipeline_config)
        return pipeline_config
    except Exception:
        return None


def get_entity_schema(config=None) -> Dict:
    """
    Return ENTITY_SCHEMA from config or infer from TARGET_ENTITIES.

    Returns:
        {"primary_field": "name", "fields": ["name", "institution"]}
    """
    if config is None:
        config = load_config()

    # Try explicit ENTITY_SCHEMA first
    schema = getattr(config, 'ENTITY_SCHEMA', None) if config else None
    if schema:
        return schema

    # Infer from TARGET_ENTITIES keys
    entities = getattr(config, 'TARGET_ENTITIES', []) if config else []
    if entities and isinstance(entities[0], dict):
        keys = list(entities[0].keys())
        return {"primary_field": keys[0], "fields": keys}

    # Ultimate fallback
    return {"primary_field": "name", "fields": ["name"]}


def get_primary_field(config=None) -> str:
    """Return the primary field name (e.g., 'name', 'company')."""
    return get_entity_schema(config)["primary_field"]


def get_entity_fields(config=None) -> List[str]:
    """Return all entity field names."""
    return get_entity_schema(config)["fields"]


def get_entity_primary_value(entity: dict, config=None) -> str:
    """Return the primary identifier value for an entity (e.g., the person's name)."""
    primary = get_primary_field(config)
    return entity.get(primary, str(entity))


def get_entity_display(entity: dict, config=None) -> str:
    """Return a human-readable display string for an entity (e.g., 'John Doe - MIT')."""
    schema = get_entity_schema(config)
    parts = [str(entity.get(f, '')) for f in schema["fields"] if entity.get(f)]
    return " - ".join(parts) if parts else str(entity)
