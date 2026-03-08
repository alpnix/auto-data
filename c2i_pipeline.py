#!/usr/bin/env python3
"""
Backward-compatibility shim — imports from research_pipeline.py.
Use research_pipeline.py directly for all new code.
"""
from research_pipeline import ResearchPipeline  # noqa: F401

# Legacy alias kept for any old references
C2IPipeline = ResearchPipeline
