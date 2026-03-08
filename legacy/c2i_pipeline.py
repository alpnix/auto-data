#!/usr/bin/env python3
"""
Backward-compatibility shim — imports from core.research_pipeline.
Use core.research_pipeline or run app.py for all new code.
"""
from core.research_pipeline import ResearchPipeline  # noqa: F401

# Legacy alias kept for any old references
C2IPipeline = ResearchPipeline
