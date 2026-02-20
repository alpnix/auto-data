# Research Pipeline — Status & Reference

## Pipeline Flowchart

```
CSV + Instructions (DOCX/TXT/MD)
         │
         ▼
┌─────────────────────┐
│   Setup UI (:8080)  │  Drop files → agentic_setup.py parses + generates pipeline_config.py
│   /setup            │  → verify_config() (separate LLM call) → auto_fix_config()
└─────────────────────┘
         │
         ▼ pipeline_config.py written
┌─────────────────────────────────────────────────────────────┐
│                  Research Pipeline (:8080/research)          │
│                                                             │
│  For each entity in TARGET_ENTITIES:                        │
│  1. generate_search_queries()  ← SEARCH_QUERY_TEMPLATES     │
│     ({{field}} substitution — both single & double brace)   │
│  2. search() → Serper (primary) → Google CSE (fallback)     │
│  3. is_url_worth_scraping() filter → scrape_content()       │
│  4. store_in_vector_db() → Milvus Lite (local .db file)     │
│  5. extract_entity_info() → LLM (DeepSeek or GPT)          │
│     + confidence scores per field                           │
│  6. cross_validate_with_profile() → re-scrape profile URL   │
│  7. retry loop → generate_retry_queries() for missing fields│
│                                                             │
│  → results saved to research_results/<timestamp>/           │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────┐
│  Results             │  entity_data.csv, full_results.json,
│  research_results/   │  extracted_data.json, SUMMARY.txt,
│                      │  sources.txt, stats.json, process_log.txt
└──────────────────────┘
```

## Tech Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| Web server | Flask | SSE for real-time progress |
| Search | Serper API / Google CSE | Serper preferred (cheaper) |
| Scraping | requests + BeautifulSoup | URL quality filter pre-screens |
| Embeddings | BAAI/bge-small-en-v1.5 | 384-dim via sentence-transformers |
| Vector DB | Milvus Lite | Local file `milvus_research.db` |
| LLM | GPT-5 Nano / DeepSeek / others | Switchable via UI dropdown; default: gpt-5-nano |
| Config agent | Same model registry | agentic_setup.py |
| Evaluation | qa-metrics (F1/EM) + PEDANTS | Optional, graceful fallback |

## Key Files

```
setup_server.py              — Main server (port 8080), all routes
faculty_research_pipeline.py — Core pipeline (ResearchPipeline class)
agentic_setup.py             — Config generation + verification agent
pipeline_config.py           — Generated config (entities, fields, prompts, templates)
config_helpers.py            — Shared helpers for reading pipeline_config
results_manager.py           — Saves session results in all formats
evaluation.py                — F1/EM/PEDANTS scoring against ground truth
templates/setup.html         — Setup UI (drop CSV + instructions)
templates/research.html      — Research dashboard (model selector, live progress)
```

## Environment Variables (.env)

```
GOOGLE_CUSTOM_SEARCH_KEY=   # Google CSE API key
GOOGLE_CSE_ID=              # Google Custom Search Engine ID
DEEPSEEK_API_KEY=           # DeepSeek API key
OPENAI_API_KEY=             # OpenAI API key (for GPT models)
SERPER_API_KEY=             # Serper.dev key (optional, preferred search)
```

## Quick Start

```bash
pip install -r requirements.txt
# fill in .env
python setup_server.py      # → http://localhost:8080
```

1. Go to `/setup` → drop CSV + instructions → Generate → Apply
2. Go to `/research` → pick model → Start Research

---

## What Works ✅

- **Agentic config generation**: CSV + instructions → full `pipeline_config.py` in one LLM call
- **Post-generation verification**: separate LLM call reviews the config for bugs before use
- **Template substitution**: `{{name}}` / `{name}` both handled correctly
- **Entity schema**: `primary_field` correctly set to `name` (not institution)
- **URL quality filter**: rejects PDFs, data dumps, irrelevant domains before scraping
- **Field-specific extraction prompts**: each field has targeted instructions (where to look, patterns)
- **Confidence scoring**: LLM returns 0.0–1.0 per field; low-confidence fields trigger retry
- **Retry logic**: targeted re-search for fields that came back "Not available"
- **Cross-field validation**: if profile URL found → re-scrape it focused on missing fields
- **Model switching**: DeepSeek ↔ GPT switchable via dropdown before each run
- **Ground truth alignment**: keyed by primary field (name), shown correctly in UI
- **Results saved**: CSV + JSON + summary per session under `research_results/`
- **Evaluation**: F1/EM scores shown inline when ground truth provided in config

## What Is Partially Working ⚠️

- **Gender extraction**: relies on pronouns appearing in scraped content — works when bio is scraped, misses when only snippets available
- **Email extraction**: works when profile page is scraped; misses when search snippets don't include it
- **PhD/Bachelor's**: works well when CV or profile page is scraped; misses for less-indexed faculty

## Known Limitations / Next Steps 🔲

- **GROUND_TRUTH in pipeline_config.py**: currently empty — populate manually or from CSV to enable accuracy scoring
- **Search quota**: Google CSE free tier = 100/day; Serper is cheaper alternative
- **Milvus collection reuse**: existing collection is reused across runs — delete `milvus_research.db` to reset
- **faculty_data_extractor.py / faculty_dashboard.py**: legacy files from earlier iteration, not used by main server
- **Playwright**: in requirements.txt but not yet used — future enhancement for JS-heavy pages
- **Batch size**: currently processes all entities sequentially; no parallelism
- **PEDANTS metric**: may fail on some sklearn versions — falls back to F1/EM gracefully

---

## Pipeline Config Structure (pipeline_config.py)

```python
DATASET_NAME = "..."
DATASET_DESCRIPTION = "..."

ENTITY_SCHEMA = {"primary_field": "name", "fields": ["name", "institution"]}

TARGET_ENTITIES = [{"name": "...", "institution": "..."}, ...]

EXTRACTION_FIELDS = [
    {"key": "phd_institution", "label": "PhD Institution",
     "description": "...", "example": "...", "required": False},
    ...
]

SEARCH_QUERY_TEMPLATES = [
    "{{name}} {{institution}} faculty profile",
    ...
]

def get_extraction_prompt(entity: dict, context: str) -> str:
    # Field-specific instructions per field
    ...

GROUND_TRUTH = {
    "Entity Name": {"field_key": "expected_value", ...},
    ...
}
```
