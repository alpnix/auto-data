# mini-rag-c2i

**Dataset-agnostic RAG pipeline: drop a CSV + instructions → auto-configure → search → vector DB → LLM extraction → structured output**

---

## What it does

Given any CSV of entities and a plain-English instruction document, the pipeline:

1. **Auto-configures itself** — an LLM reads your instructions and generates search templates, extraction fields, and prompts
2. **Searches the web** — fires targeted queries per entity using Serper (primary) / Google CSE (fallback)
3. **Scrapes & stores** — cleans page content and stores everything in a local vector database (Milvus Lite)
4. **Extracts via RAG** — for each field, semantically retrieves the most relevant chunks from the VDB, then calls an LLM to extract structured JSON
5. **Retries from VDB** — missing/low-confidence fields trigger a second round of targeted VDB queries, no extra web searches
6. **Outputs results** — JSON, CSV, evaluation scores (F1/EM vs ground truth if provided)

Works with any entity type: people, companies, universities, products, etc.

---

## Quick start

```bash
# 1. Install dependencies
pip install flask sentence-transformers pymilvus openai python-dotenv \
            beautifulsoup4 lxml requests qa-metrics

# 2. Set environment variables
cp .env.example .env   # then fill in your keys

# 3. Start the server
python setup_server.py

# 4. Open browser
open http://localhost:8080
```

---

## Environment variables

```env
# Required — at least one search API
SERPER=your_serper_api_key                  # primary (cheaper)
GOOGLE_CUSTOM_SEARCH_KEY=your_google_key    # fallback
GOOGLE_CSE_ID=your_cse_id

# Required — at least one LLM
DEEPSEEK_API=your_deepseek_key              # recommended (large output window)
OPENAI_API_KEY=your_openai_key              # optional (gpt-5-nano, gpt-4o-mini)
```

---

## Workflow

### Step 1 — Setup (`/` or `/setup`)

1. Drop your **CSV** (entities + any ground truth columns)
2. Drop your **instructions** (plain text describing what to extract)
3. Select a model from the dropdown (default: DeepSeek Chat)
4. Click **Generate Configuration** — the agent:
   - Parses the CSV schema and detects entity fields vs target fields
   - Calls the LLM to generate `pipeline_config.py` (search templates, extraction fields, prompts)
   - Verifies and auto-fixes the config
   - Injects all entities and ground truth
5. Review the generated config, then click **Apply & Save**

### Step 2 — Research (`/research`)

1. Select a model from the dropdown
2. Click **Start** — the pipeline runs per entity:
   - Generates search queries from templates
   - Searches + scrapes URLs (filtered for quality)
   - Stores all documents in Milvus with embeddings
   - RAG extraction: per-field VDB query → LLM → structured JSON + confidence scores
   - VDB retry for any missing/low-confidence fields
3. Results stream live; saved to `research_results/<session>/`

---

## File structure

```
setup_server.py          — main Flask server (/ /setup /research /api/*)
research_pipeline.py     — core pipeline: search, scrape, VDB, RAG extraction
agentic_setup.py         — agent that generates pipeline_config.py from inputs
pipeline_config.py       — auto-generated config (entities, fields, prompts)
config_helpers.py        — helpers for loading config generically
results_manager.py       — saves results to JSON/CSV/summary
evaluation.py            — F1/EM evaluation against ground truth

c2i_pipeline.py          — backward-compat shim → research_pipeline
c2i_dashboard.py         — legacy standalone dashboard (SSE)
c2i_data_extractor.py    — legacy feature-by-feature extractor
c2i_setup_check.py       — credential + dependency checker

templates/
  overview.html          — presentation/overview page (also at /)
  setup.html             — setup UI
  research.html          — research dashboard UI

INFERENCE_DATA/          — sample input files for testing
  *.csv                  — sample entity dataset
  *.txt                  — sample instructions document
```

---

## Key components

### `research_pipeline.py`

- `ResearchPipeline` — main class
- `generate_search_queries()` — builds queries from `SEARCH_QUERY_TEMPLATES` with `{{field}}` substitution
- `search()` — Serper first, Google CSE fallback
- `scrape_content()` — fetches + cleans HTML; filters bad extensions and irrelevant domains
- `store_in_vector_db()` — embeds with BGE and stores in Milvus, flushes immediately
- `semantic_search_vdb()` — COSINE similarity search filtered by `entity_id`
- `extract_entity_info()` — **RAG**: per-field VDB query → deduplicated context → LLM → JSON + confidence
- `vdb_retry_missing_fields()` — second-pass VDB queries for missing/low-confidence fields
- `process_entity()` — orchestrates all phases; `on_event` callback for SSE streaming

### `agentic_setup.py`

- `parse_csv_dataset()` — detects entity vs target fields, builds ground truth
- `parse_instructions()` — LLM extracts goal/hints/strategies; rule-based fallback
- `generate_config()` — LLM generates full `pipeline_config.py`; entities injected post-generation
- `verify_config()` — separate LLM call reviews the generated config
- `auto_fix_config()` — fixes common issues (brace syntax, field name mismatches)

### `pipeline_config.py` (auto-generated)

```python
DATASET_NAME = "..."
ENTITY_SCHEMA = {"primary_field": "name", "fields": ["name", "institution"]}
TARGET_ENTITIES = [{"name": "...", "institution": "..."}, ...]
EXTRACTION_FIELDS = [{"key": "phd", "label": "PhD Institution", ...}, ...]
SEARCH_QUERY_TEMPLATES = ["{{name}} {{institution}} profile", ...]
def get_extraction_prompt(entity: dict, context: str) -> str: ...
GROUND_TRUTH = {"Entity Name": {"phd": "Harvard", ...}, ...}
EVALUATION_CONFIG = {"metric": "f1", "threshold": 0.7}
```

---

## Model support

| Model | Provider | Token param | Best for |
|---|---|---|---|
| `deepseek-chat` | DeepSeek | `max_tokens` | Config generation, extraction (large output) |
| `gpt-5-nano` | OpenAI | `max_completion_tokens` | Fast extraction (small prompts only) |
| `gpt-4o-mini` | OpenAI | `max_tokens` | Balanced |

Both `DEEPSEEK_API` and `DEEPSEEK_API_KEY` env var names are supported.

---

## Output

Each session saves to `research_results/<timestamp>/`:

```
full_results.json     — complete results with sources
extracted_data.json   — extracted fields only
entity_data.csv       — spreadsheet format
sources.txt           — all collected URLs
stats.json            — session statistics
process_log.txt       — processing log
SUMMARY.txt           — human-readable summary
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `max_tokens` error with gpt-5-nano | Already handled — uses `max_completion_tokens` automatically |
| Empty LLM response / `finish_reason: length` | Switch to DeepSeek for config generation |
| `DEEPSEEK_API_KEY not found` | Use `DEEPSEEK_API` in your `.env` (both names supported) |
| Milvus connection error | Delete `milvus_research.db` to reset |
| No search results | Check `SERPER` key; fallback to `GOOGLE_CUSTOM_SEARCH_KEY` + `GOOGLE_CSE_ID` |
| Config generation produces 0 chars | Prompt too large for model — use DeepSeek |
