# c2i-data-collection

Dataset-agnostic research pipeline: **CSV + plain-English instructions → auto-generated config → web search → vector store → LLM extraction → scored output.**

Give it a list of entities (people, companies, universities, …) and a description of what you want to know about each one. An LLM writes the pipeline configuration, the pipeline searches the web, stores what it finds in a local vector database, and extracts the requested fields with retrieval-augmented generation. If your CSV already contains some answers, they are used as ground truth and every extraction is scored.

---

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # add at least one search key and one LLM key (see below)
python app.py                 # http://localhost:8080
```

1. **`/setup`** — drop your CSV and instruction text, pick a model, click *Generate*. Review the generated `pipeline_config.py`, then *Apply & Save*.
2. **`/research`** — pick a model and *Start*. Progress streams live; results land in `research_results/<timestamp>/`.

The repo ships with a working `pipeline_config.py` for the sample faculty dataset in `INFERENCE_DATA/`, so step 2 works immediately.

### Environment variables

```env
# web search
SERPER_API_KEY=...             # primary
GOOGLE_CUSTOM_SEARCH_KEY=...   # fallback
GOOGLE_CSE_ID=...

# LLM via Amazon Bedrock (default): any standard AWS credential source works
AWS_PROFILE=admin
AWS_REGION=us-east-1
# AWS_BEARER_TOKEN_BEDROCK=... # alternative: a Bedrock API key instead of a profile

# LLM via vendor APIs (optional)
DEEPSEEK_API_KEY=...
OPENAI_API_KEY=...
```

The model dropdowns only list models whose credentials are present. `SERPER` and `DEEPSEEK_API` are accepted as alternative names.

---

## How a run works

For each entity in `TARGET_ENTITIES`:

| Step | What happens | Where |
|---|---|---|
| 1. Query | Fill `SEARCH_QUERY_TEMPLATES` placeholders with the entity's fields | `research_pipeline.generate_search_queries` |
| 2. Search | Serper, falling back to Google CSE; 3 results per query | `research_pipeline.search` |
| 3. Scrape | Fetch full pages for profile / site-specific hits (max 5 per entity), skip PDFs and file dumps | `research_pipeline.scrape_content` |
| 4. Store | Embed snippets and pages with BGE-small, insert into Milvus Lite tagged by entity | `research_pipeline.store_in_vector_db` |
| 5. Extract | One vector query per field → deduplicated context → one LLM call → JSON with per-field confidence | `research_pipeline.extract_entity_info` |
| 6. Retry | Fields that came back empty or low-confidence get a second, focused retrieval + LLM pass (no new web searches) | `research_pipeline.vdb_retry_missing_fields` |
| 7. Score | If ground truth exists for the entity, exact match and token F1 per field | `evaluation.Evaluator` |
| 8. Save | JSON, CSV, sources, stats, summary | `results_manager.ResultsManager` |

---

## Project layout

```
app.py                     Flask server: pages + JSON/SSE API (see module docstring for routes)
pipeline_config.py         Generated config: entities, fields, templates, prompt, ground truth

core/
  llm.py                   Model registry (Bedrock + OpenAI-compatible), clients, chat_text / chat_json
  config_helpers.py        Reload pipeline_config and read the entity schema from it
  agentic_setup.py         CSV + instructions → pipeline_config.py (LLM drafts, code injects data)
  research_pipeline.py     Search → scrape → Milvus → RAG extraction → retry
  evaluation.py            Exact-match and token-F1 scoring (pure Python)
  results_manager.py       Writes one research_results/<timestamp>/ folder per run

templates/
  overview.html            Landing page describing the system
  setup.html               Setup UI
  research.html            Live dashboard (SSE) with per-field scores

scripts/
  view_sessions.py         List / show / export saved runs
  compare_results.py       Re-score a saved full_results.json against GROUND_TRUTH

INFERENCE_DATA/            Sample CSV + instruction text used to build the shipped config
```

Each module's docstring lists its components and what they do.

---

## pipeline_config.py contract

The generated file must define:

```python
DATASET_NAME, DATASET_DESCRIPTION          # strings
ENTITY_SCHEMA  = {"primary_field": "name", "fields": ["name", "institution"]}
TARGET_ENTITIES = [{"name": ..., "institution": ...}, ...]
EXTRACTION_FIELDS = [{"key": "phd", "label": "PhD", "description": ..., "example": ...}, ...]
SEARCH_QUERY_TEMPLATES = ["{{name}} {{institution}} profile", ...]
def get_extraction_prompt(entity: dict, context: str) -> str: ...
GROUND_TRUTH = {"<primary value>": {"<field key>": "<expected>", ...}, ...}
```

`EXTRACTION_FIELDS` keys, the JSON keys requested by `get_extraction_prompt`, and `GROUND_TRUTH` keys must all match. The setup agent enforces the data-bearing sections (entities, schema, fields, ground truth) in code after the LLM has drafted the rest, and appends any section the model left out.

---

## Output

```
research_results/<timestamp>/
  full_results.json     everything, including every source document
  extracted_data.json   entity + extracted fields only
  entity_data.csv       one row per entity
  sources.txt           unique URLs per entity
  stats.json            counts, searches, timing
  SUMMARY.txt           readable recap
```

---

## Models

| Model id | Provider | Credentials | Notes |
|---|---|---|---|
| `us.amazon.nova-micro-v1:0` | Bedrock | AWS | Cheapest; fine for extraction of short pages |
| `us.amazon.nova-lite-v1:0` | Bedrock | AWS | **Default.** Good cost/quality for extraction |
| `us.amazon.nova-pro-v1:0` | Bedrock | AWS | Stronger Nova tier |
| `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Bedrock | AWS | Recommended for the Setup page (writes clean configs) |
| `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | Bedrock | AWS | Highest quality, highest cost |
| `deepseek.v3.2`, `us.meta.llama3-3-70b-instruct-v1:0` | Bedrock | AWS | Open-weight alternatives |
| `deepseek-chat` | DeepSeek | `DEEPSEEK_API_KEY` | Direct API |
| `gpt-4o-mini`, `gpt-5-nano` | OpenAI | `OPENAI_API_KEY` | Direct API |

Bedrock calls use the Converse API through boto3 and the normal AWS credential chain. If `AWS_BEARER_TOKEN_BEDROCK` is set, Bedrock's OpenAI-compatible endpoint is used instead. Setup and Research can use different models; the setup agent validates the generated prompt function and substitutes a standard one if a weaker model produces something unusable. Add a model by appending to `MODEL_CONFIGS` in `core/llm.py`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Dropdown shows "No models available" | No AWS credentials (`AWS_PROFILE`) and no vendor key in `.env` |
| `AccessDeniedException` from Bedrock | Enable the model in the Bedrock console for your region, or pick another model |
| Empty or truncated generated config | Switch the setup model to Claude Haiku 4.5 |
| Milvus errors after changing schema | Delete `milvus_research.db` |
| No search results | Check `SERPER_API_KEY`, or set the Google CSE pair |
| Scores not shown on the dashboard | Entity's primary value must be a key in `GROUND_TRUTH` |
