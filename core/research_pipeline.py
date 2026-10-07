"""
Research pipeline: web search -> scrape -> vector store -> RAG extraction.

Everything dataset-specific (entities, fields, search templates, prompt) comes
from pipeline_config.py, so the same code runs for people, companies, etc.

Components (in the order process_entity() uses them):
  generate_search_queries   fill SEARCH_QUERY_TEMPLATES with the entity's field values
  search                    Serper first, Google Custom Search as fallback
  scrape_content            fetch a URL, strip boilerplate, keep the first 5000 chars
  store_in_vector_db        embed texts with BGE and insert into Milvus Lite
  semantic_search_vdb       cosine-similarity retrieval scoped to one entity
  extract_entity_info       per-field retrieval -> one LLM call -> JSON + confidence
  vdb_retry_missing_fields  second focused LLM pass for empty / low-confidence fields
"""

import json
import os
import time
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
from sentence_transformers import SentenceTransformer

from core.config_helpers import get_entity_display, get_entity_primary_value, get_entity_schema, load_config
from core.llm import DEFAULT_MODEL, MODEL_CONFIGS, build_llm_client, chat_json

load_dotenv()

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
MILVUS_URI = "./milvus_research.db"
MISSING_MARKERS = ("not available", "unable to retrieve")

# URL suffixes that are never worth fetching as HTML.
SKIP_EXTENSIONS = (
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".zip", ".gz",
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".mp3", ".mp4", ".css", ".js",
    ".json", ".xml", ".csv", ".txt",
)
# Path fragments typical of repository dumps and raw file viewers.
SKIP_PATTERNS = ("/bitstream/", "/viewcontent.cgi", "/afs/")


def _is_missing(value) -> bool:
    return isinstance(value, str) and (not value.strip() or value.lower().strip() in MISSING_MARKERS)


class ResearchPipeline:
    def __init__(self, model: str = DEFAULT_MODEL, collection_name: str = "research_data"):
        self.model = model
        self.collection_name = collection_name
        self.embedder: Optional[SentenceTransformer] = None
        self.collection: Optional[Collection] = None
        self.llm = None
        self.serper_key = os.getenv("SERPER_API_KEY") or os.getenv("SERPER")
        self.google_key = os.getenv("GOOGLE_CUSTOM_SEARCH_KEY")
        self.google_cse = os.getenv("GOOGLE_CSE_ID")
        self.search_count = 0

    # ── lifecycle ────────────────────────────────────────────────────────────

    def initialize(self):
        """Load the embedding model, the LLM client and the Milvus collection."""
        print(f"[INIT] Loading embedding model {EMBEDDING_MODEL}...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        try:
            self.llm = build_llm_client(self.model)
            print(f"[OK] LLM: {MODEL_CONFIGS[self.model]['label']}")
        except ValueError as e:
            print(f"[WARN] {e} — extraction will be skipped")
        self._init_milvus()

    def _init_milvus(self):
        connections.connect("default", uri=MILVUS_URI)
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
        else:
            schema = CollectionSchema([
                FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema("entity_id", DataType.VARCHAR, max_length=200),
                FieldSchema("text", DataType.VARCHAR, max_length=5000),
                FieldSchema("url", DataType.VARCHAR, max_length=500),
                FieldSchema("query_type", DataType.VARCHAR, max_length=100),
                FieldSchema("embedding", DataType.FLOAT_VECTOR,
                            dim=self.embedder.get_sentence_embedding_dimension()),
            ])
            self.collection = Collection(self.collection_name, schema)
            self.collection.create_index("embedding", {
                "metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 128},
            })
        self.collection.load()
        print(f"[OK] Milvus collection '{self.collection_name}' ready")

    def cleanup(self):
        try:
            if self.collection:
                self.collection.release()
            connections.disconnect("default")
        except Exception:
            pass

    # ── search ───────────────────────────────────────────────────────────────

    def generate_search_queries(self, entity: Dict) -> List[Dict]:
        """Substitute {{field}} / {field} placeholders in each template with entity values."""
        config = load_config()
        templates = getattr(config, "SEARCH_QUERY_TEMPLATES", None) or [
            "{{%s}} profile" % get_entity_schema(config)["primary_field"],
        ]
        queries = []
        for i, template in enumerate(templates):
            query = template
            for key, value in entity.items():
                query = query.replace("{{%s}}" % key, str(value)).replace("{%s}" % key, str(value))
            queries.append({"query": query, "type": self._query_type(query, i)})
        return queries

    @staticmethod
    def _query_type(query: str, index: int) -> str:
        """Label a query so process_entity() knows which result pages deserve a full scrape."""
        q = query.lower()
        if index == 0 or "profile" in q or "cv" in q or "curriculum" in q:
            return "profile"
        if "site:" in q:
            return "site_specific"
        if "email" in q or "contact" in q:
            return "contact"
        return f"search_{index}"

    def search(self, query: str, num: int = 3) -> List[Dict]:
        """Return [{title, url, description}]; Serper is primary, Google CSE is the backup."""
        results = self._search_serper(query, num) if self.serper_key else []
        if not results and self.google_key and self.google_cse:
            results = self._search_google(query, num)
        return results

    def _search_serper(self, query: str, num: int) -> List[Dict]:
        try:
            r = requests.post("https://google.serper.dev/search", timeout=10,
                              headers={"X-API-KEY": self.serper_key}, json={"q": query, "num": num})
            r.raise_for_status()
            self.search_count += 1
            return [{"title": x.get("title", ""), "url": x.get("link", ""), "description": x.get("snippet", "")}
                    for x in r.json().get("organic", [])]
        except Exception as e:
            print(f"[ERROR] Serper: {e}")
            return []

    def _search_google(self, query: str, num: int) -> List[Dict]:
        try:
            r = requests.get("https://www.googleapis.com/customsearch/v1", timeout=10,
                             params={"key": self.google_key, "cx": self.google_cse, "q": query, "num": num})
            r.raise_for_status()
            self.search_count += 1
            return [{"title": x.get("title", ""), "url": x.get("link", ""), "description": x.get("snippet", "")}
                    for x in r.json().get("items", [])]
        except Exception as e:
            print(f"[ERROR] Google CSE: {e}")
            return []

    # ── scraping ─────────────────────────────────────────────────────────────

    def scrape_content(self, url: str) -> Optional[str]:
        """Visible page text (max 5000 chars) or None if the URL is not worth keeping."""
        low = url.lower()
        if low.endswith(SKIP_EXTENSIONS) or any(p in low for p in SKIP_PATTERNS):
            return None
        try:
            r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            if "html" not in r.headers.get("content-type", "").lower():
                return None
            soup = BeautifulSoup(r.content, "lxml")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = " ".join(soup.get_text(separator=" ", strip=True).split())[:5000]
            return text if len(text) >= 200 else None
        except Exception as e:
            print(f"[WARN] Scrape failed for {url[:60]}: {type(e).__name__}")
            return None

    # ── vector store ─────────────────────────────────────────────────────────

    def _embed(self, text: str) -> List[float]:
        return self.embedder.encode(text, normalize_embeddings=True).tolist()

    def store_in_vector_db(self, entity_id: str, docs: List[Dict]):
        """Insert documents ({text, url, query_type}) for one entity and flush."""
        if not docs:
            return
        self.collection.insert([
            [entity_id] * len(docs),
            [d["text"] for d in docs],
            [d["url"] for d in docs],
            [d["query_type"] for d in docs],
            [self._embed(d["text"]) for d in docs],
        ])
        self.collection.flush()
        print(f"[OK] Stored {len(docs)} documents")

    def semantic_search_vdb(self, entity_id: str, query: str, top_k: int = 5) -> List[Dict]:
        """Top-k documents for `query`, restricted to the given entity."""
        try:
            hits = self.collection.search(
                data=[self._embed(query)], anns_field="embedding", limit=top_k,
                param={"metric_type": "COSINE", "params": {"nprobe": 16}},
                expr=f'entity_id == "{entity_id}"', output_fields=["text", "url", "query_type"],
            )[0]
            return [{"text": h.entity.get("text"), "url": h.entity.get("url"),
                     "query_type": h.entity.get("query_type"), "score": round(h.score, 4)} for h in hits]
        except Exception as e:
            print(f"[WARN] VDB search failed: {e}")
            return []

    # ── extraction (RAG) ─────────────────────────────────────────────────────

    def _retrieve_context(self, entity: Dict, entity_id: str, fields: List[Dict],
                          top_k: int, max_docs: int) -> str:
        """One VDB query per field, deduplicated by URL, formatted as numbered sources."""
        name = get_entity_primary_value(entity)
        seen, docs = set(), []
        for f in fields:
            query = f"{name} {f.get('label', '')} {f.get('description', '')}"
            for d in self.semantic_search_vdb(entity_id, query, top_k):
                if d["url"] not in seen:
                    seen.add(d["url"])
                    docs.append(d)
        print(f"[RAG] {len(docs)} unique docs retrieved for {len(fields)} fields")
        return "\n\n".join(
            f"Source {i + 1} (relevance={d['score']}): {d['url']}\n{d['text'][:1500]}"
            for i, d in enumerate(docs[:max_docs])
        )

    def extract_entity_info(self, entity: Dict, entity_id: str) -> Dict:
        """Retrieve field-relevant context from the VDB and ask the LLM for all fields at once."""
        if not self.llm:
            return {"error": "LLM not configured"}
        config = load_config()
        fields = getattr(config, "EXTRACTION_FIELDS", []) or []
        context = self._retrieve_context(entity, entity_id, fields, top_k=4, max_docs=12)

        prompt_fn = getattr(config, "get_extraction_prompt", None)
        prompt = prompt_fn(entity, context) if prompt_fn else None
        if not isinstance(prompt, str) or not prompt.strip():
            # Config prompt missing or broken: fall back to a generic one built from the field list.
            wanted = {f["key"]: f.get("description", f["label"]) for f in fields} or {"info": "Any relevant information"}
            prompt = (f"Extract information about {get_entity_display(entity, config)} from these sources.\n\n"
                      f"{context}\n\nReturn ONLY a JSON object with these fields "
                      f'(use "Not available" if not found):\n{json.dumps(wanted, indent=2)}')
        prompt += ('\n\nAlso include a "_confidence" object scoring each field 0.0-1.0 '
                   "(1.0 explicit in sources, 0.7 strongly implied, 0.4 inferred, 0.0 not found).")

        try:
            result = chat_json(self.llm, self.model,
                               "You are a precise data extraction assistant. Return only valid JSON.",
                               prompt, max_tokens=900)
        except Exception as e:
            return {"error": str(e)}
        if result is None:
            return {"error": "LLM returned invalid JSON"}

        conf = result.get("_confidence") or {}
        if conf:
            result["_avg_confidence"] = round(sum(conf.values()) / len(conf), 2)
            result["_low_confidence_fields"] = [k for k, v in conf.items() if v < 0.5]
        return result

    def vdb_retry_missing_fields(self, entity: Dict, entity_id: str, info: Dict) -> Dict:
        """Re-query the VDB only for weak fields and let a focused LLM call fill them in."""
        if not self.llm or "error" in info:
            return info
        weak = {k for k, v in info.items() if not k.startswith("_") and _is_missing(v)}
        weak |= set(info.get("_low_confidence_fields", []))
        if not weak:
            return info

        config = load_config()
        by_key = {f["key"]: f for f in getattr(config, "EXTRACTION_FIELDS", []) or []}
        fields = [by_key.get(k, {"key": k, "label": k}) for k in sorted(weak)]
        context = self._retrieve_context(entity, entity_id, fields, top_k=5, max_docs=10)
        if not context:
            return info

        print(f"[RETRY] Re-extracting {sorted(weak)}")
        prompt = (f"Extract ONLY these fields for {get_entity_primary_value(entity, config)}: "
                  f"{', '.join(sorted(weak))}\n\nSOURCES:\n{context}\n\n"
                  'Return ONLY a JSON object mapping each field to its value or "Not available".')
        try:
            found = chat_json(self.llm, self.model,
                              "You extract specific missing fields. Return only valid JSON.", prompt, 400)
        except Exception as e:
            print(f"[WARN] Retry failed: {e}")
            return info
        for key, value in (found or {}).items():
            if key in weak and not _is_missing(value):
                info[key] = value
                print(f"[RETRY] filled {key}: {str(value)[:60]}")
        return info

    # ── orchestration ────────────────────────────────────────────────────────

    def process_entity(self, entity: Dict, on_event=None) -> Dict:
        """
        Full run for one entity. `on_event(type, data)` receives progress events
        (search, result, scraping, storing, extracting) for live UIs.
        """
        emit = on_event or (lambda *_: None)
        config = load_config()
        entity_id = get_entity_primary_value(entity, config)
        print(f"\n=== {get_entity_display(entity, config)} ===")

        docs, seen = [], set()
        queries = self.generate_search_queries(entity)
        for i, q in enumerate(queries):
            emit("search", {"description": q["query"][:60], "progress": int(i / len(queries) * 70)})
            results = self.search(q["query"])
            time.sleep(1)  # be polite to the search API
            if results:
                emit("result", {"count": len(results), "query_type": q["type"]})
            for r in results:
                if r["url"] in seen:
                    continue
                seen.add(r["url"])
                docs.append({"query_type": q["type"], "url": r["url"], "text": f"{r['title']}. {r['description']}"})
                # Full-page scrape only for the most promising query types, max 5 pages per entity.
                full_pages = sum(d["query_type"].endswith("_full") for d in docs)
                if q["type"] in ("profile", "site_specific") and full_pages < 5:
                    emit("scraping", {"url": r["url"][:60]})
                    page = self.scrape_content(r["url"])
                    if page:
                        docs.append({"query_type": f"{q['type']}_full", "url": r["url"], "text": page})

        emit("storing", {"count": len(docs)})
        self.store_in_vector_db(entity_id, docs)

        emit("extracting", {})
        info = self.extract_entity_info(entity, entity_id)
        info = self.vdb_retry_missing_fields(entity, entity_id, info)

        return {"entity": entity, "documents_collected": len(docs), "sources": docs, "extracted_info": info}
