#!/usr/bin/env python3
"""
Research Data Collection Pipeline
Collects detailed entity information from web sources using strategic search queries.
Dataset-agnostic: reads field names and search templates from pipeline_config.py.
"""

import os
import requests
from typing import List, Dict, Optional
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility
)
from openai import OpenAI
from bs4 import BeautifulSoup
import json
import time
from core.results_manager import ResultsManager
import importlib
from core.config_helpers import (
    load_config, get_entity_schema, get_primary_field,
    get_entity_primary_value, get_entity_display, get_entity_fields
)

load_dotenv()


# Known model configurations
# use_completion_tokens: True  → send max_completion_tokens instead of max_tokens
# env_keys: list of env var names to try in order (first non-empty wins)
MODEL_CONFIGS = {
    "deepseek-chat": {
        "provider": "deepseek",
        "base_url": "https://api.deepseek.com",
        "env_keys": ["DEEPSEEK_API_KEY", "DEEPSEEK_API"],  # support both naming conventions
        "label": "DeepSeek Chat (V3)",
        "use_completion_tokens": False,
    },
    "gpt-5-nano": {
        "provider": "openai",
        "base_url": None,
        "env_keys": ["OPENAI_API_KEY"],
        "label": "GPT-5 Nano",
        "use_completion_tokens": True,
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "base_url": None,
        "env_keys": ["OPENAI_API_KEY"],
        "label": "GPT-4o Mini",
        "use_completion_tokens": False,
    },
}

DEFAULT_MODEL = "deepseek-chat"


def _resolve_api_key(cfg: dict) -> Optional[str]:
    """Try each env var name in cfg['env_keys'] and return the first non-empty value."""
    for key_name in cfg.get("env_keys", [cfg.get("env_key", "")]):
        val = os.getenv(key_name)
        if val:
            return val
    return None


def get_available_models() -> List[Dict]:
    """Return list of models that have API keys configured."""
    available = []
    for model_id, cfg in MODEL_CONFIGS.items():
        if _resolve_api_key(cfg):
            available.append({"id": model_id, "label": cfg["label"], "provider": cfg["provider"]})
    return available


def build_llm_client(model_id: str):
    """Build an OpenAI-compatible client for the given model ID. Returns (client, model_id)."""
    cfg = MODEL_CONFIGS.get(model_id)
    if not cfg:
        raise ValueError(f"Unknown model: {model_id}")

    api_key = _resolve_api_key(cfg)
    if not api_key:
        tried = cfg.get("env_keys", [cfg.get("env_key", "")])
        raise ValueError(f"No API key found for {model_id} (tried: {tried})")

    kwargs = {"api_key": api_key}
    if cfg.get("base_url"):
        kwargs["base_url"] = cfg["base_url"]

    return OpenAI(**kwargs), model_id


def make_token_kwargs(model_id: str, n_tokens: int) -> dict:
    """
    Return the correct token-limit kwarg for a given model.
    Newer OpenAI reasoning/nano models require 'max_completion_tokens';
    all others use 'max_tokens'.
    """
    cfg = MODEL_CONFIGS.get(model_id, {})
    if cfg.get("use_completion_tokens"):
        return {"max_completion_tokens": n_tokens}
    return {"max_tokens": n_tokens}


class ResearchPipeline:
    """Pipeline for collecting research data about any entity type"""

    def __init__(self, collection_name: str = "research_data", model: str = DEFAULT_MODEL):
        self.collection_name = collection_name
        self.model = model
        self.embedding_model = None
        self.milvus_collection = None
        self.llm_client = None
        self.llm_model_id = model
        self.serper_api_key = os.getenv('SERPER_API_KEY') or os.getenv('SERPER')
        self.google_api_key = os.getenv('GOOGLE_CUSTOM_SEARCH_KEY')
        self.google_cse_id = os.getenv('GOOGLE_CSE_ID')
        self.search_count = 0

    def initialize(self):
        """Initialize all components"""
        print("[INIT] Loading sentence transformer model...")
        self.embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        print(f"[OK] Embedding model loaded (dim={self.embedding_model.get_sentence_embedding_dimension()})")

        try:
            self.llm_client, self.llm_model_id = build_llm_client(self.model)
            cfg = MODEL_CONFIGS.get(self.model, {})
            print(f"[OK] LLM client initialized: {cfg.get('label', self.model)}")
        except ValueError as e:
            print(f"[WARN] {e} — LLM extraction will be skipped")
            self.llm_client = None

        self._init_milvus()

    def _init_milvus(self):
        """Initialize Milvus vector database"""
        try:
            connections.connect("default", uri="./milvus_research.db")
            print("[OK] Connected to Milvus Lite")

            if utility.has_collection(self.collection_name):
                print(f"[INFO] Collection '{self.collection_name}' exists, using existing")
                self.milvus_collection = Collection(self.collection_name)
            else:
                embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="entity_id", dtype=DataType.VARCHAR, max_length=200),
                    FieldSchema(name="entity_meta", dtype=DataType.VARCHAR, max_length=500),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000),
                    FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=500),
                    FieldSchema(name="query_type", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)
                ]
                schema = CollectionSchema(fields, "Research data collection")
                self.milvus_collection = Collection(self.collection_name, schema)
                print(f"[OK] Created collection: {self.collection_name}")

                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128}
                }
                self.milvus_collection.create_index("embedding", index_params)
                print("[OK] Created vector index")

            self.milvus_collection.load()
            print("[OK] Collection loaded and ready")

        except Exception as e:
            print(f"[ERROR] Failed to initialize Milvus: {e}")
            raise

    def generate_search_queries(self, entity: Dict) -> List[Dict]:
        """
        Generate search queries using SEARCH_QUERY_TEMPLATES from pipeline_config.
        Falls back to generic queries built from entity field values.
        """
        config = load_config()
        templates = getattr(config, 'SEARCH_QUERY_TEMPLATES', []) if config else []

        if templates:
            queries = []
            for i, template in enumerate(templates):
                # Substitute all entity fields into the template
                # Support both {{field}} and {field} syntax for flexibility
                query = template
                for field_key, field_value in entity.items():
                    # Replace double-brace syntax {{field}}
                    query = query.replace(f'{{{{{field_key}}}}}', str(field_value))
                    # Also replace single-brace syntax {field}
                    query = query.replace(f'{{{field_key}}}', str(field_value))

                query_type = self._infer_query_type(query, i)
                queries.append({
                    'query': query,
                    'type': query_type,
                    'description': f'Search: {template[:60]}'
                })
            return queries

        # Fallback: build generic queries from entity field values
        primary = get_entity_primary_value(entity, config)
        parts = [f'"{v}"' for v in entity.values() if v]
        base_query = " ".join(parts)

        return [
            {'query': f'{base_query} profile', 'type': 'profile', 'description': 'Profile search'},
            {'query': f'{base_query} information', 'type': 'info', 'description': 'General information'},
            {'query': f'{base_query} contact', 'type': 'contact', 'description': 'Contact information'},
            {'query': f'"{primary}"', 'type': 'direct', 'description': 'Direct name search'},
        ]

    def _infer_query_type(self, query: str, index: int) -> str:
        """Infer query type from query content for scraping decisions."""
        query_lower = query.lower()
        if 'profile' in query_lower or 'cv' in query_lower or 'curriculum' in query_lower:
            return 'profile'
        elif 'email' in query_lower or 'contact' in query_lower:
            return 'contact'
        elif 'site:' in query_lower:
            return 'site_specific'
        elif index == 0:
            return 'profile'
        else:
            return f'search_{index}'

    def search_serper(self, query: str) -> List[Dict]:
        """Search using Serper API (primary, cheaper)"""
        if not self.serper_api_key:
            return []

        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": self.serper_api_key,
            "Content-Type": "application/json"
        }
        payload = {"q": query, "num": 3}

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            self.search_count += 1

            results = data.get('organic', [])
            print(f"[OK] Serper: found {len(results)} results (total searches: {self.search_count})")

            return [
                {
                    "title": r.get('title', 'N/A'),
                    "url": r.get('link', 'N/A'),
                    "description": r.get('snippet', 'N/A')
                }
                for r in results
            ]
        except Exception as e:
            print(f"[ERROR] Serper failed: {e}")
            return []

    def search_google(self, query: str) -> List[Dict]:
        """Search using Google Custom Search API (backup)"""
        if not self.google_api_key or not self.google_cse_id:
            return []

        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.google_api_key,
            "cx": self.google_cse_id,
            "q": query,
            "num": 3
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            self.search_count += 1

            items = data.get('items', [])
            print(f"[OK] Google CSE: found {len(items)} results (total searches: {self.search_count})")

            return [
                {
                    "title": item.get('title', 'N/A'),
                    "url": item.get('link', 'N/A'),
                    "description": item.get('snippet', 'N/A')
                }
                for item in items
            ]
        except Exception as e:
            print(f"[ERROR] Google CSE failed: {e}")
            return []

    def search(self, query: str) -> List[Dict]:
        """Execute search using Serper (primary) with Google backup"""
        if self.serper_api_key:
            results = self.search_serper(query)
            if results:
                return results
        
        if self.google_api_key and self.google_cse_id:
            return self.search_google(query)
        
        return []

    def is_url_worth_scraping(self, url: str, entity: Dict = None) -> bool:
        """
        Filter URLs to avoid scraping irrelevant content.
        Returns True if URL is likely to contain useful information.
        """
        url_lower = url.lower()
        
        # Reject non-HTML file types
        bad_extensions = [
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.zip', '.tar', '.gz', '.rar', '.7z',
            '.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp',
            '.mp3', '.mp4', '.avi', '.mov', '.wav',
            '.css', '.js', '.json', '.xml', '.csv', '.txt',
            '.woff', '.woff2', '.ttf', '.eot'
        ]
        for ext in bad_extensions:
            if url_lower.endswith(ext):
                print(f"[SKIP] Rejecting {ext} file: {url[:60]}...")
                return False
        
        # Reject known irrelevant domains/paths
        bad_patterns = [
            'cmu.edu/afs/',  # CMU file system dumps
            'cmu.edu/~ark/',  # CMU research data
            'mod9.io/',  # ASR model files
            'foundationcenter.org/990',  # Tax forms
            'nih.gov/reportweb/',  # NIH reports
            'scholarworks.wmich.edu/cgi/',  # Generic repository
            'asc41.org/wp-content/uploads/',  # Conference proceedings dumps
            '/bitstream/',  # Repository bitstreams
            '/viewcontent.cgi',  # Generic content viewers
            'data.mhec.state',  # Government data files
            'dhrm.virginia.gov',  # Government HR data
            'scotscoop.com',  # Unrelated news
            'hfcl.com',  # Unrelated company
        ]
        for pattern in bad_patterns:
            if pattern in url_lower:
                print(f"[SKIP] Rejecting irrelevant URL pattern: {url[:60]}...")
                return False
        
        # Prefer URLs from the target institution domain if we know it
        if entity:
            institution = entity.get('institution', '').lower()
            # Extract domain hints from institution name
            if 'amherst' in institution and 'amherst.edu' in url_lower:
                return True  # Prioritize institution URLs
        
        # Accept by default if no red flags
        return True

    def scrape_content(self, url: str, entity: Dict = None) -> Optional[str]:
        """Scrape content from URL with quality filtering"""
        # Pre-filter URLs
        if not self.is_url_worth_scraping(url, entity):
            return None
        
        try:
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response.raise_for_status()
            
            # Check content type - only process HTML
            content_type = response.headers.get('content-type', '').lower()
            if 'html' not in content_type and 'text' not in content_type:
                print(f"[SKIP] Non-HTML content type: {content_type[:30]} for {url[:50]}...")
                return None

            soup = BeautifulSoup(response.content, 'lxml')
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            text = soup.get_text(separator=' ', strip=True)
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)

            content = text[:5000]
            
            # Post-filter: reject if content is too short or looks like an error page
            if len(content) < 200:
                print(f"[SKIP] Content too short ({len(content)} chars): {url[:50]}...")
                return None
            
            print(f"[OK] Scraped {len(content)} chars from {url[:50]}...")
            return content

        except Exception as e:
            print(f"[WARN] Failed to scrape {url}: {type(e).__name__}")
            return None

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text"""
        embedding = self.embedding_model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def store_in_vector_db(self, entity_id: str, entity_meta: str,
                          texts: List[str], urls: List[str], query_types: List[str]):
        """Store entity data in Milvus"""
        if not texts:
            print("[WARN] No texts to store")
            return

        print(f"[INFO] Generating embeddings for {len(texts)} documents...")
        embeddings = [self.embed_text(text) for text in texts]

        entity_ids = [entity_id] * len(texts)
        entity_metas = [entity_meta] * len(texts)

        data = [entity_ids, entity_metas, texts, urls, query_types, embeddings]

        self.milvus_collection.insert(data)
        self.milvus_collection.flush()
        print(f"[OK] Stored {len(texts)} documents in Milvus")

    def semantic_search_vdb(self, entity_id: str, query: str, top_k: int = 5) -> List[Dict]:
        """
        Query Milvus for the most semantically relevant documents for this entity.
        Filters by entity_id so we only retrieve docs collected for this specific entity.
        """
        if not self.milvus_collection:
            return []

        try:
            query_embedding = self.embed_text(query)

            search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}

            results = self.milvus_collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=f'entity_id == "{entity_id}"',
                output_fields=["text", "url", "query_type"]
            )

            docs = []
            for hit in results[0]:
                docs.append({
                    "text": hit.entity.get("text", ""),
                    "url": hit.entity.get("url", ""),
                    "query_type": hit.entity.get("query_type", ""),
                    "score": round(hit.score, 4)
                })

            print(f"[VDB] Retrieved {len(docs)} docs for query: '{query[:50]}...'")
            return docs

        except Exception as e:
            print(f"[WARN] VDB search failed: {e}")
            return []

    def _build_field_retrieval_query(self, entity: Dict, field_cfg: Dict) -> str:
        """
        Build a semantic retrieval query for a specific field.
        Combines entity identity with field-specific keywords so the VDB
        returns the most relevant chunks for that field.
        """
        config = load_config()
        entity_name = get_entity_primary_value(entity, config)
        field_key = field_cfg.get('key', '')
        description = field_cfg.get('description', field_cfg.get('label', field_key))

        # Field-specific keyword hints for better semantic retrieval
        field_hints = {
            'phd_institution':       'PhD doctorate degree education university graduate',
            'bachelors_institution': "bachelor's undergraduate degree education college",
            'alumni_status':         'alumnus alumna graduated degree from',
            'institutional_email':   'email contact address @',
            'academic_rank':         'professor assistant associate rank title position',
            'gender':                'pronouns she her he his profile biography',
            'department_affiliations': 'department program faculty affiliation',
            'profile_link':          'profile page faculty staff people URL',
        }
        hints = field_hints.get(field_key, description)
        return f"{entity_name} {hints}"

    def _parse_llm_json(self, text: str) -> Optional[Dict]:
        """Strip markdown fences and parse JSON, returning None on failure."""
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def extract_entity_info(self, entity: Dict, entity_id: str) -> Dict:
        """
        Proper RAG extraction: for each field, query the VDB with a
        field-specific semantic query, then pass the top-k retrieved chunks
        to the LLM. This replaces the old approach of dumping all docs at once.
        """
        if not self.llm_client:
            return {"error": "LLM not configured"}

        config = load_config()
        entity_display = get_entity_display(entity, config)
        entity_name = get_entity_primary_value(entity, config)

        # Load extraction fields from config
        try:
            import pipeline_config
            importlib.reload(pipeline_config)
            extraction_fields = getattr(pipeline_config, 'EXTRACTION_FIELDS', [])
            get_prompt_fn = getattr(pipeline_config, 'get_extraction_prompt', None)
        except Exception:
            extraction_fields = []
            get_prompt_fn = None

        # --- RAG Pass 1: per-field VDB retrieval ---
        # Each extraction field gets its own semantic query so the most relevant
        # chunks are retrieved for that specific field, then deduplicated into a
        # single context window for the LLM.
        print(f"[RAG] Issuing {len(extraction_fields)} field-specific VDB queries for '{entity_display}'")
        field_contexts = {}
        for field_cfg in extraction_fields:
            retrieval_query = self._build_field_retrieval_query(entity, field_cfg)
            print(f"  [RAG] field='{field_cfg['key']}' query='{retrieval_query[:70]}'")
            docs = self.semantic_search_vdb(entity_id, retrieval_query, top_k=4)
            if docs:
                field_contexts[field_cfg['key']] = docs
                print(f"    → {len(docs)} docs retrieved (top score={docs[0]['score']})")
            else:
                print(f"    → 0 docs retrieved")

        # Deduplicate across fields and build a unified context string
        seen_urls = set()
        all_relevant_docs = []
        for field_key, docs in field_contexts.items():
            for doc in docs:
                url = doc.get('url', '')
                if url not in seen_urls:
                    seen_urls.add(url)
                    all_relevant_docs.append(doc)

        print(f"[RAG] {len(all_relevant_docs)} unique docs after dedup across {len(field_contexts)} fields")
        if not all_relevant_docs:
            print(f"[WARN] No VDB results for {entity_display}, extraction may be empty")

        context_str = "\n\n".join([
            f"Source {i+1} (relevance={doc.get('score', '?')}, type={doc['query_type']}): {doc['url']}\n{doc['text'][:1500]}"
            for i, doc in enumerate(all_relevant_docs[:12])
        ])

        # Build extraction prompt
        if get_prompt_fn:
            try:
                prompt = get_prompt_fn(entity, context_str)
            except TypeError:
                fields_list = get_entity_fields(config)
                secondary = entity.get(fields_list[1], '') if len(fields_list) > 1 else ''
                prompt = get_prompt_fn(entity_name, secondary, context_str)
        else:
            fields_json = json.dumps(
                {f['key']: f.get('description', f['label']) for f in extraction_fields},
                indent=4
            ) if extraction_fields else '{"info": "Any relevant information"}'
            prompt = f"""Extract information about {entity_display} from the following sources.

Sources:
{context_str}

Return ONLY a JSON object with these fields (use "Not available" if not found):
{fields_json}

Return ONLY valid JSON, no additional text."""

        # Append confidence scoring instruction
        prompt += """

For each field, also return a "_confidence" object with scores 0.0-1.0:
- 1.0: Explicitly stated in sources
- 0.7: Strongly implied
- 0.4: Inferred / uncertain
- 0.0: Not found

Example: {"field_name": "value", ..., "_confidence": {"field_name": 1.0, ...}}"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model_id,
                messages=[
                    {"role": "system", "content": "You are a precise data extraction assistant. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                **make_token_kwargs(self.llm_model_id, 900)
            )
            answer = response.choices[0].message.content
            print(f"[OK] LLM extraction completed for {entity_display}")

            result = self._parse_llm_json(answer)
            if result is None:
                print(f"[WARN] Failed to parse JSON response")
                return {"raw_response": answer}

            # Summarise confidence
            if '_confidence' in result:
                confs = result['_confidence']
                if confs:
                    result['_avg_confidence'] = round(sum(confs.values()) / len(confs), 2)
                    result['_low_confidence_fields'] = [k for k, v in confs.items() if v < 0.5]

            return result

        except Exception as e:
            print(f"[ERROR] LLM extraction failed: {e}")
            return {"error": str(e)}

    def vdb_retry_missing_fields(self, entity: Dict, entity_id: str,
                                  extracted_info: Dict) -> Dict:
        """
        For fields that are still missing/low-confidence after the first pass,
        re-query the VDB with more targeted queries and run a focused LLM call.
        No new web searches — uses only what's already stored.
        """
        if not self.llm_client:
            return extracted_info

        config = load_config()
        entity_name = get_entity_primary_value(entity, config)

        # Identify fields that need improvement
        missing = [
            k for k, v in extracted_info.items()
            if not k.startswith('_') and isinstance(v, str) and
            ('not available' in v.lower() or 'unable to retrieve' in v.lower())
        ]
        low_conf = extracted_info.get('_low_confidence_fields', [])
        fields_to_retry = list(set(missing + low_conf))

        if not fields_to_retry:
            return extracted_info

        print(f"\n[VDB-RETRY] Retrying {len(fields_to_retry)} fields from VDB: {fields_to_retry}")

        try:
            import pipeline_config
            importlib.reload(pipeline_config)
            extraction_fields = getattr(pipeline_config, 'EXTRACTION_FIELDS', [])
        except Exception:
            extraction_fields = []

        field_cfg_map = {f['key']: f for f in extraction_fields}

        # For each missing field, do a targeted VDB query (top_k=5 for broader coverage)
        retry_docs = []
        seen_urls = set()
        for field_key in fields_to_retry:
            field_cfg = field_cfg_map.get(field_key, {'key': field_key, 'label': field_key})
            retrieval_query = self._build_field_retrieval_query(entity, field_cfg)
            print(f"  [VDB-RETRY] field='{field_key}' query='{retrieval_query[:70]}'")
            docs = self.semantic_search_vdb(entity_id, retrieval_query, top_k=5)
            new_docs = 0
            for doc in docs:
                url = doc.get('url', '')
                if url not in seen_urls:
                    seen_urls.add(url)
                    retry_docs.append(doc)
                    new_docs += 1
            print(f"    → {new_docs} new docs (total pool: {len(retry_docs)})")

        if not retry_docs:
            print(f"  [VDB-RETRY] No additional docs found in VDB")
            return extracted_info

        context_str = "\n\n".join([
            f"Source {i+1} (score={doc.get('score', '?')}): {doc['url']}\n{doc['text'][:1500]}"
            for i, doc in enumerate(retry_docs[:10])
        ])

        focused_prompt = f"""Extract ONLY the following missing fields for {entity_name}.

SOURCES:
{context_str}

FIELDS TO EXTRACT: {', '.join(fields_to_retry)}

For each field, return the exact value found, or "Not available" if absent.
Return ONLY a JSON object:
{{
{chr(10).join(f'  "{f}": "[value or Not available]"' for f in fields_to_retry)}
}}"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model_id,
                messages=[
                    {"role": "system", "content": "You are extracting specific missing fields. Be thorough and precise. Return only valid JSON."},
                    {"role": "user", "content": focused_prompt}
                ],
                **make_token_kwargs(self.llm_model_id, 400)
            )
            answer = response.choices[0].message.content
            new_values = self._parse_llm_json(answer)

            if new_values:
                improvements = 0
                for field, value in new_values.items():
                    if field in fields_to_retry and isinstance(value, str):
                        if 'not available' not in value.lower() and 'unable to retrieve' not in value.lower():
                            extracted_info[field] = value
                            print(f"    [VDB-FILLED] {field}: {value[:60]}")
                            improvements += 1
                if improvements:
                    print(f"  VDB retry filled {improvements} fields")

        except Exception as e:
            print(f"  [WARN] VDB retry LLM call failed: {e}")

        return extracted_info

    def process_entity(self, entity: Dict, on_event=None) -> Dict:
        """
        Process a single entity end-to-end:
          1. Generate & execute search queries
          2. Scrape relevant URLs
          3. Store everything in Milvus
          4. RAG extraction: per-field VDB retrieval → LLM
          5. VDB retry for missing/low-confidence fields (no new web searches)

        Args:
            entity:   Dict of entity fields (e.g. {'name': '...', 'institution': '...'})
            on_event: Optional callback(event_type, data) for SSE streaming
        """
        def emit(event_type, data):
            if on_event:
                on_event(event_type, data)

        config = load_config()
        entity_display = get_entity_display(entity, config)
        entity_id = get_entity_primary_value(entity, config)
        entity_meta = json.dumps(entity)

        print(f"\n{'='*80}")
        print(f"Processing: {entity_display}")
        print(f"{'='*80}\n")

        search_queries = self.generate_search_queries(entity)
        total_queries = len(search_queries)

        all_documents = []
        texts_to_store = []
        urls_to_store = []
        query_types_to_store = []
        seen_urls: set = set()

        # ── Phase 1: Search + Scrape ──────────────────────────────────────────
        for q_idx, query_info in enumerate(search_queries):
            query = query_info['query']
            query_type = query_info['type']
            progress = int((q_idx / total_queries) * 70)

            print(f"\n[SEARCH] {query_info['description']}")
            print(f"  Query: '{query}'")
            emit('search', {'description': query_info['description'], 'progress': progress,
                            'total_searches': self.search_count})

            search_results = self.search(query)
            time.sleep(1)

            if search_results:
                emit('result', {'count': len(search_results), 'query_type': query_type})

            for result in search_results:
                url = result['url']
                text = f"{result['title']}. {result['description']}"

                # Deduplicate snippets
                if url not in seen_urls:
                    seen_urls.add(url)
                    texts_to_store.append(text)
                    urls_to_store.append(url)
                    query_types_to_store.append(query_type)
                    all_documents.append({"query_type": query_type, "url": url, "text": text})

                # Scrape full content for profile/CV/site-specific URLs (cap at 5 full pages)
                full_pages = sum(1 for d in all_documents if d['query_type'].endswith('_full'))
                if query_type in ['profile', 'cv', 'site_specific'] and full_pages < 5:
                    emit('scraping', {'url': url[:60]})
                    scraped = self.scrape_content(url, entity)
                    if scraped:
                        scrape_key = f"{url}::full"
                        if scrape_key not in seen_urls:
                            seen_urls.add(scrape_key)
                            qt_full = f"{query_type}_full"
                            texts_to_store.append(scraped)
                            urls_to_store.append(url)
                            query_types_to_store.append(qt_full)
                            all_documents.append({"query_type": qt_full, "url": url, "text": scraped})

        # ── Phase 2: Store in VDB ─────────────────────────────────────────────
        if texts_to_store:
            emit('storing', {'count': len(texts_to_store)})
            print(f"\n[STORE] Storing {len(texts_to_store)} documents in vector database...")
            self.store_in_vector_db(entity_id, entity_meta,
                                    texts_to_store, urls_to_store, query_types_to_store)

        # ── Phase 3: RAG Extraction (VDB → LLM) ──────────────────────────────
        emit('extracting', {})
        print(f"\n[EXTRACT] RAG extraction via VDB for {entity_display}...")
        extracted_info = self.extract_entity_info(entity, entity_id)

        # ── Phase 4: VDB Retry for missing/low-confidence fields ──────────────
        extracted_info = self.vdb_retry_missing_fields(entity, entity_id, extracted_info)

        return {
            "entity": entity,
            "documents_collected": len(all_documents),
            "sources": all_documents,
            "extracted_info": extracted_info,
            "total_searches": self.search_count,
        }

    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.milvus_collection:
                self.milvus_collection.release()
            connections.disconnect("default")
            print("\n[CLEANUP] Resources released")
        except:
            pass


def main():
    """Test with entities from pipeline_config"""
    from datetime import datetime

    config = load_config()
    if config and hasattr(config, 'TARGET_ENTITIES') and config.TARGET_ENTITIES:
        entity_list = config.TARGET_ENTITIES[:3]  # Test with first 3
    else:
        # Fallback test data
        entity_list = [
            {"name": "Francis Couvares", "institution": "Amherst College"},
            {"name": "Solsiree Del Moral", "institution": "Amherst College"},
            {"name": "Pawan Dhingra", "institution": "Amherst College"}
        ]

    pipeline = ResearchPipeline()
    results_manager = ResultsManager()

    try:
        # Create session folder
        session_folder = results_manager.create_session_folder()
        print(f"\n[SESSION] Results will be saved to: {session_folder}\n")

        start_time = time.time()
        pipeline.initialize()

        results = []
        log_entries = []
        all_sources = set()

        for entity in entity_list:
            result = pipeline.process_entity(entity)
            results.append(result)

            # Collect unique sources
            for source in result.get('sources', []):
                all_sources.add(source.get('url', ''))

            entity_display = get_entity_display(entity, config)

            # Add to log
            log_entries.append(f"\n{'='*80}")
            log_entries.append(f"PROCESSED: {entity_display}")
            log_entries.append(f"{'='*80}")
            log_entries.append(f"Documents collected: {result['documents_collected']}")
            log_entries.append(f"Extracted Info: {json.dumps(result['extracted_info'], indent=2)}")

            print("\n" + "="*80)
            print(f"RESULT FOR {entity_display}")
            print("="*80)
            print(f"Documents collected: {result['documents_collected']}")
            print(f"\nExtracted Information:")
            print(json.dumps(result['extracted_info'], indent=2))
            print("\n")

        end_time = time.time()
        processing_time = end_time - start_time

        # Save results
        print("\n[SAVING] Saving results...")

        # Compile stats
        stats = {
            'total_entities': len(results),
            'total_documents': sum(r['documents_collected'] for r in results),
            'unique_sources': len(all_sources),
            'total_searches': pipeline.search_count,
            'processing_time': f"{processing_time:.1f} seconds",
            'processing_time_seconds': processing_time,
            'avg_time_per_entity': f"{processing_time / len(results):.1f} seconds"
        }

        # Save all formats
        saved_files = results_manager.save_results(results, session_folder)
        stats_file = results_manager.save_stats(stats, session_folder)
        log_file = results_manager.save_session_log(log_entries, session_folder)
        summary_file = results_manager.create_summary_report(results, stats, session_folder)

        print(f"\n[COMPLETE] Processed {len(results)} entities")
        print(f"Total Google searches used: {pipeline.search_count}")
        print(f"Processing time: {processing_time:.1f} seconds\n")

        print("="*80)
        print("FILES SAVED")
        print("="*80)
        for file_type, file_path in saved_files.items():
            print(f"  {file_type}: {file_path}")
        print(f"  stats: {stats_file}")
        print(f"  log: {log_file}")
        print(f"  summary: {summary_file}")
        print("\n" + "="*80)
        print(f"Session folder: {session_folder}")
        print("="*80 + "\n")

    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()
